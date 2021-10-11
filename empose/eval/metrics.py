"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import numpy as np
import quaternion
import torch

from tabulate import tabulate

from empose.helpers.configuration import CONSTANTS as C
from empose.helpers.utils import mask_from_seq_lengths
from empose.helpers.utils import local_to_global


def _procrustes(X, Y, compute_optimal_scale=True):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Args
      X: array NxM of targets, with N number of points and M point dimensionality
      Y: array NxM of inputs
      compute_optimal_scale: whether we compute optimal scale or force it to be 1
    Returns:
      d: squared error after transformation
      Z: transformed Y
      T: computed rotation
      b: scaling
      c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)
    traceTA = s.sum()
    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX
    c = muX - b * np.dot(muY, T)
    return d, Z, T, b, c


class MetricsEngine(object):
    """Helper class to compute metrics over a dataset."""

    def __init__(self, smpl_model):
        """
        :param smpl_model: The SMPL model.
        """
        self.smpl_model = smpl_model
        self.eucl_dists = []  # list of Euclidean distance to ground-truth for each sample and each joint
        self.eucl_dists_pa = []  # Same as self.eucl_dists but Procrustes-aligned
        self.angle_diffs = []  # list of angular difference for each sample

        # List of joints to consider for either metric.
        self.eucl_eval_joints = ['root', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle',
                                 'spine3', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
                                 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
        self.angle_eval_joints = ['l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'spine3',
                                  'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
                                  'l_elbow', 'r_elbow']

        self.eucl_idxs = [C.SMPL_JOINTS.index(j) for j in self.eucl_eval_joints]

        # The pose vector does not contain the root, so adjust index by -1.
        self.angle_idxs = [C.SMPL_JOINTS.index(j)-1 for j in self.angle_eval_joints]
        self.angle_glob = True

    def reset(self):
        """Reset all computations."""
        self.eucl_dists = []
        self.eucl_dists_pa = []
        self.angle_diffs = []

    def _masked_flatten(self, t, mask):
        tt = t.masked_select(mask.unsqueeze(-1))
        return tt.reshape(-1, t.shape[-1])

    def _pad_shapes(self, s, n, mask):
        if len(s.shape) == 3:
            return self._masked_flatten(s, mask)
        return self._masked_flatten(s.unsqueeze(1).repeat(1, n, 1), mask)

    def _compute_eucl_dist(self, kp3d, kp3d_hat, procrustes=False):
        """
        Compute and store the Euclidean distance.
        :param kp3d: A tensor of shape (N, J, 3).
        :param kp3d_hat: A tensor of shape (N, J, 3).
        :param procrustes: Whether or not to align the keypoints with Procrustes analysis before computing the error.
        """
        if procrustes:
            # Align estimate with reference.
            kps_hat = kp3d_hat.cpu().detach().numpy()
            kps_gt = kp3d.cpu().detach().numpy()
            n, j = kps_hat.shape[0], kps_hat.shape[1]
            kps_hat_p = []
            for i in range(n):
                d, Z, T, b, c = _procrustes(kps_gt[i], kps_hat[i])
                kps_hat_p.append(Z)
            kps_hat_p = np.stack(kps_hat_p)
        else:
            kps_hat_p = kp3d_hat.cpu().detach().numpy()
            kps_gt = kp3d.cpu().detach().numpy()

        # Compute Euclidean distance and store it.
        diff = kps_gt - kps_hat_p
        eucl_dist = np.sqrt(np.sum(diff * diff, axis=-1))

        if procrustes:
            self.eucl_dists_pa.append(eucl_dist)
        else:
            self.eucl_dists.append(eucl_dist)

    def _compute_angular_dist(self, pose, pose_hat, rep='aa'):
        """
        Compute and store the angular error.
        :param pose: A tensor of shape (..., N_JOINTS*DOF).
        :param pose_hat: A tensor of the same shape as `pose`.
        :param rep: The representation of the input, 'aa' or 'rotmat'.
        """
        assert rep in ['aa', 'rotmat']
        dof = 3 if rep == 'aa' else 9
        n_joints = pose.shape[-1] // dof
        if rep == 'aa':
            pose = pose.reshape([-1, 3])
            pose_hat = pose_hat.reshape([-1, 3])
            ja = quaternion.from_rotation_vector(pose.detach().cpu().numpy())
            ja_hat = quaternion.from_rotation_vector(pose_hat.detach().cpu().numpy())
        else:
            pose = pose.reshape([-1, 3, 3])
            pose_hat = pose_hat.reshape([-1, 3, 3])
            ja = quaternion.from_rotation_matrix(pose.detach().cpu().numpy())
            ja_hat = quaternion.from_rotation_matrix(pose_hat.detach().cpu().numpy())
        angle_diff = quaternion.rotation_intrinsic_distance(ja, ja_hat)
        angle_diff = np.rad2deg(angle_diff).reshape(-1, n_joints)
        self.angle_diffs.append(angle_diff)

    def _get_mask(self, seq_lengths, n, f, frame_mask, device):
        """Return frame mask."""
        if seq_lengths is not None:
            # Throw away the entries that are padded.
            mask = mask_from_seq_lengths(seq_lengths)
        else:
            # Keep all entries as is.
            mask = torch.ones(n, f)
        mask = mask.to(dtype=torch.bool, device=device)

        if frame_mask is not None:
            frame_mask = frame_mask.to(dtype=torch.bool, device=device)
            if len(frame_mask.shape) == 3:
                frame_mask = frame_mask.logical_not().any(dim=-1).logical_not()
            else:
                assert len(frame_mask.shape) == 2
            mask = torch.logical_and(mask, frame_mask)
        return mask

    def compute(self, pose, shape, pose_hat, shape_hat=None, seq_lengths=None, pose_root=None, pose_root_hat=None,
                frame_mask=None):
        """
        Compute the metrics.
        :param pose: The ground-truth pose without the root as a tensor of shape (N, F, N_JOINTS*3)
        :param shape: The ground-truth shape as a tensor of shape (N, N_BETAS)
        :param pose_hat: The predicted pose.
        :param shape_hat: The predicted shape. If None the ground-truth shape is assumed.
        :param seq_lengths: An optional tensor of shape (N, ) indicating the true sequence length.
        :param pose_root: An optional tensor of shape (N, F, 3) indicating the ground-truth root pose.
        :param pose_root_hat: An optional tensor of shape (N, F, 3) indicating the estimated root pose.
        :param frame_mask: An optional boolean tensor of shape (N, F) or (N, F, M) indicating whether a frame
          should be considered in the evaluation or not. If the shape is (N, F, M) the last dimension will be
          reduced and the corresponding frame is not considered if any of the M dimensions is False.
        """
        n, f = pose.shape[0], pose.shape[1]

        if shape_hat is None:
            shape_hat = shape

        mask = self._get_mask(seq_lengths, n, f, frame_mask, pose.device)
        if mask.sum() == 0:
            return

        shape = self._pad_shapes(shape, f, mask)
        shape_hat = self._pad_shapes(shape_hat, f, mask)

        pose = self._masked_flatten(pose, mask)
        pose_hat = self._masked_flatten(pose_hat, mask)

        if pose_root is None:
            pose_root = torch.zeros([pose.shape[0], 3]).to(dtype=pose.dtype, device=pose.device)
            pose_root_hat = torch.zeros([pose.shape[0], 3]).to(dtype=pose.dtype, device=pose.device)
        else:
            pose_root = self._masked_flatten(pose_root, mask)
            pose_root_hat = self._masked_flatten(pose_root_hat, mask)

        # Get joint positions.
        _, kp3d = self.smpl_model.fk(pose, shape, poses_root=pose_root, window_size=1000)
        _, kp3d_hat = self.smpl_model.fk(pose_hat, shape_hat, poses_root=pose_root_hat, window_size=1000)

        # We're only interested in the body joints without hands.
        kp3d = kp3d[:, :C.N_JOINTS + 1]
        kp3d_hat = kp3d_hat[:, :C.N_JOINTS + 1]
        self._compute_eucl_dist(kp3d, kp3d_hat)
        self._compute_eucl_dist(kp3d, kp3d_hat, procrustes=True)

        if self.angle_glob:
            n = pose.shape[0]
            dummy_root = torch.zeros((n, 3)).to(dtype=pose.dtype, device=pose.device)
            pose_w_root = torch.cat([dummy_root, pose], dim=-1)
            pose_hat_w_root = torch.cat([dummy_root, pose_hat], dim=-1)

            pose_global = local_to_global(pose_w_root, C.SMPL_PARENTS)
            pose_hat_global = local_to_global(pose_hat_w_root, C.SMPL_PARENTS)

            self._compute_angular_dist(pose_global[:, 3:], pose_hat_global[:, 3:])
        else:
            self._compute_angular_dist(pose, pose_hat)

    def compute_joint_dist(self, joints, joints_hat, seq_lengths=None, frame_mask=None):
        """
        Compute only the metric on the 3D joint positions.
        :param joints: The ground-truth joint positions as a tensor of shape (N, F, N_JOINTS*3)
        :param joints_hat: The estimated joint positions as a tensor of shape (N, F, N_JOINTS*3)
        :param seq_lengths: An optional tensor of shape (batch_size, ) indicating the true sequence length.
        """
        n, f = joints.shape[0], joints.shape[1]
        mask = self._get_mask(seq_lengths, n, f, frame_mask, joints.device)
        if mask.sum() == 0:
            return

        js = self._masked_flatten(joints, mask)
        js_hat = self._masked_flatten(joints_hat, mask)
        kp3d = js.reshape(js.shape[0], -1, 3)
        kp3d_hat = js_hat.reshape(js_hat.shape[0], -1, 3)

        # We're only interested in the body joints without hands.
        kp3d = kp3d[:, :C.N_JOINTS + 1]
        kp3d_hat = kp3d_hat[:, :C.N_JOINTS + 1]

        self._compute_eucl_dist(kp3d, kp3d_hat)
        self._compute_eucl_dist(kp3d, kp3d_hat, procrustes=True)

    def compute_angle_dist(self, pose, pose_hat, seq_lengths=None, frame_mask=None, rep='aa'):
        """
        Compute only the metric on the joint angles.
        :param pose: The ground-truth pose without the root as a tensor of shape (N, F, N_JOINTS*DOF)
        :param pose_hat: The predicted pose.
        :param seq_lengths: An optional tensor of shape (N, ) indicating the true sequence length.
        :param frame_mask: An optional boolean tensor of shape (N, F) or (N, F, M) indicating whether a frame
          should be considered in the evaluation or not. If the shape is (N, F, M) the last dimension will be
          reduced and the corresponding frame is not considered if any of the M dimensions is False.
        :param rep: The representation of the input, 'aa' or 'rotmat'.
        """
        assert rep in ['aa', 'rotmat']
        n, f = pose.shape[0], pose.shape[1]

        mask = self._get_mask(seq_lengths, n, f, frame_mask, pose.device)
        if mask.sum() == 0:
            return

        p = self._masked_flatten(pose, mask)
        p_hat = self._masked_flatten(pose_hat, mask)
        self._compute_angular_dist(p, p_hat, rep=rep)

    def get_metrics(self, eucl_idxs_select=True, angle_idxs_select=True):
        """
        Compute the aggregated metrics that we want to report.
        :return: The computed metrics in a dictionary.
        """
        # Mean and median euclidean distance over all batches and joints.
        if len(self.eucl_dists) > 0:
            eucl_dists = np.concatenate(self.eucl_dists, axis=0)
            eucl_dists_pa = np.concatenate(self.eucl_dists_pa, axis=0)
            eucl_idxs = self.eucl_idxs if eucl_idxs_select else list(range(eucl_dists.shape[1]))

            eucl_mean_per_joint = np.mean(eucl_dists, axis=0)[eucl_idxs]
            eucl_mean_all = np.mean(eucl_mean_per_joint)
            eucl_std_all = np.std(eucl_dists[:, eucl_idxs])
            eucl_mean_pa_per_joint = np.mean(eucl_dists_pa, axis=0)[eucl_idxs]
            eucl_mean_pa_all = np.mean(eucl_mean_pa_per_joint)
            eucl_std_pa_all = np.std(eucl_dists_pa[:, eucl_idxs])
        else:
            eucl_mean_all = 0.0
            eucl_std_all = 0.0
            eucl_mean_pa_all = 0.0
            eucl_std_pa_all = 0.0

        # Mean and median angular difference.
        if len(self.angle_diffs) > 0:
            angle_diffs = np.concatenate(self.angle_diffs, axis=0)
            angle_idxs = self.angle_idxs if angle_idxs_select else list(range(angle_diffs.shape[1]))

            angle_mean_per_joint = np.mean(angle_diffs, axis=0)[angle_idxs]
            angle_mean_all = np.mean(angle_mean_per_joint)
            angle_std_all = np.std(angle_diffs[:, angle_idxs])
        else:
            angle_mean_all = 0.0
            angle_std_all = 0.0

        metrics = {'MPJPE [mm]': eucl_mean_all * 1000.0,
                   'MPJPE STD': eucl_std_all * 1000.0,
                   'PA-MPJPE [mm]': eucl_mean_pa_all * 1000.0,
                   'PA-MPJPE STD': eucl_std_pa_all * 1000.0,
                   'MPJAE [deg]': angle_mean_all,
                   'MPJAE STD': angle_std_all}
        return metrics

    @staticmethod
    def to_pretty_string(metrics, model_name):
        """Print the metrics onto the console, but pretty."""
        headers, values = [], []
        for k in metrics:
            headers.append(k)
            values.append(metrics[k])
        return tabulate([[model_name] + values], headers=['Model'] + headers)

    @staticmethod
    def to_tensorboard_log(metrics, writer, global_step, prefix=''):
        """Write metrics to tensorboard."""
        writer.add_scalar('metrics/{}/mje mean'.format(prefix), metrics['MPJPE [mm]'], global_step)
        writer.add_scalar('metrics/{}/mje pa mean'.format(prefix), metrics['PA-MPJPE [mm]'], global_step)
        writer.add_scalar('metrics/{}/mae mean'.format(prefix), metrics['MPJAE [deg]'], global_step)
