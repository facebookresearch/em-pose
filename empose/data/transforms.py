"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import numpy as np
import torch
import quaternion

from empose.data.data import ABatch
from empose.data.data import RealSample
from empose.data.noise_functions import get_noise_fn
from empose.data.virtual_sensors import VirtualMarkerHelper
from empose.helpers.configuration import CONSTANTS as C
from empose.helpers.utils import get_all_offset_files
from empose.helpers.so3 import so3_exponential_map as aa2rot
from empose.helpers.so3 import so3_log_map as rot2aa
from torch.distributions.multivariate_normal import MultivariateNormal


def get_end_to_end_preprocess_fn(config, smpl_model, randomize_if_configured=False):
    """Factory function to return the preprocessing function depending on the given configuration."""
    normalize_root = NormalizeRoot()
    fk = SMPLFK(smpl_model)

    if config.use_real_offsets:
        noise_level = config.offset_noise_level if randomize_if_configured else -1
        sample_markers = SampleMarkersWithOffsets(smpl_model, list(get_all_offset_files().values()),
                                                  noise_level=noise_level)
    else:
        raise ValueError("We expect to use the real offsets.")

    noise_fn = get_noise_fn(config, randomize_if_configured)

    def _preprocess_fn(sample, mode='all', **noise_kwargs):
        if mode == 'all':
            sample = sample_markers(fk(normalize_root(sample)))
            return noise_fn(sample)
        elif mode == 'normalize_only':
            return normalize_root(sample)
        elif mode == 'after_normalize':
            return noise_fn(sample_markers(fk(sample)))
        else:
            raise ValueError("Mode '{}' unknown.".format(mode))

    return _preprocess_fn


class ToTensor(object):
    """Convert sample to torch tensors."""

    def __call__(self, sample):
        sample.to_tensor()
        return sample


class IdentityTransform(object):
    """Do nothing."""

    def __call__(self, batch):
        return batch


class ExtractWindow(object):
    """
    Extract a window of a fixed size. If the sequence is shorter than the desired window size it will return the
    entire sequence without any padding.
    """

    def __init__(self, window_size, rng=None, mode='random'):
        assert mode in ['random', 'beginning', 'middle']
        if mode == 'random':
            assert rng is not None
        self.window_size = window_size
        self.rng = rng
        self.mode = mode
        self.padding_value = 0.0

    def __call__(self, sample):
        if sample.n_frames > self.window_size:
            if self.mode == 'beginning':
                sf, ef = 0, self.window_size
            elif self.mode == 'middle':
                mid = sample.n_frames // 2
                sf = mid - self.window_size // 2
                ef = sf + self.window_size
            elif self.mode == 'random':
                sf = self.rng.randint(0, sample.n_frames - self.window_size + 1)
                ef = sf + self.window_size
            else:
                raise ValueError("Mode '{}' for window extraction unknown.".format(self.mode))
            return sample.extract_window(sf, ef)
        else:
            return sample


class NormalizeRealMarkers(object):
    """
    Normalize the sensor data.
    """

    @staticmethod
    def _normalize_points(points, root_trans, root_ori):
        """Normalize 3D points of shape (F, J, 3) with translations (F, J, 3) and rotations (F, J, 3)."""
        rs = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(root_ori))
        rs = np.transpose(rs, [0, 1, 3, 2])
        ps = points - root_trans
        ps = np.matmul(rs, ps[..., np.newaxis]).squeeze()
        return ps

    @staticmethod
    def _normalize_orientations(oris, root_ori):
        """Normalize 3D rotations of shape (F, J, 3, 3) with rotations (F, J, 3)."""
        rs = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(root_ori))
        rs = np.transpose(rs, [0, 1, 3, 2])
        os = np.matmul(rs, oris)
        return os

    def __call__(self, sample: RealSample):
        n_markers = sample.marker_pos_real.shape[-1] // 3
        root_ori = sample.smpl_poses[0:1, :3][:, np.newaxis, :]
        root_trans = sample.smpl_trans[:, np.newaxis, :]
        marker_pos = self._normalize_points(sample.marker_pos_real.reshape((-1, n_markers, 3)), root_trans, root_ori)
        marker_ori = self._normalize_orientations(sample.marker_ori_real.reshape((-1, n_markers, 3, 3)), root_ori)
        sample.marker_pos_real = marker_pos.reshape((-1, n_markers*3))
        sample.marker_ori_real = marker_ori.reshape((-1, n_markers*9))
        return sample


class SampleMarkersWithOffsets(object):
    """
    Sample virtual sensors and apply pre-defined estimated offsets.
    """

    def __init__(self, smpl_model, offset_files, noise_level=-1):
        self.smpl_model = smpl_model
        self.randomize = noise_level >= 0
        self.noise_level = noise_level

        if not isinstance(offset_files, list):
            offset_files = [offset_files]

        self.n_markers = np.load(offset_files[0])['means'].shape[0]
        self.n_offsets = len(offset_files)
        self.offset_means = np.zeros([self.n_offsets, self.n_markers, 3])
        self.offset_covs = np.zeros([self.n_offsets, self.n_markers, 3, 3])
        self.r = np.zeros([self.n_offsets, self.n_markers, 3, 3])
        offset_data = None
        for i, offset_file in enumerate(offset_files):
            offset_data = np.load(offset_file)
            self.offset_means[i] = offset_data['means']  # (M, 3)
            self.offset_covs[i] = offset_data['covs']  # (M, 3, 3)
            self.r[i] = offset_data['r']  # (M, 3, 3) mapping from local to global ori.

        self.normal_dists = MultivariateNormal(loc=torch.from_numpy(self.offset_means).to(dtype=torch.float32),
                                               covariance_matrix=torch.from_numpy(self.offset_covs).to(dtype=torch.float32))
        self.vertex_ids = offset_data['vertex_ids'].tolist()  # They are the same for all offsets.
        self.virtual_helper = VirtualMarkerHelper(smpl_model)
        self.offset_rng = np.random.RandomState(6273)

    def __call__(self, batch: ABatch):
        n, f = batch.batch_size, batch.seq_length
        vs = batch.vertices.reshape(n * f, -1, 3)
        markers, marker_oris, marker_normals = self.virtual_helper.get_virtual_pos_and_rot(vs, self.vertex_ids)

        # Store the local marker positions and orientations (certain models might have them as targets).
        batch.marker_pos_vertex = markers.clone().detach().reshape(n, f, -1)
        batch.marker_ori_vertex = marker_oris.clone().detach().reshape(n, f, -1)
        batch.marker_normal_vertex = marker_normals.clone().detach().reshape(n, f, -1)

        # Apply offsets, may be with noise.
        s_idxs = self.offset_rng.randint(0, self.n_offsets, n)
        offset_means = torch.from_numpy(self.offset_means[s_idxs]).to(dtype=markers.dtype)
        local_offsets = offset_means.clone().unsqueeze(1).repeat(1, f, 1, 1)
        s_idxs = torch.from_numpy(s_idxs).to(dtype=torch.long)
        if self.randomize:
            if self.noise_level == 0:
                offset_noise = self.normal_dists.sample((n, ))  # (N, N_OFFSETS, M, 3)
                offset_noise = offset_noise[torch.arange(n), s_idxs]  # (N, M, 3)
                local_offsets = offset_noise.unsqueeze(1).repeat(1, f, 1, 1)
            elif self.noise_level == 1:
                offset_noise = self.normal_dists.sample((n, f))  # (N, F, N_OFFSETS, M, 3)
                s = s_idxs.unsqueeze(-1).repeat(1, f).reshape(-1)
                offset_noise = offset_noise.reshape((n*f, self.n_offsets, -1, 3))[torch.arange(n*f), s]
                local_offsets = offset_noise.reshape((n, f, -1, 3))
            elif self.noise_level == 2 or self.noise_level == 3:
                local_offsets = torch.zeros_like(local_offsets)
            else:
                raise ValueError("Unknown noise level {}".format(self.noise_level))

        local_offsets = local_offsets.to(device=markers.device)

        # Apply offsets to marker position.
        ms = markers.reshape((n, f, -1, 3))
        ori_synth = marker_oris.reshape((n, f, -1, 3, 3))
        markers_new = ms + torch.matmul(ori_synth, local_offsets.unsqueeze(-1)).squeeze()
        batch.marker_pos_synth = markers_new.reshape((n, f, -1))

        # Apply offset to marker orientation.
        if isinstance(self.r, np.ndarray):
            self.r = torch.from_numpy(self.r).to(dtype=local_offsets.dtype, device=local_offsets.device)
        r = self.r[s_idxs].unsqueeze(1).repeat(1, f, 1, 1, 1)

        if self.randomize:
            if self.noise_level == 3:
                r = torch.zeros_like(r)
                r[:, :, :, 0, 0] = 1.0
                r[:, :, :, 1, 1] = 1.0
                r[:, :, :, 2, 2] = 1.0

        ori_synth = torch.matmul(ori_synth, r)
        marker_normals = ori_synth[..., 2]

        # Order is already correct since we load the vertex IDs from the offset file.
        batch.marker_pos_synth = markers_new.reshape(n, f, -1)
        batch.marker_ori_synth = ori_synth.reshape(n, f, -1)
        batch.marker_normal_synth = marker_normals.reshape(n, f, -1)

        # Store the information to revert synthetic offsets. We always take the mean of the offsets for this, since
        # this is the information we'll have during test as well.
        batch.offset_t_augmented = offset_means.clone().detach().to(device=markers.device)
        batch.offset_r_augmented = r[:, 0].clone().detach().to(device=markers.device)  # Take first frame.

        return batch


class NormalizeRoot(object):
    """
    Normalize the SMPL root such that there is no global translation and the first root orientation is always the
    identity.
    """

    def __init__(self, normalize_root_ori=True, remove_root_trans=True):
        self.normalize_root_ori = normalize_root_ori
        self.remove_root_trans = remove_root_trans

    def __call__(self, batch: ABatch):
        with torch.no_grad():
            batch.trans_source = batch.trans.clone()
            batch.root_pose_source = batch.poses_root.clone()

            if self.remove_root_trans:
                batch.trans = torch.zeros_like(batch.trans)

            if self.normalize_root_ori:
                root_pose = batch.poses_root
                n, f = root_pose.shape[0], root_pose.shape[1]
                root_ori = aa2rot(root_pose[:, 0:1].reshape(-1, 3)).reshape(n, 1, 3, 3)
                root_ori_inv = root_ori.transpose(2, 3).repeat(1, f, 1, 1)
                root_ori_all = aa2rot(root_pose.reshape(-1, 3)).reshape(n, f, 3, 3)
                new_root_ori = torch.matmul(root_ori_inv, root_ori_all)
                batch.poses_root = rot2aa(new_root_ori.reshape(-1, 3, 3)).reshape(n, f, -1)

        return batch


class SMPLFK(object):
    """
    A transform to perform forward kinematics with a given SMPL model. This sets (and possibly overrides) `joints_gt`
    and `vertices` on the sample.
    """

    def __init__(self, smpl_model):
        self.smpl_model = smpl_model
        self.max_window_size = 1000

    def __call__(self, batch: ABatch):
        n, f = batch.batch_size, batch.seq_length
        p = batch.poses_body.reshape([n * f, -1])
        s = batch.shapes.unsqueeze(1).repeat(1, f, 1).reshape([n * f, -1])
        r = batch.poses_root.reshape([n * f, -1])
        t = batch.trans.reshape([n * f, -1])
        vertices, joints = self.smpl_model(poses_body=p, betas=s, poses_root=r, trans=t,
                                           window_size=self.max_window_size)
        joints_body = joints[:, :(1 + C.N_JOINTS)]  # we don't want hands
        batch.joints_gt = joints_body.reshape(n, f, -1)
        batch.vertices = vertices.reshape(n, f, -1)
        batch.joints_hat = batch.joints_gt.clone().detach()  # IK models access `joints_hat` which is GT during training

        return batch
