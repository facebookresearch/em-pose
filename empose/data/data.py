"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from empose.helpers.configuration import CONSTANTS as C


class ABatch(object):
    """Base class for batches consisting of real and synthetic samples."""

    def __init__(self, seq_ids, seq_lengths, poses, shapes, trans, joints_gt, offset_t=None, offset_r=None):
        """
        Initializer.
        :param seq_ids: A list of IDs.
        :param seq_lengths: The actual sequence length for each batch entry.
        :param poses: A tensor of shape (N, F, (C.N_JOINTS + 1)*3), i.e. SMPL body pose parameters in angle-axis format
          including the root orientation as the first three values.
        :param shapes: A tensor of shape (N, C.N_SHAPE_PARAMS) specifying the SMPL shape parameters.
        :param trans: A tensor of shape (N, F, 3) specifying the root translation.
        :param joints_gt: A tensor of shape (N, F, C.N_JOINTS*3) specifying the SMPL joint positions corresponding to
          the given `poses`.
        :param offset_t: Translational offsets per marker as a tensor of shape (1, N_MARKERS, 3) or (N, ...).
        :param offset_r: Rotational offsets per marker as a tensor of shape (1, N_MARKERS, 3, 3) or (N, ...).
        """
        self.ids = seq_ids
        self.seq_lengths = seq_lengths
        self.poses = poses  # (N, F, (C.N_JOINTS + 1)*3)
        self.shapes = shapes  # (N, C.N_SHAPE_PARAMS)
        self.trans = trans  # (N, F, 3)
        self.joints_gt = joints_gt  # (N, F, C.N_JOINTS*3)

        # These are the offsets that we assume are given.
        self.offset_t = offset_t
        self.offset_r = offset_r

        # The following members are quantities that are set externally. Typically they are computed by some transforms
        # when pre-processing a batch.
        self.joints_hat = None  # (N, F, C.N_JOINTS*3)
        self.vertices = None  # (N, F, V*3)

        self.marker_pos_real = None  # (N, F, N_MARKERS*3)
        self.marker_ori_real = None  # (N, F, N_MARKERS*9)
        self.marker_normal_real = None  # (N, F, N_MARKERS*3)

        self.marker_pos_synth = None  # (N, F, N_MARKERS*3)
        self.marker_ori_synth = None  # (N, F, N_MARKERS*9)
        self.marker_normal_synth = None  # (N, F, N_MARKERS*3)
        self.marker_pos_vertex = None  # (N, F, N_MARKERS*3), marker position without offset, i.e. on the mesh.
        self.marker_ori_vertex = None  # (N, F, N_MARKERS*9), marker orientation without offset.

        self.marker_masks = None  # (N, F, N_MARKERS) True if available

        # If data augmentation is used, the perturbed markers are stored here.
        self.marker_pos_noisy = None
        self.marker_ori_noisy = None
        self.marker_normal_noisy = None

        # Some data augmentation might introduce new offsets, which we can store here.
        self.offset_t_augmented = None
        self.offset_r_augmented = None

    @property
    def batch_size(self):
        """Mini-batch size."""
        return self.poses.shape[0]

    @property
    def seq_length(self):
        """The sequence length of this batch (i.e. including potential padding)."""
        return self.poses.shape[1]

    @property
    def n_markers(self):
        """Number of markers."""
        raise NotImplementedError("Must be implemented by subclass.")

    @property
    def poses_body(self):
        return self.poses[:, :, 3:]

    @poses_body.setter
    def poses_body(self, value):
        self.poses[:, :, 3:] = value

    @property
    def poses_root(self):
        return self.poses[:, :, :3]

    @poses_root.setter
    def poses_root(self, value):
        self.poses[:, :, :3] = value

    def get_inputs(self, sf=None, ef=None, **kwargs):
        raise NotImplementedError('Must be implemented by subclass.')

    @staticmethod
    def from_sample_list(samples):
        raise NotImplementedError('Must be implemented by subclass')


class RealSample(object):
    """
    This is a wrapper to store real sensor data with the corresponding ground-truth SMPL poses/shapes.
    This represents a single sequence. Although timestamps are supplied, they are irrelevent at this point because
    the sensor and SMPL data have already been aligned previously.
    """

    def __init__(self, seq_id, marker_pos, marker_ori, marker_masks, smpl_poses, smpl_shape, smpl_trans, offset_data):
        """
        Initializer.
        :param seq_id: An ID for this sample.
        :param marker_pos: A np array of shape (F, N_MARKERS*3) specifying the positional measurements of
          the EM sensors.
        :param marker_ori: A np array of shape (F, N_MARKERS*3*3) specifying the rotational measurements of
          the EM sensors.
        :param marker_masks: A boolean np array of shape (F, N_MARKERS) specifying for each marker and each frame
          whether data for that marker is available (True) or missing (False).
        :param smpl_poses: A np array of shape (F, (C.N_JOINTS + 1)*3) i.e. SMPL body pose parameters in angle-axis
          format including the root orientation as the first three values.
        :param smpl_shape: A np array of shape (N, ) specifying the SMPL shape parameters.
        :param smpl_trans:  A np array of shape (F, 3) specifying the root translation.
        :param offset_data: A dictionary with keys "means", "covs", and "r" representing the translational and
          rotational offset data per marker.
        """
        assert marker_pos.shape[0] == smpl_poses.shape[0]
        self.id = seq_id

        # Sensor data.
        f1 = marker_pos.shape[0]
        self.marker_pos_real = marker_pos.reshape(f1, -1)  # (F, N_MARKERS*3)
        self.marker_ori_real = marker_ori.reshape(f1, -1)  # (F, N_MARKERS*9)
        self.marker_masks = marker_masks  # (F, N_MARKERS) True if marker available

        # SMPL data.
        self.smpl_poses = smpl_poses  # (F, (C.N_JOINTS + 1) * 3)
        self.smpl_shape = smpl_shape  # (N_BETAS, )
        self.smpl_trans = smpl_trans  # (F, 3)

        # Offsets if they are available. Expected in same format as they are stored to .npz files.
        self.offset_means = offset_data['means']  # (M, 3)
        self.offset_covs = offset_data['covs']  # (M, 3, 3)
        self.offset_r = offset_data['r']  # (M, 3, 3)

    def extract_window(self, start_frame, end_frame):
        """Extract a subsequence from `start_frame` to `end_frame` (non-inclusive)."""
        sf, ef = start_frame, end_frame
        rs = RealSample(self.id, self.marker_pos_real[sf:ef], self.marker_ori_real[sf:ef],
                        self.marker_masks[sf:ef], self.smpl_poses[sf:ef], self.smpl_shape, self.smpl_trans[sf:ef],
                        {'means': self.offset_means, 'covs': self.offset_covs, 'r': self.offset_r})
        return rs

    @classmethod
    def from_npz_clean(cls, npz_file):
        assert npz_file.endswith("_clean.npz")
        data = np.load(npz_file)
        offset_data = {'means': data['offset_means'],
                       'covs': data['offset_covs'],
                       'r': data['offset_r']}
        obj = cls(data['id'].tolist(), data['sensor_pos'], data['sensor_oris'],
                  data['sensor_masks'], data['smpl_poses'], data['smpl_shape'], data['smpl_trans'],
                  offset_data)
        return obj

    @property
    def n_frames(self):
        """Return sequence lengths."""
        return self.marker_pos_real.shape[0]

    def to_tensor(self):
        """Convert numpy arrays to torch tensor."""
        # Marker data.
        self.marker_pos_real = torch.from_numpy(self.marker_pos_real).to(dtype=C.DTYPE)
        self.marker_ori_real = torch.from_numpy(self.marker_ori_real).to(dtype=C.DTYPE)
        self.marker_masks = torch.from_numpy(self.marker_masks).to(dtype=C.DTYPE)

        # SMPL data.
        self.smpl_poses = torch.from_numpy(self.smpl_poses).to(dtype=C.DTYPE)
        self.smpl_shape = torch.from_numpy(self.smpl_shape).to(dtype=C.DTYPE)
        self.smpl_trans = torch.from_numpy(self.smpl_trans).to(dtype=C.DTYPE)

        # Offsets.
        self.offset_means = torch.from_numpy(self.offset_means).to(dtype=C.DTYPE)
        self.offset_covs = torch.from_numpy(self.offset_covs).to(dtype=C.DTYPE)
        self.offset_r = torch.from_numpy(self.offset_r).to(dtype=C.DTYPE)


class RealBatch(ABatch):
    """
    Collate `RealSample`s into batches.
    """

    def __init__(self, seq_ids, seq_lengths, smpl_poses, smpl_shape, smpl_trans,
                 marker_pos, marker_ori, marker_masks, offset_t=None, offset_r=None):
        """
        Initializer.
        :param seq_ids: A list of IDs.
        :param seq_lengths: The actual sequence length for each batch entry.
        :param smpl_poses: A tensor of shape (N, F, (C.N_JOINTS + 1)*3), i.e. SMPL body pose parameters in angle-axis
          format including the root orientation as the first three values.
        :param smpl_shape: A tensor of shape (N, C.N_SHAPE_PARAMS) specifying the SMPL shape parameters.
        :param smpl_trans: A tensor of shape (N, F, 3) specifying the root translation.
        :param marker_pos: A np array of shape (N, F, N_MARKERS*3) specifying the positional measurements of
          the EM sensors.
        :param marker_ori: A np array of shape (N, F, N_MARKERS*3*3) specifying the rotational measurements of
          the EM sensors.
        :param marker_masks: A boolean np array of shape (N, F, N_MARKERS) specifying for each marker and each frame
          whether data for that marker is available (True) or missing (False).
        :param offset_t: Translational offsets per marker as a tensor of shape (1, N_MARKERS, 3) or (N, ...).
        :param offset_r: Rotational offsets per marker as a tensor of shape (1, N_MARKERS, 3, 3) or (N, ...).
        """
        super(RealBatch, self).__init__(seq_ids, seq_lengths, smpl_poses, smpl_shape, smpl_trans, joints_gt=None)
        self.marker_pos_real = marker_pos
        self.marker_ori_real = marker_ori
        self.marker_masks = marker_masks

        m_ori = marker_ori.detach().clone().reshape(self.batch_size, self.seq_length, -1, 3, 3)
        n_markers = m_ori.shape[2]
        self.marker_normal_real = m_ori[..., 2].reshape(self.batch_size, self.seq_length, -1)

        self.offset_t = torch.zeros((self.batch_size, n_markers, 3)) if offset_t is None else offset_t  # (N, M, 3)
        if offset_r is None:
            self.offset_r = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, n_markers, 1, 1)  # (N, M, 3, 3)
        else:
            self.offset_r = offset_r

    @property
    def n_markers(self):
        """Return number of markers."""
        return self.marker_pos_real.shape[-1] // 3

    @staticmethod
    def from_sample_list(samples):
        """Collect a set of `RealSample`s into a batch."""
        ids = []
        seq_lengths = []
        marker_pos = []
        marker_ori = []
        marker_masks = []
        poses = []
        shapes = []
        trans = []
        offset_t = []
        offset_r = []
        for sample in samples:
            ids.append(sample.id)
            seq_lengths.append(sample.smpl_poses.shape[0])
            marker_pos.append(sample.marker_pos_real)
            marker_ori.append(sample.marker_ori_real)
            marker_masks.append(sample.marker_masks)
            poses.append(sample.smpl_poses)
            shapes.append(sample.smpl_shape)
            trans.append(sample.smpl_trans)
            offset_t.append(sample.offset_means)
            offset_r.append(sample.offset_r)
        return RealBatch(ids, torch.from_numpy(np.array(seq_lengths)), pad_sequence(poses, batch_first=True),
                         pad_sequence(shapes, batch_first=True), pad_sequence(trans, batch_first=True),
                         pad_sequence(marker_pos, batch_first=True), pad_sequence(marker_ori, batch_first=True),
                         pad_sequence(marker_masks, batch_first=True),
                         torch.stack(offset_t), torch.stack(offset_r))

    def to_gpu(self):
        """Move data to GPU (if configured)."""
        self.seq_lengths = self.seq_lengths.to(dtype=torch.int, device=C.DEVICE)
        self.marker_pos_real = self.marker_pos_real.to(dtype=C.DTYPE, device=C.DEVICE)
        self.marker_ori_real = self.marker_ori_real.to(dtype=C.DTYPE, device=C.DEVICE)
        self.marker_normal_real = self.marker_normal_real.to(dtype=C.DTYPE, device=C.DEVICE)
        self.marker_masks = self.marker_masks.to(dtype=C.DTYPE, device=C.DEVICE)
        self.poses = self.poses.to(dtype=C.DTYPE, device=C.DEVICE)
        self.shapes = self.shapes.to(dtype=C.DTYPE, device=C.DEVICE)
        self.trans = self.trans.to(dtype=C.DTYPE, device=C.DEVICE)
        self.offset_t = self.offset_t.to(dtype=C.DTYPE, device=C.DEVICE)
        self.offset_r = self.offset_r.to(dtype=C.DTYPE, device=C.DEVICE)
        return self

    def _suppress_missing_markers(self, mask_value):
        """Make sure the values for missing markers are the same as when we train with marker suppression noise."""
        valid_mask = (self.marker_masks == 1.0).unsqueeze(-1)
        n, f, m = self.batch_size, self.seq_length, self.n_markers

        def _mask(x):
            xr = x.reshape((n, f, m, -1))
            xm = torch.zeros_like(xr) + mask_value
            xm = xr * valid_mask + xm * ~valid_mask
            return xm.reshape((n, f, -1))

        marker_pos_supp = _mask(self.marker_pos_real)
        self.marker_pos_real = marker_pos_supp

        marker_ori_supp = _mask(self.marker_ori_real)
        self.marker_ori_real = marker_ori_supp

        marker_nor_supp = _mask(self.marker_normal_real)
        self.marker_normal_real = marker_nor_supp

    def get_inputs(self, sf=None, ef=None, **kwargs):
        # For a batch of real data, we always feed the real data into the model and without noise.
        self._suppress_missing_markers(kwargs.get('mask_value', 0.0))
        return {'marker_pos': self.marker_pos_real[:, sf:ef], 'marker_oris': self.marker_ori_real[:, sf:ef],
                'marker_normals': self.marker_normal_real[:, sf:ef], 'joints': self.joints_hat[:, sf:ef],
                'offset_t': self.offset_t, 'offset_r': self.offset_r, 'marker_masks': self.marker_masks[:, sf:ef]}


class AMASSSample(object):
    """A single sequence from AMASS."""

    def __init__(self, id, poses, shape, trans, fps, joints=None, gender='unknown'):
        """
        Initializer.
        :param id: An ID for this sample.
        :param poses: A np array of shape (F, (C.N_JOINTS + 1)*3), i.e. SMPL body pose parameters in angle-axis format
          including the root orientation as the first three values.
        :param shape: A np array of shape ( C.N_SHAPE_PARAMS, ) specifying the SMPL shape parameters.
        :param trans: A np array of shape (F, 3) specifying the root translation.
        :param fps: The FPS of this AMASS sample.
        :param joints: A np array of shape (F, C.N_JOINTS*3) specifying the SMPL joint positions corresponding to
          the given `poses`.
        :param gender: 'female', 'male', or 'unknown'.
        """
        """Poses are expected to contain the root orientation as the first three entries."""
        assert poses.shape[1] >= C.MAX_INDEX_ROOT_AND_BODY
        self.id = id
        self.poses = poses  # (N_FRAMES, (C.N_JOINTS + 1) * 3)
        self.shape = shape  # (N_BETAS, )
        self.joints = None if joints is None else joints[:, :(C.N_JOINTS + 1)*3]  # (N_FRAMES, (C.N_JOINTS + 1) * 3)
        self.trans = trans  # (N_FRAMES, 3)
        self.fps = fps
        self.gender = gender

    @staticmethod
    def from_disk(sample_path, id):
        """Load an AMASS sample from an npz file."""
        raw_data = np.load(sample_path)
        sample = AMASSSample(id,
                             raw_data['poses'][:, :C.MAX_INDEX_ROOT_AND_BODY],  # (N_FRAMES, 66)
                             raw_data['betas'][:C.N_SHAPE_PARAMS],  # (N_SHAPE_PARAMS, )
                             raw_data['trans'],
                             raw_data['mocap_framerate'].tolist())
        return sample

    @property
    def n_frames(self):
        """Return number of frames."""
        return self.poses.shape[0]

    def to_tensor(self):
        """Convert np arrays to torch tensors."""
        self.poses = torch.from_numpy(self.poses).to(dtype=C.DTYPE)
        self.shape = torch.from_numpy(self.shape).to(dtype=C.DTYPE)
        self.trans = torch.from_numpy(self.trans).to(dtype=C.DTYPE)
        self.fps = torch.scalar_tensor(self.fps).to(dtype=C.DTYPE)
        self.joints = torch.from_numpy(self.joints).to(dtype=C.DTYPE) if self.joints is not None else None

    def extract_window(self, start_frame, end_frame):
        """Select a subsequence."""
        return AMASSSample(self.id, self.poses[start_frame:end_frame], self.shape, self.trans[start_frame:end_frame],
                           self.fps, self.joints[start_frame:end_frame if self.joints is not None else None],
                           self.gender)


class AMASSBatch(ABatch):
    """
    A mini-batch of 'synthetic' AMASS sequences, padded if necessary.
    """

    def __init__(self, seq_ids, seq_lengths, poses, shapes, trans, joints_gt, genders=None):
        """
        Initializer.
        :param seq_ids: An list of IDs identifying each batch entry.
        :param seq_lengths: A list of true sequence lengths for each batch entry.
        :param poses: A tensor of shape (N, F, (C.N_JOINTS + 1)*3), i.e. SMPL body pose parameters in angle-axis format
          including the root orientation as the first three values.
        :param shapes: A tensor of shape (N, C.N_SHAPE_PARAMS, ) specifying the SMPL shape parameters.
        :param trans: A tensor of shape (N, F, 3) specifying the root translation.
        :param joints_gt: A tensor of shape (N, F, C.N_JOINTS*3) specifying the SMPL joint positions corresponding to
          the given `poses`.
        :param genders: List of genders for each batch entry.
        """
        super(AMASSBatch, self).__init__(seq_ids, seq_lengths, poses, shapes, trans, joints_gt)
        self.marker_pos_noisy = None
        self.root_marker = None
        self.root_marker_ori = None
        self.vertex_ids = None
        self.genders = ['unknown' for _ in range(self.batch_size)] if genders is None else genders

    @staticmethod
    def from_sample_list(samples):
        """Collect a set of AMASSSamples into a batch with padding if necessary."""
        ids = []
        seq_lengths = []
        poses = []
        shapes = []
        trans = []
        joints = []
        genders = []
        for sample in samples:
            ids.append(sample.id)
            seq_lengths.append(sample.n_frames)
            poses.append(sample.poses)
            shapes.append(sample.shape)
            trans.append(sample.trans)
            joints.append(sample.joints)
            genders.append(sample.gender)
        shapes = pad_sequence(shapes, batch_first=True)
        poses = pad_sequence(poses, batch_first=True)
        trans = pad_sequence(trans, batch_first=True)
        joints = pad_sequence(joints, batch_first=True)
        seq_lengths = torch.from_numpy(np.array(seq_lengths))
        return AMASSBatch(ids, seq_lengths, poses, shapes, trans, joints, genders)

    def to_gpu(self):
        """Move data to GPU (if configured)."""
        self.seq_lengths = self.seq_lengths.to(device=C.DEVICE)
        self.poses = self.poses.to(device=C.DEVICE)
        self.shapes = self.shapes.to(device=C.DEVICE)
        self.trans = self.trans.to(device=C.DEVICE)
        self.joints_gt = self.joints_gt.to(device=C.DEVICE)
        return self

    @property
    def n_markers(self):
        """Return number of markers."""
        return self.marker_pos_synth.shape[-1] // 3

    def get_inputs(self, sf=None, ef=None, **kwargs):
        """Returns the inputs that we feed to a model."""
        inputs_ = {'marker_pos': None, 'marker_oris': None, 'marker_normals': None, 'joints': None}

        if self.marker_pos_noisy is not None:
            inputs_['marker_pos'] = self.marker_pos_noisy.detach()[:, sf:ef]
        elif self.marker_pos_synth is not None:
            inputs_['marker_pos'] = self.marker_pos_synth.detach()[:, sf:ef]

        if self.marker_ori_noisy is not None:
            inputs_['marker_oris'] = self.marker_ori_noisy.detach()[:, sf:ef]
        elif self.marker_ori_synth is not None:
            inputs_['marker_oris'] = self.marker_ori_synth.detach()[:, sf:ef]

        if self.marker_normal_noisy is not None:
            inputs_['marker_normals'] = self.marker_normal_noisy.detach()[:, sf:ef]
        elif self.marker_normal_synth is not None:
            inputs_['marker_normals'] = self.marker_normal_synth.detach()[:, sf:ef]

        joints = self.joints_gt

        inputs_['joints'] = joints[:, sf:ef]
        inputs_['offset_t'] = self.offset_t_augmented
        inputs_['offset_r'] = self.offset_r_augmented
        inputs_['marker_masks'] = None

        return inputs_
