"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import numpy as np
import torch

from empose.data.data import ABatch
from empose.helpers.configuration import CONSTANTS as C


def get_noise_fn(config, randomize_if_configured, is_valid=False):
    """Factory function for convenience."""
    def no_noise(x, **kwargs):
        return x

    if randomize_if_configured:
        if config.spherical_noise_length > 0.0:
            assert config.suppression_noise_length <= 0.0, "We currently only support one noise type."
            noise_fn = SphericalMarkerNoise(config.spherical_noise_strength, config.spherical_noise_length,
                                            config.noise_num_markers)
        elif config.suppression_noise_length > 0.0:
            noise_fn = MarkerSuppressionNoise(config.suppression_noise_length, config.noise_num_markers,
                                              config.suppression_noise_value, config.n_markers)
        else:
            noise_fn = no_noise
    else:
        if is_valid and config.suppression_noise_length > 0.0:
            noise_fn = MarkerSuppressionNoise(config.suppression_noise_length, config.noise_num_markers,
                                              config.suppression_noise_value, config.n_markers)
        else:
            noise_fn = no_noise

    return noise_fn


class SphericalMarkerNoise(object):
    """Perturb a single marker with spherical random displacements. The input sample already has to represent its
      internal data as pytorch tensors!"""

    def __init__(self, sphere_size, window_size, num_markers):
        """
        Initializer.
        :param sphere_size: A float between 0 and 1 indicating the diameter of the sphere. This percentage is relative
          to the size of the thigh bone. 0 means no perturbation, 1 means the diameter of the sphere is equal to
          the length of the thigh.
        :param window_size: A float between 0 and 1 indicating to temporal length of the perturbation. 0 means no
          perturbation, 1 means the entire sequence length.
        :param num_markers: How many markers are affected.
        """
        self.max_r = min(max(0.0, sphere_size), 1.0)
        self.ws = min(max(0.0, window_size), 1.0)
        if self.max_r > 0.0 and self.ws <= 0.0:
            raise ValueError("Temporal length of spherical marker noise is 0.0 but strength is > 0.0.")
        self.num_markers = num_markers
        self.rng = torch.Generator().manual_seed(98052)

    def __call__(self, batch: ABatch, **kwargs):
        if self.max_r <= 0.0:
            return batch
        if batch.marker_pos_synth is None:
            return batch

        markers = batch.marker_pos_synth
        batch_size, seq_len, n_markers = markers.shape[0], markers.shape[1], markers.shape[-1] // 3
        ms = markers.reshape(batch_size, seq_len, n_markers, 3)

        # Select `num_marker` many markers at random. For simplicity each batch entry is treated the same.
        m_ids = torch.randperm(n_markers, generator=self.rng)[:self.num_markers].to(device=markers.device)

        # Select perturbation window at random.
        window_len = int(self.ws * seq_len)
        sf = torch.randint(0, seq_len - window_len + 1, (batch_size,), generator=self.rng).to(device=markers.device)
        ef = sf + window_len

        # Choose perturbation radius at random.
        thigh_len = torch.norm(ms[0, seq_len // 2, C.T_TO_IDX_WO_ROOT[C.T_RUL]] - ms[0, 0, C.T_TO_IDX_WO_ROOT[C.T_RLL]])
        r = torch.rand(batch_size, window_len, self.num_markers).to(device=markers.device) * self.max_r * thigh_len / 2

        # Choose spherical coordinates at random.
        thetas = torch.rand((batch_size, window_len, self.num_markers), generator=self.rng).to(
            device=markers.device) * np.pi * 2
        phis = torch.rand((batch_size, window_len, self.num_markers), generator=self.rng).to(
            device=markers.device) * np.pi

        # Create displacement vector.
        xs = r * torch.cos(thetas) * torch.sin(phis)
        ys = r * torch.sin(thetas) * torch.cos(phis)
        zs = r * torch.cos(phis)

        xs = xs.to(dtype=ms.dtype, device=ms.device)
        ys = ys.to(dtype=ms.dtype, device=ms.device)
        zs = zs.to(dtype=ms.dtype, device=ms.device)

        # Apply displacement to the markers.
        # For now we're looping since this is easier then via advanced indexing.
        ms_noisy = ms.clone()
        for i in range(batch_size):
            ms_noisy[i, sf[i]:ef[i], m_ids, 0] += xs[i]
            ms_noisy[i, sf[i]:ef[i], m_ids, 1] += ys[i]
            ms_noisy[i, sf[i]:ef[i], m_ids, 2] += zs[i]

        batch.marker_pos_noisy = ms_noisy.reshape(batch_size, seq_len, -1)
        return batch


class MarkerSuppressionNoise(object):
    """Suppress a marker for a number of frames by setting it to 0."""

    def __init__(self, window_size, num_markers, mask_value, n_markers_in=12):
        """
        Initializer.
        :param window_size: A float between 0 and 1 indicating to temporal length of the perturbation. 0 means no
          perturbation, 1 means the entire sequence length.
        :param num_markers: How many markers are affected, currently not supported (i.e. just one marker affected).
        :param mask_value: A float used to represent the suppressed marker data.
        :param n_markers_in: How many markers are used on the inputs.
        """
        assert n_markers_in in [6, 12]
        self.ws = min(max(0.0, window_size), 1.0)
        self.rng = torch.Generator().manual_seed(8004)
        self.num_markers = num_markers
        self.mask_value = mask_value
        m_ids = C.S_CONFIG_6 if n_markers_in == 6 else list(range(12))
        self.marker_ids = torch.IntTensor(m_ids).to(dtype=torch.long, device=C.DEVICE)

    def reset_rng(self):
        """Resets the random number generator."""
        self.rng.manual_seed(8004)

    def __call__(self, batch: ABatch, **kwargs):
        if kwargs.get('reset_rng', False):
            self.reset_rng()

        markers = batch.marker_pos_synth
        n, f, m = markers.shape[0], markers.shape[1], markers.shape[-1] // 3
        ms = markers.reshape(n, f, m, 3)
        ms_ori = batch.marker_ori_synth.reshape(n, f, m, 3, 3)
        ms_normal = batch.marker_normal_synth.reshape(n, f, m, 3)

        # Select a marker at random per batch entry.
        m_ids = torch.randint(0, len(self.marker_ids), (n, self.num_markers), generator=self.rng)

        # Select perturbation window at random.
        window_len = int(self.ws * f)
        sf = torch.randint(0, f - window_len + 1, (n, ), generator=self.rng)
        ef = sf + window_len

        ms_noisy = ms.clone()
        ms_ori_noisy = ms_ori.clone()
        ms_normal_noisy = ms_normal.clone()

        for i in range(n):
            ms_noisy[i, sf[i]:ef[i], self.marker_ids[m_ids[i]]] = self.mask_value
            ms_ori_noisy[i, sf[i]:ef[i], self.marker_ids[m_ids[i]]] = self.mask_value
            ms_normal_noisy[i, sf[i]:ef[i], self.marker_ids[m_ids[i]]] = self.mask_value

        batch.marker_pos_noisy = ms_noisy.reshape(n, f, -1)
        batch.marker_ori_noisy = ms_ori_noisy.reshape(n, f, -1)
        batch.marker_normal_noisy = ms_normal_noisy.reshape(n, f, -1)
        return batch
