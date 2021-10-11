"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import torch

from empose.helpers.utils import mask_from_seq_lengths


def padded_loss(gt, hat, loss_fn, seq_lengths):
    """Compute the `loss_fn` but respect the potential padding."""
    unreduced = loss_fn(gt, hat).mean(-1)
    # Compute the correct mean ignoring the padding.
    mask = mask_from_seq_lengths(seq_lengths).to(dtype=unreduced.dtype)
    n_frames = seq_lengths.to(dtype=unreduced.dtype)
    loss_per_sample = (unreduced * mask).sum(-1) / n_frames
    return loss_per_sample.mean()


def reconstruction_loss(markers_gt, markers_hat, seq_lengths=None, marker_mask=None):
    """If seq_lengths is given, the loss is respecting the padding. The per-joint error is summed over the joints
    and averaged over the sequence length."""
    # Markers expected in format (N, F, N_MARKERS, 3).
    diff = markers_hat - markers_gt
    marker_loss_per_sample = torch.sqrt((diff * diff).sum(dim=-1)).sum(dim=-1)

    # Marker mask is expected in format (N, F, M).
    if marker_mask is not None:
        assert len(marker_mask.shape) == 3
        frame_mask = marker_mask.logical_not().any(dim=-1).logical_not()  # (N, F)
        marker_loss_per_sample = marker_loss_per_sample * frame_mask

    if seq_lengths is not None:
        mask = mask_from_seq_lengths(seq_lengths).to(dtype=marker_loss_per_sample.dtype)
        n_frames = seq_lengths.to(dtype=marker_loss_per_sample.dtype)
        marker_loss_per_sample = (marker_loss_per_sample * mask).sum(-1) / n_frames

    return marker_loss_per_sample.mean()


def normal_mse(x_gt, x_hat, seq_lengths=None, marker_mask=None):
    """If seq_lengths is given, the loss is respecting the padding. The per-joint error is summed over the joints
    and averaged over the sequence length."""
    # Inputs expected in format (N, F, M, DOF).
    diff = x_hat - x_gt
    loss_per_sample = (diff * diff).sum(dim=-1).sum(dim=-1)  # (N, F)

    # Marker mask is expected in format (N, F, M).
    if marker_mask is not None:
        assert len(marker_mask.shape) == 3
        frame_mask = marker_mask.logical_not().any(dim=-1).logical_not()  # (N, F)
        loss_per_sample = loss_per_sample * frame_mask

    if seq_lengths is not None:
        mask = mask_from_seq_lengths(seq_lengths).to(dtype=loss_per_sample.dtype)
        n_frames = seq_lengths.to(dtype=loss_per_sample.dtype)
        loss_per_sample = (loss_per_sample * mask).sum(-1) / n_frames

    return loss_per_sample.mean()
