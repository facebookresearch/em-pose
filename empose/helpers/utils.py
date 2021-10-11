"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""

import glob
import numpy as np
import os
import quaternion
import torch
import zipfile

from empose.helpers.configuration import CONSTANTS as C
from empose.helpers.so3 import so3_exponential_map as aa2rot
from empose.helpers.so3 import so3_log_map as rot2aa


def zip_files(file_list, output_file):
    """Stores files in a zip."""
    if not output_file.endswith('.zip'):
        output_file += '.zip'
    ofile = output_file
    counter = 0
    while os.path.exists(ofile):
        counter += 1
        ofile = output_file.replace('.zip', '_{}.zip'.format(counter))
    zipf = zipfile.ZipFile(ofile, mode="w", compression=zipfile.ZIP_DEFLATED)
    for f in file_list:
        zipf.write(f)
    zipf.close()


def get_model_dir(experiment_dir, model_id):
    """Return the directory in `experiment_dir` that contains the given `model_id` string."""
    model_dir = glob.glob(os.path.join(experiment_dir, str(model_id) + "-*"), recursive=False)
    return None if len(model_dir) == 0 else model_dir[0]


def create_model_dir(experiment_dir, experiment_id, model_summary, other_summary=None):
    """Create a new model directory."""
    model_name = "{}-{}".format(experiment_id, model_summary)
    if other_summary:
        model_name = '{}-{}'.format(model_name, other_summary)
    model_dir = os.path.join(experiment_dir, model_name)
    if os.path.exists(model_dir):
        raise ValueError("Model directory already exists {}".format(model_dir))
    os.makedirs(model_dir)
    return model_dir


def count_parameters(model):
    """Count number of trainable parameters in `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fix_quaternions(quats):
    """
    From https://github.com/facebookresearch/QuaterNet/blob/ce2d8016f749d265da9880a8dcb20a9be1a6d69c/common/quaternion.py

    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    :param quats: A numpy array of shape (F, N, 4).
    :return: A numpy array of the same shape.
    """
    assert len(quats.shape) == 3
    assert quats.shape[-1] == 4

    result = quats.copy()
    dot_products = np.sum(quats[1:] * quats[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def resample(poses, fps_in, fps_out):
    """
    Resample a motion sequence from `fps_in` to `fps_out`.
    :param poses: A numpy array of shape (F, N, 3), i.e. in angle-axis form.
    :param fps_in: The frequency of the input sequence.
    :param fps_out: The desired frequency of the output sequence.
    :return: A numpy array of shape (F', N, 3) where F is adjusted according to the new fps.
    """
    quats = quaternion.from_rotation_vector(poses)
    quats = quaternion.from_float_array(fix_quaternions(quaternion.as_float_array(quats)))
    n_frames = quats.shape[0]
    assert n_frames > 1, "We need at least two quaternions for a resampling to make sense."
    duration = n_frames / fps_in
    ts_in = np.arange(0, duration, 1/fps_in)[:n_frames]
    ts_out = np.arange(0, duration, 1/fps_out)

    poses_out = []
    for j in range(poses.shape[1]):
        quats_n = quaternion.squad(quats[:, j:j+1], ts_in, ts_out)
        poses_out.append(quaternion.as_rotation_vector(quats_n))

    return np.concatenate(poses_out, axis=1)


def mask_from_seq_lengths(seq_lengths, max_seq_len=None):
    """
    Creates a mask of shape (len(seq_lenghts), S) where S = max(seq_lengths) such that mask[i, j] == 1
    if mask[i, j] < seq_lenghts[i] and 0 otherwise. E.g. if seq_lengths is [4, 3, 2, 4]
    the mask will look like:
       [1, 1, 1, 1]
       [1, 1, 1, 0]
       [1, 1, 0, 0]
       [1, 1, 1, 1]
    :param seq_lengths: A tensor of integers.
    :param max_seq_len: If given will be used instead of max(seq_lengths).
    :return: The described mask as a tensor of shape (len(seq_lengths), max(seq_lengths)).
    """
    if max_seq_len is None:
        max_seq_len = torch.max(seq_lengths)
    n = seq_lengths.shape[0]
    t = torch.arange(max_seq_len).repeat(n, 1).to(dtype=seq_lengths.dtype, device=seq_lengths.device)
    mask = t < seq_lengths.unsqueeze(1)
    return mask


def compute_vertex_and_face_normals(vertices, faces, vertex_faces, normalize=False):
    """
    Compute (unnormalized) vertex normals for the given vertices.
    :param vertices: A tensor of shape (N, V, 3).
    :param faces: A tensor of shape (F, 3) indexing into `vertices`.
    :param vertex_faces: A tensor of shape (V, MAX_VERTEX_DEGREE) that lists the face IDs each vertex is a part of.
    :return: The vertex and face normals as tensors of shape (N, V, 3) and (N, F, 3) respectively.
    """
    vs = vertices[:, faces.to(dtype=torch.long)]
    face_normals = torch.cross(vs[:, :, 1] - vs[:, :, 0], vs[:, :, 2] - vs[:, :, 0], dim=-1)  # (N, F, 3)

    ns_all_faces = face_normals[:, vertex_faces]  # (N, V, MAX_VERTEX_DEGREE, 3)
    ns_all_faces[:, vertex_faces == -1] = 0.0
    vertex_degrees = (vertex_faces > -1).sum(dim=-1).to(dtype=ns_all_faces.dtype)
    vertex_normals = ns_all_faces.sum(dim=-2) / vertex_degrees[None, :, None]  # (N, V, 3)

    if normalize:
        face_normals = face_normals / torch.norm(face_normals, dim=-1).unsqueeze(-1)
        vertex_normals = vertex_normals / torch.norm(vertex_normals, dim=-1).unsqueeze(-1)

    return vertex_normals, face_normals


def get_all_offset_files():
    """Find all offset files and return them as a dictionary {subject_id -> path to offset file}."""
    offset_files = glob.glob(os.path.join(C.DATA_DIR_TEST, "*_offsets.npz"))
    subject_ids = [os.path.split(o)[-1].split('_')[0] for o in offset_files]
    return dict(zip(subject_ids, offset_files))


def global_oris_from_pose(pose_root, pose_body, smpl_parents, angle_idxs):
    """Get the global orientations from the given relative body joint angles."""
    n, f = pose_root.shape[0], pose_root.shape[1]
    poses = torch.cat([pose_root.reshape((n*f, -1)), pose_body.reshape((n*f, -1))], dim=-1)
    pose_global = local_to_global(poses, smpl_parents, output_format='rotmat')
    oris_global = pose_global.reshape((n, f, -1, 3, 3))[:, :, angle_idxs]
    return oris_global.reshape((n, f, -1))


def local_to_global(poses, parents, output_format='aa', input_format='aa'):
    """
    Convert relative joint angles to global ones by unrolling the kinematic chain.
    :param poses: A tensor of shape (N, N_JOINTS*3) defining the relative poses in angle-axis format.
    :param parents: A list of parents for each joint j, i.e. parent[j] is the parent of joint j.
    :param output_format: 'aa' or 'rotmat'.
    :param input_format: 'aa' or 'rotmat'
    :return: The global joint angles as a tensor of shape (N, N_JOINTS*DOF).
    """
    assert output_format in ['aa', 'rotmat']
    assert input_format in ['aa', 'rotmat']
    dof = 3 if input_format == 'aa' else 9
    n_joints = poses.shape[-1] // dof
    if input_format == 'aa':
        local_oris = aa2rot(poses.reshape((-1, 3)))
    else:
        local_oris = poses
    local_oris = local_oris.reshape((-1, n_joints, 3, 3))
    global_oris = torch.zeros_like(local_oris)

    for j in range(n_joints):
        if parents[j] < 0:
            # root rotation
            global_oris[..., j, :, :] = local_oris[..., j, :, :]
        else:
            parent_rot = global_oris[..., parents[j], :, :]
            local_rot = local_oris[..., j, :, :]
            global_oris[..., j, :, :] = torch.matmul(parent_rot, local_rot)

    if output_format == 'aa':
        global_oris = rot2aa(global_oris.reshape((-1, 3, 3)))
        res = global_oris.reshape((-1, n_joints * 3))
    else:
        res = global_oris.reshape((-1, n_joints * 3 * 3))
    return res
