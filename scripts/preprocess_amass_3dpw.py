"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import lmdb
import numpy as np
import os
import pickle as pkl
import quaternion
import torch

from empose.bodymodels.smpl import create_default_smpl_model
from empose.helpers.configuration import CONSTANTS as C
from empose.helpers.utils import fix_quaternions
from scipy.interpolate import CubicSpline
from pathlib import Path
from tqdm import tqdm


def get_all_valid_files(dir, is_valid_file, denylist):
    """Recursively finds all files in `dir` that pass the test provided by `is_valid_file` (callable) and that are
    not on the denylist."""
    dir = os.path.expanduser(dir)
    data_paths = []
    for root, dirs, f_names in os.walk(dir):
        dirs.sort()
        for f in sorted(f_names):
            if is_valid_file(f) and f not in denylist:
                path = Path(os.path.join(root, f)).resolve()
                data_paths.append(path)

    return data_paths


def get_all_amass_file_ids(amass_dir):
    """
    Create a training and a validation set for AMASS.
    :param amass_dir: Where the AMASS data is stored.
    :return: The list of filenames for training and validation.
    """
    all_ids = []
    all_paths = get_all_valid_files(amass_dir,
                                    lambda x: x.endswith('.npz') and not x.endswith('shape.npz'),
                                    denylist=['MTR03_poses.npz', 'WalkingStraightBackwards08_poses.npz'])
    for p in all_paths:
        # The ID of the sample is its path in the AMASS folder, but without the AMASS prefix.
        amass_path = Path(amass_dir)
        parts = p.parts
        base_path = Path(parts[0])
        for i in range(1, len(parts)):
            if base_path == amass_path:
                break
            base_path = base_path / parts[i]
        path_id = '/'.join(parts[i:])
        all_ids.append(path_id)

    return all_ids


def interpolate_rotations(rotations, ts_in, ts_out):
    """
    Interpolate rotations given at timestamps `ts_in` to timestamps given at `ts_out`. This performs the equivalent
    of cubic interpolation in SO(3).
    :param rotations: A numpy array of shape (F, N, 3), i.e. in angle-axis form.
    :param ts_in: Timestamps corresponding to the given rotations, len(ts_in) == F
    :param ts_out: The desired output timestamps.
    :return: A numpy array of shape (len(ts_out), N, 3).
    """
    quats = quaternion.from_rotation_vector(rotations)
    quats = quaternion.from_float_array(fix_quaternions(quaternion.as_float_array(quats)))
    rotations_out = []
    for j in range(rotations.shape[1]):
        quats_n = quaternion.squad(quats[:, j:j + 1], ts_in, ts_out)
        rotations_out.append(quaternion.as_rotation_vector(quats_n))
    return np.concatenate(rotations_out, axis=1)


def resample_rotations(rotations, fps_in, fps_out):
    """
    Resample a motion sequence from `fps_in` to `fps_out`.
    :param rotations: A numpy array of shape (F, N, 3), i.e. in angle-axis form.
    :param fps_in: The frequency of the input sequence.
    :param fps_out: The desired frequency of the output sequence.
    :return: A numpy array of shape (F', N, 3) where F is adjusted according to the new fps.
    """
    n_frames = rotations.shape[0]
    assert n_frames > 1, "We need at least two quaternions for a resampling to make sense."
    duration = n_frames / fps_in
    ts_in = np.arange(0, duration, 1 / fps_in)[:n_frames]
    ts_out = np.arange(0, duration, 1 / fps_out)
    return interpolate_rotations(rotations, ts_in, ts_out)


def interpolate_positions(positions, ts_in, ts_out):
    """
    Interpolate positions given at timestamps `ts_in` to timestamps given at `ts_out` with a cubic spline.
    :param positions: A numpy array of shape (F, N, 3), i.e. in angle-axis form.
    :param ts_in: Timestamps corresponding to the given positions, len(ts_in) == F
    :param ts_out: The desired output timestamps.
    :return: A numpy array of shape (len(ts_out), N, 3).
    """
    cs = CubicSpline(ts_in, positions, axis=0)
    new_positions = cs(ts_out)
    return new_positions


def resample_positions(positions, fps_in, fps_out):
    """
    Resample 3D positions from `fps_in` to `fps_out`.
    :param positions: A numpy array of shape (F, ...).
    :param fps_in: The frequency of the input sequence.
    :param fps_out: The desired output frequency.
    :return: A numpy array of shape (F', ...) where F is adjusted according to the new fps.
    """
    n_frames = positions.shape[0]
    assert n_frames > 1, "Resampling with one data point does not make sense."
    duration = n_frames / fps_in
    ts_in = np.arange(0, duration, 1 / fps_in)[:n_frames]
    ts_out = np.arange(0, duration, 1 / fps_out)
    return interpolate_positions(positions, ts_in, ts_out)


def convert_amass_to_lmdb(output_file, amass_root):
    """Convert AMASS to LMDB format that we can use during training."""
    print("Converting AMASS data under {} and exporting it to {} ...".format(amass_root, output_file))
    npz_file_ids = get_all_amass_file_ids(amass_root)
    smpl_layer = create_default_smpl_model(C.DEVICE)
    max_possible_len = 1000
    env = lmdb.open(output_file, map_size=1 << 33)
    cache = dict()

    for i in tqdm(range(len(npz_file_ids))):
        file_id = npz_file_ids[i]
        sample = np.load(os.path.join(amass_root, file_id))
        poses = sample['poses'][:, :C.MAX_INDEX_ROOT_AND_BODY]  # (N_FRAMES, 66)
        betas = sample['betas'][:C.N_SHAPE_PARAMS]  # (N_SHAPE_PARAMS, )
        trans = sample['trans']  # (N_FRAMES, 30)
        fps = sample['mocap_framerate'].tolist()
        gender = sample['gender'].tolist()
        n_frames = poses.shape[0]
        n_coords = poses.shape[1]

        # Resample to 60 FPS.
        poses = resample_rotations(poses.reshape(n_frames, -1, 3), fps, C.FPS).reshape(-1, n_coords)
        trans = resample_positions(trans, fps, C.FPS)

        # Extract joint information, watch out for CUDA out of memory.
        n_frames = poses.shape[0]
        n_shards = n_frames // max_possible_len
        joints = []
        for j in range(n_shards+1):
            sf = j*max_possible_len
            ef = None if j == n_shards else (j+1)*max_possible_len
            with torch.no_grad():
                ps = torch.from_numpy(poses[sf:ef]).to(dtype=torch.float32, device=C.DEVICE)
                ts = torch.from_numpy(trans[sf:ef]).to(dtype=torch.float32, device=C.DEVICE)
                bs = torch.from_numpy(betas).to(dtype=torch.float32, device=C.DEVICE)
                _, js = smpl_layer(poses_body=ps[:, 3:], betas=bs, poses_root=ps[:, :3], trans=ts)
                joints.append(js[:, :(1 + C.N_JOINTS)].reshape(-1, (1 + C.N_JOINTS)*3).detach().cpu().numpy())

        joints = np.concatenate(joints, axis=0)  # (N_FRAMES, 66)
        assert joints.shape[0] == n_frames

        if not isinstance(gender, str):
            gender = gender.decode()

        # Store.
        cache["poses{}".format(i)] = poses.astype(np.float32).tobytes()
        cache["betas{}".format(i)] = betas.astype(np.float32).tobytes()
        cache["trans{}".format(i)] = trans.astype(np.float32).tobytes()
        cache["joints{}".format(i)] = joints.astype(np.float32).tobytes()
        cache["n_frames{}".format(i)] = "{}".format(n_frames).encode()
        cache["id{}".format(i)] = file_id.encode()
        cache["gender{}".format(i)] = gender.encode()

        if i % 1000 == 0:
            with env.begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(k.encode(), v)
                cache = dict()
                torch.cuda.empty_cache()

    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)
        txn.put('__len__'.encode(), "{}".format(len(npz_file_ids)).encode())


def convert_3dpw_to_lmdb(output_file, threedpw_root):
    """Convert 3DPW to LMDB format that we can use during training."""
    print("Converting 3DPW data under {} and exporting it to {} ...".format(threedpw_root, output_file))
    smpl_layer = create_default_smpl_model(C.DEVICE)
    max_possible_len = 1000
    env = lmdb.open(output_file, map_size=1 << 29)
    cache = dict()

    # Find all pickle files.
    pkl_files = []
    for root_dir, dirs, files in os.walk(threedpw_root):
        for f in files:
            if f.endswith('.pkl'):
                pkl_files.append(os.path.join(root_dir, f))

    idx = 0
    for i in tqdm(range(len(pkl_files))):
        file_id = os.path.split(pkl_files[i])[-1]
        sample = pkl.load(open(pkl_files[i], 'rb'), encoding='latin1')
        n_subjects = len(sample['poses_60Hz'])

        for s in range(n_subjects):
            poses = sample['poses_60Hz'][s][:, :C.MAX_INDEX_ROOT_AND_BODY]  # (N_FRAMES, 66)
            betas = sample['betas'][s][:C.N_SHAPE_PARAMS]  # (N_SHAPE_PARAMS, )
            trans = sample['trans_60Hz'][s]  # (N_FRAMES, 3)
            gender = sample['genders'][s]
            n_frames = poses.shape[0]

            n_shards = n_frames // max_possible_len
            joints = []
            for j in range(n_shards+1):
                sf = j*max_possible_len
                ef = None if j == n_shards else (j+1)*max_possible_len
                with torch.no_grad():
                    ps = torch.from_numpy(poses[sf:ef]).to(dtype=torch.float32, device=C.DEVICE)
                    ts = torch.from_numpy(trans[sf:ef]).to(dtype=torch.float32, device=C.DEVICE)
                    bs = torch.from_numpy(betas).to(dtype=torch.float32, device=C.DEVICE)
                    _, js = smpl_layer(poses_body=ps[:, 3:], betas=bs, poses_root=ps[:, :3], trans=ts)
                    joints.append(js[:, :(1 + C.N_JOINTS)].reshape(-1, (1 + C.N_JOINTS)*3).detach().cpu().numpy())

            joints = np.concatenate(joints, axis=0)  # (N_FRAMES, 66)
            assert joints.shape[0] == n_frames

            gender = 'female' if gender == 'f' else 'male'

            # Store.
            cache["poses{}".format(idx)] = poses.astype(np.float32).tobytes()
            cache["betas{}".format(idx)] = betas.astype(np.float32).tobytes()
            cache["trans{}".format(idx)] = trans.astype(np.float32).tobytes()
            cache["joints{}".format(idx)] = joints.astype(np.float32).tobytes()
            cache["n_frames{}".format(idx)] = "{}".format(n_frames).encode()
            cache["id{}".format(idx)] = file_id.encode()
            cache["gender{}".format(idx)] = gender.encode()

            if idx > 0 and idx % 1000 == 0:
                with env.begin(write=True) as txn:
                    for k, v in cache.items():
                        txn.put(k.encode(), v)
                    cache = dict()
                    torch.cuda.empty_cache()

            idx += 1

    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)
        txn.put('__len__'.encode(), "{}".format(len(pkl_files)).encode())


if __name__ == '__main__':
    amass_in = os.path.join(C.DATA_DIR, "amass")
    amass_out = os.path.join(C.DATA_DIR, "amass_lmdb_test")
    convert_amass_to_lmdb(amass_out, amass_in)

    threedpw_in = os.path.join(C.DATA_DIR, "3dpw")
    threedpw_out = os.path.join(C.DATA_DIR, "3dpw_lmdb_test")
    convert_3dpw_to_lmdb(threedpw_out, threedpw_in)
