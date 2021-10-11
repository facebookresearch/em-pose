"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import glob

import lmdb
import numpy as np
import os

from empose.data.data import AMASSSample
from empose.data.data import RealSample
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    """Access datasets that were stored in LMDB format."""

    def __init__(self, lmdb_path, transform):
        super(LMDBDataset).__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None

        self.open_lmdb()
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('__len__'.encode()).decode())
        self.env.close()
        self.env = None

    def open_lmdb(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, subdir=os.path.isdir(self.lmdb_path),
                                 readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.open_lmdb()
        poses_key = "poses{}".format(index).encode()
        betas_key = "betas{}".format(index).encode()
        trans_key = "trans{}".format(index).encode()
        joints_key = "joints{}".format(index).encode()
        n_frames_key = "n_frames{}".format(index).encode()
        id_key = "id{}".format(index).encode()
        gender_key = "gender{}".format(index).encode()
        with self.env.begin(write=False) as txn:
            n_frames = int(txn.get(n_frames_key).decode())
            sample = AMASSSample(id=txn.get(id_key).decode(),
                                 poses=np.frombuffer(txn.get(poses_key), dtype=np.float32).copy().reshape(n_frames, -1),
                                 shape=np.frombuffer(txn.get(betas_key), dtype=np.float32).copy(),
                                 joints=np.frombuffer(txn.get(joints_key), dtype=np.float32).copy().reshape(n_frames, -1),
                                 trans=np.frombuffer(txn.get(trans_key), dtype=np.float32).copy().reshape(n_frames, -1),
                                 fps=60.0,
                                 gender=txn.get(gender_key).decode())
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class RealDataset(Dataset):
    """Dataset class for our real dataset."""

    def __init__(self, base_path, transform=None):
        """
        Initializer.
        :param base_path: Path to npz files from our real dataset that should be loaded into this dataset.
        :param transform: Optional transform to be applied to every sample.
        """
        self.files = sorted(glob.glob(os.path.join(base_path, '*_clean.npz')))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        sample = RealSample.from_npz_clean(self.files[item])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
