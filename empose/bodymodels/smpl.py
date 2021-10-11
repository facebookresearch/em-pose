"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import numpy as np
import os
import torch
import torch.nn as nn
import trimesh

from abc import ABC

from empose.helpers.utils import compute_vertex_and_face_normals
from empose.helpers.configuration import CONSTANTS as C
from empose.helpers.so3 import so3_exponential_map as aa2rot
from empose.helpers.so3 import so3_log_map as rot2aa
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_vposer as poser_loader


def create_default_smpl_model(device=C.DEVICE, vposer_path=None):
    """Factory method for convenience."""
    smpl_layer = SMPLLayer(os.path.join(C.SMPL_MODELS_DIR, 'smplh_amass/neutral/model.npz'), vposer_path)
    smpl_layer = smpl_layer.to(device=device, dtype=torch.float32)
    return smpl_layer


class SMPLLayer(nn.Module, ABC):
    def __init__(self, smpl_path, device=C.DEVICE, vposer_path=None):
        """
        Initializer.
        :param smpl_path: Path to where the SMPL model is stored on disk.
        :param device: The device (CPU/GPU) for torch.
        :param vposer_path: Path to a VPoser model. If supplied, SMPL poses can be encoded/decoded to/from VPoser's
          latent space.
        """
        super(SMPLLayer, self).__init__()
        self.num_betas = C.N_SHAPE_PARAMS
        self.bm = BodyModel(smpl_path, num_betas=self.num_betas, dtype=torch.float64)
        self.bm.to(device)
        self.vposer = None
        if vposer_path is not None:
            self.vposer, _ = poser_loader(vposer_path)
            self.vposer.to(device)
        self._vertex_faces = None
        self._faces = None

    @property
    def faces(self):
        """Return the definition of the faces."""
        if self._faces is None:
            self._faces = self.bm.f.to(dtype=torch.int32)
        return self._faces

    def vertex_faces(self, n_vertices):
        """For each vertex of the mesh defined by SMPL return a list of faces this vertex is a part of."""
        if self._vertex_faces is None:
            # Since this computation is expensive and the result does not change with the articulation of the
            # body model, we cash it.
            dummy_vertices = np.zeros((n_vertices, 3))
            mesh = trimesh.Trimesh(dummy_vertices, self.faces.cpu().numpy(), process=False)
            vf = mesh.vertex_faces.copy()
            self._vertex_faces = torch.from_numpy(vf).to(dtype=torch.long, device=self.faces.device)
        return self._vertex_faces

    def vertex_normals(self, vertices, output_vertex_ids=None):
        """
        Compute vertex normals. The normals are un-normalized because normalizing them (i.e. making unit length)
        is expensive and not always needed by the callee.
        :param vertices: A tensor of shape (N, V, 3).
        :param output_vertex_ids: Optional list of vertex indices to be picked..
        :return: A tensor of shape (N, V, 3) or (N, len(output_vertex_ids), 3).
        """
        normals, _ = compute_vertex_and_face_normals(vertices, self.faces, self.vertex_faces(vertices.shape[1]))
        ns = normals[:, output_vertex_ids] if output_vertex_ids is not None else normals
        return ns

    def _fk(self, poses_body, betas, poses_root=None, trans=None, normalize_root=False):
        """
        Evaluate the SMPL body model, i.e. compute joint and vertex positions given pose and shape parameters.
        :param poses_body: A tensor of shape (N, 21*3) specifying the body pose.
        :param betas: A tensor of shape (N, N_BETAS) specifying the SMPL parameters. If N_BETAS > self.num_betas, the
          excessive parameters are shaved off.
        :param poses_root: Root orientation as a tensor of shape (N, 3) or None.
        :param trans: Root translation as a tensor of shape (N, 3) or None.
        :param normalize_root: Normalize the root into the coordinate system defined by the first frame (i.e.
          root orientation at frame 0 is the identity and the root position is at the origin).
        :return: Vertex and joint positions of the posed SMPL mesh.
        """
        assert poses_body.shape[1] >= C.N_JOINTS * 3

        batch_size = poses_body.shape[0]
        device = poses_body.device

        # The body model expects hand poses, so supply a dummy value.
        poses_hands = torch.zeros([batch_size, C.N_JOINTS_HAND * 3 * 2]).to(dtype=poses_body.dtype, device=device)

        if poses_root is None:
            poses_root = torch.zeros([batch_size, 3]).to(dtype=poses_body.dtype, device=device)

        if trans is None:
            trans = torch.zeros([batch_size, 3]).to(dtype=poses_body.dtype, device=device)

        # Broadcast shape parameters.
        if len(betas.shape) == 1 or betas.shape[0] == 1:
            betas = betas.repeat(poses_body.shape[0], 1)
        betas = betas[:, :self.num_betas]

        if normalize_root:
            # Make everything relative to the first root orientation.
            root_ori = aa2rot(poses_root)
            first_root_ori = torch.inverse(root_ori[0:1])
            root_ori = torch.matmul(first_root_ori, root_ori)
            poses_root = rot2aa(root_ori)
            trans = torch.matmul(first_root_ori.unsqueeze(0), trans.unsqueeze(-1)).squeeze()
            trans = trans - trans[0:1]

        body = self.bm(root_orient=poses_root, pose_body=poses_body, betas=betas, pose_hand=poses_hands, trans=trans)
        return body.v, body.Jtr

    def fk(self, poses_body, betas, poses_root=None, trans=None, normalize_root=False, window_size=None):
        """Wrapper for self._fk to support windowed evaluation if required."""
        if window_size is not None:
            if normalize_root:
                raise ValueError("Are you sure you want to use root normalization with windowed evaluation?")

            n = poses_body.shape[0]
            n_windows = n // window_size + int(n % window_size > 0)

            vs, js = [], []
            for i in range(n_windows):
                sf = i * window_size
                ef = min((i + 1) * window_size, n)
                r = poses_root[sf:ef] if poses_root is not None else None
                t = trans[sf:ef] if trans is not None else None
                vertices, joints = self._fk(poses_body=poses_body[sf:ef], betas=betas[sf:ef], poses_root=r, trans=t,
                                            normalize_root=normalize_root)
                vs.append(vertices)
                js.append(joints)

            return torch.cat(vs, dim=0), torch.cat(js, dim=0)

        else:
            return self._fk(poses_body, betas, poses_root, trans, normalize_root)

    def vposer_decode(self, poZ_body):
        """Decode VPoser latent space to SMPL."""
        assert self.vposer is not None
        pose = self.vposer.decode(poZ_body, output_type='aa').view(poZ_body.shape[0], -1)
        return pose

    def vposer_encode(self, pose_body):
        """Encode SMPL to VPoser latent space."""
        assert self.vposer is not None
        poZ_body = self.vposer.encode(pose_body)
        return poZ_body

    def forward(self, *args, **kwargs):
        """
        The forward pass just performs forward kinematics.
        """
        return self.fk(*args, **kwargs)
