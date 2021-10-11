"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import functools
import torch
import numpy as np
import trimesh

from empose.helpers.utils import compute_vertex_and_face_normals


def _compute_local_coordinate_frames(vertices, vertex_normals, vertex_ids, vertex_helper_ids):
    """Compute a local coordinate frame at the given vertex_ids. The vertex IDs in `vertex_helper_ids`
    are used to construct a local coordinate system."""
    assert vertex_normals.shape[1] == len(vertex_ids)
    vs = vertices[:, vertex_ids]
    ns = vertex_normals

    normals = ns / torch.norm(ns, dim=-1, keepdim=True)
    on_surface = vertices[:, vertex_helper_ids] - vs
    on_surface = on_surface / torch.norm(on_surface, dim=-1, keepdim=True)

    third_axis = torch.cross(normals, on_surface)
    third_axis = third_axis / torch.norm(third_axis, dim=-1, keepdim=True)

    on_surface = torch.cross(third_axis, normals)
    on_surface = on_surface / torch.norm(on_surface, dim=-1, keepdim=True)

    local_rot = torch.zeros([normals.shape[0], len(vertex_ids), 3, 3]).to(device=vertices.device)
    local_rot[..., :, 0] = on_surface
    local_rot[..., :, 1] = third_axis
    local_rot[..., :, 2] = normals

    return local_rot


class VirtualMarkerHelper(object):
    """A helper to sample virtual position and orientation at certain vertex IDs given SMPL meshes."""

    def __init__(self, smpl_model):
        self.smpl_model = smpl_model

    @functools.lru_cache()
    def get_vertex_helpers(self, vertex_ids):
        """For each vertex, pick a random adjacent vertex."""
        faces = self.smpl_model.faces
        n_vertices = torch.max(faces).cpu().item() + 1
        vertex_faces = self.smpl_model.vertex_faces(n_vertices)
        _vertex_helpers = []
        for v in vertex_ids:
            for v_candidate in faces[vertex_faces[v, 0]]:
                if v_candidate != v:
                    _vertex_helpers.append(v_candidate.item())
                    break
        return _vertex_helpers

    @functools.lru_cache()
    def get_sub_faces(self, vertex_ids):
        """Get only those faces that contain a vertex that is included in `vertex_ids`."""
        v_ids = list(vertex_ids)
        # First get the regular vertex_faces.
        smpl_faces = self.smpl_model.faces.cpu().numpy()
        mesh = trimesh.Trimesh(np.zeros((smpl_faces.max()+1, 3)), smpl_faces, process=False)
        # Select only those faces that are connected to any of the given vertices.
        vertex_faces = mesh.vertex_faces[v_ids]
        face_ids = np.unique(vertex_faces[np.where(vertex_faces != -1)])
        faces = smpl_faces[face_ids]
        # Create another trimesh to get the new vertex_faces for the given vertices.
        mesh = trimesh.Trimesh(np.zeros((faces.max()+1, 3)), faces, process=False)
        vertex_faces = mesh.vertex_faces[v_ids]
        return torch.from_numpy(faces).to(dtype=torch.long), torch.from_numpy(vertex_faces).to(dtype=torch.long)

    def get_vertex_normals(self, vertices, vertex_ids):
        """Compute vertex normals."""
        # To make this a bit more efficient, only compute this on the selected vertices, not the entire SMPL Mesh.
        faces, vertex_faces = self.get_sub_faces(tuple(vertex_ids))
        normals, _ = compute_vertex_and_face_normals(vertices, faces.to(device=vertices.device),
                                                     vertex_faces.to(device=vertices.device))
        return normals

    def get_virtual_pos_and_rot(self, vertices, vertex_ids):
        """
        Get virtual tracker position and orientation at the provided vertex indices.
        :param vertices: A tensor of shape (N, V, 3).
        :param vertex_ids: A list of vertex IDs.
        :return: A tensor of shape (N, len(vertex_ids), 3) and (N, len(vertex_ids), 3, 3).
        """
        marker_normals = self.get_vertex_normals(vertices, vertex_ids)
        markers = vertices[:, vertex_ids]
        vertex_helpers = self.get_vertex_helpers(tuple(vertex_ids))
        marker_oris = _compute_local_coordinate_frames(vertices, marker_normals, vertex_ids, vertex_helpers)
        return markers, marker_oris, marker_normals
