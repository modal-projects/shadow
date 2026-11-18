"""Skinned triangle surfaces and the exact collision primitives used by the hand gates."""

from __future__ import annotations

from dataclasses import dataclass

import fcl
import numpy as np

from .pose import Array


@dataclass
class Mesh:
    vertices: Array
    faces: np.ndarray
    hand: str
    weights: Array
    labels: np.ndarray
    part: str = "hand"

    @property
    def normals(self) -> Array:
        a, b, c = self.vertices[self.faces].transpose(1, 0, 2)
        face = np.cross(b-a, c-a)
        n = np.zeros_like(self.vertices)
        for k in range(3):
            np.add.at(n, self.faces[:, k], face)
        return n/np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)


def collision_object(mesh: Mesh):
    bvh = fcl.BVHModel()
    bvh.beginModel(len(mesh.vertices), len(mesh.faces))
    bvh.addSubModel(mesh.vertices, mesh.faces)
    bvh.endModel()
    return fcl.CollisionObject(bvh)


def _inside(point, mesh):
    v = mesh.vertices[mesh.faces]-point
    norm = np.linalg.norm(v, axis=2)
    numerator = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
    denominator = (np.prod(norm, axis=1)
                   + np.einsum("ij,ij->i", v[:, 0], v[:, 1])*norm[:, 2]
                   + np.einsum("ij,ij->i", v[:, 1], v[:, 2])*norm[:, 0]
                   + np.einsum("ij,ij->i", v[:, 2], v[:, 0])*norm[:, 1])
    return abs(np.sum(2*np.arctan2(numerator, denominator))) > 2*np.pi
