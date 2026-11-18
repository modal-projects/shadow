"""Two-hand pose state: 2 x 23 joint angles (radians), wrist anchors and root quaternions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
SIDES = ("right", "left")


class InvalidPose(ValueError):
    """A refused action or invalid evidence, never a successful fallback."""


def finite_array(value: Any, shape: tuple[int, ...], name: str) -> Array:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise InvalidPose(f"{name} must be a numeric array of shape {shape}")
    def contains_bool(item: Any) -> bool:
        if isinstance(item, (list, tuple)):
            return any(contains_bool(child) for child in item)
        return isinstance(item, (bool, np.bool_))
    if contains_bool(value):
        raise InvalidPose(f"{name} must not contain booleans")
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise InvalidPose(f"{name} must be a numeric array of shape {shape}; booleans/strings are not numbers")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise InvalidPose(f"{name} must contain finite numbers")
    return result.copy()


@dataclass
class Pose:
    q: Array
    positions: Array
    quaternions: Array

    def copy(self) -> Pose:
        return Pose(self.q.copy(), self.positions.copy(), self.quaternions.copy())

    def dump(self) -> dict[str, Any]:
        return {"q": self.q.tolist(), "positions": self.positions.tolist(), "quaternions": self.quaternions.tolist()}

    @classmethod
    def parse(cls, value: dict[str, Any], nq: int = 46) -> Pose:
        if not isinstance(value, dict) or set(value) != {"q", "positions", "quaternions"}:
            raise InvalidPose("pose must have exactly q, positions and quaternions")
        pose = cls(finite_array(value["q"], (nq,), "q"),
                   finite_array(value["positions"], (2, 3), "positions"),
                   finite_array(value["quaternions"], (2, 4), "quaternions"))
        if not np.allclose(np.linalg.norm(pose.quaternions, axis=1), 1, atol=1e-8, rtol=0):
            raise InvalidPose("quaternions must be unit quaternions in w,x,y,z order")
        return pose


def interpolate(a: Pose, b: Pose, fraction: float) -> Pose:
    qa, qb = a.quaternions, b.quaternions.copy()
    dots = np.sum(qa * qb, axis=1)
    qb[dots < 0] *= -1
    theta = np.arccos(np.clip(np.abs(dots), 0, 1))
    out = np.empty_like(qa)
    for i in range(2):
        if theta[i] < 1e-7:
            out[i] = (1 - fraction) * qa[i] + fraction * qb[i]
        else:
            out[i] = (np.sin((1 - fraction) * theta[i]) * qa[i]
                      + np.sin(fraction * theta[i]) * qb[i]) / np.sin(theta[i])
        out[i] /= np.linalg.norm(out[i])
    return Pose(a.q + fraction * (b.q - a.q),
                a.positions + fraction * (b.positions - a.positions), out)
