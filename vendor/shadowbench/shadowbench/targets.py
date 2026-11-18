"""Immutable source-derived silhouette tasks. No reference 3D pose is needed."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bench" / "books"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Task:
    task_id: str
    title: str
    mask: np.ndarray
    known: np.ndarray
    metadata: dict


def load(task_id: str, data: Path = DATA) -> Task:
    index = json.loads((data / "index.json").read_text())
    record = next((x for x in index["tasks"] if x["task_id"] == task_id), None)
    if record is None:
        raise ValueError(f"unknown v3 task {task_id}")
    path = data / task_id / "task.json"
    if digest(path) != record["sha256"]:
        raise ValueError("task metadata changed after dataset freeze")
    doc = json.loads(path.read_text())
    for f, sha in doc["files"].items():
        if digest(path.parent / f) != sha:
            raise ValueError(f"task asset changed: {f}")
    mask = np.array(Image.open(path.parent / "target.png").convert("L")) < 128
    known = np.array(Image.open(path.parent / "known.png").convert("L")) >= 128
    if mask.shape != (384, 512) or known.shape != mask.shape or not (mask & known).any():
        raise ValueError("invalid fixed target")
    mask.setflags(write=False); known.setflags(write=False)
    return Task(task_id, doc["title"], mask, known, doc)


def catalogue(data: Path = DATA) -> list[dict]:
    return json.loads((data / "index.json").read_text())["tasks"]
