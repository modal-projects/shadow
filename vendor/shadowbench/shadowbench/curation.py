"""Book-only vector drafts and explicit, immutable human-reviewed target revisions."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Annotated, Literal
import uuid
import xml.etree.ElementTree as ET

import cv2
from filelock import FileLock
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .targets import DATA, ROOT, Task, digest

DEFAULT_REVIEWS = ROOT / "data" / "bench" / "targets"
REVIEWS = Path(os.environ.get("SHADOWBENCH_REVIEWS", DEFAULT_REVIEWS)).expanduser().resolve()
ATTESTATION = (
    "I reviewed the complete outline and negative spaces against the original book figure, "
    "including any manually reconstructed regions. I approve this full-frame silhouette "
    "as a benchmark reference, not as proof of a historical 3D pose or human reachability."
)
Coordinate = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-8192, le=8192)]
Point = tuple[Coordinate, Coordinate]


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")
    point: Point
    incoming: Point | None = None
    outgoing: Point | None = None
    mode: Literal["corner", "smooth", "symmetric"] = "corner"


class Contour(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80, pattern=r"^[\w-]+$")
    kind: Literal["ink", "cutout"] = "ink"
    closed: bool = Field(default=True, strict=True)
    nodes: list[Node] = Field(default_factory=list, max_length=10000)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    width: Literal[512] = 512
    height: Literal[384] = 384
    paths: list[Contour] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def check_size(self):
        if sum(len(p.nodes) for p in self.paths) > 10000:
            raise ValueError("at most 10000 nodes per drawing")
        if len({p.id for p in self.paths}) != len(self.paths):
            raise ValueError("contour identifiers must be unique")
        return self


class SaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_revision: str | None = None
    document: Document
    certify: bool = Field(default=False, strict=True)
    reviewer: str = Field(default="", max_length=120)
    notes: str = Field(default="", max_length=4000)
    attestation: bool = Field(default=False, strict=True)


class RevisionConflict(ValueError):
    pass


def book_records() -> list[dict]:
    index = json.loads((DATA / "index.json").read_text())
    return sorted(
        index["tasks"] + index["excluded"],
        key=lambda r: r["task_id"],
    )


def source_metadata(task_id: str) -> dict:
    record = next((r for r in book_records() if r["task_id"] == task_id), None)
    if record is None:
        raise ValueError("unknown book plate")
    path = DATA / task_id / "task.json"
    if digest(path) != record["sha256"]:
        raise ValueError("original book metadata changed after its archive was frozen")
    doc = json.loads(path.read_text())
    for name in ("source.png", "target.svg"):
        if digest(path.parent / name) != doc["files"][name]:
            raise ValueError(f"original book asset changed: {name}")
    return doc


def initial_document(task_id: str) -> Document:
    """Import only the archived M/L/Z trace; never complete its obscured contours."""
    source_metadata(task_id)
    root = ET.fromstring((DATA / task_id / "target.svg").read_text())
    paths: list[Contour] = []
    polygons = []
    for element in root.iter("{http://www.w3.org/2000/svg}path"):
        tokens = re.findall(r"[MLZ]|[-+]?(?:\d*\.?\d+)(?:[eE][-+]?\d+)?", element.attrib["d"])
        nodes: list[Node] = []
        i = 0
        while i < len(tokens):
            command = tokens[i]
            i += 1
            if command in ("M", "L"):
                nodes.append(Node(point=(float(tokens[i]), float(tokens[i + 1]))))
                i += 2
            elif command == "Z":
                polygons.append(np.array([n.point for n in nodes], dtype=np.float32))
                paths.append(Contour(id=f"trace-{len(paths)}", nodes=nodes))
                nodes = []
            else:
                raise ValueError("unexpected command in the archived source trace")
        if nodes:
            raise ValueError("archived source trace has an unfinished contour")
    # Preserve nested negative spaces while making each contour independently editable.
    depths = [
        sum(cv2.pointPolygonTest(other, tuple(map(float, polygon[0])), False) > 0
            for j, other in enumerate(polygons) if i != j)
        for i, polygon in enumerate(polygons)
    ]
    for i, path in enumerate(paths):
        path.kind = "cutout" if depths[i] % 2 else "ink"
    return Document(paths=[paths[i] for i in sorted(range(len(paths)), key=lambda i: depths[i])])


def path_data(path: Contour) -> str:
    def point(p):
        return f"{p[0]:.6f},{p[1]:.6f}"
    if not path.nodes:
        return ""
    commands = ["M " + point(path.nodes[0].point)]
    count = len(path.nodes) if path.closed else len(path.nodes) - 1
    for i in range(count):
        a, b = path.nodes[i], path.nodes[(i + 1) % len(path.nodes)]
        if a.outgoing is None and b.incoming is None:
            commands.append("L " + point(b.point))
        else:
            commands.append("C " + " ".join(point(p) for p in (
                a.outgoing or a.point, b.incoming or b.point, b.point)))
    if path.closed:
        commands.append("Z")
    return " ".join(commands)


def svg_bytes(document: Document) -> bytes:
    paths = "".join(
        f'<path fill="{"black" if p.kind == "ink" else "white"}" d="{path_data(p)}"/>'
        for p in document.paths if p.closed and len(p.nodes) >= 3
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="384" viewBox="0 0 512 384">'
        '<rect width="512" height="384" fill="white"/>' + paths + "</svg>"
    ).encode()


def render_target(document: Document) -> Image.Image:
    import cairosvg  # only needed to save new drawings: pip install cairosvg

    rendered = cairosvg.svg2png(bytestring=svg_bytes(document), output_width=1024, output_height=768)
    if not isinstance(rendered, bytes):
        raise RuntimeError("SVG renderer did not return a PNG")
    with Image.open(io.BytesIO(rendered)) as image:
        gray = image.convert("L").resize((512, 384), Image.Resampling.BOX)
    return Image.fromarray(np.where(np.asarray(gray) < 128, 0, 255).astype(np.uint8))


def revision_directory(task_id: str, revision: str) -> Path:
    source_metadata(task_id)
    if re.fullmatch(r"[a-f0-9]{32}", revision) is None:
        raise ValueError("invalid book revision")
    return REVIEWS / task_id / "revisions" / revision


def head(task_id: str) -> dict | None:
    source_metadata(task_id)
    path = REVIEWS / task_id / "head.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_revision(task_id: str, revision: str | None = None, *, require_certified=True) -> Task:
    current = head(task_id) if revision is None else None
    revision = revision or (current["revision"] if current else None)
    if revision is None:
        raise ValueError("book is not human-certified yet; finish and certify it in /editor")
    directory = revision_directory(task_id, revision)
    commit = json.loads((directory / "commit.json").read_text())
    if current and current != commit:
        raise ValueError("book head and immutable revision commit disagree")
    if digest(directory / "task.json") != commit["sha256"]:
        raise ValueError("immutable book revision metadata was modified")
    metadata = json.loads((directory / "task.json").read_text())
    if metadata["revision"] != revision or metadata["task_id"] != task_id:
        raise ValueError("book revision identity mismatch")
    if require_certified and metadata["trace_status"] != "human_certified":
        raise ValueError("draft is not eligible for evaluation; certify the complete silhouette first")
    for name, sha in metadata["files"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"immutable book revision asset changed: {name}")
    with Image.open(directory / "target.png") as image:
        mask = np.array(image.convert("L")) < 128
    with Image.open(directory / "known.png") as image:
        known = np.array(image.convert("L")) >= 128
    if mask.shape != (384, 512) or known.shape != mask.shape or not known.all():
        raise ValueError("reviewed targets must score the full frame, with no occlusion mask")
    mask.setflags(write=False)
    known.setflags(write=False)
    return Task(task_id, metadata["title"], mask, known, metadata)


def editor_state(task_id: str) -> dict:
    source = source_metadata(task_id)
    current = head(task_id)
    if current:
        task = load_revision(task_id, current["revision"], require_certified=False)
        directory = revision_directory(task_id, current["revision"])
        document = Document.model_validate_json((directory / "document.json").read_text())
        review = task.metadata["human_review"]
        status = task.metadata["trace_status"]
    else:
        document = initial_document(task_id)
        review = None
        status = "not_reviewed"
    with Image.open(DATA / task_id / "source.png") as image:
        size = list(image.size)
    return {"task_id": task_id, "title": source["title"], "source": source["source"],
            "source_size": size, "source_to_screen": source["trace"]["source_to_screen"],
            "source_was_occluded": source["trace_status"] != "unoccluded",
            "original_exclusion": next((r.get("reason") for r in book_records() if r["task_id"] == task_id), None),
            "revision": current["revision"] if current else None, "status": status,
            "document": document.model_dump(mode="json"), "review": review, "attestation": ATTESTATION}


def review_catalogue(*, certified_only=False) -> list[dict]:
    result = []
    for record in book_records():
        current = head(record["task_id"])
        status = "not_reviewed"
        if current:
            task = load_revision(record["task_id"], current["revision"], require_certified=False)
            status = task.metadata["trace_status"]
        if certified_only and status != "human_certified":
            continue
        result.append({"task_id": record["task_id"], "title": record["title"],
                       "trace_status": status, "revision": current["revision"] if current else None,
                       "formerly_excluded": bool(record.get("reason"))})
    return result


def save_revision(task_id: str, request: SaveRequest) -> dict:
    source = source_metadata(task_id)
    if request.certify:
        if not request.attestation or not request.reviewer.strip():
            raise ValueError("certification requires your name and explicit complete-outline attestation")
        if not request.document.paths or any(not p.closed or len(p.nodes) < 3 for p in request.document.paths):
            raise ValueError("finish or remove every open/short contour before certification")
    image = render_target(request.document)
    black = int(np.count_nonzero(np.asarray(image) < 128))
    if request.certify and (black < 16 or black > 512 * 384 - 16):
        raise ValueError("certification requires a visible foreground silhouette and background")
    root = REVIEWS / task_id
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".write.lock"), timeout=10):
        current = head(task_id)
        if request.base_revision != (current["revision"] if current else None):
            raise RevisionConflict("another tab saved this plate; reload before saving (your local drawing is unchanged)")
        revision = uuid.uuid4().hex
        revisions = root / "revisions"
        revisions.mkdir(exist_ok=True)
        directory = revisions / revision
        # A failed write never becomes the active revision.
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=revisions) as temporary:
            staging = Path(temporary)
            (staging / "document.json").write_text(request.document.model_dump_json(indent=2) + "\n")
            (staging / "target.svg").write_bytes(svg_bytes(request.document))
            image.save(staging / "target.png")
            Image.new("L", (512, 384), 255).save(staging / "known.png")
            shutil.copyfile(DATA / task_id / "source.png", staging / "source.png")
            now = datetime.now(timezone.utc).isoformat()
            metadata = {
                "task_id": task_id, "title": source["title"], "kind": "fixed_book_silhouette",
                "source": source["source"], "revision": revision, "parent_revision": request.base_revision,
                "trace_status": "human_certified" if request.certify else "draft_uncertified",
                "reference_pose": None, "solvability": "not established; human review certifies the drawing, not 3D reachability",
                "uncertain_foreground_fraction": 0., "known_foreground_pixels": black,
                "trace": {"method": "manual Bezier refinement of archived source trace",
                          "source_to_screen": source["trace"]["source_to_screen"], "unknown_mask": "none; full frame scored"},
                "human_review": {"certified": request.certify, "reviewer": request.reviewer.strip() if request.certify else None,
                                 "attestation": ATTESTATION if request.certify else None,
                                 "notes": request.notes, "saved_at": now, "original_was_occluded": source["trace_status"] != "unoccluded"},
                "files": {name: digest(staging / name) for name in
                          ("document.json", "target.svg", "target.png", "known.png", "source.png")},
            }
            (staging / "task.json").write_text(json.dumps(metadata, indent=2) + "\n")
            commit = {"revision": revision, "sha256": digest(staging / "task.json")}
            (staging / "commit.json").write_text(json.dumps(commit) + "\n")
            os.replace(staging, directory)
        with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix=".head-", delete=False) as stream:
            json.dump(commit, stream)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_head = Path(stream.name)
        try:
            os.replace(temporary_head, root / "head.json")
        finally:
            temporary_head.unlink(missing_ok=True)
    return editor_state(task_id)


def history(task_id: str) -> list[dict]:
    source_metadata(task_id)
    directory = REVIEWS / task_id / "revisions"
    result = []
    if directory.exists():
        for path in directory.iterdir():
            if not re.fullmatch(r"[a-f0-9]{32}", path.name):
                continue
            task = load_revision(task_id, path.name, require_certified=False)
            result.append({"revision": path.name, "status": task.metadata["trace_status"], **task.metadata["human_review"]})
    return sorted(result, key=lambda item: item["saved_at"], reverse=True)
