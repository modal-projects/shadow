"""Deterministic projection of the same triangle surfaces used by the validity gate."""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .mesh import Mesh

WIDTH, HEIGHT = 512, 384
FRAME = (-.4, .4, -.3, .3)
LIGHT = np.array([0., 0., 2.5])


def pixels(xy, w, h):
    return np.column_stack(((xy[:, 0]-FRAME[0])*w/(FRAME[1]-FRAME[0]),
                             (FRAME[3]-xy[:, 1])*h/(FRAME[3]-FRAME[2])))


def shadow(meshes: dict[str, Mesh], width=WIDTH, height=HEIGHT, supersample=2) -> np.ndarray:
    mask = np.zeros((height*supersample, width*supersample), np.uint8)
    for mesh in meshes.values():
        vertices = mesh.vertices
        if not np.isfinite(vertices).all() or vertices[:, 2].max() >= LIGHT[2]-.005:
            raise ValueError("surface intersects lamp plane")
        xy = vertices[:, :2]*(LIGHT[2]/(LIGHT[2]-vertices[:, 2, None]))
        projected = np.rint(pixels(xy, width*supersample, height*supersample)*256).astype(np.int32)
        for triangle in projected[mesh.faces]:
            cv2.fillConvexPoly(mask, triangle, 255, shift=8)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA) >= 128


def png(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def wall(mask: np.ndarray, title="") -> Image.Image:
    image = Image.fromarray(np.where(mask[..., None], np.array([26, 31, 38], np.uint8),
                                    np.array([246, 240, 225], np.uint8)))
    if title:
        ImageDraw.Draw(image).text((10, 8), title, fill=(72, 72, 72), font=ImageFont.load_default(size=14))
    return image


def comparison(actual: np.ndarray, target: np.ndarray, known: np.ndarray | None = None) -> Image.Image:
    rgb = np.full((*target.shape, 3), [246, 240, 225], np.uint8)
    rgb[actual & target] = [26, 31, 38]
    rgb[actual & ~target] = [50, 112, 210]
    rgb[target & ~actual] = [215, 59, 45]
    if known is not None:
        rgb[~known] = [175, 175, 175]
    return Image.fromarray(rgb)


def _raster(vertices, normals, faces, colors, eye, center, width, height):
    def cross2(a, b):
        return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    f = center-eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0., 1., 0.])
    if np.linalg.norm(r) < .01:
        r = np.cross(f, [0., 0., 1.])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = vertices-eye
    depth = d@f
    focal = height*.5/math.tan(math.radians(35)/2)
    xy = np.c_[width/2+focal*(d@r)/depth, height/2-focal*(d@u)/depth]
    a, b, c = xy[faces].transpose(1, 0, 2)
    area = cross2(b-a, c-a)
    low = np.clip(np.floor(np.minimum(np.minimum(a, b), c)), [0, 0], [width-1, height-1]).astype(int)
    high = np.clip(np.ceil(np.maximum(np.maximum(a, b), c)), [0, 0], [width-1, height-1]).astype(int)
    lengths = high-low+1
    valid = (abs(area) > 1e-8) & (depth[faces].min(1) > .001)
    indices = np.flatnonzero(valid)
    counts = np.prod(lengths[indices], axis=1)
    image = np.full((height*width, 3), [239., 242., 246.])
    zbuffer = np.full(height*width, np.inf)
    inv = 1/np.maximum(depth, 1e-9)
    light = -.4*r+.7*u-.6*f
    light /= np.linalg.norm(light)
    for first in range(0, len(indices), 256):
        ids = indices[first:first+256]
        cnt = counts[first:first+256]
        t = np.repeat(ids, cnt)
        off = np.arange(cnt.sum())-np.repeat(np.cumsum(cnt)-cnt, cnt)
        x = low[t, 0]+off % lengths[t, 0]
        y = low[t, 1]+off // lengths[t, 0]
        q = np.c_[x+.5, y+.5]
        w0 = cross2(b[t]-q, c[t]-q)/area[t]
        w1 = cross2(c[t]-q, a[t]-q)/area[t]
        w2 = 1-w0-w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        t, x, y = t[inside], x[inside], y[inside]
        bary = np.c_[w0[inside], w1[inside], w2[inside]]*inv[faces[t]]
        z = 1/bary.sum(1)
        bary *= z[:, None]
        pix = y*width+x
        order = np.lexsort((z, pix))
        p = pix[order]
        unique = np.r_[True, p[1:] != p[:-1]] if len(p) else np.array([], bool)
        choice = order[unique]
        choice = choice[z[choice] < zbuffer[pix[choice]]]
        if not len(choice):
            continue
        tri, w = faces[t[choice]], bary[choice]
        n = np.einsum("ij,ijk->ik", w, normals[tri])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        col = np.einsum("ij,ijk->ik", w, colors[tri])
        headlight = np.maximum(0, n@(-f))
        diffuse = np.maximum(0, n@light)
        image[pix[choice]] = col*(.3+.45*diffuse+.25*headlight)[:, None]
        zbuffer[pix[choice]] = z[choice]
    return Image.fromarray(np.clip(image.reshape(height, width, 3), 0, 255).astype(np.uint8))


def hands(meshes: dict[str, Mesh], azimuth=25., elevation=15., width=640, height=480,
          coded=False, focus: str | None = None) -> Image.Image:
    selected = [m for m in meshes.values() if focus is None or m.hand == focus]
    focus_meshes = [m for m in selected if m.part == "hand"] or selected
    points = np.vstack([m.vertices for m in focus_meshes])
    # A hand explicitly moved outside the screen should not shrink the remaining close-up.
    visible = [m for m in focus_meshes if m.vertices[:, 1].max() > -.45]
    points = np.vstack([m.vertices for m in visible or focus_meshes])
    center = (points.min(0)+points.max(0))/2
    radius = np.linalg.norm(points.max(0)-points.min(0))/2
    angle, elev = np.radians([azimuth, elevation])
    eye = center + max(.35, radius*3.6)*np.array([np.sin(angle)*np.cos(elev), np.sin(elev), np.cos(angle)*np.cos(elev)])
    vs, ns, fs, cs, offset = [], [], [], [], 0
    palette = {"thumb": [215, 88, 65], "index": [226, 158, 58], "middle": [219, 192, 79],
               "ring": [96, 170, 111], "pinky": [88, 134, 193]}
    for m in selected:
        tone = np.array([222, 168, 128] if m.hand == "right" else [189, 138, 104])
        color = np.tile(tone, (len(m.vertices), 1))
        if coded:
            for digit, col in palette.items():
                color[m.labels == digit] = col
        vs.append(m.vertices); ns.append(m.normals); fs.append(m.faces+offset); cs.append(color)
        offset += len(m.vertices)
    return _raster(np.vstack(vs), np.vstack(ns), np.vstack(fs), np.vstack(cs), eye, center, width, height)
