"""Fixed-image fidelity first; normalized source-model mechanical effort second."""

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from .targets import Task


def grade(actual: np.ndarray, task: Task, ease: float) -> dict:
    if actual.shape != task.mask.shape or actual.dtype != bool:
        raise ValueError("mask shape/type mismatch; no per-candidate alignment is permitted")
    k, target = task.known, task.mask
    intersection = np.count_nonzero(actual & target & k)
    union = np.count_nonzero((actual | target) & k)
    iou = intersection/max(1, union)
    safe = binary_erosion(k, iterations=3, border_value=1)
    a = actual & ~binary_erosion(actual) & safe
    b = target & ~binary_erosion(target) & safe
    fs = []
    for tolerance in (1., 2., 4.):
        if a.any() and b.any():
            da,db=distance_transform_edt(~a),distance_transform_edt(~b)
            if da is None or db is None:raise RuntimeError("distance transform returned no distances")
            p = float(np.mean(np.asarray(db)[a] <= tolerance))
            r = float(np.mean(np.asarray(da)[b] <= tolerance))
            fs.append(2*p*r/max(p+r, 1e-12))
        else:fs.append(0.)
    boundary = float(np.mean(fs))
    # Internal negative-space islands (eyes etc.) are not swallowed by area-dominated IoU.
    contours,hierarchy=cv2.findContours(target.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    holes=[]
    if hierarchy is not None:
        for i,c in enumerate(contours):
            if hierarchy[0,i,3] >= 0 and 6 <= cv2.contourArea(c) <= 1500:
                region=np.zeros_like(actual,np.uint8);cv2.drawContours(region,[c],-1,1,-1)
                valid=(region>0)&k&~target
                if valid.sum()>=4:holes.append(float(np.mean(~actual[valid])))
    hole_score=float(np.mean(holes)) if holes else 1.
    shape=(.55*iou+.45*boundary) * (.9+.1*hole_score)
    if not np.isfinite(ease) or not 0<=ease<=1:raise ValueError("invalid effort proxy")
    quality=shape*(.95+.05*ease)
    return {"score":round(100*quality,4),"reward":float(quality),"shape":shape,"iou":iou,
            "boundary_f1":boundary,"hole_preservation":hole_score,"holes_scored":len(holes),
            "ease":ease,"solved":bool(iou>=.95 and boundary>=.95 and hole_score>=.95),
            "coverage":float(k.mean()),"trace_status":task.metadata["trace_status"],
            "target_is_fixed":True,"pose_similarity_used":False,
            "success_scope":"matches observed known silhouette pixels; not a canonical human pose"}
