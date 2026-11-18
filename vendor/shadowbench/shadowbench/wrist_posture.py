"""RULA worksheet steps 3/3a only: wrist posture, not pain or a full RULA score."""

import math

from .pose import Pose
from .native_environment import angles_for

METHOD = "RULA wrist position component (steps 3 and 3a)"
SOURCE = "https://ergo.human.cornell.edu/Pub/AHquest/RULAworksheet.pdf"
NUMERICAL_ZERO_DEGREES = 1e-6


def wrist_component(flexion: float, deviation: float) -> dict:
    if not math.isfinite(flexion) or not math.isfinite(deviation):
        raise ValueError("wrist angles must be finite")
    angle = abs(flexion)
    neutral = angle <= NUMERICAL_ZERO_DEGREES
    base = 1 if neutral else (2 if angle <= 15. + 1e-9 else 3)
    sideways = int(abs(deviation) > NUMERICAL_ZERO_DEGREES)
    return {"flexion_extension_degrees": flexion, "radial_ulnar_deviation_degrees": deviation,
            "position_score": base, "deviation_increment": sideways, "wrist_component": base + sideways,
            "reason": ("neutral" if neutral else "within 15 degrees" if base == 2 else "beyond 15 degrees")
                      + ("; deviated from midline (+1)" if sideways else "; no deviation")}


def posture_effort(pose: Pose) -> dict:
    hands = {}
    for i, side in enumerate(("right", "left")):
        angles = angles_for(pose, i)
        hands[side] = wrist_component(angles["wrist.flex"], angles["wrist.side"])
    worst = max(h["wrist_component"] for h in hands.values())
    penalty = (worst - 1) / 3.
    return {"ease": 1. - penalty, "posture_penalty": penalty, "method": METHOD, "hands": hands,
            "worst_wrist_component": worst, "component_range": [1, 4],
            "reward_mapping": "ease = 1 - (max(right, left) - 1) / 3; at most 5% of shape quality",
            "source": SOURCE, "numerical_zero_degrees": NUMERICAL_ZERO_DEGREES,
            "scope": "established wrist-posture categories only; not full RULA, pain, strain probability or human certification",
            "not_estimated": ["finger discomfort", "muscle force", "contact force", "hold duration",
                              "repetition/exposure", "neck/trunk/shoulder posture", "wrist-twist risk"]}
