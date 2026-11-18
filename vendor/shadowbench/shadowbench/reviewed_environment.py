"""Current book-only environment; the v3.1 native implementation it builds on remains replayable."""

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from . import curation
from .native_environment import NativeEnvironment, INDEX, Output, TOOL_SPECS as NATIVE_TOOLS, fingerprint as native_fingerprint
from .native_environment import BODY, SYSTEM_PROMPT as NATIVE_PROMPT, LOG
from .pose import InvalidPose, SIDES, finite_array
from .render import wall
from .targets import Task
from .wrist_posture import posture_effort

# Original-figure hint: each peek multiplies the final reward; the image stays in the model's context.
BOOK_PEEKS = 3
PEEK_MULTIPLIER = 0.94
PLAN_LIMITS = {"subject": 200, "arrangement": 2000}
POSE_TOOLS = ("set_fingers", "move_hand", "turn_hand")
# Fractions of the scan. The hilliar_dog page also shows a rabbit cast by its other drawn hand; per the
# reviewer only the dog and the upper hand that casts it are shown (cut in the ink-free gap at y=0.681).
BOOK_FIGURE_VIEWS = {"hilliar_dog": {"crop": (0., 0., 1., .681), "blank": [(0., .37, .555, .59)]}}

SYSTEM_PROMPT = NATIVE_PROMPT.replace(
    "Gray pixels are fixed uncertainty in the original artwork; other background and forearm shadows count.",
    "The entire human-reviewed target is scored: no uncertainty masks. All background and forearm shadows count.",
).replace(
    "Ease uses explicit joint-end-range, wrist-posture and soft finger-coupling costs. It is NOT measured pain,\n"
    "muscle activation or a clinical risk score. The empirical mechanisms and engineering constants are disclosed.",
    "The secondary term uses RULA wrist-position categories only: neutral=1; flexion/extension up to 15 degrees=2;\n"
    "beyond 15 degrees=3; add 1 for radial/ulnar deviation. The worse wrist determines ease=1-(score-1)/3.\n"
    "This is not a full RULA assessment, measured pain, or finger strain estimate. Finger anatomy remains a hard gate.",
).replace(
    "100 calls, 32 look images, plus the initial observation. Submit when finished.",
    "The target is a classic hand-shadow puppet. Work like a shadow puppeteer, top-down: decide what the\n"
    "silhouette depicts and which hand parts will form which of its features (which fingers become ears,\n"
    "beak, mouth or horns; which hand forms the head; where the forearms become a neck or body), build\n"
    "that arrangement, then refine it with look.\n"
    "Call plan before your first hand movement: name the subject and the hand arrangement you intend.\n"
    "Call plan again whenever your approach changes. Plans are recorded with the episode but never scored.\n"
    "peek_book shows the original book figure for this target, including the author's drawn hands.\n"
    f"Their arrangement is a hint, not a measured 3D pose. Each peek multiplies your final score by {PEEK_MULTIPLIER}\n"
    f"(at most {BOOK_PEEKS}). The figure stays in your context, so one peek is usually enough.\n"
    "100 calls (plan and peek_book included), 32 look images, plus the initial observation. Submit when finished.",
)
TOOL_SPECS = {
    "describe": (
        "Read exact joint coordinates, conservative anatomy bounds, each wrist's interpretable RULA position "
        "component, your current plan and remaining book peeks.",
        NATIVE_TOOLS["describe"][1]),
    "look": (
        "One image: complete human-reviewed target, your shadow, red-missing/blue-extra overlay, and native hands. "
        "All pixels count; no unknown regions. Costs one look image.",
        NATIVE_TOOLS["look"][1]),
    "plan": (
        "Record what the silhouette depicts and the hand arrangement you intend: which hand parts form which "
        "features and roughly where each hand goes. Required before your first hand movement; call again to "
        "revise. Recorded, never scored. Costs one call.",
        {**BODY, "properties": {
            "subject": {"type": "string", "maxLength": PLAN_LIMITS["subject"]},
            "arrangement": {"type": "string", "maxLength": PLAN_LIMITS["arrangement"]}},
         "required": list(PLAN_LIMITS)}),
    "peek_book": (
        "Show the original book figure for this target, including the author's drawn hand arrangement. A hint, "
        f"not a measured 3D pose. Each peek multiplies the final score by {PEEK_MULTIPLIER}; at most {BOOK_PEEKS} "
        "peeks. Costs one call, not a look image.",
        {**BODY, "properties": {}, "required": []}),
    **{name: NATIVE_TOOLS[name] for name in (*POSE_TOOLS, "undo", "submit")},
}


@lru_cache(maxsize=1)
def fingerprint():
    sha = hashlib.sha256(b"shadowbench-v3.3-reviewed-books-rula-wrist-plan-peek")
    sha.update(native_fingerprint().encode())
    for name in ("reviewed_environment.py", "wrist_posture.py"):
        sha.update(Path(__file__).with_name(name).read_bytes())
    return sha.hexdigest()


class ReviewedEnvironment(NativeEnvironment):
    def __init__(self, task: Task, log_dir: Path | None = None, max_calls=100, max_images=32, *, preview=False,
                 plan_required=True):
        if task.metadata["kind"] != "fixed_book_silhouette" or not task.known.all():
            raise ValueError("current environment requires a complete book target with no unknown pixels")
        certified = task.metadata["trace_status"] == "human_certified"
        if not certified and not preview:
            raise ValueError("only explicitly human-certified book revisions are eligible for evaluation")
        self.evaluation_enabled = certified and not preview
        # Replaced, never mutated, so OpenEnv's shallow-copied transactions stay isolated.
        self.plan: dict | None = None
        self.peeks = 0
        self.plan_required = plan_required
        super().__init__(task, None, max_calls, max_images)
        # The base is immutable for historical replay. No obsolete initialization
        # record is persisted or exposed in this separately versioned environment.
        self.hold = posture_effort(self.pose)
        self.version = fingerprint()
        self.steps = []
        self._metrics = None
        self.log_dir = log_dir
        if log_dir:
            (log_dir / "images").mkdir(parents=True, exist_ok=True)
            (log_dir / "states").mkdir(exist_ok=True)
            (log_dir / "task.json").write_text(json.dumps(task.metadata, indent=2))
            wall(task.mask).save(log_dir / "target.png")
        self._record("reset", {}, True, "Native hand rest pose; " + (
            "certified book revision." if self.evaluation_enabled else "draft preview only, not evaluation."),
            None, self.pose, self.pose, None)

    def metrics(self):
        result = super().metrics()
        multiplier = PEEK_MULTIPLIER ** self.peeks
        result.update({"evaluation_enabled": self.evaluation_enabled, "reward_profile": "rula_wrist_v1",
                       "target_revision": self.task.metadata["revision"], "book_peeks": self.peeks,
                       "peek_multiplier": multiplier, "score_before_peeks": result["score"],
                       "reward": result["reward"] * multiplier,
                       "score": round(100 * result["reward"] * multiplier, 4)})
        return result

    def describe(self):
        result = super().describe()
        result.update({"plan": self.plan, "plan_required_before_moving": self.plan_required and self.plan is None,
                       "book_peeks_left": BOOK_PEEKS - self.peeks,
                       "final_score_multiplier_from_peeks": round(PEEK_MULTIPLIER ** self.peeks, 4)})
        return result

    def book_figure(self) -> Image.Image:
        directory = curation.revision_directory(self.task.task_id, self.task.metadata["revision"])
        data = (directory / "source.png").read_bytes()
        if hashlib.sha256(data).hexdigest() != self.task.metadata["files"]["source.png"]:
            raise RuntimeError("book figure changed after the target revision was validated")
        with Image.open(io.BytesIO(data)) as image:
            figure = image.convert("RGB")
        view = BOOK_FIGURE_VIEWS.get(self.task.task_id)
        if view:
            w, h = figure.size
            paper = tuple(int(v) for v in np.percentile(np.asarray(figure).reshape(-1, 3), 75, axis=0))
            draw = ImageDraw.Draw(figure)
            for x0, y0, x1, y1 in view["blank"]:
                draw.rectangle([round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)], fill=paper)
            x0, y0, x1, y1 = view["crop"]
            figure = figure.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
        return figure

    def _candidate(self, name, args):
        candidate = self.pose.copy()
        if name == "undo":
            if not self.history:
                raise InvalidPose("nothing to undo")
            return self.history[-1].copy()
        side = args["hand"]
        if side not in SIDES:
            raise InvalidPose("hand must be right or left")
        i = SIDES.index(side)
        if name == "set_fingers":
            values = args["joints"]
            if not isinstance(values, dict) or not values:
                raise InvalidPose("joints must be a nonempty object")
            for alias, value in values.items():
                if alias not in INDEX or type(value) not in (int, float) or not np.isfinite(value):
                    raise InvalidPose("unknown joint or non-finite angle")
                j = i * len(INDEX) + INDEX[alias]
                angle = np.radians(value)
                if abs(angle - self.pose.q[j]) > np.radians(45) + 1e-12:
                    raise InvalidPose("maximum change 45 degrees per joint")
                candidate.q[j] = angle
        elif name == "move_hand":
            delta = finite_array(args["delta"], (3,), "delta")
            if np.linalg.norm(delta) > .15000001:
                raise InvalidPose("maximum translation 15cm")
            candidate.positions[i] += delta
        else:
            axis = finite_array(args["axis"], (3,), "axis")
            degrees = args["degrees"]
            if type(degrees) not in (int, float) or not np.isfinite(degrees) or abs(degrees) > 45 or np.linalg.norm(axis) < 1e-9:
                raise InvalidPose("nonzero axis and rotation -45..45 degrees required")
            rotation = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.radians(degrees))
            candidate.quaternions[i] = (
                rotation * Rotation.from_quat(candidate.quaternions[i], scalar_first=True)).as_quat(scalar_first=True)
        return candidate

    def call(self, name, arguments=None) -> Output:
        with self.lock:
            if self.done:
                return Output("Episode already ended.")
            self.calls += 1
            before = self.pose.copy()
            image = certificate = None
            ok = False
            args = {} if arguments is None else arguments
            try:
                if name not in TOOL_SPECS:
                    raise InvalidPose(f"unknown tool {name}")
                props = TOOL_SPECS[name][1]["properties"]
                if not isinstance(args, dict) or set(args) != set(props):
                    raise InvalidPose(f"{name} requires exactly {list(props)}")
                if name == "describe":
                    text = json.dumps(self.describe())
                elif name == "look":
                    if self.images >= self.max_images:
                        raise InvalidPose("image budget exhausted")
                    image = self.observe(args["view"])
                    self.images += 1
                    text = "Full book target | shadow | difference | native hands. Red missing, blue extra. No masked regions."
                elif name == "submit":
                    self.done = True
                    self.final = self.metrics()
                    text = "Submitted." if self.evaluation_enabled else "Draft preview finished; not an evaluation result."
                elif name == "plan":
                    plan = {}
                    for field, limit in PLAN_LIMITS.items():
                        value = args[field]
                        if not isinstance(value, str) or not value.strip() or len(value) > limit:
                            raise InvalidPose(f"plan {field} must be nonempty text of at most {limit} characters")
                        plan[field] = value.strip()
                    self.plan = plan
                    text = "Plan recorded (not scored). Call plan again whenever your approach changes."
                elif name == "peek_book":
                    if self.peeks >= BOOK_PEEKS:
                        raise InvalidPose(f"all {BOOK_PEEKS} book peeks used")
                    image = self.book_figure()
                    self.peeks += 1
                    text = (f"Original book figure; its drawn hands are a hint, not a measured 3D pose. The target and "
                            f"frame are unchanged. Peeks {self.peeks}/{BOOK_PEEKS}; final score multiplier now "
                            f"x{PEEK_MULTIPLIER ** self.peeks:.4f}.")
                else:
                    if self.plan_required and self.plan is None and name in POSE_TOOLS:
                        raise InvalidPose("call plan first: say what the silhouette depicts and which hand parts "
                                          "will form which features")
                    candidate = self._candidate(name, args)
                    self.gate.check(candidate)
                    certificate = self.gate.motion(self.pose, candidate)
                    if name == "undo":
                        self.history.pop()
                    else:
                        self.history.append(self.pose.copy())
                    self.pose = candidate
                    self.hold = posture_effort(candidate)
                    self._metrics = None
                    text = (f"Accepted; {certificate['surface_samples']} checked skin states. "
                            f"Worst wrist position component {self.hold['worst_wrist_component']}/4; not a pain score.")
                ok = True
            except InvalidPose as error:
                text = f"REFUSED: {error}. State unchanged."
                LOG.info("%s %s", name, text)
            if self.calls >= self.max_calls and not self.done:
                self.done = True
                self.final = self.metrics()
                text += " Call budget exhausted; submitted."
            text += f" [calls {self.calls}/{self.max_calls}; images {self.images}/{self.max_images}]"
            self._record(name, args, ok, text, image, before, self.pose, certificate)
            if self.done and self.log_dir:
                (self.log_dir / "final.json").write_text(json.dumps(self.final, indent=2))
            return Output(text, image)
