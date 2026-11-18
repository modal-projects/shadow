import asyncio
import base64
import json
import os
import random
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import modal

ROOT = Path(__file__).resolve().parent

MODEL = "qwen3-8-max-vl-thinking"
ENVIRONMENT = "demos"
PRICE_PER_MTOK = {
    "glm-5-3-flash-nvfp4": (0.15, 0.50),
    "glm-5-3-flash": (0.45, 1.50),
    "glm-5-3": (1.40, 4.40),
    "deepseek-v4-1-flash-1017": (0.25, 1.00),
    "deepseek-v4-1-flash": (0.30, 1.20),
    "muse-spark-open": (1.25, 4.25),
    "inkling-nvfp4": (1.20, 5.00),
    "qwen3-8-max": (2.00, 6.00),
    "qwen3-8-max-vl-thinking": (2.00, 6.00),
    "kimi-k3-1-pro": (2.00, 10.00),
    "kimi-k3": (3.00, 15.00),
}
BASE_URL = f"https://modal-labs-{ENVIRONMENT}--ep-{MODEL}-server.us-west.modal.direct/v1/"
RUN_LOCK = threading.Lock()

RUNS = modal.Volume.from_name(
    "shadowbench-runs", environment_name="demos", create_if_missing=True
)
REPLAYS = modal.Volume.from_name(
    "shadow-replays", environment_name="demos", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(ROOT), extra_options="--no-dev --group bench")
    .env({"OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "vendor", "/root/vendor")
)
app = modal.App("shadow-replays")

async def run_trial(task_id, output, effort="medium", base_url=BASE_URL, label=MODEL):
    from openai import AsyncOpenAI

    sys.path.insert(0, "/root/vendor/shadowbench")
    from shadowbench import agent
    from shadowbench.native_environment import state_hash

    if not RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("benchmark instrumentation requires one episode per process")
    output = Path(output)
    original_create = agent.create
    observation = {"tools": [], "model_turns": []}
    environment = None
    invocation_start = time.monotonic()

    def elapsed():
        return time.monotonic() - environment.start_time

    def create(*args, **kwargs):
        nonlocal environment
        environment = original_create(*args, **kwargs)
        observation["environment_initialized_seconds"] = elapsed()
        observation["environment_clock_offset_seconds"] = (
            environment.start_time - invocation_start
        )
        observation["initial"] = environment.metrics()
        original_call = environment.call

        def call(name, arguments=None):
            before = state_hash(environment.pose)
            calls_before = environment.calls
            started = elapsed()
            result = original_call(name, arguments)
            completed = elapsed()
            observation["tools"].append({
                "tool": name,
                "started_seconds": started,
                "completed_seconds": completed,
                "seconds": completed - started,
                "counted": environment.calls > calls_before,
                "pose_changed": before != state_hash(environment.pose),
                "metrics": environment.metrics(),
            })
            print(
                json.dumps({
                    "task": task_id,
                    "tool": name,
                    "call": environment.calls,
                    "seconds": round(completed, 2),
                    "refused": result.text.startswith("REFUSED:"),
                }),
                flush=True,
            )
            return result

        environment.call = call
        return environment

    try:
        agent.create = create
        async with AsyncOpenAI(
            base_url=base_url,
            api_key="unused",
            default_headers={
                "Modal-Key": os.environ["MODAL_PROXY_KEY"],
                "Modal-Secret": os.environ["MODAL_PROXY_SECRET"],
                "Modal-Routing-Affinity-Key": uuid.uuid4().hex,
            },
            timeout=1800,
            max_retries=1,
        ) as client:
            models = await client.models.list()
            model_id = models.data[0].id

            async def complete(**kwargs):
                started = elapsed()
                post_submit = environment.done
                succeeded = False
                try:
                    response = await client.chat.completions.create(**kwargs)
                    succeeded = True
                    return response
                finally:
                    completed = elapsed()
                    observation["model_turns"].append({
                        "started_seconds": started,
                        "completed_seconds": completed,
                        "seconds": completed - started,
                        "post_submit": post_submit,
                        "succeeded": succeeded,
                    })

            timed_client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=complete))
            )
            result = await agent.run_episode(
                timed_client,
                model_id,
                task_id,
                output,
                label=label,
                reasoning_effort=effort,
            )
        observation.update({
            "clock": "Seconds from upstream environment.start_time; includes initial observation, inference, native checks and episode logging. Excludes rig construction/initial gate, endpoint setup, Modal provisioning and post-hoc analysis. Upstream result.seconds uses its own later start and includes the closing model turn.",
            "endpoint_setup_and_invocation_seconds": time.monotonic()
            - invocation_start,
            "effort": effort,
        })
        (output / "timing.json").write_text(json.dumps(observation, indent=2) + "\n")
        print(
            json.dumps({
                "task": task_id,
                "status": result["status"],
                "calls": result["calls"],
                "score": result["final"]["score"] if result["final"] else None,
            }),
            flush=True,
        )
    finally:
        agent.create = original_create
        RUN_LOCK.release()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("shadow-endpoints", environment_name="demos")],
    volumes={"/runs": RUNS},
    cpu=8,
    memory=8192,
    timeout=7200,
)
def trial(task_id, run, effort="medium", model=MODEL):
    output = Path("/runs") / run / task_id
    output.mkdir(parents=True, exist_ok=True)
    base_url = (
        f"https://modal-labs-{ENVIRONMENT}--ep-{model}-server.us-west.modal.direct/v1/"
    )
    try:
        asyncio.run(run_trial(task_id, output, effort, base_url, model))
    except Exception as error:
        (output / "infrastructure-error.json").write_text(
            json.dumps({"type": type(error).__name__, "error": str(error)}, indent=2)
            + "\n"
        )
        RUNS.commit()
        return {"task": task_id, "error": type(error).__name__}
    RUNS.commit()
    return {"task": task_id}


@app.function(image=image, volumes={"/runs": RUNS})
def init_run(run, meta, selected):
    import shutil

    directory = Path("/runs") / run
    directory.mkdir(parents=True, exist_ok=True)
    batch = directory / "batch.json"
    if batch.exists():
        try:
            existing = json.loads(batch.read_text())
            meta["started_at_utc"] = existing["started_at_utc"]
            meta["tasks"] = sorted(set(existing.get("tasks", [])) | set(meta["tasks"]))
        except Exception:
            pass
    batch.write_text(json.dumps(meta, indent=2) + "\n")
    incomplete = []
    for task_id in selected:
        task_dir = directory / task_id
        complete = (task_dir / "result.json").exists() and (
            task_dir / "trajectory.jsonl"
        ).exists()
        if complete:
            try:
                complete = json.loads(
                    (task_dir / "result.json").read_text()
                ).get("final") is not None
            except Exception:
                complete = False
        if complete:
            continue
        if task_dir.exists():
            archive = directory / "_incomplete" / f"{task_id}-{int(time.time())}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(task_dir), archive)
        incomplete.append(task_id)
    RUNS.commit()
    return incomplete




MAX_FRAMES = 40


def frame_poses(task_dir):
    entries = [
        json.loads(line)
        for line in (task_dir / "trajectory.jsonl").read_text().splitlines()
        if line.strip()
    ]
    poses = [entries[0]["before"]]
    poses += [
        entry["after"]
        for entry in entries
        if entry.get("accepted") and entry.get("after_hash") != entry.get("before_hash")
    ]
    final = entries[-1]["after"]
    if poses[-1] != final:
        raise ValueError(f"{task_dir.name}: last frame is not the submitted final pose")
    if len(poses) > MAX_FRAMES:
        indices = sorted({
            round(i * (len(poses) - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)
        })
        poses = [poses[i] for i in indices]
    return poses


def task_dirs(run_dir):
    dirs = sorted(
        path
        for path in run_dir.iterdir()
        if (path / "trajectory.jsonl").exists()
        and (path / "result.json").exists()
        and (path / "final.json").exists()
        and (path / "task.json").exists()
    )
    if not dirs:
        raise ValueError(f"no task dirs in {run_dir}")
    return dirs


def encode_bin(frames):
    import numpy as np

    count = len(frames)
    vertices = next(iter(frames[0].values())).vertices.shape[0]
    header = np.array([count, vertices], dtype="<i4")
    body = np.stack(
        [
            np.clip(np.round(mesh.vertices * 10000), -32768, 32767).astype("<i2")
            for frame in frames
            for side in ("left", "right")
            for mesh in [frame[side]]
        ]
    )
    return header.tobytes() + body.tobytes()


@app.function(
    image=image,
    volumes={"/runs": RUNS, "/replays": REPLAYS},
    cpu=8,
    timeout=1800,
)
def export(run):
    import numpy as np

    sys.path.insert(0, "/root/vendor/shadowbench")
    from shadowbench import curation
    from shadowbench.native_environment import meshes_for
    from shadowbench.wrist_posture import posture_effort
    from shadowbench.pose import Pose
    from shadowbench.render import shadow
    from shadowbench.score import grade
    from shadowbench.ultraleap import NativeRig

    REST = Pose(
        np.zeros(46),
        np.array([[0.1, -0.12, 1.3], [-0.1, -0.12, 1.3]]),
        np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )

    RUNS.reload()
    dirs = sorted(
        d for name in run.split(",") for d in task_dirs(Path("/runs") / name)
    )
    for stale in Path("/replays").glob("*.bin"):
        stale.unlink()
    rig = NativeRig()
    order = list(range(len(dirs)))
    random.Random(0).shuffle(order)
    index = [None] * len(dirs)
    parity = []

    for slot, task_index in enumerate(order):
        task_dir = dirs[task_index]
        task_id = task_dir.name
        result = json.loads((task_dir / "result.json").read_text())
        task_json = json.loads((task_dir / "task.json").read_text())

        poses = frame_poses(task_dir)
        final_pose = Pose.parse(poses[-1])
        metrics = grade(
            shadow(meshes_for(rig, final_pose)),
            curation.load_revision(task_id, task_json["revision"]),
            posture_effort(final_pose)["ease"],
        )
        expected = json.loads((task_dir / "final.json").read_text())
        parity.append({
            "task_id": task_id,
            "iou_delta": abs(metrics["iou"] - expected["iou"]),
            "score_delta": abs(metrics["score"] - expected["score"]),
        })

        frames = [meshes_for(rig, Pose.parse(pose)) for pose in poses]
        (Path("/replays") / f"{slot}.bin").write_bytes(encode_bin(frames))

        from shadowbench.render import wall

        shots = Path("/replays") / "shots"
        shots.mkdir(exist_ok=True)
        wall(shadow(meshes_for(rig, final_pose)), title=task_json["title"]).save(
            shots / f"{task_id}.png"
        )
        index[slot] = {
            "subject": task_json["title"],
            "task_id": task_id,
            "model": result.get("model") or result.get("endpoint_model"),
            "final_score": result["final"]["score"],
            "frames": len(poses),
        }

    rest_meshes = meshes_for(rig, REST)
    rig_payload = {
        "vertices": {
            side: base64.b64encode(mesh.vertices.astype("<f4").tobytes()).decode()
            for side, mesh in rest_meshes.items()
        },
        "faces": {
            side: base64.b64encode(mesh.faces.astype("<u2").tobytes()).decode()
            for side, mesh in rest_meshes.items()
        },
    }
    (Path("/replays") / "rig.json").write_text(
        json.dumps(rig_payload, separators=(",", ":"))
    )
    (Path("/replays") / "index.json").write_text(
        json.dumps(index, separators=(",", ":")) + "\n"
    )

    worst_iou = max(p["iou_delta"] for p in parity)
    worst_score = max(p["score_delta"] for p in parity)
    print(f"built {len(index)} replays + rig.json -> shadow-replays")
    print(
        f"parity: max |iou delta| = {worst_iou:.3e}, max |score delta| = {worst_score:.3e}"
    )
    for row in parity:
        if row["iou_delta"] or row["score_delta"]:
            print("  nonzero:", row)
    if worst_iou or worst_score:
        raise ValueError(
            f"export parity drift: iou {worst_iou:.3e}, score {worst_score:.3e}"
        )
    REPLAYS.commit()


@app.local_entrypoint()
def main(
    run: str,
    build: bool = False,
    task: str = "",
    effort: str = "medium",
    model: str = MODEL,
):
    if build:
        export.remote(run)
        return
    sys.path.insert(0, str(ROOT / "vendor" / "shadowbench"))
    from shadowbench import tasks

    selected = [task] if task else tasks()
    if len(set(selected)) != len(selected) or any(
        item not in tasks() for item in selected
    ):
        raise ValueError("select certified benchmark task ids only")
    incomplete = init_run.remote(
        run,
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "tasks": selected,
            "effort": effort,
            "model": model,
            "price_usd_per_mtok": PRICE_PER_MTOK.get(model),
            "temperature": "omitted, matching upstream agent",
            "cpu_per_episode": 8,
            "episode_retries": 0,
            "environment": "demos",
            "protocol": "Published reviewed-target agent: 100 calls, 32 looks, plans, penalized book peeks; no game timer or guesses.",
        },
        selected,
    )
    if not incomplete:
        print(f"{run}: all {len(selected)} episodes already recorded", flush=True)
        return
    for task_id, result in zip(
        incomplete,
        trial.map(
            incomplete,
            kwargs={"run": run, "effort": effort, "model": model},
            order_outputs=True,
            return_exceptions=True,
        ),
        strict=True,
    ):
        if isinstance(result, Exception):
            print(f"Failed {task_id}: {type(result).__name__}", flush=True)
            continue
        if "error" in result:
            print(f"Failed {task_id}: {result['error']}", flush=True)
            continue
        print(f"Saved {task_id}", flush=True)
