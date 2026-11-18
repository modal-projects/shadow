import json
import os
import secrets
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REPLAYS_DIR = (
    Path("/replays")
    if os.access("/replays/index.json", os.R_OK)
    else ROOT / "replays"
)


def create_web(static_dir=None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles


    if not (REPLAYS_DIR / "index.json").exists():
        raise RuntimeError(
            f"no replay index at {REPLAYS_DIR}; mount /replays or pull the "
            "shadow-replays volume first: "
            "modal volume get -e demos shadow-replays / ./replays"
        )

    def replays():
        return json.loads((REPLAYS_DIR / "index.json").read_text())

    web = FastAPI()

    @web.get("/api/round")
    def round_():
        return {"round": secrets.randbelow(len(replays()))}

    @web.get("/replays/{slot}.bin")
    def replay_bin(slot: int):
        if slot < 0 or slot >= len(replays()):
            raise HTTPException(404, "unknown round")
        return FileResponse(
            REPLAYS_DIR / f"{slot}.bin",
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @web.get("/rig.json")
    def rig_json():
        return FileResponse(REPLAYS_DIR / "rig.json")

    if static_dir:
        web.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return web


REPLAYS_VOLUME = modal.Volume.from_name(
    "shadow-replays", environment_name="demos", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(ROOT), extra_options="--no-dev --no-group bench")
    .add_local_dir(ROOT / "web" / "dist", "/demo/web/dist", copy=True)
)
app = modal.App("shadow-demo")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("shadow-endpoints", environment_name="demos")],
    volumes={"/replays": REPLAYS_VOLUME},
    timeout=1800,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(requires_proxy_auth=False, custom_domains=["shadow.modal.dev"])
def serve():
    return create_web(static_dir=Path("/demo/web/dist"))
