from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from fly_emotion.connectome.skeleton import fetch_skeleton, skeleton_segments
from fly_emotion.driving import DrivingEngine
from fly_emotion.driving.engine import NEURAL_POLICY_VERSION, POLICY_VERSION

ROOT = Path(
    os.getenv(
        "AUTODRIVE_FLY_ROOT",
        os.getenv("FLY_EMOTION_ROOT", Path(__file__).parents[4]),
    )
)
_ENGINE: DrivingEngine | None = None
_ENGINE_LOCK = Lock()
_STEP_LOCK = Lock()


class ResetRequest(BaseModel):
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    keep_learning: bool = True
    scenario: Literal["highway", "city"] = "highway"
    control_mode: Literal["assisted", "neural"] = "assisted"


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=50)
    learning: bool = False
    explore: bool = False
    safety_constraints: bool = True
    control_mode: Literal["assisted", "neural"] | None = None


class RunRequest(BaseModel):
    max_steps: int = Field(default=500, ge=1, le=500)
    learning: bool = False
    explore: bool = False
    safety_constraints: bool = True
    control_mode: Literal["assisted", "neural"] | None = None


def engine() -> DrivingEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = DrivingEngine(ROOT)
    return _ENGINE


app = FastAPI(title="MaleCNS Visual Driving API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health():
    checkpoint = ROOT / "artifacts/checkpoints/driving-policy.npz"
    neural_checkpoint = ROOT / "artifacts/checkpoints/driving-policy.neural-v6.npz"
    checkpoint_version = None
    neural_checkpoint_version = None
    if checkpoint.exists():
        try:
            with np.load(checkpoint, allow_pickle=False) as payload:
                checkpoint_version = int(payload["format_version"][0])
        except (OSError, ValueError, KeyError, IndexError):
            pass
    if neural_checkpoint.exists():
        try:
            with np.load(neural_checkpoint, allow_pickle=False) as payload:
                neural_checkpoint_version = int(payload["format_version"][0])
        except (OSError, ValueError, KeyError, IndexError):
            pass
    return {
        "status": "ok",
        "connectome": "male-cns:v1.0",
        "task": "visual-obstacle-and-city-driving",
        "language_model": "retired",
        "scenarios": ["highway", "city"],
        "brain_ready": (ROOT / "data/processed/malecns-v1.0/adjacency_target_norm.npz").exists(),
        "policy_checkpoint_ready": checkpoint_version == POLICY_VERSION,
        "policy_checkpoint_version": checkpoint_version,
        "required_policy_version": POLICY_VERSION,
        "neural_policy_checkpoint_ready": (
            neural_checkpoint_version == NEURAL_POLICY_VERSION
        ),
        "neural_policy_checkpoint_version": neural_checkpoint_version,
    }


def asset(name: Literal["overview", "pathways"]):
    path = ROOT / f"data/processed/malecns-v1.0/{name}.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Build {name} before opening this view")
    return json.loads(path.read_text())


@app.get("/api/connectome/overview")
def connectome_overview():
    return asset("overview")


@app.get("/api/connectome/pathways")
def connectome_pathways():
    return asset("pathways")


@app.get("/api/skeleton/{body_id}")
def skeleton(body_id: int, max_edges: int = 20_000):
    if body_id <= 0 or not 100 <= max_edges <= 100_000:
        raise HTTPException(status_code=422, detail="invalid skeleton request")
    try:
        source = fetch_skeleton(body_id, ROOT / "data/raw/malecns-v1.0/skeletons")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    segments = skeleton_segments(source, max_edges=max_edges)
    return {
        "body_id": body_id,
        "units": "micrometres",
        "source_vertices": len(source.vertices),
        "source_edges": len(source.edges),
        "display_edges": len(segments),
        "segments": segments.tolist(),
    }


@app.get("/api/driving/state")
def driving_state():
    with _STEP_LOCK:
        return engine().state(include_activity=True)


@app.post("/api/driving/reset")
def driving_reset(request: ResetRequest):
    with _STEP_LOCK:
        return engine().reset(
            request.seed,
            keep_learning=request.keep_learning,
            scenario=request.scenario,
            control_mode=request.control_mode,
        )


@app.post("/api/driving/step")
def driving_step(request: StepRequest):
    with _STEP_LOCK:
        if request.control_mode is not None:
            engine().set_control_mode(request.control_mode)
        result = None
        for _ in range(request.steps):
            result = engine().step(
                learning=request.learning,
                explore=request.explore,
                safety_constraints=request.safety_constraints,
            )
            if result["environment"]["done"]:
                break
        return result


@app.post("/api/driving/run-stream")
def driving_run_stream(request: RunRequest):
    return StreamingResponse(
        driving_event_lines(request),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def driving_event_lines(request: RunRequest):
    try:
        for _ in range(request.max_steps):
            started = time.perf_counter()
            with _STEP_LOCK:
                if request.control_mode is not None:
                    engine().set_control_mode(request.control_mode)
                state = engine().step(
                    learning=request.learning,
                    explore=request.explore,
                    safety_constraints=request.safety_constraints,
                )
            yield json.dumps({"type": "driving_state", **state}, ensure_ascii=False) + "\n"
            if state["environment"]["done"]:
                break
            # Pace the closed loop to 20 rendered states/s; neural computation
            # time remains separately reported in each state.
            time.sleep(max(0.0, 0.05 - (time.perf_counter() - started)))
        yield json.dumps({"type": "done"}) + "\n"
    except (OSError, ValueError, RuntimeError) as error:
        yield json.dumps({"type": "error", "message": str(error)}, ensure_ascii=False) + "\n"
