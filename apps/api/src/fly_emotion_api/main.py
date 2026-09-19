from __future__ import annotations

import hashlib
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


def _verified_v7_status(root: Path) -> dict:
    path = root / "artifacts/v7-goal-audit.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Build the v7 goal audit before opening status")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        dependencies = report["protocol"]["dependencies_sha256"]
        for relative, expected in dependencies.items():
            dependency = root / relative
            digest = (
                hashlib.sha256(dependency.read_bytes()).hexdigest()
                if dependency.exists()
                else None
            )
            if digest != expected:
                raise HTTPException(
                    status_code=503, detail=f"Stale v7 audit dependency: {relative}"
                )
        checks = {item["item"]: item for item in report["checks"]}
        visual = checks[1]["observations"]
        status = {
            "version": "v7-experimental",
            "source": "hash-verified-offline-goal-audit",
            "current_stage": report["summary"]["current_stage"],
            "objective_complete": report["summary"]["objective_complete"],
            "deployment_enabled": checks[0]["observations"]["v7_deployment_enabled"],
            "default_runtime_changed": checks[0]["observations"]["default_runtime_changed"],
            "gates": {
                "T4_T5_direction_and_ON_OFF": visual["controlled_response_gates_pass"],
                "LPLC1_near_collision": visual["LPLC1_near_collision_precheck_passed"],
                "LPLC2_radial_opponency": visual["LPLC2_radial_opponency_gates_passed"],
                "LC4_angular_speed": visual["LC4_position_speed_precheck_passed"],
                "EPG_PEN_PEG_heading": checks[2]["observations"][
                    "EPG_PEN_PEG_heading_assays_passed"
                ],
                "PFL3_DNa_transparent_mapping": checks[3]["observations"][
                    "readout_action_equivalence"
                ],
                "causal_visual_navigation": checks[4]["observations"]["stage1_pass"],
                "external_final": checks[7]["observations"]["external_final_evaluated"],
            },
            "evidence_boundaries": {
                "nine_source_contract_complete": visual[
                    "source_evidence_matrix_all_nine_complete"
                ],
                "Mi4_C3_direct_numeric_voltage_candidates": visual[
                    "Mi4_C3_direct_numeric_voltage_candidates"
                ],
                "Mi4_C3_independent_numeric_voltage_candidate_count": len(
                    visual["Mi4_C3_independent_numeric_voltage_candidates"]
                ),
                "T5_voltage_field_counts": visual[
                    "T5_voltage_modality_field_counts"
                ],
                "Braun_calcium_fly_counts": visual["Braun_T5_payload_fly_counts"],
                "Braun_calcium_condition_grids_complete": visual[
                    "Braun_T5_all_condition_grids_complete"
                ],
                "Braun_calcium_allowed_voltage_sources": visual[
                    "Braun_T5_allowed_voltage_sources"
                ],
                "Gou_Dryad_archive_hash_locally_verified": visual[
                    "Gou_Dryad_archive_hash_locally_verified"
                ],
                "Gou_Dryad_local_processed_calcium_sources": visual[
                    "Gou_Dryad_local_processed_calcium_sources"
                ],
                "Gou_Dryad_flash_fly_axis_sizes": visual[
                    "Gou_Dryad_flash_fly_axis_sizes"
                ],
                "Gou_Dryad_moving_bar_fly_axis_sizes": visual[
                    "Gou_Dryad_moving_bar_fly_axis_sizes"
                ],
                "Gou_Dryad_stable_biological_individual_IDs_verified": visual[
                    "Gou_Dryad_stable_biological_individual_IDs_verified"
                ],
                "Gou_Dryad_experimental_membrane_voltage": visual[
                    "Gou_Dryad_experimental_membrane_voltage"
                ],
                "T5_record_specific_stimulus_logs_available": visual[
                    "Kohn_Portes_record_specific_stimulus_logs_available"
                ],
                "Motyxia2_public_history_branch_count": visual[
                    "Motyxia2_public_history_branch_count"
                ],
                "Motyxia2_public_history_commit_count": visual[
                    "Motyxia2_public_history_commit_count"
                ],
                "T5_record_log_found_in_Motyxia2_public_history": visual[
                    "Kohn_Portes_record_log_found_in_Motyxia2_public_history"
                ],
                "T5_external_successful_indexes_linked_log_found": visual[
                    "Kohn_Portes_external_successful_indexes_linked_log_found"
                ],
                "T5_PMC_supplement_content_inspected": visual[
                    "Kohn_Portes_PMC_supplement_content_inspected"
                ],
                "T5_publisher_supplements_inspected": visual[
                    "Kohn_Portes_publisher_supplements_inspected"
                ],
                "T5_publisher_supplements_contain_record_log": visual[
                    "Kohn_Portes_publisher_supplements_contain_record_log"
                ],
                "T5_Figshare_search_accessible": visual[
                    "Kohn_Portes_Figshare_search_accessible"
                ],
                "T5_stimulus_log_global_absence_claimed": visual[
                    "Kohn_Portes_stimulus_log_global_absence_claimed"
                ],
                "T5_generator_defaults_used_as_record_fields": visual[
                    "Kohn_Portes_generator_defaults_used_as_record_fields"
                ],
                "T5_stimulus_provenance_complete": visual[
                    "Kohn_Portes_stimulus_provenance_complete"
                ],
                "CT1_audited_candidate_count": visual[
                    "CT1_voltage_audited_candidate_count"
                ],
                "CT1_incremental_2025_2026_candidate_count": len(
                    visual["CT1_incremental_2025_2026_relevant_candidates"]
                ),
                "CT1_direct_experimental_voltage_candidate_found": visual[
                    "CT1_direct_experimental_voltage_candidate_found"
                ],
                "CT1_PuRe_archive_contents_verified": visual[
                    "CT1_PuRe_archive_contents_verified"
                ],
                "CT1_PuRe_new_numerical_payload_verified": visual[
                    "CT1_PuRe_new_numerical_payload_verified"
                ],
                "Tm9_official_synapse_coordinate": visual[
                    "MaleCNS_Tm9_532266_official_synapse_coordinate"
                ],
                "CT1_per_synapse_Lo1_columnar_retinotopy_available": visual[
                    "MaleCNS_CT1_columnar_Lo1_retinotopy_available"
                ],
                "CT1_complete_official_LO_column_coverage": visual[
                    "MaleCNS_CT1_complete_official_LO_column_coverage"
                ],
            },
        }
    except HTTPException:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Invalid v7 goal audit") from error
    return {
        **status,
        "contributions": {
            "upper_planner": {
                "status": "paused",
                "active_in_default_runtime": False,
            },
            "fly_local_core": {
                "status": "component_only_not_release_authorized",
                "active_v7_in_default_runtime": False,
            },
            "engineering_executor": {
                "status": "transparent_fixed_mapping_component_passed",
                "v7_deployment_enabled": False,
            },
        },
        "audit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


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


@app.get("/api/v7/status")
def v7_status():
    return _verified_v7_status(ROOT)


@app.get("/api/autonomy/status")
def autonomy_status():
    """Compatibility alias for clients using the autonomy status route."""
    return _verified_v7_status(ROOT)


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
