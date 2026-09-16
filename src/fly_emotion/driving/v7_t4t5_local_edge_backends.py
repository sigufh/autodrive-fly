"""Frozen existing-backend A/B for the anatomy-local T4/T5 edge precheck."""

from __future__ import annotations

from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    CONFIG as PRECHECK_CONFIG,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    IMPLEMENTATION as PRECHECK_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    evaluate_v7_t4t5_local_edge_precheck,
)

CONFIG = Path("configs/driving-v7-t4t5-local-edge-backends.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4t5_local_edge_backends.py")


def evaluate_v7_t4t5_local_edge_backends(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    if Path(config["precheck_config"]) != PRECHECK_CONFIG:
        raise ValueError("backend A/B must reuse the frozen local-edge precheck")
    results = {}
    for retinal_backend in config["retinal_backends"]:
        for dynamics_backend in config["dynamics_backends"]:
            name = f"{retinal_backend}:{dynamics_backend}"
            report = evaluate_v7_t4t5_local_edge_precheck(
                root,
                retinal_backend=retinal_backend,
                dynamics_backend=dynamics_backend,
            )
            results[name] = {
                "retinal_backend": retinal_backend,
                "dynamics_backend": dynamics_backend,
                "direction_pass_count": report["summary"]["direction_pass_count"],
                "polarity_pass_count": report["summary"]["polarity_pass_count"],
                "population_scores": report["population_scores"],
            }
    minimum = int(config["selection"]["minimum_direction_populations_to_expand"])
    advancing = [
        name for name, result in results.items() if result["direction_pass_count"] >= minimum
    ]
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(PRECHECK_CONFIG): _sha256(root / PRECHECK_CONFIG),
                str(PRECHECK_IMPLEMENTATION): _sha256(root / PRECHECK_IMPLEMENTATION),
            },
            "condition_id": config["condition_id"],
            "candidate_count": len(results),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "candidates": results,
        "advancing_candidates": advancing,
        "existing_backend_reuse_gate_passed": bool(advancing),
        "expand_to_three_conditions": bool(advancing),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
