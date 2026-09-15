from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_neural_channels import (
    CONFIG,
    TUNING_ARTIFACT,
    StructuredNeuralFeatures,
    _neural_episode,
)
from fly_emotion.driving.v7_neural_channels import (
    IMPLEMENTATION as CHANNEL_IMPLEMENTATION,
)

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_topology_controls.py")
CALIBRATION_ARTIFACT = Path("artifacts/v7-neural-channels-calibration.json")
CONTROL_SEED = 20260915


def evaluate_v7_neural_topology_controls(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    tuning = json.loads((root / TUNING_ARTIFACT).read_text())
    calibration = json.loads((root / CALIBRATION_ARTIFACT).read_text())
    if not tuning["tuning_passed"] or not calibration["calibration_passed"]:
        raise ValueError("topology controls require a frozen passed neural readout")
    if tuning["model_sha256"] != calibration["protocol"]["model_sha256"]:
        raise ValueError("tuning and calibration neural models differ")
    teacher = json.loads((root / config["adapter_source"]).read_text())["selected_candidate"]
    seeds = [int(seed) for seed in config["tuning"]["mirror_pair_seeds"]]
    controls = (
        "real_malecns",
        "shuffled_retina_coordinates",
        "source_preserving_target_shuffle",
        "shuffled_transmitter_signs",
    )
    results = {}
    for control in controls:
        features = StructuredNeuralFeatures(
            root, config, topology_control=control, control_seed=CONTROL_SEED
        )
        episodes = [_neural_episode(features, tuning["model"], teacher, seed) for seed in seeds]
        results[control] = {
            "success_count": sum(item["success"] for item in episodes),
            "total_obstacles_passed": sum(item["obstacles_passed"] for item in episodes),
            "terminal_counts": {
                terminal: sum(item["terminal_reason"] == terminal for item in episodes)
                for terminal in ("success", "obstacle", "road_boundary", "timeout")
            },
            "episodes": [
                {key: value for key, value in item.items() if key != "trace"} for item in episodes
            ],
        }
    real = results["real_malecns"]
    gates = {
        "real_topology_replays_all_successes": real["success_count"] == len(seeds),
        "retina_shuffle_reduces_success": results["shuffled_retina_coordinates"]["success_count"]
        < real["success_count"],
        "target_shuffle_reduces_success": results["source_preserving_target_shuffle"][
            "success_count"
        ]
        < real["success_count"],
        "transmitter_shuffle_reduces_success": results["shuffled_transmitter_signs"][
            "success_count"
        ]
        < real["success_count"],
    }
    return {
        "protocol": {
            "name": config["name"],
            "phase": "post_calibration_frozen_topology_controls",
            "control_seed": CONTROL_SEED,
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(CHANNEL_IMPLEMENTATION): _sha256(root / CHANNEL_IMPLEMENTATION),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(TUNING_ARTIFACT): _sha256(root / TUNING_ARTIFACT),
                str(CALIBRATION_ARTIFACT): _sha256(root / CALIBRATION_ARTIFACT),
            },
            "model_sha256": tuning["model_sha256"],
            "model_changed_after_calibration": False,
            "control_results_used_for_selection": False,
            "runtime_modified": False,
        },
        "seeds": seeds,
        "controls": results,
        "gates": gates,
        "real_topology_advantage_passed": all(gates.values()),
        "final_evaluated": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
