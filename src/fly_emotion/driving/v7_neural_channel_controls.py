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

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_channel_controls.py")
CALIBRATION_ARTIFACT = Path("artifacts/v7-neural-channels-calibration.json")


def evaluate_v7_neural_channel_controls(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    tuning = json.loads((root / TUNING_ARTIFACT).read_text())
    calibration = json.loads((root / CALIBRATION_ARTIFACT).read_text())
    if not tuning["tuning_passed"] or not calibration["calibration_passed"]:
        raise ValueError("neural channel controls require passed tuning and calibration")
    if calibration["protocol"]["model_sha256"] != tuning["model_sha256"]:
        raise ValueError("neural channel model digest changed")
    teacher = json.loads((root / config["adapter_source"]).read_text())["selected_candidate"]
    spatial = {
        name
        for name in tuning["model"]["group_names"]
        if name.startswith(("front_to_back", "back_to_front"))
    }
    arms = {
        "full": {"groups": set(), "heading": True},
        "no_T4_T5_spatial": {"groups": spatial, "heading": True},
        "no_LPLC1": {"groups": {"LPLC1_L", "LPLC1_R"}, "heading": True},
        "no_LPLC2": {"groups": {"LPLC2_L", "LPLC2_R"}, "heading": True},
        "no_LC4": {"groups": {"LC4_L", "LC4_R"}, "heading": True},
        "no_all_LPLC_LC": {
            "groups": {
                "LPLC1_L",
                "LPLC1_R",
                "LPLC2_L",
                "LPLC2_R",
                "LC4_L",
                "LC4_R",
            },
            "heading": True,
        },
        "no_heading_feedback": {"groups": set(), "heading": False},
    }
    seeds = [int(seed) for seed in config["tuning"]["mirror_pair_seeds"]]
    results = {}
    for name, arm in arms.items():
        features = StructuredNeuralFeatures(root, config)
        episodes = [
            _neural_episode(
                features,
                tuning["model"],
                teacher,
                seed,
                ablated_groups=arm["groups"],
                heading_feedback=arm["heading"],
            )
            for seed in seeds
        ]
        results[name] = {
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
    full = results["full"]
    return {
        "protocol": {
            "name": config["name"],
            "phase": "post_calibration_neural_source_controls",
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
        "arms": results,
        "reference_success_count": full["success_count"],
        "reference_total_obstacles_passed": full["total_obstacles_passed"],
        "interpretation": {
            name: (
                "necessary_or_synergistic"
                if item["success_count"] < full["success_count"]
                else "not_necessary_under_this_tuning_screen"
            )
            for name, item in results.items()
            if name != "full"
        },
        "final_evaluated": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
