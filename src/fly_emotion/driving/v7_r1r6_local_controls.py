from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_r1r6_local import (
    CONFIG,
    TUNING_ARTIFACT,
    local_visual_channels,
    reconstruct_image,
)
from fly_emotion.driving.v7_r1r6_local import (
    IMPLEMENTATION as LOCAL_IMPLEMENTATION,
)
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_r1r6_local_controls.py")
CALIBRATION_ARTIFACT = Path("artifacts/v7-r1r6-local-calibration.json")


def _episode(retina, config: dict, candidate: dict, seed: int, arm: dict) -> dict:
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    command = 0.0
    while not environment.done:
        channels = local_visual_channels(reconstruct_image(retina, retina.encode(image)), config)
        target = np.tanh(
            (
                float(candidate["obstacle_gain"])
                * channels["danger"]
                * channels["obstacle_asymmetry"]
                if arm["obstacle"]
                else 0.0
            )
            + (float(candidate["road_gain"]) * channels["road_centroid"] if arm["road"] else 0.0)
            - (float(candidate["heading_gain"]) * environment.heading if arm["heading"] else 0.0)
        )
        adapter = config["adapter"]
        command = float(
            float(adapter["smoothing_previous"]) * command
            + float(adapter["smoothing_current"]) * target
        )
        image, _, _ = environment.step(command, float(candidate["throttle"]), 0.0)
    return {
        "seed": seed,
        "terminal_reason": environment.terminal_reason,
        "success": environment.terminal_reason == "success",
        "obstacles_passed": environment.obstacles_passed,
        "distance": environment.y,
    }


def evaluate_v7_r1r6_local_controls(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    tuning = json.loads((root / TUNING_ARTIFACT).read_text())
    calibration = json.loads((root / CALIBRATION_ARTIFACT).read_text())
    if not calibration["calibration_passed"]:
        raise ValueError("local-channel controls require passed calibration")
    candidate = tuning["selected_candidate"]
    retina = build_mass_balanced_retina(root)
    seeds = [int(seed) for seed in config["tuning"]["mirror_pair_seeds"]]
    arms = {
        "full": {"obstacle": True, "road": True, "heading": True},
        "no_obstacle": {"obstacle": False, "road": True, "heading": True},
        "no_road": {"obstacle": True, "road": False, "heading": True},
        "no_heading": {"obstacle": True, "road": True, "heading": False},
        "zero_action": {"obstacle": False, "road": False, "heading": False},
    }
    results = {}
    for name, arm in arms.items():
        episodes = [_episode(retina, config, candidate, seed, arm) for seed in seeds]
        results[name] = {
            "success_count": sum(item["success"] for item in episodes),
            "total_obstacles_passed": sum(item["obstacles_passed"] for item in episodes),
            "terminal_counts": {
                terminal: sum(item["terminal_reason"] == terminal for item in episodes)
                for terminal in ("success", "obstacle", "road_boundary", "timeout")
            },
            "episodes": episodes,
        }
    full = results["full"]
    gates = {
        "full_replays_all_successes": full["success_count"] == len(seeds),
        "obstacle_channel_is_causal": results["no_obstacle"]["success_count"]
        < full["success_count"],
        "road_channel_improves_completion": results["no_road"]["success_count"]
        < full["success_count"],
        "heading_feedback_is_causal": results["no_heading"]["success_count"]
        < full["success_count"],
        "zero_action_fails": results["zero_action"]["success_count"] == 0,
    }
    return {
        "protocol": {
            "name": config["name"],
            "phase": "post_calibration_channel_controls",
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(LOCAL_IMPLEMENTATION): _sha256(root / LOCAL_IMPLEMENTATION),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(TUNING_ARTIFACT): _sha256(root / TUNING_ARTIFACT),
                str(CALIBRATION_ARTIFACT): _sha256(root / CALIBRATION_ARTIFACT),
            },
            "candidate_sha256": tuning["selected_candidate_sha256"],
            "candidate_parameters_changed": False,
            "control_results_used_for_selection": False,
            "runtime_modified": False,
        },
        "seeds": seeds,
        "arms": results,
        "gates": gates,
        "causal_controls_passed": all(gates.values()),
        "engineering_input_upper_bound_only": True,
        "final_evaluated": False,
        "advance_to_navigation_release": False,
    }
