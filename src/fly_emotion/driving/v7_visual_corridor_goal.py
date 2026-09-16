from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_r1r6_local import local_visual_channels, reconstruct_image
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina

CONFIG = Path("configs/driving-v7-visual-corridor-goal.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_visual_corridor_goal.py")


def obstacle_distance_profile(image: np.ndarray, visual_config: dict) -> np.ndarray:
    obstacle = image > float(
        visual_config["channels"]["obstacle"]["luminance_threshold"]
    )
    height, width = image.shape
    horizon = height // 3
    obstacle_height = np.zeros(width, dtype=np.float64)
    for column in range(width):
        positions = np.flatnonzero(obstacle[:, column])
        if positions.size:
            obstacle_height[column] = height - int(positions.min())
    return 1.0 - obstacle_height / (height - horizon)


def _corridor_goal(
    distance_profile: np.ndarray, width: int, previous_goal: float, switch_margin: float
) -> tuple[float, bool]:
    profile = np.asarray(distance_profile, dtype=np.float64)
    if profile.shape != (48,) or width <= 0 or width > 24:
        raise ValueError("visual corridor requires a valid 48-column distance profile")
    starts = np.arange(12, 36 - width + 1)
    scores = np.asarray([np.mean(profile[start : start + width]) for start in starts])
    centers = starts + (width - 1) / 2.0
    goals = (centers - 23.5) / 12.0
    best = int(np.argmax(scores - 0.03 * np.abs(goals)))
    previous_index = int(np.argmin(np.abs(goals - previous_goal)))
    if scores[best] < scores[previous_index] + switch_margin:
        return float(goals[previous_index]), False
    return float(goals[best]), best != previous_index


def _episode(
    retina, visual_config: dict, policy: dict, candidate: dict, seed: int, arm: str = "full"
) -> dict:
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    command = 0.0
    goal = 0.0
    switches = 0
    trace = []
    while not environment.done:
        reconstructed = reconstruct_image(retina, retina.encode(image))
        channels = local_visual_channels(reconstructed, visual_config)
        distance_profile = obstacle_distance_profile(reconstructed, visual_config)
        if channels["danger"] >= float(candidate["active_danger_threshold"]):
            goal, switched = _corridor_goal(
                distance_profile,
                int(candidate["corridor_width_columns"]),
                goal,
                float(candidate["switch_margin"]),
            )
            switches += int(switched)
        else:
            goal *= 0.85
        goal_signal = goal
        if arm == "no_visual_goal":
            goal_signal = 0.0
        elif arm == "reversed_visual_goal":
            goal_signal = -goal_signal
        elif arm != "full":
            raise ValueError(f"unknown corridor goal arm: {arm}")
        target = np.tanh(
            float(candidate["goal_gain"]) * channels["danger"] * goal_signal
            + float(policy["road_gain"]) * channels["road_centroid"]
            - float(policy["heading_gain"]) * environment.heading
        )
        command = (
            float(policy["smoothing_previous"]) * command
            + float(policy["smoothing_current"]) * target
        )
        image, _, _ = environment.step(command, float(policy["throttle"]), 0.0)
        trace.append(
            {
                "danger": channels["danger"],
                "visual_goal": goal,
                "steering_command": command,
                "vehicle_x": environment.x,
                "vehicle_y": environment.y,
            }
        )
    return {
        "seed": seed,
        "terminal_reason": environment.terminal_reason,
        "success": environment.terminal_reason == "success",
        "obstacles_passed": environment.obstacles_passed,
        "distance": environment.y,
        "steps": environment.steps,
        "goal_switch_count": switches,
        "maximum_absolute_lateral_position": float(
            max(abs(item["vehicle_x"]) for item in trace)
        ),
    }


def _grid(config: dict) -> list[dict]:
    values = config["candidate_grid"]
    return [
        {
            "corridor_width_columns": int(width),
            "goal_gain": float(gain),
            "switch_margin": float(margin),
            "active_danger_threshold": float(threshold),
        }
        for width in values["corridor_width_columns"]
        for gain in values["goal_gain"]
        for margin in values["switch_margin"]
        for threshold in values["active_danger_threshold"]
    ]


def _summary(candidate: dict, episodes: list[dict]) -> dict:
    return {
        "candidate": candidate,
        "success_count": sum(item["success"] for item in episodes),
        "total_obstacles_passed": sum(item["obstacles_passed"] for item in episodes),
        "road_boundary_count": sum(
            item["terminal_reason"] == "road_boundary" for item in episodes
        ),
        "mean_maximum_absolute_lateral_position": float(
            np.mean([item["maximum_absolute_lateral_position"] for item in episodes])
        ),
        "total_goal_switch_count": sum(item["goal_switch_count"] for item in episodes),
        "episodes": episodes,
    }


def _key(summary: dict) -> tuple:
    candidate = summary["candidate"]
    return (
        -summary["success_count"],
        -summary["total_obstacles_passed"],
        summary["road_boundary_count"],
        summary["mean_maximum_absolute_lateral_position"],
        summary["total_goal_switch_count"],
        candidate["corridor_width_columns"],
        candidate["goal_gain"],
        candidate["switch_margin"],
        candidate["active_danger_threshold"],
    )


def evaluate_v7_visual_corridor_goal(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_path = Path(config["nested_protocol"])
    nested = yaml.safe_load((root / nested_path).read_text())
    visual_path = Path(config["input_upper_bound_config"])
    visual = yaml.safe_load((root / visual_path).read_text())
    tuning_seeds = [
        int(seed)
        for item in nested["conditions"]
        if item["role"] == "tuning"
        for seed in item["mirror_pair_seeds"]
    ]
    if tuning_seeds != [int(seed) for seed in config["tuning_seeds"]]:
        raise ValueError("corridor tuning seeds differ from nested protocol")
    retina = build_mass_balanced_retina(root)
    summaries = [
        _summary(
            candidate,
            [
                _episode(retina, visual, config["fixed_policy"], candidate, seed)
                for seed in tuning_seeds
            ],
        )
        for candidate in _grid(config)
    ]
    selected = min(summaries, key=_key)
    controls = {
        arm: _summary(
            selected["candidate"],
            [
                _episode(
                    retina, visual, config["fixed_policy"], selected["candidate"], seed, arm
                )
                for seed in tuning_seeds
            ],
        )
        for arm in config["controls"]
    }
    passed = selected["success_count"] == len(tuning_seeds) and selected[
        "total_obstacles_passed"
    ] == 9 * len(tuning_seeds)
    controls_passed = all(
        item["success_count"] < selected["success_count"] for item in controls.values()
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(nested_path): _sha256(root / nested_path),
                str(visual_path): _sha256(root / visual_path),
            },
            "candidate_count": len(summaries),
            "tuning_only": True,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "environment_geometry_used": False,
            "runtime_modified": False,
        },
        "candidate_summaries": summaries,
        "selected_candidate": selected["candidate"],
        "selected_summary": selected,
        "controls": controls,
        "tuning_passed": bool(passed),
        "causal_controls_passed": bool(controls_passed),
        "advance_to_neural_cv": bool(passed and controls_passed),
        "advance_to_calibration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
