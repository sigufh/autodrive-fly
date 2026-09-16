from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_r1r6_local import local_visual_channels, reconstruct_image
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina

CONFIG = Path("configs/driving-v7-danger-throttle.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_danger_throttle.py")


def _throttle(candidate: dict, base: float, danger: float) -> float:
    formula = candidate["formula"]
    if formula == "base":
        value = base
    elif formula == "base_times_one_minus_gain_danger":
        value = base * (1.0 - float(candidate["gain"]) * danger)
    elif formula == "base_times_one_minus_danger_squared":
        value = base * (1.0 - danger**2)
    elif formula == "zero_above_threshold":
        value = 0.0 if danger >= float(candidate["threshold"]) else base
    elif formula == "base_times_one_plus_gain_danger":
        value = base * (1.0 + float(candidate["gain"]) * danger)
    else:
        raise ValueError(f"unknown danger throttle formula: {formula}")
    return float(np.clip(value, 0.0, 1.0))


def _episode(retina, visual_config: dict, policy: dict, candidate: dict, seed: int) -> dict:
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    command = 0.0
    base_throttle = float(policy["throttle"])
    throttle_changes = []
    steering_digest = []
    while not environment.done:
        channels = local_visual_channels(
            reconstruct_image(retina, retina.encode(image)), visual_config
        )
        target = np.tanh(
            float(policy["obstacle_gain"])
            * channels["danger"]
            * channels["obstacle_asymmetry"]
            + float(policy["road_gain"]) * channels["road_centroid"]
            - float(policy["heading_gain"]) * environment.heading
        )
        command = 0.4 * command + 0.6 * target
        throttle = _throttle(candidate, base_throttle, channels["danger"])
        throttle_changes.append(abs(throttle - base_throttle))
        steering_digest.append(command)
        image, _, _ = environment.step(command, throttle, 0.0)
    return {
        "seed": seed,
        "terminal_reason": environment.terminal_reason,
        "success": environment.terminal_reason == "success",
        "obstacles_passed": environment.obstacles_passed,
        "distance": environment.y,
        "steps": environment.steps,
        "mean_absolute_throttle_change": float(np.mean(throttle_changes)),
        "maximum_absolute_lateral_position": float(
            max(abs(item[0]) for item in environment.trajectory)
        ),
        "steering_policy_trace_sha256": __import__("hashlib").sha256(
            np.asarray(steering_digest, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def _summary(candidate: dict, episodes: list[dict]) -> dict:
    return {
        "candidate": candidate,
        "success_count": sum(item["success"] for item in episodes),
        "total_obstacles_passed": sum(item["obstacles_passed"] for item in episodes),
        "timeout_count": sum(item["terminal_reason"] == "timeout" for item in episodes),
        "mean_absolute_throttle_change": float(
            np.mean([item["mean_absolute_throttle_change"] for item in episodes])
        ),
        "episodes": episodes,
    }


def _key(summary: dict) -> tuple:
    return (
        -summary["success_count"],
        -summary["total_obstacles_passed"],
        summary["timeout_count"],
        summary["mean_absolute_throttle_change"],
        summary["candidate"]["name"],
    )


def evaluate_v7_danger_throttle(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_path = Path(config["nested_protocol"])
    nested = yaml.safe_load((root / nested_path).read_text())
    visual_path = Path(config["input_upper_bound_config"])
    visual = yaml.safe_load((root / visual_path).read_text())
    policy = json.loads((root / "artifacts/v7-r1r6-local-tuning.json").read_text())[
        "selected_candidate"
    ]
    if float(policy["throttle"]) != float(config["base_throttle"]):
        raise ValueError("danger throttle base changed from frozen local policy")
    protocol_tuning = [
        int(seed)
        for item in nested["conditions"]
        if item["role"] == "tuning"
        for seed in item["mirror_pair_seeds"]
    ]
    if protocol_tuning != [int(seed) for seed in config["tuning_seeds"]]:
        raise ValueError("danger throttle tuning seeds differ from nested protocol")
    retina = build_mass_balanced_retina(root)
    summaries = [
        _summary(
            candidate,
            [_episode(retina, visual, policy, candidate, seed) for seed in protocol_tuning],
        )
        for candidate in config["candidates"]
    ]
    selected = min(summaries, key=_key)
    negative = _summary(
        config["negative_control"],
        [
            _episode(retina, visual, policy, config["negative_control"], seed)
            for seed in protocol_tuning
        ],
    )
    passed = selected["success_count"] == len(protocol_tuning) and selected[
        "total_obstacles_passed"
    ] == 9 * len(protocol_tuning)
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(nested_path): _sha256(root / nested_path),
                str(visual_path): _sha256(root / visual_path),
            },
            "tuning_only": True,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "steering_policy_changed": False,
            "runtime_modified": False,
        },
        "candidate_summaries": summaries,
        "selected_candidate": selected["candidate"],
        "selected_summary": selected,
        "negative_control": negative,
        "tuning_passed": bool(passed),
        "negative_control_reduces_success": negative["success_count"]
        < selected["success_count"],
        "advance_to_neural_cv": bool(passed),
        "advance_to_calibration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
