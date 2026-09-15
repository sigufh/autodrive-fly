from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_closed_loop import (
    IMPLEMENTATION as CLOSED_LOOP_IMPLEMENTATION,
)
from fly_emotion.driving.v7_closed_loop import (
    ClosedLoopVisualCore,
    EventParameters,
    NeuralEventAdapter,
)
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-closed-loop-multi.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_closed_loop_multi.py")


def _frozen_episode(root: Path, closed_loop: dict, candidate: dict, seed: int) -> dict:
    environment = DrivingEnvironment(
        width=int(closed_loop["visual_core"]["width"]),
        height=int(closed_loop["visual_core"]["height"]),
    )
    image = environment.reset(seed)
    core = ClosedLoopVisualCore(root, closed_loop)
    core.reset(image)
    adapter = NeuralEventAdapter(
        EventParameters(
            signal_threshold=float(candidate["signal_threshold"]),
            turn_steps=int(candidate["turn_steps"]),
            counter_turn_steps=int(candidate["counter_turn_steps"]),
            steering_amplitude=float(candidate["steering_amplitude"]),
            fixed_throttle=float(candidate["fixed_throttle"]),
        )
    )
    signals: list[float] = []
    actions: list[float] = []
    trajectory_x: list[float] = []
    while not environment.done:
        signal = core.step(image)["neural_spatial_moment"]
        steering, throttle, reverse = adapter.step(signal)
        image, _, _ = environment.step(steering, throttle, reverse)
        signals.append(float(signal))
        actions.append(float(steering))
        trajectory_x.append(float(environment.x))
    return {
        "seed": seed,
        "pair_seed": environment.pair_seed,
        "mirror": environment.mirror,
        "terminal_reason": environment.terminal_reason,
        "success": environment.terminal_reason == "success",
        "obstacles_passed": environment.obstacles_passed,
        "first_obstacle_passed": 0 in environment.passed_obstacle_indices,
        "distance": environment.y,
        "steps": environment.steps,
        "event_count": int(any(value != 0 for value in actions)),
        "maximum_absolute_neural_signal": float(np.max(np.abs(signals))),
        "maximum_absolute_lateral_position": float(np.max(np.abs(trajectory_x))),
        "signal_trace_sha256": hashlib.sha256(
            np.asarray(signals, dtype="<f8").tobytes()
        ).hexdigest(),
        "action_trace_sha256": hashlib.sha256(
            np.asarray(actions, dtype="<f8").tobytes()
        ).hexdigest(),
        "trajectory_x": trajectory_x,
        "signals": signals,
        "actions": actions,
    }


def _mirror_checks(episodes: list[dict]) -> list[dict]:
    checks = []
    for first, second in zip(episodes[::2], episodes[1::2], strict=True):
        count = min(first["steps"], second["steps"])
        checks.append(
            {
                "seeds": [first["seed"], second["seed"]],
                "paired_steps": count,
                "maximum_signal_odd_error": float(
                    np.max(
                        np.abs(
                            np.asarray(first["signals"][:count])
                            + np.asarray(second["signals"][:count])
                        )
                    )
                ),
                "maximum_action_odd_error": float(
                    np.max(
                        np.abs(
                            np.asarray(first["actions"][:count])
                            + np.asarray(second["actions"][:count])
                        )
                    )
                ),
                "maximum_trajectory_mirror_error": float(
                    np.max(
                        np.abs(
                            np.asarray(first["trajectory_x"][:count])
                            + np.asarray(second["trajectory_x"][:count])
                        )
                    )
                ),
            }
        )
    return checks


def evaluate_v7_closed_loop_multi(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    closed_loop_path = Path(config["closed_loop_config"])
    tuning_path = Path(config["tuning_evidence"])
    calibration_path = Path(config["calibration_evidence"])
    closed_loop = yaml.safe_load((root / closed_loop_path).read_text())
    tuning = json.loads((root / tuning_path).read_text())
    calibration = json.loads((root / calibration_path).read_text())
    if not calibration["calibration_passed"]:
        raise ValueError("multi-obstacle diagnostic requires passed calibration")
    if calibration["protocol"]["candidate_sha256"] != tuning["selected_candidate_sha256"]:
        raise ValueError("frozen candidate mismatch")
    if config["parameter_search_allowed"] or config["candidate_changes_allowed"]:
        raise ValueError("multi-obstacle diagnostic must not tune the candidate")
    seeds = [int(seed) for seed in config["diagnostic_mirror_pair_seeds"]]
    episodes = [
        _frozen_episode(root, closed_loop, tuning["selected_candidate"], seed) for seed in seeds
    ]
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(CLOSED_LOOP_IMPLEMENTATION): _sha256(root / CLOSED_LOOP_IMPLEMENTATION),
                str(closed_loop_path): _sha256(root / closed_loop_path),
                str(tuning_path): _sha256(root / tuning_path),
                str(calibration_path): _sha256(root / calibration_path),
            },
            "candidate_sha256": tuning["selected_candidate_sha256"],
            "candidate_parameters_changed": False,
            "parameter_search_performed": False,
            "visual_input_only_via_R1_R6": True,
            "environment_state_used_by_controller": False,
            "runtime_modified": False,
        },
        "seeds": seeds,
        "episodes": episodes,
        "mirror_checks": _mirror_checks(episodes),
        "summary": {
            "success_count": sum(item["success"] for item in episodes),
            "episode_count": len(episodes),
            "first_obstacle_pass_count": sum(item["first_obstacle_passed"] for item in episodes),
            "total_obstacles_passed": sum(item["obstacles_passed"] for item in episodes),
            "mean_obstacles_passed": float(
                np.mean([item["obstacles_passed"] for item in episodes])
            ),
            "mean_distance": float(np.mean([item["distance"] for item in episodes])),
            "terminal_counts": {
                name: sum(item["terminal_reason"] == name for item in episodes)
                for name in ("success", "obstacle", "road_boundary", "timeout")
            },
        },
        "diagnostic_passed": all(item["success"] for item in episodes),
        "failure_interpretation": (
            "the frozen one-shot adapter cannot express repeated hazard responses"
            if not all(item["success"] for item in episodes)
            else None
        ),
        "next_candidate_requirement": (
            "retriggerable neural hazard state with self-motion feedback, fitted only on a new "
            "multi-obstacle development split"
        ),
        "boundary": config["boundary"],
        "final_evaluated": False,
        "advance_to_navigation_release": False,
    }
