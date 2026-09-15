from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_closed_loop import (
    CALIBRATION_ARTIFACT,
    CONFIG,
    TUNING_ARTIFACT,
    ClosedLoopVisualCore,
    EventParameters,
    NeuralEventAdapter,
)
from fly_emotion.driving.v7_closed_loop import (
    IMPLEMENTATION as CLOSED_LOOP_IMPLEMENTATION,
)
from fly_emotion.driving.v7_geometry_sign import _sha256

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_closed_loop_controls.py")


def _control_episode(
    root: Path,
    config: dict,
    candidate: dict,
    seed: int,
    *,
    signal_scale: float = 1.0,
    action_scale: float = 1.0,
    receptive_field_radius: int | None = None,
) -> dict:
    local_config = copy.deepcopy(config)
    if receptive_field_radius is not None:
        local_config["visual_core"]["same_eye_receptive_field"]["radius_pixels"] = int(
            receptive_field_radius
        )
    core = ClosedLoopVisualCore(root, local_config)
    environment = DrivingEnvironment(
        width=int(config["visual_core"]["width"]),
        height=int(config["visual_core"]["height"]),
    )
    image = environment.reset(seed)
    environment.configure_curriculum(config["tuning"]["curriculum_stage"])
    image = environment.observe()
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
    neural_signals: list[float] = []
    steering_commands: list[float] = []
    while not environment.done:
        neural = core.step(image)["neural_spatial_moment"]
        steering, throttle, reverse = adapter.step(signal_scale * neural)
        steering *= action_scale
        image, _, _ = environment.step(steering, throttle, reverse)
        neural_signals.append(float(neural))
        steering_commands.append(float(steering))
    return {
        "seed": seed,
        "terminal_reason": environment.terminal_reason,
        "success": environment.terminal_reason == "success",
        "first_obstacle_passed": 0 in environment.passed_obstacle_indices,
        "distance": environment.y,
        "steps": environment.steps,
        "maximum_absolute_neural_signal": float(np.max(np.abs(neural_signals))),
        "maximum_absolute_steering_command": float(np.max(np.abs(steering_commands))),
    }


def evaluate_v7_closed_loop_controls(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    tuning = json.loads((root / TUNING_ARTIFACT).read_text())
    calibration = json.loads((root / CALIBRATION_ARTIFACT).read_text())
    if not tuning["tuning_passed"] or not calibration["calibration_passed"]:
        raise ValueError("closed-loop controls require passed tuning and calibration evidence")
    if calibration["protocol"]["candidate_sha256"] != tuning["selected_candidate_sha256"]:
        raise ValueError("calibration and tuning candidates differ")
    candidate = tuning["selected_candidate"]
    seeds = [int(seed) for pair in config["tuning"]["condition_pairs"].values() for seed in pair]
    arms = {
        "frozen_candidate": {
            "signal_scale": 1.0,
            "action_scale": 1.0,
            "receptive_field_radius": None,
        },
        "zero_action": {
            "signal_scale": 1.0,
            "action_scale": 0.0,
            "receptive_field_radius": None,
        },
        "wrong_action_sign": {
            "signal_scale": -1.0,
            "action_scale": 1.0,
            "receptive_field_radius": None,
        },
        "point_sampled_R1_R6": {
            "signal_scale": 1.0,
            "action_scale": 1.0,
            "receptive_field_radius": 0,
        },
    }
    results = {}
    for name, values in arms.items():
        episodes = [_control_episode(root, config, candidate, seed, **values) for seed in seeds]
        results[name] = {
            "success_count": sum(item["success"] for item in episodes),
            "first_obstacle_pass_count": sum(item["first_obstacle_passed"] for item in episodes),
            "terminal_counts": {
                terminal: sum(item["terminal_reason"] == terminal for item in episodes)
                for terminal in ("success", "obstacle", "road_boundary", "timeout")
            },
            "mean_distance": float(np.mean([item["distance"] for item in episodes])),
            "episodes": episodes,
        }
    reference = results["frozen_candidate"]
    gates = {
        "frozen_candidate_replays_all_successes": reference["success_count"] == len(seeds),
        "zero_action_reduces_success": results["zero_action"]["success_count"]
        < reference["success_count"],
        "wrong_sign_reduces_success": results["wrong_action_sign"]["success_count"]
        < reference["success_count"],
        "point_sampling_reduces_success": results["point_sampled_R1_R6"]["success_count"]
        < reference["success_count"],
    }
    return {
        "protocol": {
            "name": config["name"],
            "phase": "post_calibration_causal_controls",
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(CLOSED_LOOP_IMPLEMENTATION): _sha256(root / CLOSED_LOOP_IMPLEMENTATION),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(TUNING_ARTIFACT): _sha256(root / TUNING_ARTIFACT),
                str(CALIBRATION_ARTIFACT): _sha256(root / CALIBRATION_ARTIFACT),
            },
            "candidate_sha256": tuning["selected_candidate_sha256"],
            "candidate_parameters_changed": False,
            "control_results_used_for_selection": False,
            "visual_input_only_via_R1_R6": True,
            "environment_state_used_by_controller": False,
            "runtime_modified": False,
        },
        "seeds": seeds,
        "arms": results,
        "gates": gates,
        "causal_controls_passed": all(gates.values()),
        "final_evaluated": False,
        "advance_to_navigation_release": False,
    }
