from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_heading_ring import EPGPENPEGHeadingRing
from fly_emotion.driving.v7_lamina_goal import _collect
from fly_emotion.driving.v7_lamina_goal_symmetry import _fit_goal, _raw_goal
from fly_emotion.driving.v7_neural_channels import StructuredNeuralFeatures, _predict
from fly_emotion.driving.v7_neural_corridor import _current_mean_features
from fly_emotion.driving.v7_neural_episode_cv import (
    _collect_teacher_features,
    _feature_mask,
    _fit_from_cache,
)

CONFIG = Path("configs/driving-v7-neural-goal-fusion.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_goal_fusion.py")


def _episode(
    local_features, global_features, lamina_model, global_model, teacher, heading, weight, seed
):
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    local_features.reset()
    global_features.reset()
    ring = EPGPENPEGHeadingRing(heading)
    decoded_heading = ring.decode()
    command = 0.0
    goal_trace = []
    while not environment.done:
        local_even, local_odd = local_features.step(image)
        global_even, global_odd = global_features.step(image)
        danger, asymmetry, road = _predict(global_model, global_even, global_odd)
        lamina_goal = _raw_goal(
            lamina_model, _current_mean_features(local_features, local_even, local_odd)
        )
        global_goal = (
            float(teacher["obstacle_gain"]) * danger * asymmetry
            + float(teacher["road_gain"]) * road
        ) / float(teacher["heading_gain"])
        goal = float((1.0 - weight) * global_goal + weight * lamina_goal)
        target = np.tanh(float(teacher["heading_gain"]) * (goal - decoded_heading))
        command = 0.4 * command + 0.6 * target
        image, _, _ = environment.step(command, float(teacher["throttle"]), 0.0)
        decoded_heading = ring.step(
            environment.last_yaw_rate, environment.dt, cue_visible=False
        )
        goal_trace.append(goal)
    return {
        "seed": seed,
        "terminal_reason": environment.terminal_reason,
        "success": environment.terminal_reason == "success",
        "obstacles_passed": environment.obstacles_passed,
        "distance": environment.y,
        "steps": environment.steps,
        "maximum_absolute_lateral_position": float(
            max(abs(item[0]) for item in environment.trajectory)
        ),
        "goal_trace": goal_trace,
    }


def _mirror_error(episodes: list[dict]) -> float:
    errors = []
    for first, second in zip(episodes[::2], episodes[1::2], strict=True):
        count = min(first["steps"], second["steps"])
        errors.extend(
            abs(left + right)
            for left, right in zip(
                first["goal_trace"][:count], second["goal_trace"][:count], strict=True
            )
        )
    return float(max(errors))


def _key(result: dict) -> tuple:
    return (
        -result["held_out_success_count"],
        -result["held_out_obstacles_passed"],
        result["maximum_mirror_goal_error"],
        result["lamina_weight"],
    )


def evaluate_v7_neural_goal_fusion(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_path = Path(config["nested_protocol"])
    base_path = Path(config["base_neural_config"])
    lamina_path = Path(config["lamina_goal_config"])
    symmetry_path = Path(config["lamina_symmetry_evidence"])
    heading_path = Path(config["heading_config"])
    nested = yaml.safe_load((root / nested_path).read_text())
    base = deepcopy(yaml.safe_load((root / base_path).read_text()))
    base["input"]["visual_graph"] = "typed_visual_leak_v1"
    lamina = yaml.safe_load((root / lamina_path).read_text())
    symmetry = json.loads((root / symmetry_path).read_text())
    heading = yaml.safe_load((root / heading_path).read_text())
    if symmetry["selected_candidate"]["feature_parity"] != config[
        "lamina_feature_parity"
    ]:
        raise ValueError("fusion lamina parity differs from frozen screen")
    teacher = json.loads((root / base["adapter_source"]).read_text())["selected_candidate"]
    pairs = [[int(seed) for seed in pair] for pair in config["tuning_pairs"]]
    protocol_pairs = [
        [int(seed) for seed in item["mirror_pair_seeds"]]
        for item in nested["conditions"]
        if item["role"] == "tuning"
    ]
    if pairs != protocol_pairs:
        raise ValueError("fusion pairs differ from nested protocol")
    seeds = [seed for pair in pairs for seed in pair]
    local_features, local_cache = _collect(root, base, lamina, seeds, teacher)
    global_features, global_cache = _collect_teacher_features(root, base, seeds, teacher)
    global_variant = {
        "name": "current_mean",
        "statistics": ["mean"],
        "temporal_terms": ["current"],
    }
    global_mask = _feature_mask(base, global_features, global_variant)
    summaries = []
    for weight in config["fusion_weights"]:
        folds = []
        for held_out in pairs:
            train = [seed for seed in seeds if seed not in held_out]
            lamina_model = _fit_goal(
                local_cache,
                train,
                float(config["ridge_alpha"]),
                config["lamina_feature_parity"],
                len(local_features.groups),
            )
            global_model = _fit_from_cache(
                global_features,
                global_cache,
                train,
                float(config["ridge_alpha"]),
                feature_mask=global_mask,
                feature_variant=global_variant["name"],
            )
            episodes = [
                _episode(
                    local_features,
                    StructuredNeuralFeatures(root, base),
                    lamina_model,
                    global_model,
                    teacher,
                    heading,
                    float(weight),
                    seed,
                )
                for seed in held_out
            ]
            folds.append(
                {
                    "training_seeds": train,
                    "held_out_seeds": held_out,
                    "success_count": sum(item["success"] for item in episodes),
                    "total_obstacles_passed": sum(
                        item["obstacles_passed"] for item in episodes
                    ),
                    "maximum_mirror_goal_error": _mirror_error(episodes),
                    "episodes": [
                        {key: value for key, value in item.items() if key != "goal_trace"}
                        for item in episodes
                    ],
                }
            )
        all_episodes = [episode for fold in folds for episode in fold["episodes"]]
        summaries.append(
            {
                "lamina_weight": float(weight),
                "held_out_success_count": sum(item["success"] for item in all_episodes),
                "held_out_obstacles_passed": sum(
                    item["obstacles_passed"] for item in all_episodes
                ),
                "maximum_mirror_goal_error": max(
                    item["maximum_mirror_goal_error"] for item in folds
                ),
                "folds": folds,
            }
        )
    selected = min(summaries, key=_key)
    passed = (
        selected["held_out_success_count"] == len(seeds)
        and selected["held_out_obstacles_passed"] == 9 * len(seeds)
        and selected["maximum_mirror_goal_error"]
        <= float(config["gates"]["maximum_mirror_goal_error"])
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(nested_path): _sha256(root / nested_path),
                str(base_path): _sha256(root / base_path),
                str(lamina_path): _sha256(root / lamina_path),
                str(symmetry_path): _sha256(root / symmetry_path),
                str(heading_path): _sha256(root / heading_path),
            },
            "tuning_only": True,
            "one_global_fusion_weight_across_folds": True,
            "mirror_pair_is_indivisible_cv_unit": True,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "candidate_summaries": summaries,
        "selected_lamina_weight": selected["lamina_weight"],
        "selected_summary": selected,
        "cross_validation_passed": bool(passed),
        "advance_to_full_tuning": bool(passed),
        "advance_to_calibration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
