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
from fly_emotion.driving.v7_lamina_goal_symmetry import _transform_features
from fly_emotion.driving.v7_neural_channels import StructuredNeuralFeatures, _predict
from fly_emotion.driving.v7_neural_corridor import _current_mean_features
from fly_emotion.driving.v7_neural_episode_cv import (
    _collect_teacher_features,
    _feature_mask,
    _fit_from_cache,
)

CONFIG = Path("configs/driving-v7-fc2-goal-memory.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_fc2_goal_memory.py")


def _fit_innovation(
    cache: dict, seeds: list[int], alpha: float, parity: str, group_count: int, rho: float
) -> dict:
    matrices, targets = [], []
    for seed in seeds:
        matrix = _transform_features(cache[seed]["features"], parity, group_count)
        target = cache[seed]["targets"]
        previous = np.r_[0.0, target[:-1]]
        matrices.append(matrix)
        targets.append(target - rho * previous)
    matrix = np.concatenate(matrices)
    target = np.concatenate(targets)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-7] = 1.0
    normalized = (matrix - mean) / scale
    zero_intercept = parity != "even_and_odd"
    if zero_intercept:
        coefficient = np.linalg.solve(
            normalized.T @ normalized + float(alpha) * np.eye(normalized.shape[1]),
            normalized.T @ target,
        )
        intercept = 0.0
        prediction = normalized @ coefficient
    else:
        design = np.column_stack((np.ones(len(normalized)), normalized))
        penalty = np.diag([0.0] + [float(alpha)] * normalized.shape[1])
        fitted = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        intercept = float(fitted[0])
        coefficient = fitted[1:]
        prediction = design @ fitted
    denominator = np.sum((target - np.mean(target)) ** 2)
    return {
        "alpha": float(alpha),
        "memory_rho": float(rho),
        "feature_parity": parity,
        "group_count": group_count,
        "zero_intercept": zero_intercept,
        "intercept": intercept,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "fit_r2": float(
            1.0 - np.sum((target - prediction) ** 2) / max(float(denominator), 1e-12)
        ),
        "training_seeds": seeds,
    }


def _innovation(model: dict, feature: np.ndarray) -> float:
    transformed = _transform_features(
        feature[None, :], model["feature_parity"], int(model["group_count"])
    )[0]
    normalized = (transformed - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return float(model["intercept"] + normalized @ np.asarray(model["coefficient"]))


def _episode(local_features, global_features, goal_model, global_model, teacher, heading, seed):
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    local_features.reset()
    global_features.reset()
    ring = EPGPENPEGHeadingRing(heading)
    decoded_heading = ring.decode()
    command = 0.0
    goal = 0.0
    goal_trace = []
    while not environment.done:
        local_even, local_odd = local_features.step(image)
        global_even, global_odd = global_features.step(image)
        danger, _asymmetry, road = _predict(global_model, global_even, global_odd)
        update = _innovation(
            goal_model, _current_mean_features(local_features, local_even, local_odd)
        )
        goal = float(
            np.clip(float(goal_model["memory_rho"]) * goal + update, -1.0, 1.0)
        )
        target = np.tanh(
            float(teacher["obstacle_gain"]) * danger * goal
            + float(teacher["road_gain"]) * road
            - float(teacher["heading_gain"]) * decoded_heading
        )
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


def _mirror_goal_error(episodes: list[dict]) -> float:
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
    candidate = result["candidate"]
    return (
        -result["held_out_success_count"],
        -result["held_out_obstacles_passed"],
        result["maximum_mirror_goal_error"],
        candidate["memory_rho"],
        candidate["feature_parity"],
    )


def evaluate_v7_fc2_goal_memory(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    lamina_path = Path(config["lamina_goal_config"])
    nested_path = Path(config["nested_protocol"])
    base_path = Path(config["base_neural_config"])
    corridor_path = Path(config["neural_corridor_config"])
    lamina_config = yaml.safe_load((root / lamina_path).read_text())
    nested = yaml.safe_load((root / nested_path).read_text())
    base = deepcopy(yaml.safe_load((root / base_path).read_text()))
    base["input"]["visual_graph"] = "typed_visual_leak_v1"
    corridor = yaml.safe_load((root / corridor_path).read_text())
    heading = yaml.safe_load((root / corridor["heading_config"]).read_text())
    teacher = json.loads((root / base["adapter_source"]).read_text())["selected_candidate"]
    pairs = [[int(seed) for seed in pair] for pair in config["tuning_pairs"]]
    protocol_pairs = [
        [int(seed) for seed in item["mirror_pair_seeds"]]
        for item in nested["conditions"]
        if item["role"] == "tuning"
    ]
    if pairs != protocol_pairs:
        raise ValueError("FC2 memory pairs differ from nested protocol")
    seeds = [seed for pair in pairs for seed in pair]
    local_features, cache = _collect(root, base, lamina_config, seeds, teacher)
    group_count = len(local_features.groups)
    global_features, global_cache = _collect_teacher_features(root, base, seeds, teacher)
    global_variant = {
        "name": "current_mean",
        "statistics": ["mean"],
        "temporal_terms": ["current"],
    }
    global_mask = _feature_mask(base, global_features, global_variant)
    candidates = [
        {"feature_parity": parity, "memory_rho": float(rho)}
        for parity in config["candidate_grid"]["feature_parity"]
        for rho in config["candidate_grid"]["memory_rho"]
    ]
    summaries = []
    for candidate in candidates:
        folds = []
        for held_out in pairs:
            train = [seed for seed in seeds if seed not in held_out]
            goal_model = _fit_innovation(
                cache,
                train,
                float(config["ridge_alpha"]),
                candidate["feature_parity"],
                group_count,
                candidate["memory_rho"],
            )
            global_model = _fit_from_cache(
                global_features,
                global_cache,
                train,
                1.0,
                feature_mask=global_mask,
                feature_variant=global_variant["name"],
            )
            episodes = [
                _episode(
                    local_features,
                    StructuredNeuralFeatures(root, base),
                    goal_model,
                    global_model,
                    teacher,
                    heading,
                    seed,
                )
                for seed in held_out
            ]
            folds.append(
                {
                    "training_seeds": train,
                    "held_out_seeds": held_out,
                    "goal_model": goal_model,
                    "success_count": sum(item["success"] for item in episodes),
                    "total_obstacles_passed": sum(
                        item["obstacles_passed"] for item in episodes
                    ),
                    "maximum_mirror_goal_error": _mirror_goal_error(episodes),
                    "episodes": [
                        {key: value for key, value in item.items() if key != "goal_trace"}
                        for item in episodes
                    ],
                }
            )
        summaries.append(
            {
                "candidate": candidate,
                "held_out_success_count": sum(item["success_count"] for item in folds),
                "held_out_obstacles_passed": sum(
                    item["total_obstacles_passed"] for item in folds
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
                str(lamina_path): _sha256(root / lamina_path),
                str(nested_path): _sha256(root / nested_path),
                str(base_path): _sha256(root / base_path),
                str(corridor_path): _sha256(root / corridor_path),
            },
            "tuning_only": True,
            "mirror_pair_is_indivisible_cv_unit": True,
            "odd_heads_have_zero_intercept": True,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "candidate_summaries": summaries,
        "selected_candidate": selected["candidate"],
        "selected_summary": selected,
        "cross_validation_passed": bool(passed),
        "advance_to_full_tuning": bool(passed),
        "advance_to_calibration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
