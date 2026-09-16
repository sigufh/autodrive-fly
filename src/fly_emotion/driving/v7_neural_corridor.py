from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_heading_ring import EPGPENPEGHeadingRing
from fly_emotion.driving.v7_neural_channels import (
    StructuredNeuralFeatures,
    _predict,
    _teacher_step,
)
from fly_emotion.driving.v7_r1r6_local import reconstruct_image
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina
from fly_emotion.driving.v7_visual_corridor_goal import obstacle_distance_profile

CONFIG = Path("configs/driving-v7-neural-corridor.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_corridor.py")


def _current_mean_features(features: StructuredNeuralFeatures, even, odd) -> np.ndarray:
    group_count = len(features.group_names)
    indices = np.arange(group_count) * features.stat_count
    return np.concatenate((even[indices], odd[indices]))


def _safety_targets(profile: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    if values.shape != (48,) or 48 % bins:
        raise ValueError("local safety target requires divisible 48-column profile")
    return np.mean(values.reshape(bins, 48 // bins), axis=1)


def _collect(root: Path, base: dict, seeds: list[int], teacher: dict, bins: int):
    teacher_config = yaml.safe_load((root / "configs/driving-v7-r1r6-local.yaml").read_text())
    retina = build_mass_balanced_retina(root)
    features = StructuredNeuralFeatures(root, base)
    cache = {}
    for seed in seeds:
        environment = DrivingEnvironment()
        image = environment.reset(seed)
        features.reset()
        command = 0.0
        rows, targets = [], []
        while not environment.done:
            even, odd = features.step(image)
            rows.append(_current_mean_features(features, even, odd))
            profile = obstacle_distance_profile(
                reconstruct_image(retina, retina.encode(image)), teacher_config
            )
            image, command, channels = _teacher_step(
                environment, image, command, retina, teacher_config, teacher
            )
            targets.append(_safety_targets(profile, bins))
        cache[seed] = {"features": np.asarray(rows), "targets": np.asarray(targets)}
    return features, cache


def _fit(cache: dict, seeds: list[int], alpha: float) -> dict:
    matrix = np.concatenate([cache[seed]["features"] for seed in seeds])
    targets = np.concatenate([cache[seed]["targets"] for seed in seeds])
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-7] = 1.0
    normalized = (matrix - mean) / scale
    design = np.column_stack((np.ones(len(normalized)), normalized))
    penalty = np.diag([0.0] + [float(alpha)] * normalized.shape[1])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    prediction = design @ coefficients
    denominator = np.sum((targets - np.mean(targets, axis=0)) ** 2, axis=0)
    fit_r2 = 1.0 - np.sum((targets - prediction) ** 2, axis=0) / np.maximum(
        denominator, 1e-12
    )
    return {
        "alpha": float(alpha),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "fit_r2": fit_r2.tolist(),
        "training_seeds": seeds,
    }


def _predict_safety(model: dict, values: np.ndarray) -> np.ndarray:
    normalized = (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
    prediction = np.r_[1.0, normalized] @ np.asarray(model["coefficients"])
    return np.clip(prediction, 0.0, 1.0)


def _goal(safety: np.ndarray, width: int, previous: float, margin: float) -> tuple[float, bool]:
    starts = np.arange(0, len(safety) - width + 1)
    scores = np.asarray([np.mean(safety[start : start + width]) for start in starts])
    centers = starts + (width - 1) / 2.0
    goals = (centers - (len(safety) - 1) / 2.0) / (len(safety) / 2.0)
    best = int(np.argmax(scores - 0.03 * np.abs(goals)))
    old = int(np.argmin(np.abs(goals - previous)))
    if scores[best] < scores[old] + margin:
        return float(goals[old]), False
    return float(goals[best]), best != old


def _episode(
    features, global_model: dict, corridor_model: dict, teacher: dict, heading, config, seed
):
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    features.reset()
    ring = EPGPENPEGHeadingRing(heading)
    decoded_heading = ring.decode()
    command = 0.0
    goal = 0.0
    switches = 0
    while not environment.done:
        even, odd = features.step(image)
        danger, _asymmetry, road = _predict(global_model, even, odd)
        safety = _predict_safety(
            corridor_model, _current_mean_features(features, even, odd)
        )
        if danger >= float(config["corridor"]["active_danger_threshold"]):
            goal, changed = _goal(
                safety,
                int(config["corridor"]["width_bins"]),
                goal,
                float(config["corridor"]["switch_margin"]),
            )
            switches += int(changed)
        else:
            goal *= 0.85
        target = np.tanh(
            float(config["corridor"]["goal_gain"]) * danger * goal
            + float(config["fixed_policy"]["road_gain"]) * road
            - float(config["fixed_policy"]["heading_gain"]) * decoded_heading
        )
        command = (
            float(config["fixed_policy"]["smoothing_previous"]) * command
            + float(config["fixed_policy"]["smoothing_current"]) * target
        )
        image, _, _ = environment.step(
            command, float(config["fixed_policy"]["throttle"]), 0.0
        )
        decoded_heading = ring.step(
            environment.last_yaw_rate, environment.dt, cue_visible=False
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
            max(abs(item[0]) for item in environment.trajectory)
        ),
    }


def evaluate_v7_neural_corridor(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_path = Path(config["nested_protocol"])
    nested = yaml.safe_load((root / nested_path).read_text())
    base_path = Path(config["base_neural_config"])
    heading_path = Path(config["heading_config"])
    upper_path = Path(config["upper_bound_evidence"])
    base = yaml.safe_load((root / base_path).read_text())
    heading = yaml.safe_load((root / heading_path).read_text())
    upper = json.loads((root / upper_path).read_text())
    if not upper["advance_to_neural_cv"]:
        raise ValueError("neural corridor requires passed receptor-level corridor evidence")
    global_candidate = json.loads((root / "artifacts/v7-neural-episode-cv.json").read_text())
    global_model = global_candidate["model"]
    teacher = json.loads((root / base["adapter_source"]).read_text())["selected_candidate"]
    pairs = [[int(seed) for seed in pair] for pair in config["tuning_pairs"]]
    protocol_pairs = [
        [int(seed) for seed in item["mirror_pair_seeds"]]
        for item in nested["conditions"]
        if item["role"] == "tuning"
    ]
    if pairs != protocol_pairs:
        raise ValueError("neural corridor tuning pairs differ from nested protocol")
    seeds = [seed for pair in pairs for seed in pair]
    features, cache = _collect(
        root, base, seeds, teacher, int(config["targets"]["local_safety_bins"])
    )
    folds = []
    for held_out in pairs:
        train = [seed for seed in seeds if seed not in held_out]
        model = _fit(cache, train, float(config["readout"]["ridge_alpha"]))
        episodes = [
            _episode(features, global_model, model, teacher, heading, config, seed)
            for seed in held_out
        ]
        folds.append(
            {
                "training_seeds": train,
                "held_out_seeds": held_out,
                "model": model,
                "success_count": sum(item["success"] for item in episodes),
                "total_obstacles_passed": sum(
                    item["obstacles_passed"] for item in episodes
                ),
                "episodes": episodes,
            }
        )
    success_count = sum(item["success_count"] for item in folds)
    obstacles = sum(item["total_obstacles_passed"] for item in folds)
    passed = success_count == len(seeds) and obstacles == 9 * len(seeds)
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(nested_path): _sha256(root / nested_path),
                str(base_path): _sha256(root / base_path),
                str(heading_path): _sha256(root / heading_path),
                str(upper_path): _sha256(root / upper_path),
            },
            "mirror_pair_is_indivisible_cv_unit": True,
            "tuning_only": True,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "folds": folds,
        "success_count": success_count,
        "total_obstacles_passed": obstacles,
        "cross_validation_passed": bool(passed),
        "advance_to_full_tuning": bool(passed),
        "advance_to_calibration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
