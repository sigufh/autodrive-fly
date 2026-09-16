from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_heading_ring import EPGPENPEGHeadingRing
from fly_emotion.driving.v7_neural_channels import StructuredNeuralFeatures, _predict
from fly_emotion.driving.v7_neural_corridor import _current_mean_features
from fly_emotion.driving.v7_neural_episode_cv import (
    _collect_teacher_features,
    _feature_mask,
    _fit_from_cache,
)
from fly_emotion.driving.v7_r1r6_local import (
    local_visual_channels,
    reconstruct_image,
)
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina
from fly_emotion.driving.v7_visual_corridor_goal import (
    _corridor_goal,
    obstacle_distance_profile,
)
from fly_emotion.driving.v7_visual_layer_locality import VisualLayerFeatures

CONFIG = Path("configs/driving-v7-lamina-goal.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_lamina_goal.py")


def _collect(root: Path, base: dict, config: dict, seeds: list[int], teacher: dict):
    features = VisualLayerFeatures(
        root,
        base,
        config["family"],
        list(config["cell_types"]),
        int(config["spatial_bins"]),
    )
    visual_config = yaml.safe_load((root / "configs/driving-v7-r1r6-local.yaml").read_text())
    corridor_evidence = json.loads((root / config["visual_corridor_evidence"]).read_text())
    corridor = corridor_evidence["selected_candidate"]
    retina = build_mass_balanced_retina(root)
    cache = {}
    for seed in seeds:
        environment = DrivingEnvironment()
        image = environment.reset(seed)
        features.reset()
        command = 0.0
        goal = 0.0
        rows, targets = [], []
        while not environment.done:
            even, odd = features.step(image)
            rows.append(_current_mean_features(features, even, odd))
            reconstructed = reconstruct_image(retina, retina.encode(image))
            channels = local_visual_channels(reconstructed, visual_config)
            if channels["danger"] >= float(corridor["active_danger_threshold"]):
                goal, _changed = _corridor_goal(
                    obstacle_distance_profile(reconstructed, visual_config),
                    int(corridor["corridor_width_columns"]),
                    goal,
                    float(corridor["switch_margin"]),
                )
            else:
                goal *= 0.85
            targets.append(goal)
            target = np.tanh(
                float(teacher["obstacle_gain"])
                * channels["danger"]
                * channels["obstacle_asymmetry"]
                + float(teacher["road_gain"]) * channels["road_centroid"]
                - float(teacher["heading_gain"]) * environment.heading
            )
            command = 0.4 * command + 0.6 * target
            image, _, _ = environment.step(command, float(teacher["throttle"]), 0.0)
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
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    prediction = design @ coefficient
    denominator = np.sum((targets - np.mean(targets)) ** 2)
    return {
        "alpha": float(alpha),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "fit_r2": float(
            1.0 - np.sum((targets - prediction) ** 2) / max(float(denominator), 1e-12)
        ),
        "training_seeds": seeds,
    }


def _predict_goal(model: dict, feature: np.ndarray) -> float:
    normalized = (feature - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return float(np.clip(np.r_[1.0, normalized] @ np.asarray(model["coefficient"]), -1.0, 1.0))


def _episode(local_features, global_features, goal_model, global_model, teacher, heading, seed):
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    local_features.reset()
    global_features.reset()
    ring = EPGPENPEGHeadingRing(heading)
    decoded_heading = ring.decode()
    command = 0.0
    while not environment.done:
        local_even, local_odd = local_features.step(image)
        global_even, global_odd = global_features.step(image)
        danger, _asymmetry, road = _predict(global_model, global_even, global_odd)
        goal = _predict_goal(
            goal_model, _current_mean_features(local_features, local_even, local_odd)
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
    }


def evaluate_v7_lamina_goal(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_path = Path(config["nested_protocol"])
    base_path = Path(config["base_neural_config"])
    local_path = Path(config["local_column_config"])
    corridor_path = Path(config["neural_corridor_config"])
    upper_path = Path(config["visual_corridor_evidence"])
    layer_path = Path(config["visual_layer_evidence"])
    nested = yaml.safe_load((root / nested_path).read_text())
    base = deepcopy(yaml.safe_load((root / base_path).read_text()))
    base["input"]["visual_graph"] = "typed_visual_leak_v1"
    heading = yaml.safe_load(
        (root / yaml.safe_load((root / corridor_path).read_text())["heading_config"]).read_text()
    )
    layer = json.loads((root / layer_path).read_text())
    if layer["selected_family"] != config["family"]:
        raise ValueError("direct goal family differs from frozen layer screen")
    teacher = json.loads((root / base["adapter_source"]).read_text())["selected_candidate"]
    pairs = [[int(seed) for seed in pair] for pair in config["tuning_pairs"]]
    protocol_pairs = [
        [int(seed) for seed in item["mirror_pair_seeds"]]
        for item in nested["conditions"]
        if item["role"] == "tuning"
    ]
    if pairs != protocol_pairs:
        raise ValueError("lamina goal pairs differ from nested protocol")
    seeds = [seed for pair in pairs for seed in pair]
    local_features, cache = _collect(root, base, config, seeds, teacher)
    global_features, global_cache = _collect_teacher_features(root, base, seeds, teacher)
    variant = {
        "name": "current_mean",
        "statistics": ["mean"],
        "temporal_terms": ["current"],
    }
    mask = _feature_mask(base, global_features, variant)
    folds = []
    for held_out in pairs:
        train = [seed for seed in seeds if seed not in held_out]
        goal_model = _fit(cache, train, float(config["ridge_alpha"]))
        global_model = _fit_from_cache(
            global_features,
            global_cache,
            train,
            1.0,
            feature_mask=mask,
            feature_variant=variant["name"],
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
        held_out_features = np.concatenate([cache[seed]["features"] for seed in held_out])
        held_out_targets = np.concatenate([cache[seed]["targets"] for seed in held_out])
        predictions = np.asarray([_predict_goal(goal_model, row) for row in held_out_features])
        folds.append(
            {
                "training_seeds": train,
                "held_out_seeds": held_out,
                "goal_model": goal_model,
                "held_out_goal_mae": float(np.mean(np.abs(predictions - held_out_targets))),
                "success_count": sum(item["success"] for item in episodes),
                "total_obstacles_passed": sum(
                    item["obstacles_passed"] for item in episodes
                ),
                "episodes": episodes,
            }
        )
    successes = sum(item["success_count"] for item in folds)
    obstacles = sum(item["total_obstacles_passed"] for item in folds)
    passed = successes == len(seeds) and obstacles == 9 * len(seeds)
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(nested_path): _sha256(root / nested_path),
                str(base_path): _sha256(root / base_path),
                str(local_path): _sha256(root / local_path),
                str(corridor_path): _sha256(root / corridor_path),
                str(upper_path): _sha256(root / upper_path),
                str(layer_path): _sha256(root / layer_path),
            },
            "tuning_only": True,
            "held_out_pairs_never_labeled_for_fit": True,
            "teacher_actions_injected": False,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "structure": {
            "family": config["family"],
            "declared_target_count": local_features.declared_target_count,
            "unmapped_target_count": local_features.unmapped_target_count,
            "minimum_bin_count": min(local_features.bin_counts),
            "feature_dimension": len(local_features.groups) * 2,
        },
        "folds": folds,
        "success_count": successes,
        "total_obstacles_passed": obstacles,
        "cross_validation_passed": bool(passed),
        "advance_to_full_tuning": bool(passed),
        "advance_to_calibration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
