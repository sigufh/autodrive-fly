from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_heading_ring import _heading_episode
from fly_emotion.driving.v7_neural_channels import (
    StructuredNeuralFeatures,
    _mirror_checks,
    _model_digest,
    _teacher_step,
)
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina

CONFIG = Path("configs/driving-v7-neural-episode-cv.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_episode_cv.py")


def _collect_teacher_features(
    root: Path, base_config: dict, seeds: list[int], teacher: dict
) -> tuple[StructuredNeuralFeatures, dict[int, dict[str, np.ndarray]]]:
    teacher_config = yaml.safe_load((root / "configs/driving-v7-r1r6-local.yaml").read_text())
    retina = build_mass_balanced_retina(root)
    features = StructuredNeuralFeatures(root, base_config)
    by_seed = {}
    for seed in seeds:
        environment = DrivingEnvironment()
        image = environment.reset(seed)
        features.reset()
        command = 0.0
        even_rows, odd_rows, targets = [], [], []
        while not environment.done:
            even, odd = features.step(image)
            image, command, channels = _teacher_step(
                environment, image, command, retina, teacher_config, teacher
            )
            even_rows.append(even)
            odd_rows.append(odd)
            targets.append(
                [channels["danger"], channels["obstacle_asymmetry"], channels["road_centroid"]]
            )
        by_seed[seed] = {
            "even": np.asarray(even_rows),
            "odd": np.asarray(odd_rows),
            "targets": np.asarray(targets),
        }
    return features, by_seed


def _fit_from_cache(
    features: StructuredNeuralFeatures,
    cache: dict[int, dict[str, np.ndarray]],
    seeds: list[int],
    alpha: float,
    feature_mask: np.ndarray | None = None,
    feature_variant: str = "full",
) -> dict:
    even_matrix = np.concatenate([cache[seed]["even"] for seed in seeds])
    odd_matrix = np.concatenate([cache[seed]["odd"] for seed in seeds])
    target_matrix = np.concatenate([cache[seed]["targets"] for seed in seeds])
    coefficients, scales, fit_r2 = [], [], []
    if feature_mask is None:
        feature_mask = np.ones(even_matrix.shape[1], dtype=bool)
    if feature_mask.shape != (even_matrix.shape[1],) or not np.any(feature_mask):
        raise ValueError("feature mask must retain at least one readout feature")
    for output_index, matrix in enumerate((even_matrix, odd_matrix, odd_matrix)):
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale[scale < 1e-7] = 1.0
        normalized = ((matrix - mean) / scale)[:, feature_mask]
        design = np.column_stack((np.ones(len(normalized)), normalized))
        penalty = np.diag([0.0] + [float(alpha)] * normalized.shape[1])
        coefficient = np.linalg.solve(
            design.T @ design + penalty, design.T @ target_matrix[:, output_index]
        )
        prediction = design @ coefficient
        denominator = np.sum(
            (target_matrix[:, output_index] - np.mean(target_matrix[:, output_index])) ** 2
        )
        fit_r2.append(
            float(
                1.0
                - np.sum((target_matrix[:, output_index] - prediction) ** 2)
                / max(float(denominator), 1e-12)
            )
        )
        expanded = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
        expanded[0] = coefficient[0]
        expanded[1:][feature_mask] = coefficient[1:]
        coefficients.append(expanded.tolist())
        scales.append({"mean": mean.tolist(), "scale": scale.tolist()})
    return {
        "output_names": ["danger", "obstacle_asymmetry", "road_center"],
        "alpha": float(alpha),
        "coefficients": coefficients,
        "scales": scales,
        "fit_r2": fit_r2,
        "group_names": features.group_names,
        "group_mirror": features.group_mirror.tolist(),
        "feature_dimension_per_parity": int(even_matrix.shape[1]),
        "training_samples": int(len(target_matrix)),
        "training_seeds": seeds,
        "feature_variant": feature_variant,
        "retained_feature_count": int(np.count_nonzero(feature_mask)),
    }


def _feature_mask(
    base_config: dict, features: StructuredNeuralFeatures, variant: dict
) -> np.ndarray:
    statistics = list(base_config["features"]["per_group_statistics"])
    temporal = list(base_config["features"]["temporal_terms"])
    selected_statistics = set(variant["statistics"])
    selected_temporal = set(variant["temporal_terms"])
    if not selected_statistics <= set(statistics) or not selected_temporal <= set(temporal):
        raise ValueError("feature complexity candidate references an unknown feature")
    group_count = len(features.group_names)
    mask = []
    for temporal_name in temporal:
        for _group in range(group_count):
            for statistic in statistics:
                mask.append(
                    temporal_name in selected_temporal and statistic in selected_statistics
                )
    return np.asarray(mask, dtype=bool)


def _summary(alpha: float, folds: list[dict]) -> dict:
    episodes = [episode for fold in folds for episode in fold["held_out_episodes"]]
    return {
        "alpha": float(alpha),
        "held_out_success_count": sum(item["success"] for item in episodes),
        "held_out_obstacles_passed": sum(item["obstacles_passed"] for item in episodes),
        "held_out_mean_maximum_absolute_lateral_position": float(
            np.mean([item["maximum_absolute_lateral_position"] for item in episodes])
        ),
        "folds": folds,
    }


def _feature_summary(name: str, retained_features: int, folds: list[dict]) -> dict:
    summary = _summary(1.0, folds)
    summary.pop("alpha")
    return {
        "name": name,
        "retained_feature_count": retained_features,
        **summary,
    }


def _feature_selection_key(summary: dict) -> tuple:
    return (
        -summary["held_out_success_count"],
        -summary["held_out_obstacles_passed"],
        summary["held_out_mean_maximum_absolute_lateral_position"],
        summary["retained_feature_count"],
        summary["name"],
    )


def _selection_key(summary: dict) -> tuple:
    return (
        -summary["held_out_success_count"],
        -summary["held_out_obstacles_passed"],
        summary["held_out_mean_maximum_absolute_lateral_position"],
        -summary["alpha"],
    )


def evaluate_v7_neural_episode_cv(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    base_path = Path(config["base_neural_config"])
    heading_path = Path(config["heading_config"])
    base = yaml.safe_load((root / base_path).read_text())
    heading = yaml.safe_load((root / heading_path).read_text())
    teacher_path = Path(config["teacher_tuning_evidence"])
    teacher = json.loads((root / teacher_path).read_text())["selected_candidate"]
    pairs = [[int(seed) for seed in pair] for pair in config["tuning_mirror_pairs"]]
    seeds = [seed for pair in pairs for seed in pair]
    features, cache = _collect_teacher_features(root, base, seeds, teacher)
    summaries = []
    for alpha in config["ridge_alpha_candidates"]:
        folds = []
        for held_out in pairs:
            train = [seed for seed in seeds if seed not in held_out]
            model = _fit_from_cache(features, cache, train, float(alpha))
            episodes = [
                _heading_episode(features, model, teacher, heading, seed, "neural_heading")
                for seed in held_out
            ]
            folds.append(
                {
                    "training_seeds": train,
                    "held_out_seeds": held_out,
                    "model_sha256": _model_digest(model),
                    "fit_r2": model["fit_r2"],
                    "held_out_episodes": [
                        {key: value for key, value in item.items() if key != "trace"}
                        for item in episodes
                    ],
                    "held_out_mirror_checks": _mirror_checks(episodes),
                }
            )
        summaries.append(_summary(float(alpha), folds))
    selected = min(summaries, key=_selection_key)
    feature_summaries = []
    feature_masks = {}
    for variant in config["feature_complexity_candidates"]:
        mask = _feature_mask(base, features, variant)
        feature_masks[variant["name"]] = mask
        folds = []
        for held_out in pairs:
            train = [seed for seed in seeds if seed not in held_out]
            model = _fit_from_cache(
                features,
                cache,
                train,
                float(config["feature_complexity_alpha"]),
                feature_mask=mask,
                feature_variant=variant["name"],
            )
            episodes = [
                _heading_episode(features, model, teacher, heading, seed, "neural_heading")
                for seed in held_out
            ]
            folds.append(
                {
                    "training_seeds": train,
                    "held_out_seeds": held_out,
                    "model_sha256": _model_digest(model),
                    "fit_r2": model["fit_r2"],
                    "held_out_episodes": [
                        {key: value for key, value in item.items() if key != "trace"}
                        for item in episodes
                    ],
                    "held_out_mirror_checks": _mirror_checks(episodes),
                }
            )
        feature_summaries.append(
            _feature_summary(variant["name"], int(np.count_nonzero(mask)), folds)
        )
    selected_feature = min(feature_summaries, key=_feature_selection_key)
    selected_mask = feature_masks[selected_feature["name"]]
    feature_cv_passed = (
        selected_feature["held_out_success_count"] == len(seeds)
        and selected_feature["held_out_obstacles_passed"] == 9 * len(seeds)
    )
    model = _fit_from_cache(
        features,
        cache,
        seeds,
        float(config["feature_complexity_alpha"]),
        feature_mask=selected_mask,
        feature_variant=selected_feature["name"],
    )
    tuning = [
        _heading_episode(features, model, teacher, heading, seed, "neural_heading")
        for seed in seeds
    ]
    cv_passed = bool(feature_cv_passed)
    tuning_passed = all(item["success"] and item["obstacles_passed"] == 9 for item in tuning)
    calibration_seeds = [int(seed) for seed in config["calibration"]["mirror_pair_seeds"]]
    calibration = (
        [
            _heading_episode(features, model, teacher, heading, seed, "neural_heading")
            for seed in calibration_seeds
        ]
        if cv_passed and tuning_passed
        else []
    )
    calibration_passed = bool(calibration) and all(
        item["success"] and item["obstacles_passed"] == 9 for item in calibration
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(base_path): _sha256(root / base_path),
                str(heading_path): _sha256(root / heading_path),
                str(teacher_path): _sha256(root / teacher_path),
            },
            "mirror_pair_is_indivisible_cv_unit": True,
            "training_r2_used_for_selection": False,
            "raw_heading_read_by_controller": False,
            "target_activity_injection": False,
            "calibration_run_once_after_candidate_freeze": True,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "candidate_summaries": summaries,
        "selected_alpha": selected["alpha"],
        "selected_cv_summary": selected,
        "ridge_only_cross_validation_passed": bool(
            selected["held_out_success_count"] == len(seeds)
            and selected["held_out_obstacles_passed"] == 9 * len(seeds)
        ),
        "feature_complexity_summaries": feature_summaries,
        "selected_feature_variant": selected_feature["name"],
        "selected_feature_cv_summary": selected_feature,
        "cross_validation_passed": bool(cv_passed),
        "model": model,
        "model_sha256": _model_digest(model),
        "tuning_episodes": [
            {key: value for key, value in item.items() if key != "trace"} for item in tuning
        ],
        "tuning_passed": bool(tuning_passed),
        "calibration_episodes": [
            {key: value for key, value in item.items() if key != "trace"}
            for item in calibration
        ],
        "calibration_passed": bool(calibration_passed),
        "advance_to_topology_controls": bool(calibration_passed),
        "advance_to_fc2_pfl_comparison": bool(calibration_passed),
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
