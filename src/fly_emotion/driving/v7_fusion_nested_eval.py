from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lamina_goal import _collect
from fly_emotion.driving.v7_lamina_goal_symmetry import _fit_goal
from fly_emotion.driving.v7_neural_channels import StructuredNeuralFeatures, _model_digest
from fly_emotion.driving.v7_neural_episode_cv import (
    _collect_teacher_features,
    _feature_mask,
    _fit_from_cache,
)
from fly_emotion.driving.v7_neural_goal_fusion import _episode, _mirror_error

CONFIG = Path("configs/driving-v7-fusion-nested.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_fusion_nested_eval.py")
PROTOCOL_ARTIFACT = Path("artifacts/v7-fusion-nested.json")
BASE_CONFIG = Path("configs/driving-v7-neural-channels.yaml")
LAMINA_CONFIG = Path("configs/driving-v7-lamina-goal.yaml")
HEADING_CONFIG = Path("configs/driving-v7-heading-ring.yaml")


def _compact(episodes: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in item.items() if key != "goal_trace"}
        for item in episodes
    ]


def evaluate_v7_fusion_nested_candidate(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    protocol = json.loads((root / PROTOCOL_ARTIFACT).read_text())
    for path, digest in protocol["protocol"]["dependencies_sha256"].items():
        if _sha256(root / path) != digest:
            raise ValueError(f"fusion nested dependency changed: {path}")
    architecture = config["frozen_architecture"]
    weight = float(architecture["lamina_weight"])
    alpha = float(architecture["ridge_alpha"])
    if weight != 0.75 or alpha != 1.0:
        raise ValueError("fusion nested architecture parameters drifted")
    base = deepcopy(yaml.safe_load((root / BASE_CONFIG).read_text()))
    base["input"]["visual_graph"] = "typed_visual_leak_v1"
    lamina = yaml.safe_load((root / LAMINA_CONFIG).read_text())
    heading = yaml.safe_load((root / HEADING_CONFIG).read_text())
    teacher = json.loads((root / base["adapter_source"]).read_text())["selected_candidate"]
    tuning_conditions = [item for item in config["conditions"] if item["role"] == "tuning"]
    pairs = [[int(seed) for seed in item["mirror_pair_seeds"]] for item in tuning_conditions]
    tuning_seeds = [seed for pair in pairs for seed in pair]
    local_features, local_cache = _collect(root, base, lamina, tuning_seeds, teacher)
    global_features, global_cache = _collect_teacher_features(
        root, base, tuning_seeds, teacher
    )
    global_variant = {
        "name": "current_mean",
        "statistics": ["mean"],
        "temporal_terms": ["current"],
    }
    global_mask = _feature_mask(base, global_features, global_variant)
    folds = []
    for held_out in pairs:
        train = [seed for seed in tuning_seeds if seed not in held_out]
        lamina_model = _fit_goal(
            local_cache,
            train,
            alpha,
            "odd_plus_paired_even_times_odd",
            len(local_features.groups),
        )
        global_model = _fit_from_cache(
            global_features,
            global_cache,
            train,
            alpha,
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
                weight,
                seed,
            )
            for seed in held_out
        ]
        folds.append(
            {
                "training_seeds": train,
                "held_out_seeds": held_out,
                "lamina_model_sha256": _model_digest(lamina_model),
                "global_model_sha256": _model_digest(global_model),
                "success_count": sum(item["success"] for item in episodes),
                "total_obstacles_passed": sum(
                    item["obstacles_passed"] for item in episodes
                ),
                "maximum_mirror_goal_error": _mirror_error(episodes),
                "episodes": _compact(episodes),
            }
        )
    cv_success = sum(item["success_count"] for item in folds)
    cv_obstacles = sum(item["total_obstacles_passed"] for item in folds)
    cv_mirror = max(item["maximum_mirror_goal_error"] for item in folds)
    cv_passed = cv_success == 6 and cv_obstacles == 54 and cv_mirror <= 1e-8
    lamina_model = _fit_goal(
        local_cache,
        tuning_seeds,
        alpha,
        "odd_plus_paired_even_times_odd",
        len(local_features.groups),
    )
    global_model = _fit_from_cache(
        global_features,
        global_cache,
        tuning_seeds,
        alpha,
        feature_mask=global_mask,
        feature_variant=global_variant["name"],
    )
    tuning = [
        _episode(
            local_features,
            StructuredNeuralFeatures(root, base),
            lamina_model,
            global_model,
            teacher,
            heading,
            weight,
            seed,
        )
        for seed in tuning_seeds
    ]
    tuning_passed = (
        all(item["success"] and item["obstacles_passed"] == 9 for item in tuning)
        and _mirror_error(tuning) <= 1e-8
    )
    calibration_condition = next(
        item for item in config["conditions"] if item["role"] == "calibration"
    )
    calibration_seeds = [int(seed) for seed in calibration_condition["mirror_pair_seeds"]]
    calibration = (
        [
            _episode(
                local_features,
                StructuredNeuralFeatures(root, base),
                lamina_model,
                global_model,
                teacher,
                heading,
                weight,
                seed,
            )
            for seed in calibration_seeds
        ]
        if cv_passed and tuning_passed
        else []
    )
    calibration_passed = bool(calibration) and (
        all(item["success"] and item["obstacles_passed"] == 9 for item in calibration)
        and _mirror_error(calibration) <= 1e-8
    )
    model = {
        "lamina_weight": weight,
        "lamina_model": lamina_model,
        "global_model": global_model,
    }
    model_sha256 = _model_digest(model)
    final = protocol["external_final"]
    return {
        "protocol": {
            "name": "v7-fused-neural-navigation-nested-evaluation",
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(PROTOCOL_ARTIFACT): _sha256(root / PROTOCOL_ARTIFACT),
                str(BASE_CONFIG): _sha256(root / BASE_CONFIG),
                str(LAMINA_CONFIG): _sha256(root / LAMINA_CONFIG),
                str(HEADING_CONFIG): _sha256(root / HEADING_CONFIG),
            },
            "architecture_changed_after_protocol_freeze": False,
            "historical_regression_used_for_fit_or_selection": False,
            "calibration_used_for_fit_or_selection": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "cross_validation": {
            "folds": folds,
            "success_count": cv_success,
            "total_obstacles_passed": cv_obstacles,
            "maximum_mirror_goal_error": cv_mirror,
            "passed": bool(cv_passed),
        },
        "model": model,
        "model_sha256": model_sha256,
        "tuning_episodes": _compact(tuning),
        "tuning_mirror_goal_error": _mirror_error(tuning),
        "tuning_passed": bool(tuning_passed),
        "calibration_episodes": _compact(calibration),
        "calibration_mirror_goal_error": _mirror_error(calibration) if calibration else None,
        "calibration_receipt": {
            "condition_id": calibration_condition["condition_id"],
            "model_sha256": model_sha256,
            "attempt_count": 1 if calibration else 0,
            "passed": bool(calibration_passed),
        },
        "calibration_passed": bool(calibration_passed),
        "external_final": final,
        "advance_to_final": bool(calibration_passed and final["committed"]),
        "advance_to_navigation_release": False,
    }
