from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_heading_ring import _heading_episode
from fly_emotion.driving.v7_neural_channels import _mirror_checks, _model_digest
from fly_emotion.driving.v7_neural_episode_cv import (
    _collect_teacher_features,
    _feature_mask,
    _fit_from_cache,
)
from fly_emotion.driving.v7_r1r6_local import _episode as _r1r6_local_episode
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina

CONFIG = Path("configs/driving-v7-navigation-nested.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_navigation_nested_eval.py")
NESTED_ARTIFACT = Path("artifacts/v7-navigation-nested.json")
BASE_CONFIG = Path("configs/driving-v7-neural-channels.yaml")
HEADING_CONFIG = Path("configs/driving-v7-heading-ring.yaml")


def _fold_summary(episodes: list[dict]) -> dict:
    return {
        "success_count": sum(item["success"] for item in episodes),
        "total_obstacles_passed": sum(item["obstacles_passed"] for item in episodes),
        "episodes": [
            {key: value for key, value in item.items() if key != "trace"}
            for item in episodes
        ],
        "mirror_checks": _mirror_checks(episodes),
    }


def evaluate_v7_navigation_nested_candidate(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested = json.loads((root / NESTED_ARTIFACT).read_text())
    for path, digest in nested["protocol"]["dependencies_sha256"].items():
        if _sha256(root / path) != digest:
            raise ValueError(f"navigation nested protocol dependency changed: {path}")
    if not nested["advance_to_tuning"] or nested["advance_to_calibration"]:
        raise ValueError("navigation nested protocol has an invalid starting state")
    base = yaml.safe_load((root / BASE_CONFIG).read_text())
    heading = yaml.safe_load((root / HEADING_CONFIG).read_text())
    teacher = json.loads((root / base["adapter_source"]).read_text())["selected_candidate"]
    tuning_conditions = [item for item in config["conditions"] if item["role"] == "tuning"]
    pairs = [[int(seed) for seed in item["mirror_pair_seeds"]] for item in tuning_conditions]
    tuning_seeds = [seed for pair in pairs for seed in pair]
    features, cache = _collect_teacher_features(root, base, tuning_seeds, teacher)
    variant = {
        "name": "current_mean",
        "statistics": ["mean"],
        "temporal_terms": ["current"],
    }
    mask = _feature_mask(base, features, variant)
    if int(np.count_nonzero(mask)) != 18:
        raise ValueError("frozen navigation architecture is no longer 18-dimensional")
    folds = []
    for held_out in pairs:
        train = [seed for seed in tuning_seeds if seed not in held_out]
        model = _fit_from_cache(
            features, cache, train, 1.0, feature_mask=mask, feature_variant=variant["name"]
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
                **_fold_summary(episodes),
            }
        )
    cv_success = sum(fold["success_count"] for fold in folds)
    cv_obstacles = sum(fold["total_obstacles_passed"] for fold in folds)
    cv_passed = cv_success == len(tuning_seeds) and cv_obstacles == 9 * len(tuning_seeds)
    model = _fit_from_cache(
        features,
        cache,
        tuning_seeds,
        1.0,
        feature_mask=mask,
        feature_variant=variant["name"],
    )
    tuning = [
        _heading_episode(features, model, teacher, heading, seed, "neural_heading")
        for seed in tuning_seeds
    ]
    tuning_passed = all(item["success"] and item["obstacles_passed"] == 9 for item in tuning)
    calibration_condition = next(
        item for item in config["conditions"] if item["role"] == "calibration"
    )
    calibration_seeds = [int(seed) for seed in calibration_condition["mirror_pair_seeds"]]
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
    r1r6_config = yaml.safe_load((root / "configs/driving-v7-r1r6-local.yaml").read_text())
    retina = build_mass_balanced_retina(root)
    upper_bound = [
        _r1r6_local_episode(retina, r1r6_config, teacher, seed) for seed in tuning_seeds
    ]
    upper_obstacles = sum(item["obstacles_passed"] for item in upper_bound)
    upper_successes = sum(item["success"] for item in upper_bound)
    attribution = {
        "role": "post_tuning_failure_attribution_only_not_parameter_selection",
        "r1r6_local_upper_bound": [
            {key: value for key, value in item.items() if key != "trace"}
            for item in upper_bound
        ],
        "success_count": upper_successes,
        "total_obstacles_passed": upper_obstacles,
        "diagnosis": (
            "fixed_local_action_policy_limit_present"
            if upper_successes < len(tuning_seeds)
            else "neural_readout_generalization_gap_only"
        ),
    }
    receipt = {
        "condition_id": calibration_condition["condition_id"],
        "model_sha256": _model_digest(model),
        "scorer": "success_and_nine_obstacles",
        "attempt_count": 1 if calibration else 0,
        "passed": bool(calibration_passed),
    }
    return {
        "protocol": {
            "name": "v7-navigation-nested-candidate-evaluation",
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(NESTED_ARTIFACT): _sha256(root / NESTED_ARTIFACT),
                str(BASE_CONFIG): _sha256(root / BASE_CONFIG),
                str(HEADING_CONFIG): _sha256(root / HEADING_CONFIG),
            },
            "architecture_changed_after_protocol_freeze": False,
            "historical_regression_used_for_fit_or_selection": False,
            "calibration_used_for_fit_or_selection": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "architecture": nested["architecture"],
        "cross_validation": {
            "folds": folds,
            "success_count": cv_success,
            "total_obstacles_passed": cv_obstacles,
            "passed": bool(cv_passed),
        },
        "model": model,
        "model_sha256": _model_digest(model),
        "tuning_episodes": [
            {key: value for key, value in item.items() if key != "trace"} for item in tuning
        ],
        "tuning_passed": bool(tuning_passed),
        "tuning_failure_attribution": attribution,
        "calibration_episodes": [
            {key: value for key, value in item.items() if key != "trace"}
            for item in calibration
        ],
        "calibration_receipt": receipt,
        "calibration_passed": bool(calibration_passed),
        "external_final": nested["external_final"],
        "advance_to_final": bool(
            calibration_passed and nested["external_final"]["committed"]
        ),
        "advance_to_navigation_release": False,
    }
