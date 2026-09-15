from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-stage1-scoring.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_stage1_scoring.py")


def strict_contrast_summary(
    cell_ids: np.ndarray,
    preferred: np.ndarray,
    comparator: np.ndarray,
    thresholds: dict,
) -> dict:
    ids = np.asarray(cell_ids, dtype=np.int64)
    preferred = np.asarray(preferred, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    if ids.ndim != 1 or preferred.shape != ids.shape or comparator.shape != ids.shape:
        raise ValueError("cell ids and responses must be equal one-dimensional arrays")
    if np.unique(ids).size != ids.size:
        raise ValueError("cell ids must be unique within a comparison")
    finite = np.isfinite(preferred) & np.isfinite(comparator)
    denominator = np.abs(preferred) + np.abs(comparator)
    valid = finite & (denominator >= float(thresholds["minimum_valid_denominator"]))
    contrast = np.full(ids.shape, np.nan, dtype=np.float64)
    contrast[valid] = (preferred[valid] - comparator[valid]) / denominator[valid]
    valid_values = contrast[valid]
    valid_fraction = float(np.mean(valid)) if ids.size else 0.0
    median = float(np.median(valid_values)) if valid_values.size else None
    positive_fraction = float(np.mean(valid_values > 0)) if valid_values.size else 0.0
    gates = {
        "valid_cell_fraction": valid_fraction >= float(thresholds["minimum_valid_cell_fraction"]),
        "median_signed_contrast": median is not None
        and median >= float(thresholds["minimum_median_signed_contrast"]),
        "positive_cell_fraction": positive_fraction
        >= float(thresholds["minimum_positive_cell_fraction"]),
    }
    return {
        "cell_count": int(ids.size),
        "valid_cell_count": int(np.count_nonzero(valid)),
        "valid_cell_fraction": valid_fraction,
        "median_signed_contrast": median,
        "positive_cell_fraction": positive_fraction,
        "minimum_valid_denominator": (float(np.min(denominator[valid])) if np.any(valid) else None),
        "median_valid_denominator": (
            float(np.median(denominator[valid])) if np.any(valid) else None
        ),
        "invalid_cell_ids": ids[~valid].tolist(),
        "cell_ids": ids.tolist(),
        "preferred": preferred.tolist(),
        "comparator": comparator.tolist(),
        "denominator": denominator.tolist(),
        "contrast": [float(value) if np.isfinite(value) else None for value in contrast],
        "gates": gates,
        "passed": all(gates.values()),
    }


def looming_contrast_summary(
    cell_ids: np.ndarray,
    expansion: np.ndarray,
    receding: np.ndarray,
    static: np.ndarray,
    thresholds: dict,
) -> dict:
    receding = np.asarray(receding, dtype=np.float64)
    static = np.asarray(static, dtype=np.float64)
    if receding.shape != static.shape:
        raise ValueError("receding and static responses must have equal shape")
    result = strict_contrast_summary(
        cell_ids, np.asarray(expansion, dtype=np.float64), np.maximum(receding, static), thresholds
    )
    result["receding"] = receding.tolist()
    result["static"] = static.tolist()
    result["comparator_definition"] = "elementwise_max_of_receding_and_same_size_static"
    return result


def mirror_error_summary(
    original: np.ndarray, mirrored_counterpart: np.ndarray, thresholds: dict
) -> dict:
    original = np.asarray(original, dtype=np.float64)
    counterpart = np.asarray(mirrored_counterpart, dtype=np.float64)
    if original.shape != counterpart.shape or original.ndim != 2:
        raise ValueError("mirror traces must have equal population-by-time shapes")
    finite = np.all(np.isfinite(original), axis=1) & np.all(np.isfinite(counterpart), axis=1)
    absolute_error = np.mean(np.abs(original - counterpart), axis=1)
    scale = np.mean(np.abs(original), axis=1) + np.mean(np.abs(counterpart), axis=1)
    active = finite & (scale >= float(thresholds["minimum_valid_denominator"]))
    active_fraction = float(np.mean(active)) if original.shape[0] else 0.0
    weighted_error = (
        float(np.sum(absolute_error[active]) / np.sum(scale[active])) if np.any(active) else None
    )
    gates = {
        "active_pair_fraction": active_fraction
        >= float(thresholds["minimum_active_mirror_pair_fraction"]),
        "energy_weighted_error": weighted_error is not None
        and weighted_error <= float(thresholds["maximum_energy_weighted_mirror_error"]),
    }
    return {
        "population_pair_count": int(original.shape[0]),
        "active_population_pair_count": int(np.count_nonzero(active)),
        "active_population_pair_fraction": active_fraction,
        "energy_weighted_mirror_error": weighted_error,
        "per_pair_scale": scale.tolist(),
        "per_pair_absolute_error": absolute_error.tolist(),
        "gates": gates,
        "passed": all(gates.values()),
    }


def authorize_split(
    split: str,
    *,
    development_passed: bool = False,
    validation_passed: bool = False,
    stage1_passed: bool = False,
    topology_passed: bool = False,
    external_final_custody: bool = False,
) -> bool:
    if split == "development":
        return True
    if split == "validation":
        return development_passed
    if split == "ood":
        return development_passed and validation_passed
    if split == "final":
        return stage1_passed and topology_passed and external_final_custody
    raise ValueError(f"unknown stage-1 split: {split}")


def _synthetic_controls(thresholds: dict) -> dict:
    ids = np.arange(10, dtype=np.int64)
    positive = strict_contrast_summary(ids, np.ones(10), np.full(10, 0.5), thresholds)
    reversed_labels = strict_contrast_summary(ids, np.full(10, 0.5), np.ones(10), thresholds)
    silent = strict_contrast_summary(ids, np.zeros(10), np.zeros(10), thresholds)
    partial = strict_contrast_summary(
        ids, np.r_[np.ones(7), np.zeros(3)], np.r_[np.full(7, 0.5), np.zeros(3)], thresholds
    )
    looming = looming_contrast_summary(
        ids, np.ones(10), np.full(10, 0.4), np.full(10, 0.6), thresholds
    )
    static_equal = looming_contrast_summary(
        ids, np.ones(10), np.full(10, 0.4), np.ones(10), thresholds
    )
    mirror = mirror_error_summary(
        np.tile([0.2, 0.4, 0.1], (10, 1)),
        np.tile([0.2, 0.4, 0.1], (10, 1)),
        thresholds,
    )
    silent_mirror = mirror_error_summary(np.zeros((10, 3)), np.zeros((10, 3)), thresholds)
    return {
        "positive_response_passes": positive["passed"],
        "reversed_labels_fail": not reversed_labels["passed"],
        "silent_response_fails": not silent["passed"],
        "insufficient_valid_coverage_fails": not partial["passed"],
        "looming_above_receding_and_static_passes": looming["passed"],
        "static_equal_to_looming_fails": not static_equal["passed"],
        "exact_active_mirror_passes": mirror["passed"],
        "silent_mirror_fails": not silent_mirror["passed"],
    }


def evaluate_v7_stage1_scoring(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    split_path = Path(config["split_evidence"])
    split = json.loads((root / split_path).read_text(encoding="utf-8"))
    final = split["split_manifests"]["final"]
    if final["evaluable"] or final["evaluated"] or final["blinded"]:
        raise ValueError("stage-1 final split boundary changed")
    controls = _synthetic_controls(config["thresholds"])
    if not all(controls.values()):
        raise ValueError("strict stage-1 scoring controls failed")
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(split_path): _sha256(root / split_path),
            },
            "model_evaluated": False,
            "parameters_fitted": False,
            "existing_artifacts_rescored": False,
        },
        "population_contract": {
            "direction_populations": config["direction_populations"],
            "polarity_populations": config["polarity_populations"],
            "looming_populations": config["looming_populations"],
            "required_direction_groups": len(config["direction_populations"]),
            "required_polarity_groups": len(config["polarity_populations"]),
            "required_looming_groups": len(config["looming_populations"]),
        },
        "thresholds": config["thresholds"],
        "aggregation": config["aggregation"],
        "execution_order": config["execution_order"],
        "synthetic_controls": controls,
        "final_lock": {
            "reserved_final_evaluable": final["evaluable"],
            "reserved_final_evaluated": final["evaluated"],
            "reserved_final_blinded": final["blinded"],
            "authorization_without_prerequisites": authorize_split("final"),
            "authorization_with_all_prerequisites": authorize_split(
                "final",
                stage1_passed=True,
                topology_passed=True,
                external_final_custody=True,
            ),
        },
        "boundary": config["boundary"],
        "limitations": [
            "Synthetic controls validate metric behavior, not biological model behavior.",
            "No existing response artifact is reinterpreted under the new thresholds.",
            "The reserved final generator is visible and cannot serve as a blinded final test.",
            "Independent cells or animals remain necessary for biological generalization.",
        ],
        "advance_to_model_fit": False,
        "advance_to_final_test": False,
        "advance_to_central_complex": False,
    }
