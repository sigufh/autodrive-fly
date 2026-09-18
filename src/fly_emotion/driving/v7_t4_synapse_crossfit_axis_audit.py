"""Cross-fit T4 synapse-cluster axes without target-label leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import V7Contract
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_synapse_axis_calibration import (
    DIRECTION_ORDER,
    DIRECTION_VECTORS,
    _collect_axes,
    _fit,
    _split,
)

CONFIG = Path("configs/driving-v7-t4-synapse-crossfit-axis-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_synapse_crossfit_axis_audit.py"
)
SOURCE_IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_synapse_axis_calibration.py"
)


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _metrics(predicted: np.ndarray, expected: np.ndarray) -> dict:
    cardinal = np.stack([DIRECTION_VECTORS[name] for name in DIRECTION_ORDER])
    labels = np.argmax(predicted @ cardinal.T, axis=1)
    truth = np.argmax(expected @ cardinal.T, axis=1)
    angles = np.degrees(
        np.arccos(np.clip(np.sum(predicted * expected, axis=1), -1.0, 1.0))
    )
    return {
        "count": len(predicted),
        "accuracy": float(np.mean(labels == truth)),
        "median_angle_error_degrees": float(np.median(angles)),
        "mean_angle_error_degrees": float(np.mean(angles)),
    }


def _random_transform(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 2))
    left, _, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right


def evaluate_v7_t4_synapse_crossfit_axis_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "synapse_spatial_protocol",
            "synapse_spatial_evidence",
            "prior_axis_calibration",
            "prior_axis_protocol",
            "axis_gate_source",
        )
    }
    spatial = json.loads((root / paths["synapse_spatial_evidence"]).read_text())
    prior = json.loads((root / paths["prior_axis_calibration"]).read_text())
    if not spatial["strict_synapse_spatial_structure_gate_passed"]:
        raise ValueError("T4 cross-fit requires passed synapse structure gate")
    if not prior["authorize_T4_single_condition_precheck"]:
        raise ValueError("T4 cross-fit requires passed prior held-out axis gate")
    spatial_config = yaml.safe_load((root / paths["synapse_spatial_protocol"]).read_text())
    dataset = _collect_axes(root, spatial_config)
    t4 = dataset["family"] == "T4"
    dataset = {name: values[t4] for name, values in dataset.items()}
    first, second = _split(
        dataset, int(config["split_seed"]), float(config["fit_fraction"])
    )
    predicted = np.zeros((len(dataset["body_id"]), 2), dtype=np.float64)
    transforms = {side: {} for side in "LR"}
    counts = {side: {} for side in "LR"}
    for train_name, train_mask, application_mask in (
        ("fit_fold_0_apply_fold_1", first, second),
        ("fit_fold_1_apply_fold_0", second, first),
    ):
        fitted = _fit(dataset, train_mask)
        for side in "LR":
            side_mask = dataset["side"] == side
            train_side = train_mask & side_mask
            apply_side = application_mask & side_mask
            transform = fitted[side]
            predicted[apply_side] = dataset["offset"][apply_side] @ transform
            transforms[side][train_name] = transform.tolist()
            counts[side][train_name] = {
                "fit_count": int(np.count_nonzero(train_side)),
                "application_count": int(np.count_nonzero(apply_side)),
                "body_id_overlap": int(
                    np.intersect1d(
                        dataset["body_id"][train_side], dataset["body_id"][apply_side]
                    ).size
                ),
            }
    predicted = _normalize(predicted)
    by_population = {}
    target_denominators = spatial["families"]["T4"]["population_results"]
    for subtype in "abcd":
        for side in "LR":
            population = f"T4{subtype}_{side}"
            mask = (dataset["subtype"] == subtype) & (dataset["side"] == side)
            result = _metrics(predicted[mask], dataset["expected"][mask])
            result["fixed_target_denominator"] = target_denominators[population][
                "target_count"
            ]
            result["valid_target_fraction"] = result["count"] / result[
                "fixed_target_denominator"
            ]
            by_population[population] = result
    overall = _metrics(predicted, dataset["expected"])
    mirror = {}
    for subtype in "abcd":
        means = {}
        for side in "LR":
            values = predicted[(dataset["subtype"] == subtype) & (dataset["side"] == side)]
            center = np.mean(values, axis=0)
            means[side] = center / np.linalg.norm(center)
        reflected = np.asarray((-means["R"][0], means["R"][1]))
        mirror[subtype] = float(
            np.degrees(np.arccos(np.clip(means["L"] @ reflected, -1.0, 1.0)))
        )
    rng = np.random.default_rng(int(config["split_seed"]))
    random_scores = []
    for _ in range(int(config["random_orthogonal_baselines"])):
        random_prediction = np.zeros_like(predicted)
        for side in "LR":
            for apply_mask in (
                first & (dataset["side"] == side),
                second & (dataset["side"] == side),
            ):
                random_prediction[apply_mask] = (
                    dataset["offset"][apply_mask] @ _random_transform(rng)
                )
        random_prediction = _normalize(random_prediction)
        random_scores.append(_metrics(random_prediction, dataset["expected"])["accuracy"])
    canonical = np.argsort(dataset["body_id"])
    digest = hashlib.sha256()
    digest.update(dataset["body_id"][canonical].astype(np.int64).tobytes())
    digest.update(predicted[canonical].astype(np.float64).tobytes())
    gates_config = config["gates"]
    gates = {
        "valid_target_fraction_each_population": min(
            item["valid_target_fraction"] for item in by_population.values()
        ) >= float(gates_config["minimum_valid_target_fraction_each_population"]),
        "overall_out_of_fold_accuracy": overall["accuracy"]
        >= float(gates_config["minimum_overall_out_of_fold_accuracy"]),
        "out_of_fold_accuracy_each_population": min(
            item["accuracy"] for item in by_population.values()
        ) >= float(gates_config["minimum_out_of_fold_accuracy_each_population"]),
        "overall_out_of_fold_angle": overall["median_angle_error_degrees"]
        <= float(gates_config["maximum_overall_out_of_fold_median_angle_error_degrees"]),
        "cross_eye_mirror": max(mirror.values())
        <= float(gates_config["maximum_cross_eye_mirror_angle_error_degrees"]),
        "accuracy_above_random_p95": overall["accuracy"]
        > float(np.quantile(random_scores, 0.95)),
    }
    passed = all(gates.values())
    contract = V7Contract.load(root)
    fixed_denominator = sum(item["target_count"] for item in target_denominators.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(SOURCE_IMPLEMENTATION): _sha256(root / SOURCE_IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in paths.values()},
            },
            "parameter_fit": True,
            "fit_uses_neural_response": False,
            "functional_stimulus_evaluated": False,
            "runtime_modified": False,
        },
        "mapping": config["mapping"],
        "valid_target_count": len(dataset["body_id"]),
        "fixed_target_denominator": int(fixed_denominator),
        "fold_counts": {
            "0": int(np.count_nonzero(first)),
            "1": int(np.count_nonzero(second)),
        },
        "transforms_by_eye_and_training_fold": transforms,
        "fit_application_counts": counts,
        "out_of_fold_predictions_sha256": digest.hexdigest(),
        "out_of_fold_overall": overall,
        "out_of_fold_by_population": by_population,
        "cross_eye_mirror_angle_error_degrees_by_subtype": mirror,
        "random_orthogonal_baseline": {
            "count": len(random_scores),
            "out_of_fold_accuracy_p95": float(np.quantile(random_scores, 0.95)),
        },
        "gates": gates,
        "T4_synapse_crossfit_axis_passed": passed,
        "authorize_T4_crossfit_single_condition_precheck": passed,
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "v7_deployment_enabled": contract.payload["deployment_enabled"],
        "stop_reason": None if passed else "T4_synapse_crossfit_axis_gate_failed",
        "boundary": config["boundary"],
    }
