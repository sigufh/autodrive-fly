"""Cross-fit the T5-only CT1 terminal axis without target-label leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_source_audit import CARDINAL_ORDER, CARDINAL_VECTORS
from fly_emotion.driving.v7_t5_ct1_axis_calibration import _fit_split, _transform

CONFIG = Path("configs/driving-v7-t5-ct1-crossfit-axis-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_ct1_crossfit_axis_audit.py"
)


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _accuracy_and_angles(predicted: np.ndarray, expected: np.ndarray) -> dict:
    cardinal = np.stack([CARDINAL_VECTORS[name] for name in CARDINAL_ORDER])
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
    angle = rng.uniform(-np.pi, np.pi)
    reflection = rng.choice((-1.0, 1.0))
    return np.asarray(
        (
            (np.cos(angle), -reflection * np.sin(angle)),
            (np.sin(angle), reflection * np.cos(angle)),
        )
    )


def evaluate_v7_t5_ct1_crossfit_axis_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "terminal_axis_evidence",
            "terminal_axis_protocol",
            "prior_axis_calibration",
            "prior_axis_protocol",
            "prior_axis_implementation",
            "source_axis_implementation",
            "scoring_config",
        )
    }
    terminal = json.loads((root / paths["terminal_axis_evidence"]).read_text())
    prior = json.loads((root / paths["prior_axis_calibration"]).read_text())
    if not prior["authorize_T5_single_condition_precheck"]:
        raise ValueError("cross-fit audit requires passed T5-only axis calibration")
    records = [item for item in terminal["target_records"] if item["unit_axis"] is not None]
    body_ids = np.asarray([item["body_id"] for item in records], dtype=np.int64)
    populations = np.asarray([item["population"] for item in records])
    sides = np.asarray([name[-1] for name in populations])
    subtypes = np.asarray([name[2] for name in populations])
    offsets = np.asarray([item["unit_axis"] for item in records], dtype=np.float64)
    scoring = yaml.safe_load((root / paths["scoring_config"]).read_text())
    expected_names = [scoring["direction_populations"][name] for name in populations]
    expected = np.stack([CARDINAL_VECTORS[name] for name in expected_names])
    fold = np.asarray(
        [_fit_split(int(body), int(config["split_seed"])) for body in body_ids]
    )
    predicted = np.zeros_like(offsets)
    transforms = {}
    fit_application_counts = {}
    for side in ("L", "R"):
        transforms[side] = {}
        fit_application_counts[side] = {}
        for train_value in (False, True):
            train = (sides == side) & (fold == train_value)
            apply = (sides == side) & (fold != train_value)
            transform = _transform(offsets, expected, subtypes, train)
            predicted[apply] = offsets[apply] @ transform
            name = "fit_fold_1_apply_fold_0" if train_value else "fit_fold_0_apply_fold_1"
            transforms[side][name] = transform.tolist()
            fit_application_counts[side][name] = {
                "fit_count": int(np.count_nonzero(train)),
                "application_count": int(np.count_nonzero(apply)),
                "body_id_overlap": int(
                    np.intersect1d(body_ids[train], body_ids[apply]).size
                ),
            }
    predicted = _normalize(predicted)
    canonical = np.argsort(body_ids)
    prediction_digest = hashlib.sha256()
    prediction_digest.update(body_ids[canonical].astype(np.int64).tobytes())
    prediction_digest.update(predicted[canonical].astype(np.float64).tobytes())
    by_population = {}
    valid_fractions = {}
    for population in [f"T5{subtype}_{side}" for subtype in "abcd" for side in "LR"]:
        mask = populations == population
        total = terminal["population_results"][population]["target_count"]
        metrics = _accuracy_and_angles(predicted[mask], expected[mask])
        metrics["fixed_target_denominator"] = total
        metrics["valid_target_fraction"] = metrics["count"] / total
        by_population[population] = metrics
        valid_fractions[population] = metrics["valid_target_fraction"]
    overall = _accuracy_and_angles(predicted, expected)
    mirror = {}
    for subtype in "abcd":
        means = {}
        for side in ("L", "R"):
            values = predicted[(subtypes == subtype) & (sides == side)]
            center = np.mean(values, axis=0)
            means[side] = center / np.linalg.norm(center)
        reflected_right = np.asarray((-means["R"][0], means["R"][1]))
        mirror[subtype] = float(
            np.degrees(
                np.arccos(np.clip(np.dot(means["L"], reflected_right), -1.0, 1.0))
            )
        )
    rng = np.random.default_rng(int(config["split_seed"]))
    random_scores = []
    for _ in range(int(config["random_orthogonal_baselines"])):
        random_prediction = np.zeros_like(offsets)
        for side in ("L", "R"):
            for train_value in (False, True):
                apply = (sides == side) & (fold != train_value)
                random_prediction[apply] = offsets[apply] @ _random_transform(rng)
        random_prediction = _normalize(random_prediction)
        random_scores.append(
            _accuracy_and_angles(random_prediction, expected)["accuracy"]
        )
    thresholds = config["gates"]
    gates = {
        "valid_target_fraction_each_population": min(valid_fractions.values())
        >= float(thresholds["minimum_valid_target_fraction_each_population"]),
        "overall_out_of_fold_accuracy": overall["accuracy"]
        >= float(thresholds["minimum_overall_out_of_fold_accuracy"]),
        "out_of_fold_accuracy_each_population": min(
            item["accuracy"] for item in by_population.values()
        )
        >= float(thresholds["minimum_out_of_fold_accuracy_each_population"]),
        "overall_out_of_fold_angle": overall["median_angle_error_degrees"]
        <= float(thresholds["maximum_overall_out_of_fold_median_angle_error_degrees"]),
        "cross_eye_mirror": max(mirror.values())
        <= float(thresholds["maximum_cross_eye_mirror_angle_error_degrees"]),
        "accuracy_above_random_p95": overall["accuracy"]
        > float(np.quantile(random_scores, 0.95)),
    }
    passed = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in paths.values()},
            },
            "parameter_fit": True,
            "fit_uses_neural_response": False,
            "functional_stimulus_evaluated": False,
            "runtime_modified": False,
        },
        "mapping": config["mapping"],
        "valid_target_count": len(records),
        "fixed_target_denominator": int(
            sum(item["target_count"] for item in terminal["population_results"].values())
        ),
        "fold_counts": {
            "0": int(np.count_nonzero(~fold)),
            "1": int(np.count_nonzero(fold)),
        },
        "transforms_by_eye_and_training_fold": transforms,
        "out_of_fold_predictions_sha256": prediction_digest.hexdigest(),
        "fit_application_counts": fit_application_counts,
        "out_of_fold_overall": overall,
        "out_of_fold_by_population": by_population,
        "cross_eye_mirror_angle_error_degrees_by_subtype": mirror,
        "random_orthogonal_baseline": {
            "count": len(random_scores),
            "out_of_fold_accuracy_p95": float(np.quantile(random_scores, 0.95)),
        },
        "gates": gates,
        "T5_CT1_crossfit_axis_passed": passed,
        "authorize_axis_aware_single_condition_precheck": passed,
        "advance_to_functional_calibration": False,
        "advance_to_runtime_integration": False,
        "stop_reason": None if passed else "T5_CT1_crossfit_axis_gate_failed",
        "boundary": config["boundary"],
    }
