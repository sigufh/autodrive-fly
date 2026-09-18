"""Fit a T5-only structural transform for the CT1 terminal axis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_source_audit import CARDINAL_ORDER, CARDINAL_VECTORS

CONFIG = Path("configs/driving-v7-t5-ct1-axis-calibration.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_ct1_axis_calibration.py")


def _fit_split(body_id: int, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{body_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2 == 0


def _transform(
    offsets: np.ndarray,
    expected: np.ndarray,
    subtypes: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    covariance = np.zeros((2, 2), dtype=np.float64)
    for subtype in "abcd":
        selected = mask & (subtypes == subtype)
        normalized = offsets[selected] / np.linalg.norm(
            offsets[selected], axis=1, keepdims=True
        )
        covariance += normalized.T @ expected[selected] / np.count_nonzero(selected)
    left, _, right = np.linalg.svd(covariance, full_matrices=False)
    return left @ right


def _metrics(offsets: np.ndarray, expected: np.ndarray, mask: np.ndarray, transform) -> dict:
    predicted = offsets[mask] @ transform
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    truth = expected[mask]
    cardinal = np.stack([CARDINAL_VECTORS[name] for name in CARDINAL_ORDER])
    predicted_labels = np.argmax(predicted @ cardinal.T, axis=1)
    true_labels = np.argmax(truth @ cardinal.T, axis=1)
    angles = np.degrees(
        np.arccos(np.clip(np.sum(predicted * truth, axis=1), -1.0, 1.0))
    )
    return {
        "count": int(np.count_nonzero(mask)),
        "accuracy": float(np.mean(predicted_labels == true_labels)),
        "median_angle_error_degrees": float(np.median(angles)),
        "mean_angle_error_degrees": float(np.mean(angles)),
    }


def evaluate_v7_t5_ct1_axis_calibration(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_path = Path(config["axis_evidence"])
    evidence = json.loads((root / evidence_path).read_text())
    protocol_path = Path(config["axis_protocol"])
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    if float(config["gates"]["minimum_held_out_accuracy"]) != 0.75:
        raise ValueError("T5 CT1 axis held-out accuracy gate changed")
    if evidence["authorize_CT1_spatial_dynamics_candidate"]:
        raise ValueError("T5 CT1 calibration requires preserved native-axis failure")
    records = [item for item in evidence["target_records"] if item["unit_axis"] is not None]
    body_ids = np.asarray([item["body_id"] for item in records], dtype=np.int64)
    sides = np.asarray([item["population"][-1] for item in records])
    subtypes = np.asarray([item["population"][2] for item in records])
    offsets = np.asarray([item["unit_axis"] for item in records], dtype=np.float64)
    expected_names = [scoring["direction_populations"][item["population"]] for item in records]
    expected = np.stack([CARDINAL_VECTORS[name] for name in expected_names])
    fit = np.asarray(
        [_fit_split(int(body), int(config["split_seed"])) for body in body_ids]
    )
    transforms, fit_metrics, held_metrics = {}, {}, {}
    for side in ("L", "R"):
        fit_mask = fit & (sides == side)
        held_mask = ~fit & (sides == side)
        transforms[side] = _transform(offsets, expected, subtypes, fit_mask)
        fit_metrics[side] = _metrics(offsets, expected, fit_mask, transforms[side])
        held_metrics[side] = _metrics(offsets, expected, held_mask, transforms[side])
    mirror_errors = {}
    for subtype in "abcd":
        means = {}
        for side in ("L", "R"):
            mask = (~fit) & (sides == side) & (subtypes == subtype)
            values = offsets[mask] @ transforms[side]
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            center = np.mean(values, axis=0)
            means[side] = center / np.linalg.norm(center)
        reflected_right = np.asarray((-means["R"][0], means["R"][1]))
        mirror_errors[subtype] = float(np.linalg.norm(means["L"] - reflected_right))
    rng = np.random.default_rng(int(config["split_seed"]))
    random_scores = []
    held = ~fit
    for _ in range(int(config["random_orthogonal_baselines"])):
        random_transforms = {}
        for side in ("L", "R"):
            angle = rng.uniform(-np.pi, np.pi)
            reflection = rng.choice((-1.0, 1.0))
            random_transforms[side] = np.asarray(
                (
                    (np.cos(angle), -reflection * np.sin(angle)),
                    (np.sin(angle), reflection * np.cos(angle)),
                )
            )
        correct = []
        for side in ("L", "R"):
            mask = held & (sides == side)
            correct.append(
                _metrics(offsets, expected, mask, random_transforms[side])["accuracy"]
            )
        random_scores.append(np.mean(correct))
    held_accuracy = float(np.mean([item["accuracy"] for item in held_metrics.values()]))
    held_angle = float(
        np.mean([item["median_angle_error_degrees"] for item in held_metrics.values()])
    )
    thresholds = config["gates"]
    gates = {
        "held_out_accuracy": held_accuracy
        >= float(thresholds["minimum_held_out_accuracy"]),
        "held_out_angle": held_angle
        <= float(thresholds["maximum_held_out_median_angle_error_degrees"]),
        "cross_eye_mirror": max(mirror_errors.values())
        <= float(thresholds["maximum_cross_eye_mirror_angle_error_degrees"])
        * np.pi
        / 180.0,
        "held_out_accuracy_above_random_p95": held_accuracy
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
                str(evidence_path): _sha256(root / evidence_path),
                str(protocol_path): _sha256(root / protocol_path),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "parameter_fit": True,
            "fit_uses_neural_response": False,
            "runtime_modified": False,
        },
        "dataset": {
            "valid_target_count": len(records),
            "fit_count": int(np.count_nonzero(fit)),
            "held_out_count": int(np.count_nonzero(~fit)),
            "fit_held_out_body_id_overlap": int(
                np.intersect1d(body_ids[fit], body_ids[~fit]).size
            ),
        },
        "transforms_by_eye": {
            side: transforms[side].tolist() for side in ("L", "R")
        },
        "fit_by_eye": fit_metrics,
        "held_out_by_eye": held_metrics,
        "held_out_accuracy_mean": held_accuracy,
        "held_out_median_angle_error_mean_degrees": held_angle,
        "cross_eye_mirror_error_by_subtype": mirror_errors,
        "random_orthogonal_baseline": {
            "count": len(random_scores),
            "held_out_accuracy_p95": float(np.quantile(random_scores, 0.95)),
        },
        "gates": gates,
        "T5_CT1_axis_calibration_passed": passed,
        "authorize_T5_single_condition_precheck": passed,
        "advance_to_calibration_stimulus": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
