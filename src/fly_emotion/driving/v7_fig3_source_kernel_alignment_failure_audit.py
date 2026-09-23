"""Decompose Fig. 3 source-kernel failures without replacing their gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path(
    "configs/driving-v7-fig3-source-kernel-alignment-failure-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_fig3_source_kernel_alignment_failure_audit.py"
)


def _unit(values: np.ndarray) -> np.ndarray:
    centered = values - values[0]
    return centered / max(float(np.linalg.norm(centered)), 1e-12)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.corrcoef(first, second)[0, 1])


def _lagged_correlation(
    held_out: np.ndarray, training: np.ndarray, lag: int
) -> float:
    if lag < 0:
        return _correlation(held_out[-lag:], training[:lag])
    if lag > 0:
        return _correlation(held_out[:-lag], training[lag:])
    return _correlation(held_out, training)


def evaluate_v7_fig3_source_kernel_alignment_failure_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_protocol_path = Path(config["source_protocol"])
    robustness_protocol_path = Path(config["robustness_protocol"])
    robustness_evidence_path = Path(config["robustness_evidence"])
    source_protocol = yaml.safe_load(
        (root / source_protocol_path).read_text(encoding="utf-8")
    )
    robustness_protocol = yaml.safe_load(
        (root / robustness_protocol_path).read_text(encoding="utf-8")
    )
    robustness = json.loads(
        (root / robustness_evidence_path).read_text(encoding="utf-8")
    )
    if robustness["all_source_kernel_robustness_gates_passed"]:
        raise ValueError("alignment audit requires the retained robustness failure")
    workbook_path = root / source_protocol["source_data"]["path"]
    baseline_start, baseline_stop = map(
        float, robustness_protocol["baseline_seconds"]
    )
    maximum_lag_milliseconds = int(config["maximum_oracle_lag_milliseconds"])
    conditions = {}
    original_negative_count = 0
    oracle_negative_count = 0
    for condition_name, condition in robustness_protocol["conditions"].items():
        frame = pd.read_excel(workbook_path, sheet_name=condition["sheet"])
        time = frame.iloc[:, 0].to_numpy(dtype=np.float64)
        sample_interval_milliseconds = float(np.median(np.diff(time)) * 1000.0)
        maximum_lag_samples = int(
            round(maximum_lag_milliseconds / sample_interval_milliseconds)
        )
        baseline_mask = (time >= baseline_start) & (time < baseline_stop)
        response_mask = (time >= float(condition["onset_seconds"])) & (
            time
            <= float(condition["onset_seconds"])
            + float(robustness_protocol["response_duration_seconds"])
        )
        source_results = {}
        for source_type in config["source_types"]:
            columns = [
                name for name in frame if str(name).startswith(f"{source_type}-")
            ]
            values = frame[columns].to_numpy(dtype=np.float64)
            baseline = np.mean(values[baseline_mask], axis=0)
            oriented = float(condition["sign"]) * (values[response_mask] - baseline)
            zero_lag = []
            oracle = []
            best_lags = []
            for cell_index in range(oriented.shape[1]):
                held_out = _unit(oriented[:, cell_index])
                training = _unit(
                    np.mean(np.delete(oriented, cell_index, axis=1), axis=1)
                )
                zero_lag.append(_correlation(held_out, training))
                candidates = [
                    _lagged_correlation(held_out, training, lag)
                    for lag in range(-maximum_lag_samples, maximum_lag_samples + 1)
                ]
                best_index = int(np.nanargmax(candidates))
                oracle.append(candidates[best_index])
                best_lags.append(best_index - maximum_lag_samples)
            original = robustness["conditions"][condition_name]["sources"][
                source_type
            ]
            zero_array = np.asarray(zero_lag)
            oracle_array = np.asarray(oracle)
            lag_array = np.asarray(best_lags)
            if not math.isclose(
                float(np.median(zero_array)),
                float(original["median_leave_one_cell_out_correlation"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(np.min(zero_array)),
                float(original["minimum_leave_one_cell_out_correlation"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{condition_name} {source_type} zero-lag robustness changed"
                )
            source_original_negative_count = int(np.count_nonzero(zero_array < 0.0))
            source_oracle_negative_count = int(np.count_nonzero(oracle_array < 0.0))
            original_negative_count += source_original_negative_count
            oracle_negative_count += source_oracle_negative_count
            source_results[source_type] = {
                "cell_count": int(oriented.shape[1]),
                "sample_interval_milliseconds": sample_interval_milliseconds,
                "zero_lag_median_correlation": float(np.median(zero_array)),
                "zero_lag_minimum_correlation": float(np.min(zero_array)),
                "zero_lag_negative_cell_count": source_original_negative_count,
                "oracle_shift_median_correlation": float(np.median(oracle_array)),
                "oracle_shift_minimum_correlation": float(np.min(oracle_array)),
                "oracle_shift_negative_cell_count": source_oracle_negative_count,
                "negative_cells_rescued_by_oracle_shift": (
                    source_original_negative_count - source_oracle_negative_count
                ),
                "best_lag_milliseconds": {
                    "median": float(np.median(lag_array) * sample_interval_milliseconds),
                    "IQR": float(
                        (np.quantile(lag_array, 0.75) - np.quantile(lag_array, 0.25))
                        * sample_interval_milliseconds
                    ),
                    "minimum": float(np.min(lag_array) * sample_interval_milliseconds),
                    "maximum": float(np.max(lag_array) * sample_interval_milliseconds),
                    "boundary_fraction": float(
                        np.mean(np.abs(lag_array) == maximum_lag_samples)
                    ),
                },
                "original_robustness_passed": bool(original["passed"]),
            }
        conditions[condition_name] = {
            "sources": source_results,
            "original_condition_all_sources_passed": bool(
                robustness["conditions"][condition_name]["all_sources_passed"]
            ),
        }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(source_protocol_path): _sha256(root / source_protocol_path),
                str(robustness_protocol_path): _sha256(
                    root / robustness_protocol_path
                ),
                str(robustness_evidence_path): _sha256(
                    root / robustness_evidence_path
                ),
                str(source_protocol["source_data"]["path"]): _sha256(
                    workbook_path
                ),
                "pyproject.toml": _sha256(root / "pyproject.toml"),
            },
            "maximum_oracle_lag_milliseconds": maximum_lag_milliseconds,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "preexisting_normalization": {
            "baseline_offset_removed": True,
            "L2_gain_removed": True,
        },
        "conditions": conditions,
        "original_negative_leave_one_out_cell_count": original_negative_count,
        "oracle_shift_negative_leave_one_out_cell_count": oracle_negative_count,
        "bounded_oracle_shift_eliminates_all_negative_leave_one_out_cells": (
            oracle_negative_count == 0
        ),
        "baseline_or_gain_mismatch_explains_original_failure": False,
        "bounded_latency_jitter_explains_every_negative_cell": False,
        "original_source_kernel_robustness_failure_retained": True,
        "authorize_aligned_source_kernel": False,
        "authorize_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            "negative_leave_one_out_shapes_remain_after_baseline_gain_and_"
            "bounded_oracle_latency_diagnostics"
        ),
        "boundary": config["boundary"],
    }
