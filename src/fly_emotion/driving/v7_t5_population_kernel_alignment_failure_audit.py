"""Bound the role of latency jitter in T5 population-kernel failure."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_kohn_portes_t5_source_kernel_audit import (
    _author_rescaled_temporal,
)
from fly_emotion.driving.v7_t5_population_kernel_robustness_audit import (
    _unique_near_equal_partitions,
)

CONFIG = Path(
    "configs/driving-v7-t5-population-kernel-alignment-failure-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_population_kernel_alignment_failure_audit.py"
)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    value = float(np.corrcoef(first, second)[0, 1])
    if not np.isfinite(value):
        raise ValueError("non-finite aligned T5 kernel correlation")
    return value


def _oracle_shift(
    first: np.ndarray, second: np.ndarray, maximum_lag_samples: int
) -> tuple[float, int]:
    correlations = []
    for lag in range(-maximum_lag_samples, maximum_lag_samples + 1):
        if lag < 0:
            compared = (first[-lag:], second[:lag])
        elif lag > 0:
            compared = (first[:-lag], second[lag:])
        else:
            compared = (first, second)
        correlations.append(_correlation(*compared))
    index = int(np.argmax(correlations))
    return correlations[index], index - maximum_lag_samples


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def evaluate_v7_t5_population_kernel_alignment_failure_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_config_path = Path(config["source_config"])
    robustness_protocol_path = Path(config["robustness_protocol"])
    robustness_evidence_path = Path(config["robustness_evidence"])
    source_config = yaml.safe_load(
        (root / source_config_path).read_text(encoding="utf-8")
    )
    robustness_protocol = yaml.safe_load(
        (root / robustness_protocol_path).read_text(encoding="utf-8")
    )
    robustness = json.loads(
        (root / robustness_evidence_path).read_text(encoding="utf-8")
    )
    if robustness["all_population_kernel_robustness_gates_passed"]:
        raise ValueError("alignment audit requires retained T5 robustness failure")
    thresholds = robustness["thresholds"]
    if thresholds != robustness_protocol["gates"]:
        raise ValueError("saved and configured T5 robustness thresholds differ")
    maximum_lag_milliseconds = int(config["maximum_oracle_lag_milliseconds"])
    kernel_interval_milliseconds = 10.0
    maximum_lag_samples = int(
        round(maximum_lag_milliseconds / kernel_interval_milliseconds)
    )
    source_results = {}
    source_paths = {}
    for source in config["source_order"]:
        spec = source_config["white_noise_files"][source]
        source_path = _verify_file(root, spec)
        source_paths[source] = source_path
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for record in _load_restricted(source_path):
            grouped[str(record["recording_id"])].append(
                _author_rescaled_temporal(record, int(config["kernel_length"]))
            )
        recording_ids = sorted(grouped)
        kernels = np.stack(
            [np.mean(grouped[recording_id], axis=0) for recording_id in recording_ids]
        )
        loo_zero = []
        loo_oracle = []
        loo_lags = []
        loo_folds = []
        for index, recording_id in enumerate(recording_ids):
            held_out = kernels[index]
            rest = np.mean(np.delete(kernels, index, axis=0), axis=0)
            zero = _correlation(held_out, rest)
            oracle, lag = _oracle_shift(held_out, rest, maximum_lag_samples)
            loo_zero.append(zero)
            loo_oracle.append(oracle)
            loo_lags.append(lag)
            loo_folds.append(
                {
                    "held_out_recording_id": recording_id,
                    "zero_lag_correlation": zero,
                    "oracle_shift_correlation": oracle,
                    "oracle_lag_milliseconds": lag * kernel_interval_milliseconds,
                }
            )
        partition_zero = []
        partition_oracle = []
        partition_lags = []
        for first, second in _unique_near_equal_partitions(len(recording_ids)):
            first_mean = np.mean(kernels[np.asarray(first)], axis=0)
            second_mean = np.mean(kernels[np.asarray(second)], axis=0)
            zero = _correlation(first_mean, second_mean)
            oracle, lag = _oracle_shift(
                first_mean, second_mean, maximum_lag_samples
            )
            partition_zero.append(zero)
            partition_oracle.append(oracle)
            partition_lags.append(lag)
        original = robustness["source_results"][source]
        expected_loo = original["held_out_recording_id_vs_rest"]["summary"]
        expected_partitions = original[
            "exhaustive_near_equal_recording_id_partitions"
        ]["summary"]
        if not math.isclose(
            float(np.median(loo_zero)),
            float(expected_loo["median"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(np.quantile(partition_zero, 0.05)),
            float(expected_partitions["p05"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{source} zero-lag robustness changed")
        oracle_gates = {
            "mean_median_kernel_correlation": original["gates"][
                "mean_median_kernel_correlation"
            ],
            "median_leave_one_recording_id_out_correlation": (
                float(np.median(loo_oracle))
                >= float(
                    thresholds[
                        "minimum_median_leave_one_recording_id_out_correlation"
                    ]
                )
            ),
            "minimum_leave_one_recording_id_out_correlation": (
                min(loo_oracle)
                >= float(
                    thresholds["minimum_leave_one_recording_id_out_correlation"]
                )
            ),
            "partition_correlation_p05": (
                float(np.quantile(partition_oracle, 0.05))
                >= float(thresholds["minimum_partition_correlation_p05"])
            ),
        }
        all_lags = np.asarray(loo_lags + partition_lags)
        source_results[source] = {
            "recording_id_count": len(recording_ids),
            "leave_one_recording_id_out": {
                "folds": loo_folds,
                "zero_lag_summary": _summary(loo_zero),
                "oracle_shift_summary": _summary(loo_oracle),
            },
            "near_equal_partitions": {
                "partition_count": len(partition_zero),
                "zero_lag_summary": _summary(partition_zero),
                "oracle_shift_summary": _summary(partition_oracle),
            },
            "oracle_lag_boundary_fraction": float(
                np.mean(np.abs(all_lags) == maximum_lag_samples)
            ),
            "original_gates": original["gates"],
            "oracle_diagnostic_gates": oracle_gates,
            "oracle_all_shape_gates_passed": bool(all(oracle_gates.values())),
        }
    tm2 = source_results["Tm2"]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(source_config_path): _sha256(root / source_config_path),
                str(robustness_protocol_path): _sha256(
                    root / robustness_protocol_path
                ),
                str(robustness_evidence_path): _sha256(
                    root / robustness_evidence_path
                ),
                **{
                    source_config["white_noise_files"][source]["path"]: _sha256(
                        path
                    )
                    for source, path in source_paths.items()
                },
            },
            "kernel_sample_interval_milliseconds": kernel_interval_milliseconds,
            "maximum_oracle_lag_milliseconds": maximum_lag_milliseconds,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "thresholds": thresholds,
        "source_results": source_results,
        "Tm2_zero_lag_LOO_median": tm2["leave_one_recording_id_out"][
            "zero_lag_summary"
        ]["median"],
        "Tm2_oracle_shift_LOO_median": tm2["leave_one_recording_id_out"][
            "oracle_shift_summary"
        ]["median"],
        "Tm2_zero_lag_partition_p05": tm2["near_equal_partitions"][
            "zero_lag_summary"
        ]["p05"],
        "Tm2_oracle_shift_partition_p05": tm2["near_equal_partitions"][
            "oracle_shift_summary"
        ]["p05"],
        "Tm2_oracle_all_shape_gates_passed": tm2[
            "oracle_all_shape_gates_passed"
        ],
        "bounded_latency_jitter_explains_Tm2_robustness_failure": False,
        "original_population_kernel_robustness_failure_retained": True,
        "authorize_oracle_aligned_population_kernel": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            "Tm2_partition_p05_remains_below_frozen_threshold_after_bounded_"
            "oracle_alignment"
        ),
        "boundary": config["boundary"],
    }
