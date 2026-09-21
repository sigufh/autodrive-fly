"""Audit T5 population-kernel robustness across recording identifiers."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
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

CONFIG = Path("configs/driving-v7-t5-population-kernel-robustness-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_population_kernel_robustness_audit.py"
)


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    value = float(np.corrcoef(first, second)[0, 1])
    if not np.isfinite(value):
        raise ValueError("non-finite T5 kernel correlation")
    return value


def _unique_near_equal_partitions(count: int):
    smaller = count // 2
    for selected in combinations(range(count), smaller):
        if count % 2 == 0 and 0 not in selected:
            continue
        selected_set = set(selected)
        complement = tuple(index for index in range(count) if index not in selected_set)
        yield selected, complement


def evaluate_v7_t5_population_kernel_robustness_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_path = Path(config["source_kernel_evidence"])
    source_protocol_path = Path(config["source_kernel_protocol"])
    source_config_path = Path(config["source_config"])
    threshold_path = Path(config["threshold_source"])
    evidence = json.loads((root / evidence_path).read_text(encoding="utf-8"))
    source_config = yaml.safe_load((root / source_config_path).read_text(encoding="utf-8"))
    inherited = yaml.safe_load((root / threshold_path).read_text(encoding="utf-8"))[
        "gates"
    ]
    gates = config["gates"]
    inherited_mapping = {
        "minimum_mean_median_kernel_correlation": inherited[
            "minimum_mean_median_kernel_correlation"
        ],
        "minimum_median_leave_one_recording_id_out_correlation": inherited[
            "minimum_median_leave_one_cell_out_correlation"
        ],
        "minimum_leave_one_recording_id_out_correlation": inherited[
            "minimum_leave_one_cell_out_correlation"
        ],
        "minimum_partition_correlation_p05": inherited[
            "minimum_bootstrap_split_correlation_p05"
        ],
        "require_every_source": inherited["require_every_source_type_and_condition"],
    }
    if gates != inherited_mapping:
        raise ValueError("T5 kernel robustness thresholds diverged from source-kernel contract")
    if evidence["aggregate"]["kernel_lengths"] != [499, 500]:
        raise ValueError("source-kernel length inventory changed")

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
        population_mean = np.mean(kernels, axis=0)
        population_median = np.median(kernels, axis=0)
        peak_index = int(np.argmax(np.abs(population_mean)))
        held_out_vs_rest = []
        jackknife_population = []
        peak_shifts = []
        peak_sign_stable = []
        for index, recording_id in enumerate(recording_ids):
            rest = np.mean(np.delete(kernels, index, axis=0), axis=0)
            held_out_vs_rest.append(
                {
                    "held_out_recording_id": recording_id,
                    "correlation": _correlation(kernels[index], rest),
                }
            )
            jackknife_population.append(
                {
                    "excluded_recording_id": recording_id,
                    "correlation_to_full_population_mean": _correlation(
                        rest, population_mean
                    ),
                }
            )
            rest_peak = int(np.argmax(np.abs(rest)))
            peak_shifts.append(rest_peak - peak_index)
            peak_sign_stable.append(
                bool(np.sign(rest[peak_index]) == np.sign(population_mean[peak_index]))
            )
        partitions = []
        for first, second in _unique_near_equal_partitions(len(recording_ids)):
            partitions.append(
                {
                    "first_recording_ids": [recording_ids[index] for index in first],
                    "second_recording_ids": [recording_ids[index] for index in second],
                    "correlation": _correlation(
                        np.mean(kernels[np.asarray(first)], axis=0),
                        np.mean(kernels[np.asarray(second)], axis=0),
                    ),
                }
            )
        loo_values = [item["correlation"] for item in held_out_vs_rest]
        partition_values = [item["correlation"] for item in partitions]
        gate_values = {
            "mean_median_kernel_correlation": _correlation(
                population_mean, population_median
            )
            >= float(gates["minimum_mean_median_kernel_correlation"]),
            "median_leave_one_recording_id_out_correlation": float(
                np.median(loo_values)
            )
            >= float(gates["minimum_median_leave_one_recording_id_out_correlation"]),
            "minimum_leave_one_recording_id_out_correlation": min(loo_values)
            >= float(gates["minimum_leave_one_recording_id_out_correlation"]),
            "partition_correlation_p05": float(np.quantile(partition_values, 0.05))
            >= float(gates["minimum_partition_correlation_p05"]),
        }
        source_results[source] = {
            "recording_id_count": len(recording_ids),
            "recording_ids": recording_ids,
            "mean_median_kernel_correlation": _correlation(
                population_mean, population_median
            ),
            "held_out_recording_id_vs_rest": {
                "folds": held_out_vs_rest,
                "summary": _summary(loo_values),
            },
            "exhaustive_near_equal_recording_id_partitions": {
                "partition_count": len(partitions),
                "partitions": partitions,
                "summary": _summary(partition_values),
            },
            "population_mean_jackknife": {
                "folds": jackknife_population,
                "minimum_correlation_to_full_mean": min(
                    item["correlation_to_full_population_mean"]
                    for item in jackknife_population
                ),
                "maximum_absolute_peak_shift_samples": int(
                    np.max(np.abs(peak_shifts))
                ),
                "peak_sign_stable_in_every_fold": all(peak_sign_stable),
                "used_as_transfer_gate": False,
            },
            "gates": gate_values,
            "passed": all(gate_values.values()),
        }
    all_passed = all(item["passed"] for item in source_results.values())
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(evidence_path): _sha256(root / evidence_path),
        str(source_protocol_path): _sha256(root / source_protocol_path),
        str(source_config_path): _sha256(root / source_config_path),
        str(threshold_path): _sha256(root / threshold_path),
        **{
            source_config["white_noise_files"][source]["path"]: _sha256(path)
            for source, path in source_paths.items()
        },
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "threshold_source": str(threshold_path),
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "thresholds": gates,
        "source_results": source_results,
        "passing_sources": [
            source for source, item in source_results.items() if item["passed"]
        ],
        "failing_sources": [
            source for source, item in source_results.items() if not item["passed"]
        ],
        "all_population_kernel_robustness_gates_passed": all_passed,
        "recording_id_is_biological_individual_id": False,
        "independent_biological_validation_available": False,
        "authorize_population_kernel_transfer": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            None
            if all_passed
            else "Tm2_population_kernel_failed_recording_id_robustness_gates"
        ),
        "boundary": config["boundary"],
    }
