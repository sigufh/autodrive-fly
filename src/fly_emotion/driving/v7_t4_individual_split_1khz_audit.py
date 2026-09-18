"""Recompute the frozen T4 individual split on verified 1-kHz arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t4_individual_split_audit import _unit, _validation_folds

CONFIG = Path("configs/driving-v7-t4-individual-split-1khz-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_individual_split_1khz_audit.py")


def _condition_metrics(
    values: np.ndarray, source: str, condition: dict, protocol: dict
) -> dict:
    time = np.arange(values.shape[1], dtype=np.float64) / 1000.0
    baseline_start, baseline_stop = map(float, protocol["baseline_seconds"])
    baseline_mask = (time >= baseline_start) & (time < baseline_stop)
    response_mask = (time >= float(condition["onset_seconds"])) & (
        time
        <= float(condition["onset_seconds"]) + float(protocol["response_duration_seconds"])
    )
    baseline = np.mean(values[:, baseline_mask], axis=1)
    oriented = float(condition["sign"]) * (values[:, response_mask] - baseline[:, None])
    width = int(protocol["split"]["validation_individual_count"])
    folds = _validation_folds(values.shape[0], width)
    fold_results = []
    individual_correlations = []
    validation_coverage = set()
    ids = [f"{source}-{index + 1}" for index in range(values.shape[0])]
    for fold_index, validation_indices in enumerate(folds):
        validation_coverage.update(validation_indices)
        training_indices = [
            index for index in range(values.shape[0]) if index not in validation_indices
        ]
        training_kernel = _unit(np.mean(oriented[training_indices], axis=0))
        validation_kernel = _unit(np.mean(oriented[validation_indices], axis=0))
        validation_correlation = float(np.corrcoef(training_kernel, validation_kernel)[0, 1])
        per_individual = [
            float(np.corrcoef(training_kernel, _unit(oriented[index]))[0, 1])
            for index in validation_indices
        ]
        individual_correlations.extend(per_individual)
        fold_results.append(
            {
                "fold_index": fold_index,
                "training_individual_ids": [ids[index] for index in training_indices],
                "validation_individual_ids": [ids[index] for index in validation_indices],
                "training_validation_disjoint": not bool(
                    set(training_indices) & set(validation_indices)
                ),
                "validation_kernel_correlation": validation_correlation,
                "validation_individual_correlations": per_individual,
            }
        )
    gates_config = protocol["gates"]
    gate_values = {
        "minimum_validation_fold_correlation": min(
            item["validation_kernel_correlation"] for item in fold_results
        )
        >= float(gates_config["minimum_validation_fold_correlation"]),
        "median_individual_correlation": float(np.median(individual_correlations))
        >= float(gates_config["minimum_median_individual_correlation"]),
        "minimum_individual_correlation": min(individual_correlations)
        >= float(gates_config["minimum_individual_correlation"]),
        "every_individual_validated": validation_coverage == set(range(values.shape[0])),
        "every_fold_training_validation_disjoint": all(
            item["training_validation_disjoint"] for item in fold_results
        ),
    }
    return {
        "individual_count": values.shape[0],
        "sample_interval_milliseconds": 1.0,
        "response_sample_count": int(response_mask.sum()),
        "fold_count": len(fold_results),
        "validation_coverage_count": len(validation_coverage),
        "minimum_validation_fold_correlation": min(
            item["validation_kernel_correlation"] for item in fold_results
        ),
        "median_validation_fold_correlation": float(
            np.median([item["validation_kernel_correlation"] for item in fold_results])
        ),
        "median_individual_correlation": float(np.median(individual_correlations)),
        "minimum_individual_correlation": min(individual_correlations),
        "negative_individual_correlation_fraction": float(
            np.mean(np.asarray(individual_correlations) < 0.0)
        ),
        "folds": fold_results,
        "gates": gate_values,
        "passed": all(gate_values.values()),
    }


def evaluate_v7_t4_individual_split_1khz_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    protocol_path = Path(config["frozen_split_protocol"])
    protocol = yaml.safe_load((root / protocol_path).read_text(encoding="utf-8"))
    retrieval_path = Path(config["retrieval_evidence"])
    identity_path = Path(config["identity_evidence"])
    prior_path = Path(config["prior_10ms_evidence"])
    retrieval = json.loads((root / retrieval_path).read_text(encoding="utf-8"))
    identity = json.loads((root / identity_path).read_text(encoding="utf-8"))
    prior = json.loads((root / prior_path).read_text(encoding="utf-8"))
    if not retrieval["gates"]["full_resolution_fixed_split_recompute_authorized"]:
        raise ValueError("verified Edmond arrays and ordinal identity are required")
    if not identity["transfer_gates"][
        "stable_pseudonymous_biological_individual_ID_available"
    ]:
        raise ValueError("paper-backed pseudonymous biological identity is required")
    source_dir = root / config["source_directory"]
    arrays = {}
    payloads = {}
    for source in protocol["source_types"]:
        filename = config["source_files"][source]
        expected = retrieval["frozen_file_manifest"][filename]
        path = source_dir / filename
        if _sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"1-kHz source payload changed: {filename}")
        values = np.load(path, allow_pickle=False)
        if list(values.shape) != expected["shape"] or str(values.dtype) != expected["dtype"]:
            raise ValueError(f"1-kHz source structure changed: {filename}")
        arrays[source] = values
        payloads[filename] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        }
    conditions = {}
    deltas = {}
    for condition_index, (condition_name, condition) in enumerate(protocol["conditions"].items()):
        source_results = {}
        deltas[condition_name] = {}
        for source in protocol["source_types"]:
            result = _condition_metrics(
                arrays[source][condition_index], source, condition, protocol
            )
            source_results[source] = result
            old = prior["conditions"][condition_name]["sources"][source]
            deltas[condition_name][source] = {
                key: result[key] - old[key]
                for key in (
                    "minimum_validation_fold_correlation",
                    "median_validation_fold_correlation",
                    "median_individual_correlation",
                    "minimum_individual_correlation",
                    "negative_individual_correlation_fraction",
                )
            } | {"pass_changed": result["passed"] != old["passed"]}
        conditions[condition_name] = {
            "sources": source_results,
            "all_sources_passed": all(item["passed"] for item in source_results.values()),
        }
    passing = sorted(
        f"{condition}:{source}"
        for condition, details in conditions.items()
        for source, result in details["sources"].items()
        if result["passed"]
    )
    prior_passing = sorted(
        f"{condition}:{source}"
        for condition, details in prior["conditions"].items()
        for source, result in details["sources"].items()
        if result["passed"]
    )
    all_passed = all(item["all_sources_passed"] for item in conditions.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(protocol_path): _sha256(root / protocol_path),
                str(retrieval_path): _sha256(root / retrieval_path),
                str(identity_path): _sha256(root / identity_path),
                str(prior_path): _sha256(root / prior_path),
            },
            "sample_interval_milliseconds": config["sample_interval_milliseconds"],
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "verified_payloads": payloads,
        "identity_evidence": retrieval["array_to_workbook_identity"],
        "frozen_split_contract": protocol["split"],
        "frozen_conditions": protocol["conditions"],
        "frozen_gates": protocol["gates"],
        "conditions": conditions,
        "comparison_to_10ms": {
            "metric_deltas_1khz_minus_10ms": deltas,
            "prior_passing_source_conditions": prior_passing,
            "full_resolution_passing_source_conditions": passing,
            "pass_fail_pattern_changed": passing != prior_passing,
        },
        "all_source_condition_individual_split_gates_passed": all_passed,
        "authorize_T4_source_dynamics_fit": False,
        "authorize_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            None if all_passed else "seven_of_eight_1khz_source_condition_individual_splits_fail"
        ),
        "boundary": config["boundary"],
    }
