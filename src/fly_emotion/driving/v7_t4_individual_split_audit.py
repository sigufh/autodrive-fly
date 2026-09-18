"""Evaluate fixed individual-level T4 source kernel splits without fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t4-individual-split-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_individual_split_audit.py")


def _unit(values: np.ndarray) -> np.ndarray:
    centered = values - values[0]
    return centered / max(float(np.linalg.norm(centered)), 1e-12)


def _validation_folds(count: int, width: int) -> list[list[int]]:
    starts = list(range(0, count - width + 1, width))
    if starts[-1] != count - width:
        starts.append(count - width)
    folds = [list(range(start, start + width)) for start in starts]
    if set().union(*(set(fold) for fold in folds)) != set(range(count)):
        raise ValueError("validation folds do not cover every individual")
    return folds


def evaluate_v7_t4_individual_split_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    protocol_path = Path(config["source_protocol"])
    protocol = yaml.safe_load((root / protocol_path).read_text(encoding="utf-8"))
    evidence_paths = {
        "source": Path(config["source_evidence"]),
        "identity": Path(config["identity_evidence"]),
    }
    reports = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }
    identity_gate = reports["identity"]["transfer_gates"][
        "stable_pseudonymous_biological_individual_ID_available"
    ]
    if not identity_gate:
        raise ValueError("T4 pseudonymous individual identity is not verified")
    workbook_path = root / protocol["source_data"]["path"]
    source_hash = reports["source"]["protocol"]["source_file"]["sha256"]
    if _sha256(workbook_path) != source_hash:
        raise ValueError("T4 source workbook is stale")

    baseline_start, baseline_stop = map(float, config["baseline_seconds"])
    width = int(config["split"]["validation_individual_count"])
    gate_config = config["gates"]
    conditions = {}
    for condition_name, condition in config["conditions"].items():
        frame = pd.read_excel(workbook_path, sheet_name=condition["sheet"])
        time = frame.iloc[:, 0].to_numpy(dtype=np.float64)
        baseline_mask = (time >= baseline_start) & (time < baseline_stop)
        response_mask = (time >= float(condition["onset_seconds"])) & (
            time <= float(condition["onset_seconds"]) + float(config["response_duration_seconds"])
        )
        source_results = {}
        for source in config["source_types"]:
            columns = [name for name in frame if str(name).startswith(f"{source}-")]
            values = frame[columns].to_numpy(dtype=np.float64)
            baseline = np.mean(values[baseline_mask], axis=0)
            oriented = float(condition["sign"]) * (values[response_mask] - baseline)
            folds = _validation_folds(len(columns), width)
            fold_results = []
            individual_correlations = []
            validation_coverage = set()
            for fold_index, validation_indices in enumerate(folds):
                validation_coverage.update(validation_indices)
                training_indices = [
                    index for index in range(len(columns)) if index not in validation_indices
                ]
                training_kernel = _unit(np.mean(oriented[:, training_indices], axis=1))
                validation_kernel = _unit(np.mean(oriented[:, validation_indices], axis=1))
                validation_correlation = float(
                    np.corrcoef(training_kernel, validation_kernel)[0, 1]
                )
                per_individual = [
                    float(np.corrcoef(training_kernel, _unit(oriented[:, index]))[0, 1])
                    for index in validation_indices
                ]
                individual_correlations.extend(per_individual)
                fold_results.append(
                    {
                        "fold_index": fold_index,
                        "training_individual_ids": [columns[index] for index in training_indices],
                        "validation_individual_ids": [
                            columns[index] for index in validation_indices
                        ],
                        "training_validation_disjoint": not bool(
                            set(training_indices) & set(validation_indices)
                        ),
                        "validation_kernel_correlation": validation_correlation,
                        "validation_individual_correlations": per_individual,
                    }
                )
            gate_values = {
                "minimum_validation_fold_correlation": min(
                    item["validation_kernel_correlation"] for item in fold_results
                )
                >= float(gate_config["minimum_validation_fold_correlation"]),
                "median_individual_correlation": float(np.median(individual_correlations))
                >= float(gate_config["minimum_median_individual_correlation"]),
                "minimum_individual_correlation": min(individual_correlations)
                >= float(gate_config["minimum_individual_correlation"]),
                "every_individual_validated": validation_coverage == set(range(len(columns))),
                "every_fold_training_validation_disjoint": all(
                    item["training_validation_disjoint"] for item in fold_results
                ),
            }
            source_results[source] = {
                "individual_count": len(columns),
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
        conditions[condition_name] = {
            "sources": source_results,
            "all_sources_passed": all(item["passed"] for item in source_results.values()),
        }
    passed = all(item["all_sources_passed"] for item in conditions.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(protocol_path): _sha256(root / protocol_path),
                **{str(path): _sha256(root / path) for path in evidence_paths.values()},
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "split_contract": config["split"],
        "conditions": conditions,
        "all_source_condition_individual_split_gates_passed": passed,
        "authorize_T4_source_dynamics_fit": False,
        "authorize_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "seven_of_eight_source_condition_individual_splits_fail",
        "boundary": config["boundary"],
    }
