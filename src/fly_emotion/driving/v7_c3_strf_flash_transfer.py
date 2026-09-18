"""Test a zero-fit C3 STRF step prediction on non-overlapping flash flies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from fly_emotion.driving.v7_c3_flash_preregistration import _author_aggregate
from fly_emotion.driving.v7_c3_measured_filter_robustness import _raw_units
from fly_emotion.driving.v7_c3_strf_source_dynamics_audit import _condition_by_name
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-c3-strf-flash-transfer.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_c3_strf_flash_transfer.py")


def _unit(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - float(values[0])
    return centered / max(float(np.linalg.norm(centered)), np.finfo(np.float64).tiny)


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "minimum": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _flash_fly_means(records: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fly_means, mean_stimulus, _ = _author_aggregate(records)
    # _author_aggregate returns flies in np.unique order; recover exactly that identity order.
    duration = fly_means.shape[1]
    ids, eligible_traces = [], []
    for record in records:
        stimulus = np.asarray(record.istim, dtype=np.int16)
        anchors = np.flatnonzero(np.diff(stimulus[:-duration]) < 0) + 1
        signal = np.asarray(record.idSignal1, dtype=np.float64)
        signal = signal / np.mean(signal) - 1.0
        trace = np.mean([signal[i : i + duration] for i in anchors], axis=0)
        if np.sum(trace) != 0 and np.all(np.isfinite(trace)):
            eligible_traces.append(trace)
            ids.append(str(record.name).split("_Image")[0])
    traces = np.stack(eligible_traces)
    ids = np.asarray(ids)
    correlations = np.corrcoef(mean_stimulus[None, :], traces)[0, 1:]
    traces, ids = traces[correlations > 0], ids[correlations > 0]
    fly_ids = np.unique(ids)
    recovered = np.stack([np.mean(traces[ids == fly], axis=0) for fly in fly_ids])
    if not np.allclose(recovered, fly_means):
        raise ValueError("C3 flash fly identity recovery differs from author aggregation")
    return recovered, mean_stimulus, fly_ids


def evaluate_v7_c3_strf_flash_transfer(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    strf_config_path = Path(config["strf_protocol"])
    strf_config = yaml.safe_load((root / strf_config_path).read_text())
    strf_evidence_path = Path(config["strf_evidence"])
    strf_evidence = json.loads((root / strf_evidence_path).read_text())
    flash_path = Path(config["flash_preregistration"])
    flash_evidence = json.loads((root / flash_path).read_text())
    strf_source = root / strf_config["source_data"]["path"]
    if _sha256(strf_source) != strf_evidence["protocol"]["source_data"]["sha256"]:
        raise ValueError("C3 STRF source differs from verified evidence")
    flash_source = root / flash_evidence["protocol"]["source_data"]["path"]
    if _sha256(flash_source) != flash_evidence["protocol"]["source_data"]["sha256"]:
        raise ValueError("C3 flash source differs from preregistration")
    records = scipy.io.loadmat(strf_source, simplify_cells=True)["RF_DATA"]
    records = records if isinstance(records, list) else list(records)
    units = _raw_units(
        _condition_by_name(records, strf_config["analysis_contract"]["condition"]),
        strf_config["analysis_contract"],
    )
    source_flies = sorted({item["fly"] for item in units})
    if source_flies != config["source_cohort"]["fly_ids"]:
        raise ValueError("C3 STRF source cohort differs from frozen IDs")
    if len(units) != int(config["source_cohort"]["expected_unit_count"]):
        raise ValueError("C3 STRF source unit count differs from frozen contract")
    flash_records = np.atleast_1d(
        scipy.io.loadmat(flash_source, struct_as_record=False, squeeze_me=True)[
            flash_evidence["author_analysis_contract"]["source_struct"]
        ]
    )
    flash_means, mean_stimulus, flash_flies = _flash_fly_means(flash_records)
    overlap = sorted(set(source_flies) & set(flash_flies.tolist()))
    external_mask = ~np.isin(flash_flies, source_flies)
    external = flash_means[external_mask]
    external_flies = flash_flies[external_mask]
    if len(external_flies) != int(config["external_flash_cohort"]["expected_fly_count"]):
        raise ValueError("independent C3 flash cohort size differs from frozen contract")
    if overlap != source_flies:
        raise ValueError("unexpected C3 source/flash cohort overlap")
    onset = int(np.flatnonzero(mean_stimulus >= 0.5)[0])
    flash_time = (np.arange(len(mean_stimulus)) - onset) / float(
        flash_evidence["author_analysis_contract"]["interpolated_rate_hz"]
    )
    start, stop = map(float, config["transfer"]["comparison_window_seconds"])
    mask = (flash_time >= start) & (flash_time < stop)
    target_time = flash_time[mask]
    external_shapes = np.stack([_unit(trace[mask]) for trace in external])
    source_dt = float(config["transfer"]["source_sample_interval_seconds"])
    predictions = []
    for item in units:
        impulse = np.asarray(item["trace"], dtype=np.float64)
        step = np.cumsum(impulse) * source_dt
        prediction_time = np.arange(len(step)) * source_dt
        predictions.append(_unit(np.interp(target_time, prediction_time, step)))
    predictions = np.stack(predictions)
    mean_prediction = _unit(np.mean(predictions, axis=0))
    mean_external = _unit(np.mean(external_shapes, axis=0))
    external_correlations = [
        float(np.corrcoef(mean_prediction, trace)[0, 1]) for trace in external_shapes
    ]
    source_lopo = []
    for fly in source_flies:
        kept = [
            prediction for prediction, item in zip(predictions, units, strict=True)
            if item["fly"] != fly
        ]
        source_lopo.append(
            {
                "held_out_fly": fly,
                "training_unit_count": len(kept),
                "external_mean_correlation": float(
                    np.corrcoef(_unit(np.mean(kept, axis=0)), mean_external)[0, 1]
                ),
            }
        )
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    bootstrap = []
    for _ in range(int(config["bootstrap"]["replicates"])):
        sampled_prediction = _unit(
            np.mean(predictions[rng.integers(0, len(predictions), len(predictions))], axis=0)
        )
        sampled_external = _unit(
            np.mean(
                external_shapes[
                    rng.integers(0, len(external_shapes), len(external_shapes))
                ],
                axis=0,
            )
        )
        bootstrap.append(float(np.corrcoef(sampled_prediction, sampled_external)[0, 1]))
    gates = config["gates"]
    aggregate_correlation = float(np.corrcoef(mean_prediction, mean_external)[0, 1])
    gate_values = {
        "external_mean_correlation": aggregate_correlation
        >= float(gates["minimum_external_mean_correlation"]),
        "median_external_fly_correlation": float(np.median(external_correlations))
        >= float(gates["minimum_median_external_fly_correlation"]),
        "minimum_external_fly_correlation": min(external_correlations)
        >= float(gates["minimum_external_fly_correlation"]),
        "bootstrap_correlation_p05": float(np.quantile(bootstrap, 0.05))
        >= float(gates["minimum_bootstrap_correlation_p05"]),
        "source_leave_one_fly_out_correlation": min(
            item["external_mean_correlation"] for item in source_lopo
        )
        >= float(gates["minimum_source_leave_one_fly_out_correlation"]),
        "source_and_external_fly_ids_disjoint": not set(source_flies) & set(external_flies),
    }
    passed = all(gate_values.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(strf_config_path): _sha256(root / strf_config_path),
                str(strf_evidence_path): _sha256(root / strf_evidence_path),
                str(flash_path): _sha256(root / flash_path),
                "pyproject.toml": _sha256(root / "pyproject.toml"),
            },
            "parameter_fit": False,
            "T4_response_used": False,
            "direction_label_used": False,
            "runtime_modified": False,
        },
        "transfer_contract": config["transfer"],
        "cohorts": {
            "source_fly_ids": source_flies,
            "source_fly_axis_unit_count": len(units),
            "flash_fly_count_before_exclusion": len(flash_flies),
            "overlapping_fly_ids_excluded": overlap,
            "external_flash_fly_ids": external_flies.tolist(),
            "external_flash_fly_count": len(external_flies),
        },
        "external_validation": {
            "mean_waveform_correlation": aggregate_correlation,
            "per_fly_correlations": dict(
                zip(external_flies.tolist(), external_correlations, strict=True)
            ),
            "per_fly_correlation_summary": _summary(external_correlations),
            "source_leave_one_fly_out": source_lopo,
            "bootstrap": {
                "seed": int(config["bootstrap"]["seed"]),
                "replicates": int(config["bootstrap"]["replicates"]),
                "correlation_p05": float(np.quantile(bootstrap, 0.05)),
                "correlation_median": float(np.median(bootstrap)),
            },
        },
        "transfer_gates": gate_values,
        "C3_STRF_to_flash_transfer_passed": passed,
        "C3_source_kernel_candidate_authorized": False,
        "advance_to_T4_functional_precheck": False,
        "stop_reason": (
            "validated_shape_still_lacks_physical_v7_and_MaleCNS_state_mapping"
            if passed
            else "C3_STRF_step_prediction_failed_independent_flash_robustness"
        ),
        "boundary": config["boundary"],
    }
