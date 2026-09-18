"""Audit source-type empirical kernels in the official Fig. 3 source-data workbook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-fig3-source-kernel-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_fig3_source_kernel_audit.py")


def _kernel_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _condition(
    frame: pd.DataFrame,
    source_types: dict[str, int],
    onset_seconds: float,
    expected_sign: float,
    protocol: dict,
    gates: dict,
) -> dict:
    time = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    baseline_start, baseline_stop = map(float, protocol["baseline_seconds"])
    baseline_mask = (time >= baseline_start) & (time < baseline_stop)
    onset_index = int(np.argmin(np.abs(time - onset_seconds)))
    onset = float(time[onset_index])
    response_mask = (time >= onset) & (
        time <= onset + float(protocol["response_duration_seconds"])
    )
    response_time = time[response_mask] - onset
    results = {}
    for source_type, expected_count in source_types.items():
        columns = [name for name in frame if str(name).startswith(f"{source_type}-")]
        if len(columns) != int(expected_count):
            raise ValueError(f"unexpected {source_type} cell count in Fig. 3 workbook")
        values = frame[columns].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite {source_type} source voltage in Fig. 3 workbook")
        baseline = np.mean(values[baseline_mask], axis=0)
        baseline_std = np.std(values[baseline_mask], axis=0, ddof=1)
        oriented = float(expected_sign) * (values[response_mask] - baseline)
        amplitude = np.max(oriented, axis=0)
        peak_indices = np.argmax(oriented, axis=0)
        peak_latency = response_time[peak_indices]
        population_mean = np.mean(oriented, axis=1)
        peak = float(np.max(population_mean))
        normalized = population_mean / max(peak, 1e-12)
        first = np.mean(oriented[:, ::2], axis=1)
        second = np.mean(oriented[:, 1::2], axis=1)
        split_correlation = float(np.corrcoef(first, second)[0, 1])
        snr = amplitude / np.maximum(baseline_std, 1e-12)
        peak_iqr_ms = float(
            (np.quantile(peak_latency, 0.75) - np.quantile(peak_latency, 0.25)) * 1000.0
        )
        gate_values = {
            "median_cell_SNR": float(np.median(snr))
            >= float(gates["minimum_median_cell_SNR"]),
            "alternating_split_kernel_correlation": split_correlation
            >= float(gates["minimum_alternating_split_kernel_correlation"]),
            "cell_peak_latency_IQR": peak_iqr_ms
            <= float(gates["maximum_cell_peak_latency_IQR_milliseconds"]),
        }
        results[source_type] = {
            "cell_count": len(columns),
            "expected_response_sign": int(expected_sign),
            "baseline_window_seconds": [baseline_start, baseline_stop],
            "response_onset_seconds": onset,
            "response_sample_count": len(response_time),
            "response_sample_interval_milliseconds": float(
                np.median(np.diff(response_time)) * 1000.0
            ),
            "population_positive_peak_millivolts": peak,
            "median_cell_SNR": float(np.median(snr)),
            "positive_peak_cell_fraction": float(np.mean(amplitude > 0.0)),
            "population_mean_peak_latency_milliseconds": float(
                response_time[int(np.argmax(population_mean))] * 1000.0
            ),
            "median_cell_peak_latency_milliseconds": float(
                np.median(peak_latency) * 1000.0
            ),
            "cell_peak_latency_IQR_milliseconds": peak_iqr_ms,
            "alternating_split_kernel_correlation": split_correlation,
            "normalized_population_kernel_sha256": _kernel_hash(normalized),
            "gates": gate_values,
            "ready": bool(all(gate_values.values())),
        }
    return {
        "source_results": results,
        "all_source_types_ready": all(item["ready"] for item in results.values()),
    }


def evaluate_v7_fig3_source_kernel_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_spec = config["source_data"]
    source_path = root / source_spec["path"]
    if source_path.stat().st_size != int(source_spec["bytes"]):
        raise ValueError("Fig. 3 source-data workbook size mismatch")
    if _sha256(source_path) != source_spec["sha256"]:
        raise ValueError("Fig. 3 source-data workbook SHA-256 mismatch")
    ephys_path = Path(config["ephys_evidence"])
    ephys = json.loads((root / ephys_path).read_text(encoding="utf-8"))
    crossings = ephys["paper_model_replay"]["led_half_intensity_crossing_milliseconds"]
    conditions = {}
    workbook = pd.ExcelFile(source_path)
    expected_sheets = [item["sheet"] for item in config["conditions"].values()]
    for condition_name, condition in config["conditions"].items():
        keys = condition["onset_evidence_keys"]
        onset_ms = float(np.mean([crossings[name] for name in keys]))
        frame = pd.read_excel(source_path, sheet_name=condition["sheet"])
        conditions[condition_name] = _condition(
            frame,
            config["source_types"],
            onset_ms / 1000.0,
            float(condition["expected_sign"]),
            config["kernel_protocol"],
            config["gates"],
        )
    on = conditions["on"]["source_results"]
    fast = config["source_groups"]["prior_fast"]
    delayed = config["source_groups"]["prior_delayed"]
    pairwise_delay_ms = {
        f"{slow}_minus_{quick}": (
            on[slow]["median_cell_peak_latency_milliseconds"]
            - on[quick]["median_cell_peak_latency_milliseconds"]
        )
        for slow in delayed
        for quick in fast
    }
    prior_group_ordering_passed = all(value > 0.0 for value in pairwise_delay_ms.values())
    all_ready = all(item["all_source_types_ready"] for item in conditions.values())
    source_specific_ready = conditions["on"]["all_source_types_ready"]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(ephys_path): _sha256(root / ephys_path),
                "pyproject.toml": _sha256(root / "pyproject.toml"),
            },
            "source_file": {
                "path": source_spec["path"],
                "url": source_spec["url"],
                "bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
                "workbook_sheet_count": len(workbook.sheet_names),
                "required_source_sheets": expected_sheets,
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "conditions": conditions,
        "prior_fast_delayed_grouping": {
            "fast_sources": fast,
            "delayed_sources": delayed,
            "pairwise_ON_median_peak_latency_differences_milliseconds": pairwise_delay_ms,
            "every_delayed_source_later_than_every_fast_source": prior_group_ordering_passed,
        },
        "all_condition_source_kernels_ready": all_ready,
        "ON_source_specific_kernels_ready": source_specific_ready,
        "prior_two_pool_kernel_authorized": bool(
            source_specific_ready and prior_group_ordering_passed
        ),
        "source_specific_kernel_candidate_authorized": False,
        "stop_reason": (
            "source_kernel_stability_and_prior_fast_delayed_ordering_gates_failed"
        ),
        "boundary": config["boundary"],
    }
