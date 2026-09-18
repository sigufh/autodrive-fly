"""Test C3 measured-filter robustness under the TimingModels processing protocol."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from fly_emotion.driving.v7_c3_strf_source_dynamics_audit import (
    _condition_by_name,
    _first_corrcoef_row,
)
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-c3-measured-filter-robustness.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_c3_measured_filter_robustness.py"
)


def _unit(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - float(values[0])
    return centered / max(float(np.linalg.norm(centered)), np.finfo(float).tiny)


def _laguerre_basis(length: int, order: int, alpha: float) -> np.ndarray:
    basis = np.zeros((length, order), dtype=np.float64)
    basis[0, 0] = np.sqrt(1.0 - alpha)
    for column in range(1, order):
        basis[0, column] = np.sqrt(alpha) * basis[0, column - 1]
    for row in range(1, length):
        basis[row, 0] = np.sqrt(alpha) * basis[row - 1, 0]
        for column in range(1, order):
            basis[row, column] = (
                np.sqrt(alpha) * basis[row - 1, column]
                + np.sqrt(alpha) * basis[row, column - 1]
                - basis[row - 1, column - 1]
            )
    return basis


def _raw_units(c3: dict, contract: dict) -> list[dict]:
    sample_interval = float(contract["sample_interval_seconds"])
    time = np.arange(
        float(contract["time_start_seconds"]),
        float(contract["time_stop_seconds"]) + sample_interval / 2.0,
        sample_interval,
    )
    causal_bins = int(contract["causal_peak_bin_count"])
    threshold = float(contract["author_prediction_correlation_threshold"])
    units = []
    for fly in c3["DATA"]:
        for axis in contract["axes"]:
            if axis not in fly:
                continue
            data = fly[axis]
            traces = []
            for score, strf in zip(
                _first_corrcoef_row(data["corrcoefs"]),
                np.asarray(data["STRFs"], dtype=np.float64),
                strict=True,
            ):
                if score < threshold:
                    continue
                space_by_time = strf.T
                if axis == "Az" and contract["azimuth_spatial_flip"]:
                    space_by_time = np.flipud(space_by_time)
                peak_row = int(np.argmax(np.max(space_by_time, axis=1)))
                peak_index = int(np.argmax(space_by_time[peak_row, :causal_bins]))
                if time[peak_index] <= float(contract["retain_peak_after_seconds"]):
                    continue
                start = max(0, peak_row - 1)
                stop = min(space_by_time.shape[0], peak_row + 2)
                traces.append(
                    np.mean(space_by_time[start:stop, :causal_bins], axis=0)[::-1]
                )
            if traces:
                units.append(
                    {
                        "fly": str(fly["Flyname"]),
                        "axis": axis,
                        "ROI_count": len(traces),
                        "trace": np.mean(traces, axis=0),
                    }
                )
    return units


def _process(
    values: np.ndarray,
    tau_seconds: float,
    sample_interval_seconds: float,
    projection: np.ndarray,
) -> np.ndarray:
    time = np.arange(len(values), dtype=np.float64) * sample_interval_seconds
    calcium_kernel = np.exp(-time / tau_seconds) / tau_seconds
    deconvolved = np.fft.ifft(
        np.fft.fft(values) / np.fft.fft(calcium_kernel)
    ).real
    return _unit(projection @ deconvolved)


def evaluate_v7_c3_measured_filter_robustness(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_config_path = Path(config["source_config"])
    source_config = yaml.safe_load(
        (root / source_config_path).read_text(encoding="utf-8")
    )
    source_evidence_path = Path(config["source_evidence"])
    source_evidence = json.loads(
        (root / source_evidence_path).read_text(encoding="utf-8")
    )
    timing_evidence_path = Path(config["timing_models_evidence"])
    timing_evidence = json.loads(
        (root / timing_evidence_path).read_text(encoding="utf-8")
    )
    code_path = _verify_code(root, config["timing_models_code"])
    source_path = root / source_config["source_data"]["path"]
    if _sha256(source_path) != source_evidence["protocol"]["source_data"]["sha256"]:
        raise ValueError("C3 source payload differs from verified evidence")
    assumptions = config["processing_contract"]["calcium_low_pass_seconds"]
    expected = [value / 1000.0 for value in timing_evidence["filter_data"][
        "calcium_deconvolution_milliseconds"
    ]]
    if [float(value) for value in assumptions[:3]] != expected:
        raise ValueError("C3 deconvolution assumptions differ from TimingModels")
    records = scipy.io.loadmat(source_path, simplify_cells=True)["RF_DATA"]
    if not isinstance(records, list):
        records = list(records)
    c3 = _condition_by_name(records, source_config["analysis_contract"]["condition"])
    units = _raw_units(c3, source_config["analysis_contract"])
    length = int(source_config["analysis_contract"]["causal_peak_bin_count"])
    order = int(config["processing_contract"]["laguerre_basis_order"])
    alpha = float(config["processing_contract"]["laguerre_alpha"])
    basis = _laguerre_basis(length, order, alpha)
    projection = basis @ basis.T
    sample_interval = float(config["processing_contract"]["sample_interval_seconds"])
    processed = {
        float(tau): [
            {
                **{key: item[key] for key in ("fly", "axis", "ROI_count")},
                "trace": _process(item["trace"], float(tau), sample_interval, projection),
            }
            for item in units
        ]
        for tau in assumptions
    }
    cross_assumption = {}
    for index, item in enumerate(units):
        values = [
            float(
                np.corrcoef(
                    processed[float(first)][index]["trace"],
                    processed[float(second)][index]["trace"],
                )[0, 1]
            )
            for first, second in combinations(assumptions, 2)
        ]
        cross_assumption[f"{item['fly']}:{item['axis']}"] = min(values)
    threshold = float(config["gates"]["minimum_held_out_fly_axis_correlation"])
    results = {}
    for tau, tau_units in processed.items():
        folds = []
        for fly in sorted({item["fly"] for item in tau_units}):
            training = _unit(
                np.mean([item["trace"] for item in tau_units if item["fly"] != fly], axis=0)
            )
            for item in (entry for entry in tau_units if entry["fly"] == fly):
                folds.append(
                    {
                        "held_out_fly": fly,
                        "axis": item["axis"],
                        "ROI_count": item["ROI_count"],
                        "correlation": float(np.corrcoef(item["trace"], training)[0, 1]),
                    }
                )
        correlations = [item["correlation"] for item in folds]
        gates = {
            "minimum_held_out_fly_axis_correlation": min(correlations) >= threshold,
            "median_held_out_fly_axis_correlation": float(np.median(correlations))
            >= float(config["gates"]["minimum_median_held_out_fly_axis_correlation"]),
            "every_held_out_fly_axis": all(value >= threshold for value in correlations),
        }
        results[f"{int(round(tau * 1000))}ms"] = {
            "folds": folds,
            "minimum_held_out_correlation": min(correlations),
            "median_held_out_correlation": float(np.median(correlations)),
            "gates": gates,
            "passed": all(gates.values()),
        }
    cross_threshold = float(config["gates"]["minimum_cross_deconvolution_correlation"])
    cross_passed = all(value >= cross_threshold for value in cross_assumption.values())
    all_assumptions_passed = all(item["passed"] for item in results.values())
    passed = cross_passed and all_assumptions_passed
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(source_config_path): _sha256(root / source_config_path),
                str(source_evidence_path): _sha256(root / source_evidence_path),
                str(timing_evidence_path): _sha256(root / timing_evidence_path),
                str(code_path.relative_to(root)): _sha256(code_path),
                "pyproject.toml": _sha256(root / "pyproject.toml"),
            },
            "parameter_fit": False,
            "T4_response_used": False,
            "direction_label_used": False,
            "runtime_modified": False,
        },
        "processing_contract": config["processing_contract"],
        "source_units": [
            {key: item[key] for key in ("fly", "axis", "ROI_count")}
            for item in units
        ],
        "minimum_cross_deconvolution_correlation_by_unit": cross_assumption,
        "cross_deconvolution_stability_passed": cross_passed,
        "leave_one_fly_out_by_assumption": results,
        "every_deconvolution_assumption_passed": all_assumptions_passed,
        "C3_measured_filter_robustness_passed": passed,
        "C3_type_shared_kernel_authorized": passed,
        "advance_to_T4_functional_precheck": passed,
        "stop_reason": (
            None if passed else "C3_deconvolved_kernel_failed_cross_fly_generalization"
        ),
        "boundary": config["boundary"],
    }


def _verify_code(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError("TimingModels Laguerre implementation mismatch")
    return path
