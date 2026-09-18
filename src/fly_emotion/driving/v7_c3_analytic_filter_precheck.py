"""Precheck analytic temporal-filter families against published C3 STRFs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io
import yaml
from scipy.optimize import least_squares
from scipy.signal import TransferFunction, impulse

from fly_emotion.driving.v7_c3_strf_source_dynamics_audit import (
    _condition_by_name,
    _first_corrcoef_row,
)
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-c3-analytic-filter-precheck.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_c3_analytic_filter_precheck.py"
)


def _unit(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    return centered / max(float(np.linalg.norm(centered)), np.finfo(float).tiny)


def _fly_axis_units(c3: dict, source_contract: dict) -> list[dict]:
    sample_interval = float(source_contract["sample_interval_seconds"])
    time = np.arange(
        float(source_contract["time_start_seconds"]),
        float(source_contract["time_stop_seconds"]) + sample_interval / 2.0,
        sample_interval,
    )
    causal_bins = int(source_contract["causal_peak_bin_count"])
    threshold = float(source_contract["author_prediction_correlation_threshold"])
    units = []
    for fly in c3["DATA"]:
        for axis in source_contract["axes"]:
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
                if axis == "Az" and source_contract["azimuth_spatial_flip"]:
                    space_by_time = np.flipud(space_by_time)
                peak_row = int(np.argmax(np.max(space_by_time, axis=1)))
                peak_index = int(np.argmax(space_by_time[peak_row, :causal_bins]))
                if time[peak_index] <= float(
                    source_contract["retain_peak_after_seconds"]
                ):
                    continue
                start = max(0, peak_row - 1)
                stop = min(space_by_time.shape[0], peak_row + 2)
                traces.append(np.mean(space_by_time[start:stop, :causal_bins], axis=0))
            if traces:
                units.append(
                    {
                        "fly": str(fly["Flyname"]),
                        "axis": axis,
                        "ROI_count": len(traces),
                        "trace": _unit(np.mean(traces, axis=0)[::-1]),
                    }
                )
    return units


def _basis(family: str, parameters: np.ndarray, time: np.ndarray, calcium: float) -> np.ndarray:
    if family == "low_pass":
        low_pass = float(np.exp(parameters[0]))
        system = TransferFunction(
            [1.0], [low_pass * calcium, low_pass + calcium, 1.0]
        )
    elif family == "band_pass":
        low_pass = float(np.exp(parameters[0]))
        high_pass = low_pass + float(np.exp(parameters[1]))
        denominator = np.polymul(
            [high_pass, 1.0],
            [low_pass * calcium, low_pass + calcium, 1.0],
        )
        system = TransferFunction([high_pass, 0.0], denominator)
    else:
        raise ValueError(f"unknown filter family: {family}")
    return _unit(impulse(system, T=time)[1])


def _initial_points(family: str, model: dict) -> list[np.ndarray]:
    if family == "low_pass":
        return [
            np.log([float(value)]) for value in model["low_pass_initial_seconds"]
        ]
    return [
        np.log([float(low_pass), float(separation)])
        for low_pass in model["band_pass_low_initial_seconds"]
        for separation in model["band_pass_separation_initial_seconds"]
    ]


def _fit(family: str, units: list[dict], time: np.ndarray, model: dict) -> dict:
    parameter_count = 1 if family == "low_pass" else 2
    lower, upper = map(float, model["filter_parameter_bounds_seconds"])

    def residual(parameters: np.ndarray) -> np.ndarray:
        candidate = _basis(
            family,
            parameters,
            time,
            float(model["calcium_low_pass_seconds"]),
        )
        return np.concatenate([item["trace"] - candidate for item in units])

    fits = [
        least_squares(
            residual,
            start,
            bounds=(np.log([lower] * parameter_count), np.log([upper] * parameter_count)),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=5000,
        )
        for start in _initial_points(family, model)
    ]
    best = min(fits, key=lambda item: float(np.sum(item.fun**2)))
    raw = np.exp(best.x)
    parameters = (
        {"tLP_seconds": float(raw[0])}
        if family == "low_pass"
        else {
            "tLP_seconds": float(raw[0]),
            "tHP_seconds": float(raw[0] + raw[1]),
        }
    )
    residual_sum_squares = float(np.sum(best.fun**2))
    sample_count = len(best.fun)
    bic = float(
        sample_count * np.log(residual_sum_squares / sample_count)
        + parameter_count * np.log(sample_count)
    )
    return {
        "parameters": parameters,
        "residual_sum_squares": residual_sum_squares,
        "BIC": bic,
        "basis": _basis(
            family,
            best.x,
            time,
            float(model["calcium_low_pass_seconds"]),
        ),
    }


def _cross_validate(family: str, units: list[dict], time: np.ndarray, model: dict) -> dict:
    folds = []
    for fly in sorted({item["fly"] for item in units}):
        training = [item for item in units if item["fly"] != fly]
        held_out = [item for item in units if item["fly"] == fly]
        fit = _fit(family, training, time, model)
        fold_results = [
            {
                "axis": item["axis"],
                "ROI_count": item["ROI_count"],
                "correlation": float(np.corrcoef(item["trace"], fit["basis"])[0, 1]),
            }
            for item in held_out
        ]
        folds.append(
            {
                "held_out_fly": fly,
                "training_unit_count": len(training),
                "parameters": fit["parameters"],
                "held_out": fold_results,
            }
        )
    correlations = [
        item["correlation"] for fold in folds for item in fold["held_out"]
    ]
    return {
        "folds": folds,
        "held_out_unit_count": len(correlations),
        "minimum_held_out_correlation": float(np.min(correlations)),
        "median_held_out_correlation": float(np.median(correlations)),
        "held_out_correlations": correlations,
    }


def evaluate_v7_c3_analytic_filter_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_config_path = Path(config["source_config"])
    source_config = yaml.safe_load(
        (root / source_config_path).read_text(encoding="utf-8")
    )
    source_evidence_path = Path(config["source_evidence"])
    source_evidence = json.loads(
        (root / source_evidence_path).read_text(encoding="utf-8")
    )
    arenz_path = Path(config["arenz_evidence"])
    arenz = json.loads((root / arenz_path).read_text(encoding="utf-8"))
    robustness_path = Path(config["inherited_robustness_protocol"])
    robustness = yaml.safe_load((root / robustness_path).read_text(encoding="utf-8"))
    threshold = float(config["gates"]["minimum_held_out_correlation"])
    if threshold != float(robustness["gates"]["minimum_bootstrap_split_correlation_p05"]):
        raise ValueError("C3 analytic-filter correlation gate was not inherited")
    if float(config["model_contract"]["calcium_low_pass_seconds"]) != float(
        arenz["paper_model"]["GCaMP6f_deconvolution_low_pass_seconds"]
    ):
        raise ValueError("C3 calcium low-pass differs from Arenz evidence")
    source_path = root / source_config["source_data"]["path"]
    if _sha256(source_path) != source_evidence["protocol"]["source_data"]["sha256"]:
        raise ValueError("C3 STRF payload differs from verified evidence")
    records = scipy.io.loadmat(source_path, simplify_cells=True)["RF_DATA"]
    if not isinstance(records, list):
        records = list(records)
    c3 = _condition_by_name(records, source_config["analysis_contract"]["condition"])
    units = _fly_axis_units(c3, source_config["analysis_contract"])
    time = np.arange(
        0.0,
        float(config["model_contract"]["causal_duration_seconds"]),
        float(config["model_contract"]["sample_interval_seconds"]),
    )
    families = {}
    for family in config["model_contract"]["families"]:
        full = _fit(family, units, time, config["model_contract"])
        cross_validation = _cross_validate(
            family, units, time, config["model_contract"]
        )
        correlations = cross_validation["held_out_correlations"]
        family_gates = {
            "minimum_held_out_correlation": min(correlations) >= threshold,
            "median_held_out_correlation": float(np.median(correlations)) >= threshold,
            "every_held_out_fly_axis": all(value >= threshold for value in correlations),
        }
        families[family] = {
            "full_data_parameters": full["parameters"],
            "full_data_residual_sum_squares": full["residual_sum_squares"],
            "full_data_BIC": full["BIC"],
            "cross_validation": cross_validation,
            "gates": family_gates,
            "passed": all(family_gates.values()),
        }
    band_pass_better = (
        families["band_pass"]["full_data_BIC"]
        < families["low_pass"]["full_data_BIC"]
    )
    band_pass_passed = bool(families["band_pass"]["passed"] and band_pass_better)
    transfer_gates = {
        "band_pass_source_shape_cross_fly_validated": band_pass_passed,
        "same_experiment_calcium_kernel_available": False,
        "physical_v7_timebase_available": False,
        "stable_recorded_cell_to_MaleCNS_mapping_available": False,
    }
    transferable = all(transfer_gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(source_config_path): _sha256(root / source_config_path),
                str(source_evidence_path): _sha256(root / source_evidence_path),
                str(arenz_path): _sha256(root / arenz_path),
                str(robustness_path): _sha256(root / robustness_path),
                "pyproject.toml": _sha256(root / "pyproject.toml"),
            },
            "parameter_fit": True,
            "fit_scope": "C3_source_STRF_only",
            "T4_response_used": False,
            "direction_label_used": False,
            "runtime_modified": False,
        },
        "unit_contract": config["unit_contract"],
        "model_contract": config["model_contract"],
        "source_units": [
            {key: item[key] for key in ("fly", "axis", "ROI_count")}
            for item in units
        ],
        "model_families": families,
        "band_pass_BIC_below_low_pass_BIC": band_pass_better,
        "C3_analytic_filter_family_precheck_passed": band_pass_passed,
        "transfer_gates": transfer_gates,
        "C3_analytic_filter_transfer_authorized": transferable,
        "advance_to_T4_functional_precheck": transferable,
        "blocking_requirements": [
            name for name, passed in transfer_gates.items() if not passed
        ],
        "stop_reason": (
            None
            if transferable
            else "C3_analytic_filter_not_cross_fly_and_cross_paper_transfer_ready"
        ),
        "boundary": config["boundary"],
    }
