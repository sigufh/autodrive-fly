"""Audit FlyVis effective C3 flash dynamics against preregistered evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-flyvis-c3-effective-dynamics-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_flyvis_c3_effective_dynamics_audit.py"
)


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        return None
    if np.ptp(first) <= np.finfo(np.float64).tiny:
        return None
    if np.ptp(second) <= np.finfo(np.float64).tiny:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _unit(values: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        return None
    return values / norm


def _summary(values: list[float | None]) -> dict:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    result = {
        "fixed_denominator": len(values),
        "finite_count": len(finite),
        "undefined_count": len(values) - len(finite),
    }
    if len(finite):
        result.update(
            minimum=float(np.min(finite)),
            q25=float(np.quantile(finite, 0.25)),
            median=float(np.median(finite)),
            q75=float(np.quantile(finite, 0.75)),
            maximum=float(np.max(finite)),
        )
    return result


def _response(model: dict, protocol: str, dt: float) -> np.ndarray:
    matches = [
        item
        for item in model["responses"]
        if item["name"] == protocol and float(item["dt_seconds"]) == dt
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {protocol} response at dt={dt}")
    values = np.asarray(matches[0]["C3_activity"], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite C3 response in raw response payload")
    expected = sum(
        round(float(duration) / dt) for duration in matches[0]["durations_seconds"]
    )
    if len(values) != expected:
        raise ValueError("C3 response length differs from protocol")
    return values


def _center_and_sample(
    values: np.ndarray, dt: float, baseline_seconds: float, target_time: np.ndarray
) -> np.ndarray:
    time = np.arange(len(values), dtype=np.float64) * dt - baseline_seconds
    baseline = np.mean(values[(time >= -1.0) & (time < 0.0)])
    return np.interp(target_time, time, values - baseline)


def _forward_calcium(values: np.ndarray, dt: float, tau: float) -> np.ndarray:
    alpha = float(np.exp(-dt / tau))
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        filtered[index] = alpha * filtered[index - 1] + (1.0 - alpha) * values[index]
    return filtered


def _ensemble_features(trace: np.ndarray, time: np.ndarray, prereg: dict) -> dict:
    windows = prereg["comparison_contract"]
    masks = {
        "peak": (time >= windows["onset_peak_window_seconds"][0])
        & (time < windows["onset_peak_window_seconds"][1]),
        "plateau": (time >= windows["late_plateau_window_seconds"][0])
        & (time < windows["late_plateau_window_seconds"][1]),
        "recovery": (time >= windows["recovery_window_seconds"][0])
        & (time < windows["recovery_window_seconds"][1]),
    }
    peak_values = trace[masks["peak"]]
    peak = float(np.max(peak_values))
    if not np.isfinite(peak) or peak <= 0:
        return {
            "defined_positive_ON_peak": False,
            "onset_peak_latency_seconds": None,
            "late_plateau_to_peak_ratio": None,
            "absolute_recovery_to_peak_ratio": None,
        }
    return {
        "defined_positive_ON_peak": True,
        "onset_peak_latency_seconds": float(time[masks["peak"]][np.argmax(peak_values)]),
        "late_plateau_to_peak_ratio": float(np.mean(trace[masks["plateau"]]) / peak),
        "absolute_recovery_to_peak_ratio": float(
            abs(np.mean(trace[masks["recovery"]])) / peak
        ),
    }


def _in_empirical_interval(value: float | None, empirical: dict) -> bool:
    return bool(
        value is not None
        and np.isfinite(value)
        and float(empirical["q025"]) <= value <= float(empirical["q975"])
    )


def evaluate_v7_flyvis_c3_effective_dynamics_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    prereg_path = Path(config["preregistration_evidence"])
    prereg = json.loads((root / prereg_path).read_text(encoding="utf-8"))
    static_path = Path(config["static_parameter_evidence"])
    static = json.loads((root / static_path).read_text(encoding="utf-8"))
    raw_spec = config["raw_response_payload"]
    raw_path = root / raw_spec["path"]
    if raw_path.stat().st_size != int(raw_spec["bytes"]):
        raise ValueError("FlyVis C3 response payload size mismatch")
    if _sha256(raw_path) != raw_spec["sha256"]:
        raise ValueError("FlyVis C3 response payload SHA-256 mismatch")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    generator_path = root / config["generator"]["path"]
    if _sha256(generator_path) != config["generator"]["sha256"]:
        raise ValueError("FlyVis C3 response generator mismatch")
    if raw["generator_sha256"] != config["generator"]["sha256"]:
        raise ValueError("raw response generator hash mismatch")
    for spec in config["flyvis_code"]:
        if _sha256(root / spec["path"]) != spec["sha256"]:
            raise ValueError(f"FlyVis code mismatch: {spec['path']}")
    if raw["flyvis_repository_commit"] != static["protocol"]["repository_commit"]:
        raise ValueError("FlyVis response and static audit revisions differ")
    contract = config["analysis_contract"]
    models = raw["models"]
    model_count = int(contract["model_count"])
    if len(models) != model_count or int(contract["fixed_denominator"]) != model_count:
        raise ValueError("FlyVis model count differs from fixed denominator")
    if len({model["model_name"] for model in models}) != model_count:
        raise ValueError("duplicate FlyVis model names")
    expected_input = prereg["comparison_contract"]["permitted_external_input_cell_types"]
    input_contract = raw["input_contract"]
    if input_contract["permitted_input_cell_types"] != expected_input:
        raise ValueError("FlyVis external input differs from preregistration")
    if input_contract["R7_R8_external_activity"] != 0.0:
        raise ValueError("FlyVis R7/R8 external input was not zero")
    if input_contract["target_activity_injected"]:
        raise ValueError("FlyVis C3 target activity was injected")

    comparison = prereg["comparison_contract"]
    target_time_all = np.asarray(prereg["empirical_C3_flash"]["time_seconds"])
    waveform_bounds = comparison["waveform_window_seconds"]
    waveform_mask = (target_time_all >= waveform_bounds[0]) & (
        target_time_all < waveform_bounds[1]
    )
    target_time = target_time_all[waveform_mask]
    empirical_trace = np.asarray(
        prereg["empirical_C3_flash"]["mean_baseline_centered_trace"]
    )[waveform_mask]
    empirical_unit = _unit(empirical_trace)
    if empirical_unit is None:
        raise ValueError("empirical C3 comparator is silent")
    dts = [float(value) for value in comparison["model_sample_intervals_seconds"]]
    protocol = contract["comparison_protocol"]
    sampled: dict[float, np.ndarray] = {}
    unit_shapes: dict[float, list[np.ndarray | None]] = {}
    for dt in dts:
        sampled[dt] = np.stack(
            [
                _center_and_sample(_response(model, protocol, dt), dt, 5.0, target_time)
                for model in models
            ]
        )
        unit_shapes[dt] = [_unit(trace) for trace in sampled[dt]]

    cross_dt = [
        _correlation(sampled[dts[0]][index], sampled[dts[1]][index])
        for index in range(model_count)
    ]
    threshold = float(config["gates"]["minimum_correlation"])
    cross_dt_pass = all(value is not None and value >= threshold for value in cross_dt)
    stability = {}
    for dt in dts:
        shapes = unit_shapes[dt]
        lomo = []
        raw_lomo = []
        for index in range(model_count):
            held_out = shapes[index]
            others = [
                shape
                for j, shape in enumerate(shapes)
                if j != index and shape is not None
            ]
            equal_shape_mean = (
                _unit(np.mean(others, axis=0))
                if len(others) == model_count - 1
                else None
            )
            lomo.append(
                None
                if held_out is None or equal_shape_mean is None
                else _correlation(held_out, equal_shape_mean)
            )
            raw_lomo.append(
                _correlation(
                    sampled[dt][index],
                    np.mean(np.delete(sampled[dt], index, axis=0), axis=0),
                )
            )
        stability[f"{int(round(dt * 1000))}ms"] = {
            "unit_shape_leave_one_model_out_correlations": lomo,
            "unit_shape_leave_one_model_out_summary": _summary(lomo),
            "raw_activity_weighted_leave_one_model_out_summary": _summary(raw_lomo),
            "every_model_passed": all(
                value is not None and value >= threshold for value in lomo
            ),
        }

    empirical = prereg["empirical_C3_flash"]
    feature_targets = {
        "onset_peak_latency_seconds": empirical["onset_peak_latency_seconds"],
        "late_plateau_to_peak_ratio": empirical["late_plateau_to_peak_ratio"],
        "absolute_recovery_to_peak_ratio": empirical["absolute_recovery_to_peak_ratio"],
    }
    external = {}
    all_external_correlations = []
    all_external_features = []
    calcium_assumptions = comparison["calcium_forward_time_constants_seconds"]
    for dt in dts:
        by_tau = {}
        for tau_value in calcium_assumptions:
            tau = float(tau_value)
            calcium_shapes = []
            individual_correlations = []
            positive_peak_count = 0
            for model in models:
                raw_trace = _response(model, protocol, dt)
                filtered = _forward_calcium(raw_trace, dt, tau)
                sampled_trace = _center_and_sample(filtered, dt, 5.0, target_time)
                shape = _unit(sampled_trace)
                calcium_shapes.append(shape)
                individual_correlations.append(
                    None if shape is None else _correlation(shape, empirical_unit)
                )
                peak_mask = (target_time >= comparison["onset_peak_window_seconds"][0]) & (
                    target_time < comparison["onset_peak_window_seconds"][1]
                )
                if shape is not None and float(np.max(shape[peak_mask])) > 0:
                    positive_peak_count += 1
            if any(shape is None for shape in calcium_shapes):
                ensemble_shape = None
            else:
                ensemble_shape = _unit(np.mean(calcium_shapes, axis=0))
            ensemble_correlation = (
                None
                if ensemble_shape is None
                else _correlation(ensemble_shape, empirical_unit)
            )
            features = (
                _ensemble_features(ensemble_shape, target_time, prereg)
                if ensemble_shape is not None
                else _ensemble_features(np.zeros_like(target_time), target_time, prereg)
            )
            feature_gates = {
                name: _in_empirical_interval(features[name], feature_targets[name])
                for name in feature_targets
            }
            correlation_passed = bool(
                ensemble_correlation is not None and ensemble_correlation >= threshold
            )
            features_passed = all(feature_gates.values())
            all_external_correlations.append(correlation_passed)
            all_external_features.append(features_passed)
            by_tau[f"{int(round(tau * 1000))}ms"] = {
                "individual_model_correlation_summary": _summary(individual_correlations),
                "individual_models_at_or_above_threshold": sum(
                    value is not None and value >= threshold
                    for value in individual_correlations
                ),
                "positive_ON_peak_model_count": positive_peak_count,
                "ensemble_equal_shape_correlation": ensemble_correlation,
                "ensemble_features": features,
                "ensemble_feature_gates": feature_gates,
                "ensemble_correlation_passed": correlation_passed,
                "ensemble_features_passed": features_passed,
            }
        external[f"{int(round(dt * 1000))}ms"] = by_tau

    timing_protocol = contract["timing_sensitivity_protocol"]
    timing_cross_dt = []
    for model in models:
        first = _response(model, timing_protocol, dts[0])
        second = _response(model, timing_protocol, dts[1])
        first_time = np.arange(len(first)) * dts[0] - 1.0
        second_time = np.arange(len(second)) * dts[1] - 1.0
        timing_cross_dt.append(_correlation(np.interp(second_time, first_time, first), second))

    gates = {
        "all_raw_responses_finite": True,
        "all_50_models_in_fixed_denominator": len(models) == model_count,
        "external_input_only_R1_R6": input_contract["permitted_input_cell_types"] == expected_input,
        "R7_R8_external_activity_zero": input_contract["R7_R8_external_activity"] == 0.0,
        "C3_target_activity_not_injected": not input_contract["target_activity_injected"],
        "every_model_cross_dt_correlation": cross_dt_pass,
        "every_model_leave_one_model_out_at_each_dt": all(
            item["every_model_passed"] for item in stability.values()
        ),
        "every_calcium_assumption_ensemble_external_correlation": all(
            all_external_correlations
        ),
        "every_calcium_assumption_ensemble_features_in_empirical_intervals": all(
            all_external_features
        ),
        "MaleCNS_topology_and_body_mapping_available": False,
    }
    authorized = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(prereg_path): _sha256(root / prereg_path),
                str(static_path): _sha256(root / static_path),
                str(config["generator"]["path"]): _sha256(generator_path),
            },
            "raw_response_payload": {
                "path": raw_spec["path"],
                "bytes": raw_path.stat().st_size,
                "sha256": _sha256(raw_path),
            },
            "flyvis_repository_commit": raw["flyvis_repository_commit"],
            "environment": raw["environment"],
            "model_count": len(models),
            "fixed_denominator": model_count,
            "parameter_fit": False,
            "direction_label_used": False,
            "runtime_modified": False,
        },
        "input_contract": input_contract,
        "analysis_contract": contract,
        "cross_dt_model_correlations": cross_dt,
        "cross_dt_summary": _summary(cross_dt),
        "ensemble_stability": stability,
        "external_C3_flash_consistency": external,
        "flyvis_official_timing_cross_dt_summary": _summary(timing_cross_dt),
        "transfer_gates": gates,
        "FlyVis_C3_effective_dynamics_transfer_authorized": authorized,
        "advance_to_T4_functional_precheck": authorized,
        "stop_reason": (
            None
            if authorized
            else "FlyVis_C3_effective_dynamics_failed_stability_and_external_validation"
        ),
        "boundary": config["boundary"],
    }
