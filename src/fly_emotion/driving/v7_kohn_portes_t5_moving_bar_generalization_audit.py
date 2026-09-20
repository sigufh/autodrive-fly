"""Audit static-flash to moving-bar Tm-to-T5 generalization evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_kohn_portes_tm_to_t5_model_audit import _git_blob_sha1

CONFIG = Path(
    "configs/driving-v7-kohn-portes-t5-moving-bar-generalization-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_kohn_portes_t5_moving_bar_generalization_audit.py"
)


def _flash_predictors(
    flashes: dict[str, dict], sources: list[str], duration: str, state: str, window: int
) -> np.ndarray:
    duration_ms = int(float(duration) * 1000)
    values = []
    for source in sources:
        key = f"FFF_uniform_{duration_ms}msimpulse_10int {state}"
        trace = np.asarray(flashes[source][key]["mean"], dtype=np.float64)[::50][:window]
        values.append(trace / np.max(trace))
    return np.stack(values, axis=1)


def _static_coefficients(
    target: dict,
    flashes: dict[str, dict],
    sources: list[str],
    duration: str,
    width: str,
    state: str,
    alpha: float,
) -> np.ndarray:
    x_data = _flash_predictors(flashes, sources, duration, state, 120)
    coefficients = []
    for trace in target[duration][width].values():
        y_data = np.asarray(trace["mean"], dtype=np.float64)[18000::200][:120].copy()
        y_data[~np.isfinite(y_data)] = 0.0
        model = Lasso(
            alpha=alpha,
            fit_intercept=False,
            precompute=True,
            max_iter=1000,
            positive=True,
            random_state=9999,
            selection="random",
        ).fit(x_data, y_data)
        coefficients.append(model.coef_.astype(np.float64))
    return np.stack(coefficients)


def _moving_prediction(
    source_traces: np.ndarray, coefficients: np.ndarray, duration: str
) -> tuple[np.ndarray, np.ndarray]:
    shift = int(float(duration) * 100)
    output_size = source_traces.shape[0] + 5000 - 1
    pd_components = []
    nd_components = []
    for trace, source_coefficients in zip(source_traces.T, coefficients.T, strict=True):
        preferred = np.zeros(output_size)
        null = np.zeros(output_size)
        for index, coefficient in enumerate(source_coefficients):
            preferred_start = index * shift
            null_start = (len(source_coefficients) - index - 1) * shift
            preferred[preferred_start : preferred_start + len(trace)] += coefficient * trace
            null[null_start : null_start + len(trace)] += coefficient * trace
        pd_components.append(preferred)
        nd_components.append(null)
    return np.sum(pd_components, axis=0), np.sum(nd_components, axis=0)


def _dsi(pd: np.ndarray, nd: np.ndarray) -> float:
    pd_peak = float(np.max(pd))
    nd_peak = float(np.max(nd))
    return (pd_peak - nd_peak) / (pd_peak + nd_peak)


def evaluate_v7_kohn_portes_t5_moving_bar_generalization_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    expected = config["expected"]
    model_evidence_path = Path(config["author_model_evidence"])
    source_config_path = Path(config["source_config"])
    direction_path = Path(config["direction_evidence"])
    model_evidence = json.loads((root / model_evidence_path).read_text(encoding="utf-8"))
    source_config = yaml.safe_load((root / source_config_path).read_text(encoding="utf-8"))
    direction = json.loads((root / direction_path).read_text(encoding="utf-8"))
    code_paths = {
        name: _verify_file(root, spec) for name, spec in config["author_code"].items()
    }
    code = "".join(path.read_text(encoding="utf-8") for path in code_paths.values())
    fragments = (
        "Fit to static flashes, and use corresponding weights to simulated moving bar",
        "since there is no data for 0.02 or for 0.08, use 0.04 and 0.16 coefs respectively",
        "model.fit(x_data, y_data)",
        "scale = model.coef_[0]",
        "pd_score=r2_score",
        "nd_score=r2_score",
    )
    if any(fragment not in code for fragment in fragments):
        raise ValueError("Kohn-Portes moving-bar generalization method changed")
    tree = {
        spec["repository_path"]: spec["git_blob"]
        for spec in config["model_inputs"].values()
    }
    input_paths = {}
    for name, spec in config["model_inputs"].items():
        path = _verify_file(root, spec)
        if _git_blob_sha1(path) != tree[spec["repository_path"]]:
            raise ValueError(f"Kohn-Portes moving-bar Git blob mismatch: {name}")
        input_paths[name] = path
    target = _load_restricted(input_paths["T5_stationary_flash"])
    moving = _load_restricted(input_paths["T5_moving_bar"])
    flashes = {
        source: _load_restricted(_verify_file(root, source_config["source_files"][source]))
        for source in expected["source_pair"]
    }

    conditions = []
    for state in expected["source_states"]:
        observed_dsi = []
        target_dsi = []
        for duration in expected["durations_seconds"]:
            static_duration = expected["static_fit_duration_by_moving_duration"][duration]
            for width in expected["widths_degrees"]:
                coefficients = _static_coefficients(
                    target,
                    flashes,
                    expected["source_pair"],
                    static_duration,
                    width,
                    state,
                    float(expected["lasso_alpha"]),
                )
                source_traces = _flash_predictors(
                    flashes,
                    expected["source_pair"],
                    duration,
                    state,
                    int(expected["source_trace_samples"]),
                )
                predicted_pd, predicted_nd = _moving_prediction(
                    source_traces, coefficients, duration
                )
                target_payload = moving[width][duration]
                target_pd = np.asarray(target_payload["pd_mean"], dtype=np.float64)[::2]
                target_nd = np.asarray(target_payload["nd_mean"], dtype=np.float64)[::2]
                if target_payload["pd_data"].shape[1] != int(expected["target_cell_count"]):
                    raise ValueError("moving-bar target cell count changed")
                window = int(expected["R2_window_by_duration"][duration])
                fit_x = np.concatenate(
                    (predicted_pd[:window], predicted_nd[:window])
                ).reshape(-1, 1)
                fit_y = np.concatenate((target_pd[:window], target_nd[:window]))
                gain_model = Lasso(
                    alpha=float(expected["lasso_alpha"]),
                    fit_intercept=False,
                    precompute=True,
                    max_iter=1000,
                    positive=True,
                    random_state=9999,
                    selection="random",
                ).fit(fit_x, fit_y)
                gain = float(gain_model.coef_[0])
                predicted = _dsi(predicted_pd, predicted_nd)
                measured = _dsi(target_pd[:300], target_nd[:300])
                conditions.append(
                    {
                        "source_state": state,
                        "duration_seconds": duration,
                        "width_degrees": width,
                        "static_coefficient_duration_seconds": static_duration,
                        "gain": gain,
                        "predicted_DSI_before_moving_bar_gain_fit": predicted,
                        "target_DSI": measured,
                        "absolute_DSI_error": abs(predicted - measured),
                        "PD_R2_after_moving_bar_gain_fit": float(
                            r2_score(
                                target_pd[:window], gain * predicted_pd[:window]
                            )
                        ),
                        "ND_R2_after_moving_bar_gain_fit": float(
                            r2_score(
                                target_nd[:window], gain * predicted_nd[:window]
                            )
                        ),
                        "moving_bar_gain_fit_sample_count": 2 * window,
                        "moving_bar_gain_fit_and_score_samples_disjoint": False,
                    }
                )
                observed_dsi.append(predicted)
                target_dsi.append(measured)
        if not np.allclose(
            observed_dsi, expected["predicted_DSI"][state], rtol=0.0, atol=5e-13
        ):
            raise ValueError(f"Kohn-Portes predicted moving-bar DSI changed: {state}")
        if not np.allclose(target_dsi, expected["target_DSI"], rtol=0.0, atol=5e-13):
            raise ValueError("Gruntman moving-bar target DSI changed")

    summaries = {}
    for state in expected["source_states"]:
        selected = [item for item in conditions if item["source_state"] == state]
        predicted = [item["predicted_DSI_before_moving_bar_gain_fit"] for item in selected]
        measured = [item["target_DSI"] for item in selected]
        summary = {
            "condition_count": len(selected),
            "all_predicted_DSI_positive": all(value > 0.0 for value in predicted),
            "mean_absolute_DSI_error": float(
                np.mean(
                    [
                        abs(first - second)
                        for first, second in zip(predicted, measured, strict=True)
                    ]
                )
            ),
            "predicted_target_DSI_correlation": float(np.corrcoef(predicted, measured)[0, 1]),
            "mean_PD_R2_after_moving_bar_gain_fit": float(
                np.mean([item["PD_R2_after_moving_bar_gain_fit"] for item in selected])
            ),
            "mean_ND_R2_after_moving_bar_gain_fit": float(
                np.mean([item["ND_R2_after_moving_bar_gain_fit"] for item in selected])
            ),
        }
        if not np.isclose(
            summary["mean_absolute_DSI_error"],
            expected["mean_absolute_DSI_error"][state],
            rtol=0.0,
            atol=5e-13,
        ) or not np.isclose(
            summary["predicted_target_DSI_correlation"],
            expected["predicted_target_DSI_correlation"][state],
            rtol=0.0,
            atol=5e-13,
        ):
            raise ValueError(f"moving-bar generalization summary changed: {state}")
        summaries[state] = summary

    target_direction = direction["provenance_gates"]
    gates = {
        "static_bar_coefficients_reused_without_refit": True,
        "gain_free_predicted_DSI_available": True,
        "all_gain_free_predicted_DSI_positive": all(
            summary["all_predicted_DSI_positive"] for summary in summaries.values()
        ),
        "moving_bar_gain_fit_and_score_samples_disjoint": False,
        "stable_target_cell_ids_available": False,
        "independent_target_cell_holdout_available": False,
        "absolute_physical_direction_mapping_verified": target_direction[
            "figure4_direction_code_to_absolute_physical_motion_verified"
        ],
        "source_absolute_gain_preserved": False,
    }
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(model_evidence_path): _sha256(root / model_evidence_path),
        str(source_config_path): _sha256(root / source_config_path),
        str(direction_path): _sha256(root / direction_path),
        **{
            spec["path"]: _sha256(path)
            for spec, path in zip(
                config["author_code"].values(), code_paths.values(), strict=True
            )
        },
        **{
            spec["path"]: _sha256(path)
            for spec, path in zip(
                config["model_inputs"].values(), input_paths.values(), strict=True
            )
        },
        **{
            source_config["source_files"][source]["path"]: _sha256(
                root / source_config["source_files"][source]["path"]
            )
            for source in expected["source_pair"]
        },
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "parameter_fit_for_v7": False,
            "runtime_modified": False,
        },
        "condition_order": [
            f"{state}:{duration}:{width}"
            for state in expected["source_states"]
            for duration in expected["durations_seconds"]
            for width in expected["widths_degrees"]
        ],
        "condition_results": conditions,
        "state_summaries": summaries,
        "gates": gates,
        "cross_stimulus_relative_shape_candidate_available": True,
        "independent_moving_bar_validation_available": False,
        "authorize_moving_bar_generalization_transfer_to_v7": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "moving_bar_shape_candidate_exists_but_gain_is_fit_on_test_data_and_identity_is_missing"
        ),
        "cross_check": {
            "author_model_transfer_authorized": model_evidence[
                "authorize_Tm_to_T5_model_transfer_to_v7"
            ]
        },
        "boundary": config["boundary"],
    }
