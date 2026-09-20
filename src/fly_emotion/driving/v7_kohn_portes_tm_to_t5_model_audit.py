"""Audit the author Tm-to-T5 model without authorizing v7 transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Lasso

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)

CONFIG = Path("configs/driving-v7-kohn-portes-tm-to-t5-model-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_kohn_portes_tm_to_t5_model_audit.py"
)


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}".encode() + bytes([0])
    return hashlib.sha1(header + payload).hexdigest()


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return chr(10).join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def _verify_model_input(root: Path, spec: dict, tree: dict[str, str]) -> Path:
    path = _verify_file(root, spec)
    if _git_blob_sha1(path) != spec["git_blob"]:
        raise ValueError(f"Kohn-Portes model input Git blob mismatch: {spec['path']}")
    repository_path = spec["repository_path"]
    if tree.get(repository_path) != spec["git_blob"]:
        raise ValueError(f"Kohn-Portes tree manifest mismatch: {repository_path}")
    return path


def _flash_predictor(flash: dict, duration: str, state: str, window: int) -> np.ndarray:
    key = f"FFF_uniform_{int(float(duration) * 1000)}msimpulse_10int {state}"
    trace = np.asarray(flash[key]["mean"], dtype=np.float64)[::50][:window]
    peak = float(np.max(trace))
    if trace.shape != (window,) or not np.isfinite(trace).all() or peak <= 0.0:
        raise ValueError(f"invalid normalized source flash predictor: {key}")
    return trace / peak


def _fit_condition(
    target: dict,
    flashes: dict[str, dict],
    *,
    duration: str,
    width: str,
    state: str,
    expected: dict,
) -> dict:
    source_pair = expected["source_pair"]
    window = int(expected["fit_window_samples"])
    x_data = np.stack(
        [_flash_predictor(flashes[source], duration, state, window) for source in source_pair],
        axis=1,
    )
    r2_values = []
    correlations = []
    coefficients = []
    target_nan_count = 0
    for location, trace in target[duration][width].items():
        values = np.asarray(trace["mean"], dtype=np.float64)
        y_data = values[
            int(expected["target_start_index"]) :: int(expected["target_downsample_factor"])
        ][:window].copy()
        target_nan_count += int(np.count_nonzero(~np.isfinite(y_data)))
        y_data[~np.isfinite(y_data)] = 0.0
        model = Lasso(
            alpha=float(expected["lasso_alpha"]),
            fit_intercept=bool(expected["fit_intercept"]),
            precompute=True,
            max_iter=1000,
            positive=bool(expected["positive_weights"]),
            random_state=9999,
            selection="random",
        ).fit(x_data, y_data)
        prediction = model.predict(x_data)
        r2_values.append(float(model.score(x_data, y_data)))
        correlations.append(float(np.corrcoef(prediction, y_data)[0, 1]))
        coefficients.append(model.coef_.astype(np.float64))
        if not np.isfinite(prediction).all():
            raise ValueError(
                f"non-finite Tm-to-T5 prediction: {state}/{duration}/{width}/{location}"
            )
    coefficient_array = np.stack(coefficients)
    r2 = np.asarray(r2_values)
    return {
        "location_count": len(r2_values),
        "fit_sample_count_per_location": window,
        "fit_window_nan_count_before_author_zero_fill": target_nan_count,
        "R2": {
            "minimum": float(np.min(r2)),
            "median": float(np.median(r2)),
            "mean": float(np.mean(r2)),
            "maximum": float(np.max(r2)),
            "values": r2_values,
        },
        "Pearson_r_mean": float(np.nanmean(correlations)),
        "coefficient_mean_by_source": {
            source: float(value)
            for source, value in zip(source_pair, np.mean(coefficient_array, axis=0), strict=True)
        },
        "zero_coefficient_count_by_source": {
            source: int(value)
            for source, value in zip(
                source_pair,
                np.count_nonzero(coefficient_array == 0.0, axis=0),
                strict=True,
            )
        },
        "fit_and_score_sample_identity": "same_120_time_samples_per_location",
    }


def _synapse_counts(path: Path) -> list[dict[str, float]]:
    frame = pd.read_csv(
        path,
        header=None,
        skiprows=np.arange(0, 58, 3),
        nrows=40,
        usecols=np.arange(0, 30),
    )
    tags = ("Tm1", "Tm2", "Tm4", "Tm9", "CT1")
    rows = []
    for index in range(0, frame.shape[0], 2):
        names = frame.loc[index, :]
        counts = frame.loc[index + 1, :]
        rows.append(
            {
                tag: float(pd.to_numeric(counts[names.str.contains(tag, regex=False)]).sum())
                for tag in tags
            }
        )
    return rows


def evaluate_v7_kohn_portes_tm_to_t5_model_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    expected = config["expected"]
    source_config_path = Path(config["source_config"])
    target_evidence_path = Path(config["t5_target_evidence"])
    kernel_evidence_path = Path(config["source_kernel_evidence"])
    state_unit_path = Path(config["state_unit_evidence"])
    source_config = yaml.safe_load((root / source_config_path).read_text(encoding="utf-8"))
    target_evidence = json.loads((root / target_evidence_path).read_text(encoding="utf-8"))
    kernel_evidence = json.loads((root / kernel_evidence_path).read_text(encoding="utf-8"))
    state_unit = json.loads((root / state_unit_path).read_text(encoding="utf-8"))

    tree_path = _verify_file(root, config["repository"]["tree_manifest"])
    tree = {
        item["path"]: item["id"]
        for item in (json.loads(line) for line in tree_path.read_text().splitlines())
        if item["type"] == "blob"
    }
    code_paths = {
        name: _verify_file(root, spec) for name, spec in config["author_code"].items()
    }
    input_paths = {
        name: _verify_model_input(root, spec, tree)
        for name, spec in config["model_inputs"].items()
    }
    figure5 = _notebook_source(code_paths["figure5_notebook"])
    regression = code_paths["linear_regression"].read_text(encoding="utf-8")
    figure6_model1 = _notebook_source(code_paths["figure6_model1_notebook"])
    figure6_model2 = _notebook_source(code_paths["figure6_model2_notebook"])
    figure5_fragments = (
        "positive weights, no offset",
        "model.fit(x_data, y_data)",
        "model.score(x_data,y_data)",
        "concat_list.append(lr[key]/np.max(lr[key]))",
        "trace_ds = trace_cut[::200][:window]",
        "a single",
        "gain factor",
        "obtained by a separate linear regression",
    )
    if any(fragment not in figure5 + regression for fragment in figure5_fragments):
        raise ValueError("Kohn-Portes Figure 5 model method changed")
    figure6_fragments = (
        "from axolotl.tmodel import models, stimuli",
        "apply_weight_scaling(simulations",
        "del df_count_ratio['CT1']",
        "Tm2_nd_norm = Tm1_nd/np.max(Tm2_pd[:,window])",
        "weights set before the simulation are not relevant",
    )
    if any(fragment not in figure6_model1 + figure6_model2 for fragment in figure6_fragments):
        raise ValueError("Kohn-Portes Figure 6 model method changed")

    target = _load_restricted(input_paths["T5_stationary_flash"])
    moving = _load_restricted(input_paths["T5_moving_bar"])
    if list(target) != expected["durations_seconds"]:
        raise ValueError("Gruntman stationary-flash duration inventory changed")
    if any(list(target[duration]) != expected["widths_degrees"] for duration in target):
        raise ValueError("Gruntman stationary-flash width inventory changed")
    if any(
        len(target[duration][width]) != int(expected["target_location_count_per_condition"])
        for duration in target
        for width in target[duration]
    ):
        raise ValueError("Gruntman stationary-flash location inventory changed")
    full_target_nan_count = sum(
        np.count_nonzero(~np.isfinite(np.asarray(trace["mean"])))
        for duration in target.values()
        for width in duration.values()
        for trace in width.values()
    )
    moving_shapes = {
        name: [
            list(shape)
            for shape in sorted(
                {
                    tuple(np.asarray(payload[name]).shape)
                    for width in moving.values()
                    for payload in width.values()
                }
            )
        ]
        for name in ("pd_data", "nd_data", "pd_mean", "nd_mean", "pd_std", "nd_std")
    }

    flashes = {
        source: _load_restricted(_verify_file(root, source_config["source_files"][source]))
        for source in expected["source_pair"]
    }
    fits = {}
    all_r2 = {state: [] for state in expected["source_states"]}
    all_zero = {state: 0 for state in expected["source_states"]}
    for state in expected["source_states"]:
        fits[state] = {}
        for duration in expected["durations_seconds"]:
            for width in expected["widths_degrees"]:
                key = f"{duration}:{width}"
                result = _fit_condition(
                    target,
                    flashes,
                    duration=duration,
                    width=width,
                    state=state,
                    expected=expected,
                )
                fits[state][key] = result
                all_r2[state].extend(result["R2"]["values"])
                all_zero[state] += sum(result["zero_coefficient_count_by_source"].values())
    representative = fits["saline"]["0.16:9"]["R2"]["values"]
    if not np.allclose(
        representative,
        expected["representative_saline_160ms_9deg_r2"],
        rtol=0.0,
        atol=5e-13,
    ):
        raise ValueError("Kohn-Portes representative Figure 5 regression changed")
    aggregate = {}
    for state, values in all_r2.items():
        observed = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "zero_coefficient_count": all_zero[state],
        }
        declared = expected["aggregate_training_r2"][state]
        for key, value in observed.items():
            if key == "count" or key == "zero_coefficient_count":
                if value != int(declared[key]):
                    raise ValueError(f"Kohn-Portes aggregate regression changed: {state}/{key}")
            elif not np.isclose(value, declared[key], rtol=0.0, atol=5e-13):
                raise ValueError(f"Kohn-Portes aggregate regression changed: {state}/{key}")
        aggregate[state] = observed

    counts = _synapse_counts(input_paths["Shinomiya_T5_input_Lo1"])
    count_sums = {source: int(sum(row[source] for row in counts)) for source in counts[0]}
    if len(counts) != int(expected["Shinomiya_T5_count"]):
        raise ValueError("Shinomiya T5 count changed")
    if count_sums != expected["Shinomiya_source_count_sums"]:
        raise ValueError("Shinomiya source count sums changed")
    four_sources = expected["source_pair"][:1] + ["Tm2", "Tm4"] + expected["source_pair"][1:]
    ratios = [
        {source: row[source] / sum(row[name] for name in four_sources) for source in four_sources}
        for row in counts
    ]
    mean_ratios = {
        source: float(np.mean([row[source] for row in ratios])) for source in four_sources
    }
    if not all(
        np.isclose(mean_ratios[source], value, rtol=0.0, atol=5e-13)
        for source, value in expected["Shinomiya_four_source_mean_ratios"].items()
    ):
        raise ValueError("Shinomiya four-source mean ratios changed")

    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(source_config_path): _sha256(root / source_config_path),
        str(target_evidence_path): _sha256(root / target_evidence_path),
        str(kernel_evidence_path): _sha256(root / kernel_evidence_path),
        str(state_unit_path): _sha256(root / state_unit_path),
        config["repository"]["tree_manifest"]["path"]: _sha256(tree_path),
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
    target_ready = target_evidence["direction_and_identity_readiness"]
    gates = {
        "fixed_commit_model_inputs_hash_and_git_blob_verified": True,
        "Figure5_representative_regression_reproduced": True,
        "Figure5_all_six_conditions_recomputed": True,
        "Figure5_fit_and_score_samples_disjoint": False,
        "Figure5_independent_cell_validation_available": target_ready[
            "independent_cell_holdout_available"
        ],
        "Figure5_absolute_input_gain_preserved": False,
        "Figure5_target_to_MaleCNS_identity_available": target_ready[
            "stable_biological_cell_ids_available"
        ],
        "Figure6_model_package_source_available": False,
        "Figure6_CT1_included_in_weight_normalization": False,
        "Figure6_Tm2_ND_source_reference_correct": False,
        "source_to_v7_state_mapping_available": state_unit[
            "millivolts_or_filter_output_to_v7_state_mapping_available"
        ],
        "source_kernel_gain_transferable": kernel_evidence["gates"][
            "raw_temporal_filter_absolute_gain_transferable"
        ],
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "repository_commit": config["repository"]["commit"],
            "dependencies_sha256": dependencies,
            "parameter_fit_for_v7": False,
            "runtime_modified": False,
        },
        "recovered_model_inputs": {
            "stationary_flash": {
                "condition_count": 6,
                "location_trace_count": 84,
                "samples_per_trace": 45000,
                "full_trace_nan_count": int(full_target_nan_count),
                "response_quantity": target_evidence["data_semantics"]["response_quantity"],
                "response_unit": target_evidence["data_semantics"]["response_unit"],
            },
            "moving_bar": {
                "condition_count": 12,
                "recorded_cell_count": 17,
                "array_shapes": moving_shapes,
                "used_as_independent_final_test": False,
            },
            "Shinomiya_fib19_T5_input": {
                "T5_count": len(counts),
                "source_count_sums": count_sums,
                "four_source_mean_ratios": mean_ratios,
                "CT1_removed_before_four_source_ratio": True,
                "MaleCNS_mapping": False,
            },
        },
        "Figure5_Tm1_Tm9_static_flash_regression": {
            "source_pair": expected["source_pair"],
            "source_input_normalization": "each_source_trace_divided_by_own_positive_maximum",
            "target_nonfinite_policy": "replace_with_zero_before_fit_and_score",
            "fit": {
                "estimator": "sklearn_Lasso",
                "alpha": expected["lasso_alpha"],
                "fit_intercept": expected["fit_intercept"],
                "positive_weights": expected["positive_weights"],
            },
            "condition_results": fits,
            "aggregate_training_R2": aggregate,
            "same_samples_used_for_fit_and_score": True,
        },
        "Figure6_model_boundary": {
            "external_axolotl_tmodel_source_in_fixed_repository_tree": False,
            "preset_weights_declared_irrelevant_after_numeric_scaling": True,
            "per_source_simulation_output_peak_normalized": True,
            "connectome_weight_source": "Shinomiya_2019_fib19_twenty_T5_cells",
            "CT1_removed_before_four_source_weight_ratios": True,
            "Tm2_ND_assignment_observed": "Tm1_nd / max(Tm2_pd[:, window])",
            "Tm2_ND_assignment_matches_Tm2_source": False,
        },
        "gates": gates,
        "author_Tm_to_T5_model_reproducibility_evidence_available": True,
        "author_Tm_to_T5_model_independently_validated": False,
        "authorize_Tm_to_T5_model_transfer_to_v7": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "author_model_is_training_fit_with_normalized_inputs_and_unavailable_model_package"
        ),
        "boundary": config["boundary"],
    }
