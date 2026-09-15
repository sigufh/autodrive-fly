from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_ephys_audit import (
    SOURCE_ORDER,
    _download_and_verify,
    _literal_assignments,
    _notebook_source,
    _shift,
)
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-fig5-validation.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_fig5_validation.py")


def direction_selectivity(curves: np.ndarray) -> np.ndarray:
    values = np.asarray(curves, dtype=float)
    if values.ndim != 2 or values.shape[1] != 36:
        raise ValueError("directional tuning requires cells by 36 directions")
    normalized = values - values.min(axis=1, keepdims=True)
    normalized /= np.maximum(normalized.max(axis=1, keepdims=True), 1e-12)
    angles = np.arange(36) * 2 * np.pi / 36
    x = np.sum(normalized * np.cos(angles), axis=1)
    y = np.sum(normalized * np.sin(angles), axis=1)
    return np.hypot(x, y) / np.maximum(np.abs(normalized.sum(axis=1)), 1e-12)


def paper_direction_curve(source_dir: Path, parameters: dict, *, mi9_enabled: bool) -> np.ndarray:
    source_means = {}
    for name in SOURCE_ORDER:
        values = np.load(source_dir / f"fig3_{name}.npy", allow_pickle=False)
        mean = np.nanmean(values, axis=1)
        source_means[name] = (mean - mean.min()) / (mean - mean.min()).max()
    gains = np.asarray([parameters["gains"][name] for name in SOURCE_ORDER])
    thresholds = np.asarray([parameters["thresholds"][name] for name in SOURCE_ORDER])
    reversal = parameters["reversal_potentials_millivolts"]
    reversals = np.asarray(
        (
            reversal["glutamate"],
            reversal["acetylcholine"],
            reversal["acetylcholine"],
            reversal["gaba"],
            reversal["gaba"],
        )
    )
    result = []
    for direction_index in range(36):
        delay = int(160 * np.cos(direction_index * 2 * np.pi / 36))
        inputs = []
        for name in SOURCE_ORDER:
            values = source_means[name][0]
            if name == "Mi9":
                values = _shift(values, -delay)
            elif name in {"Mi4", "C3"}:
                values = _shift(values, delay)
            inputs.append(values)
        conductances = gains[:, None] * np.maximum(np.asarray(inputs) - thresholds[:, None], 0)
        if not mi9_enabled:
            conductances[0] = 0
        leak = parameters["leak_conductance"]
        voltage = (np.sum(reversals[:, None] * conductances, axis=0) + reversal["leak"] * leak) / (
            np.sum(conductances, axis=0) + leak
        )
        result.append(float(voltage.max() - voltage[1000:2000].mean()))
    return np.asarray(result)


def curve_comparison(model: np.ndarray, measured: np.ndarray) -> dict:
    observation = np.asarray(measured, dtype=float).mean(axis=0)
    return {
        "model_curve": model.tolist(),
        "observed_mean_curve": observation.tolist(),
        "rmse_millivolts": float(np.sqrt(np.mean((model - observation) ** 2))),
        "pearson_correlation": float(np.corrcoef(model, observation)[0, 1]),
        "model_direction_selectivity": float(direction_selectivity(model[None])[0]),
        "observed_mean_direction_selectivity": float(direction_selectivity(observation[None])[0]),
        "observed_cell_direction_selectivity": direction_selectivity(measured).tolist(),
        "observed_cell_median_direction_selectivity": float(
            np.median(direction_selectivity(measured))
        ),
        "model_peak_direction_index": int(model.argmax()),
        "observed_peak_direction_index": int(observation.argmax()),
    }


def _without_notebook_magics(source: str) -> str:
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))


def evaluate_v7_fig5_validation(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    dataset_config = yaml.safe_load((root / config["dataset_config"]).read_text(encoding="utf-8"))
    ephys = json.loads((root / config["electrophysiology_evidence"]).read_text())
    if not config["exploratory"] or config["advance_allowed"]:
        raise ValueError("Fig. 5 validation must remain exploratory and non-advancing")
    fig3_names = [f"fig3_{name}.npy" for name in SOURCE_ORDER]
    download = {
        "dataset": dataset_config["dataset"],
        "files": {
            **{name: dataset_config["files"][name] for name in fig3_names},
            **config["files"],
        },
    }
    with tempfile.TemporaryDirectory(prefix="autodrive-v7-fig5-") as temporary:
        source_dir = Path(temporary)
        files = _download_and_verify(download, source_dir)
        notebook = _notebook_source(source_dir / "fig5.ipynb")
        required = (
            "ndirs = 36",
            "delay = int(4.8*fs/30 * np.cos(direction))",
            "T4model_tuning[0,:] = np.linspace(0, 360, T4model.shape[1])",
            "dvm_gfp = np.roll(np.load('fig5_dvm_gfp.npy'), 18)",
            "pvm_gfp = np.load('fig5c_pvm_gfp.npy')",
        )
        if not all(snippet in notebook for snippet in required):
            raise ValueError("Fig. 5 notebook protocol differs from frozen validation")
        assignments = _literal_assignments(
            _without_notebook_magics(notebook),
            {
                "Mi9gain",
                "Tm3gain",
                "Mi1gain",
                "Mi4gain",
                "C3gain",
                "Mi9trld",
                "Tm3trld",
                "Mi1trld",
                "Mi4trld",
                "C3trld",
                "gleak",
                "Eleak",
                "EGlu",
                "EGABA",
                "EnAChR",
            },
        )
        parameters = {
            "gains": {name: assignments[f"{name}gain"] for name in SOURCE_ORDER},
            "thresholds": {name: assignments[f"{name}trld"] for name in SOURCE_ORDER},
            "reversal_potentials_millivolts": {
                "glutamate": assignments["EGlu"],
                "acetylcholine": assignments["EnAChR"],
                "gaba": assignments["EGABA"],
                "leak": assignments["Eleak"],
            },
            "leak_conductance": assignments["gleak"],
        }
        full_model = paper_direction_curve(source_dir, parameters, mi9_enabled=True)
        mi9_block = paper_direction_curve(source_dir, parameters, mi9_enabled=False)
        dvm = {
            name: np.load(source_dir / f"fig5_dvm_{name}.npy", allow_pickle=False)
            for name in ("gfp", "gluclarnai", "nmdar1rnai")
        }
        pvm = {
            name: np.load(source_dir / f"fig5c_pvm_{name}.npy", allow_pickle=False)
            for name in ("gfp", "gluclarnai", "nmdar1rnai")
        }
        exemplar = {
            name: np.load(source_dir / f"fig5b_vm_{name}.npy", allow_pickle=False)
            for name in ("gfp", "gluclarnai", "nmdar1rnai")
        }
        flat_roll = np.roll(dvm["gfp"], 18)
        axis_roll = np.roll(dvm["gfp"], 18, axis=1)
        validation = {
            "wild_type": curve_comparison(full_model, dvm["gfp"]),
            "mi9_block_proxy_vs_gluclar_rnai": curve_comparison(mi9_block, dvm["gluclarnai"]),
        }
        measured = {
            name: {
                "cells": int(values.shape[0]),
                "directions": int(values.shape[1]),
                "delta_vm_direction_selectivity": direction_selectivity(values).tolist(),
                "delta_vm_median_direction_selectivity": float(
                    np.median(direction_selectivity(values))
                ),
                "absolute_peak_vm_direction_selectivity": direction_selectivity(pvm[name]).tolist(),
                "absolute_peak_vm_median_direction_selectivity": float(
                    np.median(direction_selectivity(pvm[name]))
                ),
                "delta_vm_mean_curve": values.mean(axis=0).tolist(),
                "absolute_peak_vm_mean_curve": pvm[name].mean(axis=0).tolist(),
                "exemplar_trace_samples": int(exemplar[name].size),
            }
            for name, values in dvm.items()
        }
    model_drop = (
        validation["wild_type"]["model_direction_selectivity"]
        - validation["mi9_block_proxy_vs_gluclar_rnai"]["model_direction_selectivity"]
    )
    measured_drop = (
        measured["gfp"]["delta_vm_median_direction_selectivity"]
        - measured["gluclarnai"]["delta_vm_median_direction_selectivity"]
    )
    internal_degrees = np.arange(36) * 10.0
    plotted_degrees = np.linspace(0.0, 360.0, 36)
    return {
        "protocol": {
            "name": config["name"],
            "exploratory": True,
            "advance_allowed": False,
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                config["dataset_config"]: _sha256(root / config["dataset_config"]),
                config["electrophysiology_evidence"]: _sha256(
                    root / config["electrophysiology_evidence"]
                ),
            },
            "parameter_fitting": False,
            "post_inspection_protocol": True,
            "raw_files_committed": False,
            "temporary_download_deleted_after_audit": True,
        },
        "dataset": dataset_config["dataset"],
        "verified_files": files,
        "data_roles": config["data_roles"],
        "excluded_files": config["excluded_files"],
        "validation_boundary": config["validation_boundary"],
        "notebook_protocol": {
            "required_snippets_verified": list(required),
            "internal_direction_degrees": internal_degrees.tolist(),
            "plot_direction_degrees": plotted_degrees.tolist(),
            "plot_coordinate_maximum_error_degrees": float(
                np.max(np.abs(plotted_degrees - internal_degrees))
            ),
            "flat_roll_18_preserves_group_mean": bool(
                np.allclose(flat_roll.mean(axis=0), axis_roll.mean(axis=0))
            ),
            "flat_roll_18_preserves_group_sem": bool(
                np.allclose(flat_roll.std(axis=0, ddof=1), axis_roll.std(axis=0, ddof=1))
            ),
            "flat_roll_18_individual_max_error_millivolts": float(
                np.max(np.abs(flat_roll - axis_roll))
            ),
            "canonical_comparison_uses_unrolled_0_to_350_degree_indices": True,
        },
        "parameter_match_with_fig3_audit": parameters
        == ephys["parameter_provenance"]["paper_parameters"],
        "validation": validation,
        "measured_groups": measured,
        "intervention_effect": {
            "model_direction_selectivity_drop_full_to_mi9_block": float(model_drop),
            "measured_median_direction_selectivity_drop_gfp_to_gluclar_rnai": float(measured_drop),
            "same_direction_of_change": bool(model_drop > 0 and measured_drop > 0),
            "mechanistic_equivalence_claimed": False,
        },
        "interpretation": {
            "supports_external_stimulus_condition_generalization": True,
            "supports_independent_cell_holdout": False,
            "supports_independent_input_recordings": False,
            "supports_T5_validation": False,
            "supports_MaleCNS_state_calibration": False,
            "supports_driving_behavior": False,
        },
        "limitations": [
            "Protocol and metrics were frozen only after inspecting Fig. 5 data and code.",
            "Fig. 5 target arrays lack IDs; disjointness from Fig. 3 cells is unverified.",
            "The model reuses Fig. 3 class means, not independent Fig. 5 input recordings.",
            "Mi9 removal is a proxy for GluCl-alpha RNAi, not the same intervention.",
            "The notebook's flattened roll preserves group mean/SEM but not individual rows.",
            "The notebook's plotted 0-360 coordinate differs from its internal 0-350 directions.",
            "The large behavior CSV files were excluded because they do not validate T4 voltage.",
        ],
        "advance_to_time_calibrated_fit": False,
        "advance_to_central_complex": False,
    }
