from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-electrophysiology-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_ephys_audit.py")
CURRENT_CONFIG = Path("configs/driving-v7-t4-conductance.yaml")
SOURCE_ORDER = ("Mi9", "Tm3", "Mi1", "Mi4", "C3")


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_and_verify(config: dict, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, expected in config["files"].items():
        target = destination / name
        request = urllib.request.Request(
            f"{config['dataset']['api_base']}/{expected['id']}",
            headers={"User-Agent": "AutoDrive-Fly-v7-ephys-audit/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as sink:
            while block := response.read(1024 * 1024):
                sink.write(block)
        actual = {
            "id": int(expected["id"]),
            "bytes": target.stat().st_size,
            "md5": _digest(target, "md5"),
            "sha256": _digest(target, "sha256"),
        }
        for key in ("size", "md5", "sha256"):
            actual_key = "bytes" if key == "size" else key
            if actual[actual_key] != expected[key]:
                raise ValueError(f"electrophysiology source mismatch: {name} {key}")
        report[name] = actual
    return report


def _npy_header(path: Path) -> dict:
    with path.open("rb") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
    return {
        "format_version": list(version),
        "shape": list(shape),
        "fortran_order": bool(fortran),
        "dtype": str(dtype),
        "contains_python_objects": bool(dtype.hasobject),
    }


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _literal_assignments(source: str, names: set[str]) -> dict:
    result = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                with contextlib.suppress(ValueError, TypeError):
                    result[target.id] = ast.literal_eval(node.value)
    return result


def _shift(values: np.ndarray, samples: int) -> np.ndarray:
    shifted = np.roll(values, samples)
    if samples > 0:
        shifted[:samples] = values[0]
    elif samples < 0:
        shifted[len(values) + samples :] = values[-1]
    return shifted


def replay_paper_model(source_dir: Path, protocol: dict, parameters: dict) -> dict:
    inputs = {
        name: np.load(source_dir / f"fig3_{name}.npy", allow_pickle=False) for name in SOURCE_ORDER
    }
    target_cells = np.load(source_dir / "fig3_T4v.npy", allow_pickle=False)
    led = np.load(source_dir / "fig3_led.npy", allow_pickle=False)
    normalized = {}
    for name, values in inputs.items():
        mean = np.nanmean(values, axis=1)
        normalized[name] = (mean - mean.min()) / (mean - mean.min()).max()
    shift = int(
        protocol["inter_ommatidial_angle_degrees"]
        * protocol["repository_rate_hz"]
        / protocol["edge_speed_degrees_per_second"]
    )
    shifted = np.empty((len(SOURCE_ORDER), 2, 2, 8000), dtype=float)
    for source_index, name in enumerate(SOURCE_ORDER):
        for stimulus in range(2):
            base = normalized[name][stimulus]
            if name == "Mi9":
                shifted[source_index, stimulus, 0] = _shift(base, -shift)
                shifted[source_index, stimulus, 1] = _shift(base, shift)
            elif name in {"Mi4", "C3"}:
                shifted[source_index, stimulus, 0] = _shift(base, shift)
                shifted[source_index, stimulus, 1] = _shift(base, -shift)
            else:
                shifted[source_index, stimulus, 0] = base
                shifted[source_index, stimulus, 1] = base
    gains = np.asarray([parameters["gains"][name] for name in SOURCE_ORDER])[:, None, None, None]
    thresholds = np.asarray([parameters["thresholds"][name] for name in SOURCE_ORDER])[
        :, None, None, None
    ]
    reversal = parameters["reversal_potentials_millivolts"]
    reversals = np.asarray(
        (
            reversal["glutamate"],
            reversal["acetylcholine"],
            reversal["acetylcholine"],
            reversal["gaba"],
            reversal["gaba"],
        )
    )[:, None, None, None]
    conductances = gains * np.maximum(shifted - thresholds, 0.0)
    leak = parameters["leak_conductance"]
    predicted = (np.sum(reversals * conductances, axis=0) + reversal["leak"] * leak) / (
        np.sum(conductances, axis=0) + leak
    )
    observed = target_cells.mean(axis=0)
    margin = int(protocol["notebook_display_margin_milliseconds"])
    window = slice(margin, protocol["trace_milliseconds"] - margin)
    conditions = {}
    for stimulus, stimulus_name in enumerate(protocol["stimuli"]):
        for direction, direction_name in enumerate(protocol["directions"]):
            prediction = predicted[stimulus, direction, window]
            observation = observed[stimulus, direction, window]
            error = prediction - observation
            conditions[f"{stimulus_name}_{direction_name}"] = {
                "rmse_millivolts": float(np.sqrt(np.mean(error**2))),
                "mae_millivolts": float(np.mean(np.abs(error))),
                "pearson_correlation": float(np.corrcoef(prediction, observation)[0, 1]),
                "observed_peak_millivolts": float(observation.max()),
                "predicted_peak_millivolts": float(prediction.max()),
                "observed_trough_millivolts": float(observation.min()),
                "predicted_trough_millivolts": float(prediction.min()),
            }
    direction = {}
    for stimulus, name in enumerate(protocol["stimuli"]):
        direction[name] = {
            "observed_peak_pd_minus_nd_millivolts": float(
                observed[stimulus, 0, window].max() - observed[stimulus, 1, window].max()
            ),
            "predicted_peak_pd_minus_nd_millivolts": float(
                predicted[stimulus, 0, window].max() - predicted[stimulus, 1, window].max()
            ),
        }
    pooled_prediction = predicted[:, :, window].ravel()
    pooled_observation = observed[:, :, window].ravel()
    led_crossings = {}
    for stimulus, stimulus_name in enumerate(protocol["stimuli"]):
        for direction_index, direction_name in enumerate(protocol["directions"]):
            values = led[stimulus, direction_index]
            crossing = np.flatnonzero(values >= 0.5 if stimulus_name == "on" else values < 0.5)
            led_crossings[f"{stimulus_name}_{direction_name}"] = int(crossing[0])
    return {
        "input_cells": {name: int(values.shape[1]) for name, values in inputs.items()},
        "T4_cells": int(target_cells.shape[0]),
        "array_axes": {
            "inputs": ["stimulus_on_off", "cell", "time_milliseconds"],
            "T4_voltage": ["cell", "stimulus_on_off", "direction_pd_nd", "time_ms"],
        },
        "normalization": "mean over cells, then one min-max range across ON/OFF and time",
        "direction_synthesis": {
            "shift_samples": shift,
            "pd": {"Mi9": -shift, "Mi4": shift, "C3": shift},
            "nd": {"Mi9": shift, "Mi4": -shift, "C3": -shift},
            "Mi1_Tm3_shift_samples": 0,
        },
        "display_window_milliseconds": [margin, protocol["trace_milliseconds"] - margin],
        "led_half_intensity_crossing_milliseconds": led_crossings,
        "conditions": conditions,
        "pooled": {
            "rmse_millivolts": float(
                np.sqrt(np.mean((pooled_prediction - pooled_observation) ** 2))
            ),
            "pearson_correlation": float(np.corrcoef(pooled_prediction, pooled_observation)[0, 1]),
        },
        "direction_peak_difference": direction,
        "interpretation": "retrospective replay on the same averaged traces used for fitting",
    }


def evaluate_v7_electrophysiology_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if not config["exploratory"] or config["advance_allowed"]:
        raise ValueError("electrophysiology audit must remain exploratory and non-advancing")
    with tempfile.TemporaryDirectory(prefix="autodrive-v7-ephys-") as temporary:
        source_dir = Path(temporary)
        files = _download_and_verify(config, source_dir)
        headers = {
            name: _npy_header(source_dir / name)
            for name in config["files"]
            if name.endswith(".npy")
        }
        fig1_source = _notebook_source(source_dir / "fig1.ipynb")
        fig3_source = _notebook_source(source_dir / "fig3.ipynb")
        required_fig1 = (
            "pw = 180/64",
            "tmin = -6",
            "RF_T4_t = time_avg(RF_T4, -16, -10)",
            "RF_T4_p = time_avg(RF_T4, -6, 0)",
        )
        required_fig3 = (
            "fs = 1000",
            "stims = ['on', 'off']",
            "dirs = ['pd', 'nd']",
            "shift = int(4.8*fs/30)",
            "T4v = T4va.mean(0)",
            "m = 2500",
            "T4model[:] = T4model[:] / gtotal[:]",
        )
        if not all(snippet in fig1_source for snippet in required_fig1):
            raise ValueError("Fig. 1 notebook protocol differs from frozen audit")
        if not all(snippet in fig3_source for snippet in required_fig3):
            raise ValueError("Fig. 3 notebook protocol differs from frozen audit")
        assignments = _literal_assignments(
            fig3_source,
            {
                "fs",
                "stims",
                "dirs",
                "tf",
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
                "m",
            },
        )
        paper_parameters = {
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
        replay = replay_paper_model(source_dir, config["paper_protocol"], paper_parameters)
    current = yaml.safe_load((root / CURRENT_CONFIG).read_text(encoding="utf-8"))
    model = current["published_single_compartment_model"]
    paper_parameters = {
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
    parameter_match = (
        all(
            model["source_parameters"][name][key] == paper_parameters[f"{key}s"][name]
            for name in SOURCE_ORDER
            for key in ("gain", "threshold")
        )
        and all(
            model["reversal_potentials_millivolts"][name] == value
            for name, value in paper_parameters["reversal_potentials_millivolts"].items()
        )
        and model["leak_conductance"] == paper_parameters["leak_conductance"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "exploratory": True,
            "advance_allowed": False,
            "config_sha256": _sha256(root / CONFIG),
            "implementation_sha256": _sha256(root / IMPLEMENTATION),
            "current_conductance_config_sha256": _sha256(root / CURRENT_CONFIG),
            "raw_files_committed": False,
            "temporary_download_deleted_after_audit": True,
            "parameter_fitting_performed": False,
        },
        "dataset": config["dataset"],
        "verified_files": files,
        "array_headers": headers,
        "safe_loading": {
            "fig1d_object_array_loaded": False,
            "reason": "object dtype requires pickle; header and notebook semantics only",
            "replay_arrays_loaded_with_allow_pickle_false": True,
            "T4_resistance_array_used_in_replay": False,
        },
        "paper_protocol": {
            **config["paper_protocol"],
            "notebook_assignments": assignments,
            "fig1_required_snippets_verified": list(required_fig1),
            "fig3_required_snippets_verified": list(required_fig3),
            "actual_source_array_order": list(SOURCE_ORDER),
            "onoff_comment_order": ["Mi9", "Mi1", "Tm3", "Mi4", "C3"],
            "onoff_comment_disagrees_with_array_order": True,
            "onoff_display_run_all_enabled": True,
            "fig1e_T4_cells": headers["fig1e_receptive_fields.npy"]["shape"][-1],
            "fig1d_cell_counts_not_loaded": True,
        },
        "paper_model_replay": replay,
        "parameter_provenance": {
            "paper_parameters": paper_parameters,
            "current_config_parameter_values_match": bool(parameter_match),
            "paper_fit_method": (
                "least squares with scipy.optimize.minimize plus manual FaderPort tuning"
            ),
            "optimizer_code_present_in_fig3_notebook": False,
            "replay_is_independent_validation": False,
        },
        "current_v7_model_differences": {
            "input_data": "MaleCNS recurrent states from image-driven R1-R6, not measured input Vm",
            "normalization": (
                "per source node ranges from synthetic sweep/noise, not class-mean global min-max"
            ),
            "spatial_timing": (
                "real source-target adjacency with no paper fixed +/-160 ms shift in the equation"
            ),
            "target_integration": (
                "per-T4 row-normalized source matrices, not five class-average traces"
            ),
            "temporal_update": (
                "recurrent tanh plus leak integration, not instantaneous steady-state voltage"
            ),
            "output": (
                "paper voltage normalized to [-1,1] drive then leak-updated, not membrane mV"
            ),
            "direct_fig3_reproduction": False,
        },
        "fitting_boundary": config["fitting_boundary"],
        "limitations": [
            "Paper Fig. 3 parameters were fitted and hand-tuned on the same average traces.",
            "The Fig. 3 notebook has simulation code, not the optimizer objective/history.",
            "The 2.5-5.5 s interval is a display crop, not a proven original fit window.",
            "Repository arrays are processed/aligned at 1 kHz; raw acquisition was 10 kHz.",
            "Input directions are synthesized from class means, not recorded PD/ND arrays.",
            "Fig. 1 T4 fields were rotated post hoc into preferred-direction coordinates.",
            "The paper cannot validate T5, MaleCNS per-cell states or driving behavior.",
        ],
        "advance_to_parameter_fit": False,
        "advance_to_central_complex": False,
    }
