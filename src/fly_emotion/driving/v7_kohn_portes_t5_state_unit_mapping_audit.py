"""Audit Kohn-Portes voltage/filter units against v7 simulated state."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)

CONFIG = Path("configs/driving-v7-kohn-portes-t5-state-unit-mapping-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_kohn_portes_t5_state_unit_mapping_audit.py"
)


def _softplus(x: np.ndarray, parameters: list[float]) -> np.ndarray:
    a, b, c, d, k = parameters
    return c * np.log1p(np.exp(a * x + b)) ** k + d


def _range(values: list[float]) -> list[float]:
    return [float(min(values)), float(max(values))]


def evaluate_v7_kohn_portes_t5_state_unit_mapping_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_config_path = Path(config["source_config"])
    source_kernel_path = Path(config["source_kernel_evidence"])
    v7_contract_path = Path(config["v7_contract"])
    v7_implementation_path = Path(config["v7_implementation"])
    runtime_engine_path = Path(config["runtime_engine"])
    source_config = yaml.safe_load((root / source_config_path).read_text())
    source_kernel = json.loads((root / source_kernel_path).read_text())
    v7 = yaml.safe_load((root / v7_contract_path).read_text())
    expected = config["expected"]
    if list(source_kernel["source_results"]) != expected["source_order"]:
        raise ValueError("Kohn-Portes source-kernel order changed")

    code_paths = {
        name: _verify_file(root, spec) for name, spec in config["author_code"].items()
    }
    convolve = code_paths["convolve"].read_text(encoding="utf-8")
    returnmeans = code_paths["returnmeans"].read_text(encoding="utf-8")
    plotting = code_paths["plotting"].read_text(encoding="utf-8")
    required_fragments = (
        "cut_traces -= trace_mean",
        "soft = c*np.log(1+np.exp(a*x+b))**k + d",
        "scale = np.max(np.abs(wn['complete_filter']))",
        "trace = scale*trace[:499]/np.max(np.abs(trace))",
        "# convert to mV",
        "sc = 1e3",
    )
    combined_source = convolve + returnmeans + plotting
    if any(fragment not in combined_source for fragment in required_fragments):
        raise ValueError("Kohn-Portes voltage-unit processing code changed")
    runtime_source = (root / v7_implementation_path).read_text(encoding="utf-8")
    engine_source = (root / runtime_engine_path).read_text(encoding="utf-8")
    if "np.tanh(1.8 * recurrent + drive)" not in runtime_source:
        raise ValueError("v7 signed-tanh state semantics changed")
    if '"simulated_activation_not_millivolts"' not in engine_source:
        raise ValueError("runtime state-unit disclosure changed")
    if v7["deployment_enabled"] or v7["city_expansion_enabled"]:
        raise ValueError("v7 deployment boundary changed")

    raw_minima = []
    raw_maxima = []
    baselines = []
    linear_minima = []
    linear_maxima = []
    full_minima = []
    full_maxima = []
    reconstruction_errors = []
    scores = []
    source_state_counts: Counter[tuple[str, str]] = Counter()
    source_recording_ids = {source: set() for source in expected["source_order"]}
    records = []
    groups = (
        ("saline", "white_noise_files"),
        ("OA", "white_noise_OA_files"),
    )
    for state, config_key in groups:
        for source in expected["source_order"]:
            spec = source_config[config_key][source]
            payload = _load_restricted(_verify_file(root, spec))
            if len(payload) != int(expected["record_counts"][state][source]):
                raise ValueError(f"Kohn-Portes {state} record count changed: {source}")
            for record in payload:
                if record["cell_type"] != source or record["bath_solution"] != state:
                    raise ValueError(f"Kohn-Portes source/state label changed: {source}")
                raw = np.asarray(record["ephys_trace"], dtype=np.float64)
                linear = np.asarray(record["linear_prediction"], dtype=np.float64)
                full = np.asarray(record["full_prediction"], dtype=np.float64)
                parameters = [float(record[f"soft_{name}"]) for name in "abcdk"]
                reconstructed = _softplus(linear, parameters)
                error = float(np.max(np.abs(full - reconstructed)))
                if not all(np.isfinite(values).all() for values in (raw, linear, full)):
                    raise ValueError(f"Kohn-Portes voltage values are non-finite: {source}")
                source_state_counts[(source, state)] += 1
                source_recording_ids[source].add(str(record["recording_id"]))
                raw_minima.append(float(np.min(raw)))
                raw_maxima.append(float(np.max(raw)))
                baselines.append(float(record["cut_trace_mean"]))
                linear_minima.append(float(np.min(linear)))
                linear_maxima.append(float(np.max(linear)))
                full_minima.append(float(np.min(full)))
                full_maxima.append(float(np.max(full)))
                reconstruction_errors.append(error)
                scores.append(float(record["score"]))
                records.append(
                    {
                        "source": source,
                        "state": state,
                        "recording_id": str(record["recording_id"]),
                        "subrecording_number": str(record["subrecording_number"]),
                        "cut_trace_mean_volts": float(record["cut_trace_mean"]),
                        "raw_voltage_minimum_volts": float(np.min(raw)),
                        "raw_voltage_maximum_volts": float(np.max(raw)),
                        "linear_prediction_minimum_volts": float(np.min(linear)),
                        "linear_prediction_maximum_volts": float(np.max(linear)),
                        "full_prediction_minimum_volts": float(np.min(full)),
                        "full_prediction_maximum_volts": float(np.max(full)),
                        "stored_softplus_reconstruction_max_abs_error_volts": error,
                        "author_reported_LN_R2": float(record["score"]),
                    }
                )

    if len(records) != int(expected["total_record_count"]):
        raise ValueError("Kohn-Portes total white-noise record count changed")
    for state in expected["states"]:
        for source, count in expected["record_counts"][state].items():
            if source_state_counts[(source, state)] != int(count):
                raise ValueError(f"Kohn-Portes source/state count changed: {source}/{state}")
    actual_unique_ids = {
        source: len(recording_ids)
        for source, recording_ids in source_recording_ids.items()
    }
    if actual_unique_ids != expected["unique_recording_id_counts"]:
        raise ValueError("Kohn-Portes combined recording-ID counts changed")
    observed_ranges = {
        "raw_voltage_volts": [min(raw_minima), max(raw_maxima)],
        "cut_trace_mean_volts": _range(baselines),
        "linear_prediction_volts": [min(linear_minima), max(linear_maxima)],
        "full_prediction_volts": [min(full_minima), max(full_maxima)],
        "author_reported_LN_R2": _range(scores),
    }
    expected_ranges = {
        "raw_voltage_volts": expected["raw_voltage_range_volts"],
        "cut_trace_mean_volts": expected["cut_trace_mean_range_volts"],
        "linear_prediction_volts": expected["linear_prediction_range_volts"],
        "full_prediction_volts": expected["full_prediction_range_volts"],
        "author_reported_LN_R2": expected["author_reported_LN_R2_range"],
    }
    for name, values in observed_ranges.items():
        if not np.allclose(values, expected_ranges[name], rtol=0.0, atol=1e-15):
            raise ValueError(f"Kohn-Portes voltage range changed: {name}")
    maximum_error = max(reconstruction_errors)
    if not np.isclose(
        maximum_error,
        expected["maximum_stored_softplus_reconstruction_error_volts"],
        rtol=0.0,
        atol=1e-18,
    ):
        raise ValueError("Kohn-Portes stored softplus reconstruction error changed")

    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(source_config_path): _sha256(root / source_config_path),
        str(source_kernel_path): _sha256(root / source_kernel_path),
        str(v7_contract_path): _sha256(root / v7_contract_path),
        str(v7_implementation_path): _sha256(root / v7_implementation_path),
        str(runtime_engine_path): _sha256(root / runtime_engine_path),
    }
    dependencies.update(
        {spec["path"]: _sha256(path) for spec, path in zip(
            config["author_code"].values(), code_paths.values(), strict=True
        )}
    )
    dependencies.update(
        {
            spec["path"]: _sha256(root / spec["path"])
            for key in ("white_noise_files", "white_noise_OA_files")
            for spec in source_config[key].values()
        }
    )
    gates = {
        "raw_voltage_unit_volts_verified": True,
        "exact_volts_to_millivolts_scale_verified": True,
        "centered_prediction_output_in_volts_verified": True,
        "stored_softplus_parameters_reconstruct_predictions_with_declared_tolerance": True,
        "single_cross_record_or_cross_source_state_mapping_declared": False,
        "raw_temporal_filter_absolute_gain_transferable": False,
        "stimulus_invariant_source_kernel_verified": False,
        "v7_signed_state_semantics_match_author_voltage_semantics": False,
        "held_out_state_mapping_validated": False,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "record_inventory": {
            "record_count": len(records),
            "source_state_counts": {
                f"{source}:{state}": source_state_counts[(source, state)]
                for state in expected["states"]
                for source in expected["source_order"]
            },
            "unique_recording_id_counts_by_source": actual_unique_ids,
            "records": records,
        },
        "unit_evidence": {
            "stored_raw_voltage_unit": "volts",
            "stored_linear_prediction_unit": "volts_relative_to_cut_trace_mean",
            "stored_full_prediction_unit": "volts_relative_to_cut_trace_mean",
            "display_conversion": "millivolts = volts * 1000",
            "observed_ranges": observed_ranges,
            "maximum_stored_softplus_reconstruction_error_volts": maximum_error,
        },
        "mapping_boundary": {
            "v7_state_activation": "signed_tanh",
            "v7_runtime_reported_unit": "simulated_activation_not_millivolts",
            "author_population_kernel_rescaling": source_kernel["author_method"][
                "author_population_rescaling"
            ],
            "author_declares_raw_temporal_filter_correct_scale": source_kernel[
                "author_method"
            ]["raw_temporal_filter_declared_correct_scale"],
            "author_reports_cross_stimulus_flash_mismatch": source_kernel[
                "author_method"
            ]["paper_reports_cross_stimulus_flash_mismatch"],
            "candidate_normalization_formula": None,
            "candidate_clipping_rule": None,
        },
        "gates": gates,
        "volts_to_millivolts_mapping_available": True,
        "millivolts_or_filter_output_to_v7_state_mapping_available": False,
        "authorize_source_kernel_gain_transfer": False,
        "authorize_source_dynamics_transfer": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "exact_voltage_units_exist_but_no_shared_held_out_mapping_to_v7_signed_state"
        ),
        "boundary": config["boundary"],
    }
