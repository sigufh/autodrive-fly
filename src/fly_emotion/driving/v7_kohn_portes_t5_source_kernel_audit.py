"""Audit voltage-derived T5 source kernels without authorizing transfer."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)

CONFIG = Path("configs/driving-v7-kohn-portes-t5-source-kernel-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_kohn_portes_t5_source_kernel_audit.py")


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or not np.isfinite(first).all():
        raise ValueError("invalid source-kernel vectors")
    if not np.isfinite(second).all():
        raise ValueError("invalid source-kernel template")
    return float(np.corrcoef(first, second)[0, 1])


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _author_rescaled_temporal(record: dict, length: int) -> np.ndarray:
    temporal = np.asarray(record["temporal_filter"], dtype=np.float64).reshape(-1)
    complete = np.asarray(record["complete_filter"], dtype=np.float64)
    scale = float(np.max(np.abs(complete)))
    temporal_peak = float(np.max(np.abs(temporal)))
    if scale <= 0 or temporal_peak <= 0:
        raise ValueError("zero-amplitude Kohn-Portes source filter")
    return scale * temporal[:length] / temporal_peak


def evaluate_v7_kohn_portes_t5_source_kernel_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_config_path = Path(config["source_config"])
    source_config = yaml.safe_load((root / source_config_path).read_text())
    expected = config["expected"]

    paper_spec = config["paper"]
    paper_path = _verify_file(root, paper_spec)
    paper_text = " ".join(
        (element.text or "")
        for element in ElementTree.parse(paper_path).getroot().findall(".//text")
    )
    paper_phrases = (
        "Membrane potential was measured in current-clamp mode",
        "Traces were downsampled to 100 Hz",
        "filters were extracted for a duration of 5 seconds",
        "linear temporal filters of OFF-pathway inputs Tm1/Tm2/Tm4/Tm9",
        "linear filters combined with a static nonlinearity are poor approximators",
    )
    if any(phrase not in paper_text for phrase in paper_phrases):
        raise ValueError("Kohn-Portes source-kernel Methods evidence changed")

    code_paths = {name: _verify_file(root, spec) for name, spec in config["author_code"].items()}
    convolve = code_paths["convolve"].read_text(encoding="utf-8")
    returnmeans = code_paths["returnmeans"].read_text(encoding="utf-8")
    notebook = code_paths["figure2_notebook"].read_text(encoding="utf-8")
    derivation_fragments = (
        "cov = np.mean(",
        "autocov_stim = np.mean(",
        "crosscorr = cov/autocov_normalized",
        "self.linear_filter = self._retrieve_filter(X, y)",
        "self.x_weights = self._x_regression(X, y)",
        "np.dot(self.linear_filter, self.x_weights)",
    )
    if any(fragment not in convolve for fragment in derivation_fragments):
        raise ValueError("Kohn-Portes filter derivation code changed")
    rescaling_fragments = (
        "scale = np.max(np.abs(wn['complete_filter']))",
        "trace = scale*trace[:499]/np.max(np.abs(trace))",
        "does NOT return the temporal vector at the max amplitude",
        'not at the "correct" scale',
    )
    if any(fragment not in convolve + returnmeans for fragment in rescaling_fragments):
        raise ValueError("Kohn-Portes temporal-filter scale semantics changed")
    notebook_phrases = (
        "`linear_filter`: 2D spatiotemporal filter",
        "`complete_filter`: 2D spatiotemporal filter with slight denoising",
        "`temporal_filter`: 1D temporal filter",
    )
    if any(phrase not in notebook for phrase in notebook_phrases):
        raise ValueError("Kohn-Portes Figure 2 filter description changed")

    source_results = {}
    orientation_counts: Counter[str] = Counter()
    all_kernel_lengths = set()
    all_temporal_errors = []
    all_complete_errors = []
    for source in expected["source_order"]:
        spec = source_config["white_noise_files"][source]
        payload = _load_restricted(_verify_file(root, spec))
        if len(payload) != int(expected["saline_record_counts"][source]):
            raise ValueError(f"Kohn-Portes saline record count changed: {source}")
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        records = []
        for record in payload:
            required = {
                "recording_id",
                "subrecording_number",
                "cell_type",
                "bath_solution",
                "sampling_rate",
                "downsampling_factor",
                "filter_time",
                "linear_filter",
                "complete_filter",
                "temporal_filter",
                "x_weights",
                "ephys_trace",
                "timestamps",
                "wn_temporal_freq",
                "wn_bar_width",
                "wn_orientation",
                "score",
            }
            if not required.issubset(record):
                raise ValueError(f"Kohn-Portes source-kernel fields changed: {source}")
            if record["cell_type"] != source or record["bath_solution"] != "saline":
                raise ValueError(f"Kohn-Portes source or state label changed: {source}")
            raw_dt = float(record["sampling_rate"])
            factor = int(record["downsampling_factor"])
            kernel_dt = raw_dt * factor
            if not np.isclose(raw_dt, expected["raw_sample_interval_seconds"]):
                raise ValueError(f"Kohn-Portes raw sample interval changed: {source}")
            if factor != int(expected["downsampling_factor"]):
                raise ValueError(f"Kohn-Portes downsampling factor changed: {source}")
            if not np.isclose(kernel_dt, expected["kernel_sample_interval_seconds"]):
                raise ValueError(f"Kohn-Portes kernel interval changed: {source}")
            if not np.isclose(record["filter_time"], expected["filter_duration_seconds"]):
                raise ValueError(f"Kohn-Portes filter duration changed: {source}")
            if int(record["wn_temporal_freq"]) != int(
                expected["white_noise_temporal_frequency_hz"]
            ) or int(record["wn_bar_width"]) != int(expected["bar_width_degrees"]):
                raise ValueError(f"Kohn-Portes white-noise protocol changed: {source}")

            linear = np.asarray(record["linear_filter"])
            weights = np.asarray(record["x_weights"])
            complete = np.asarray(record["complete_filter"], dtype=np.float64)
            temporal = np.asarray(record["temporal_filter"], dtype=np.float64)
            if linear.ndim != 2 or weights.shape != (linear.shape[1], 1):
                raise ValueError(f"Kohn-Portes filter shape changed: {source}")
            if complete.shape != linear.shape or temporal.shape != (linear.shape[0], 1):
                raise ValueError(f"Kohn-Portes derived-filter shape changed: {source}")
            if linear.shape[0] not in set(expected["kernel_lengths"]):
                raise ValueError(f"Kohn-Portes kernel length changed: {source}")
            arrays = (linear.real, linear.imag, weights.real, weights.imag, complete, temporal)
            if not all(np.isfinite(array).all() for array in arrays):
                raise ValueError(f"Kohn-Portes filter contains non-finite values: {source}")

            temporal_error = float(np.max(np.abs(temporal - np.real(linear @ weights))))
            complete_error = float(np.max(np.abs(complete - np.real(linear) * np.real(weights).T)))
            if temporal_error > 1e-15 or complete_error > 1e-15:
                raise ValueError(f"Kohn-Portes filter algebra changed: {source}")
            common_length = min(expected["kernel_lengths"])
            rescaled = _author_rescaled_temporal(record, common_length)
            recording_id = str(record["recording_id"])
            grouped[recording_id].append(rescaled)
            orientation = (
                "null" if record["wn_orientation"] is None else str(record["wn_orientation"])
            )
            orientation_counts[orientation] += 1
            all_kernel_lengths.add(linear.shape[0])
            all_temporal_errors.append(temporal_error)
            all_complete_errors.append(complete_error)
            records.append(
                {
                    "recording_id": recording_id,
                    "subrecording_number": str(record["subrecording_number"]),
                    "orientation": orientation,
                    "kernel_length": int(linear.shape[0]),
                    "kernel_sample_interval_seconds": kernel_dt,
                    "kernel_support_seconds": float(linear.shape[0] * kernel_dt),
                    "temporal_formula_max_abs_error": temporal_error,
                    "complete_formula_max_abs_error": complete_error,
                    "author_reported_LN_R2": float(record["score"]),
                    "author_rescaled_kernel_minimum": float(np.min(rescaled)),
                    "author_rescaled_kernel_maximum": float(np.max(rescaled)),
                }
            )
        if len(grouped) != int(expected["saline_unique_recording_id_counts"][source]):
            raise ValueError(f"Kohn-Portes recording-ID count changed: {source}")
        recording_kernels = {
            recording_id: np.mean(kernels, axis=0) for recording_id, kernels in grouped.items()
        }
        folds = []
        for recording_id, kernel in recording_kernels.items():
            comparison = np.mean(
                [
                    other
                    for other_id, other in recording_kernels.items()
                    if other_id != recording_id
                ],
                axis=0,
            )
            folds.append(
                {
                    "held_out_recording_id": recording_id,
                    "correlation": _correlation(kernel, comparison),
                }
            )
        correlations = [item["correlation"] for item in folds]
        source_results[source] = {
            "file": spec,
            "record_count": len(payload),
            "unique_recording_id_count": len(grouped),
            "records": records,
            "author_reported_LN_R2": _distribution([float(record["score"]) for record in payload]),
            "descriptive_leave_one_recording_id_out_shape_correlation": {
                "folds": folds,
                "summary": _distribution(correlations),
                "used_as_transfer_gate": False,
            },
            "voltage_derived_temporal_kernels_verified": True,
        }

    if dict(sorted(orientation_counts.items())) != expected["orientation_counts"]:
        raise ValueError("Kohn-Portes white-noise orientation inventory changed")
    if sorted(all_kernel_lengths) != expected["kernel_lengths"]:
        raise ValueError("Kohn-Portes kernel-length inventory changed")
    all_four = all(
        item["voltage_derived_temporal_kernels_verified"] for item in source_results.values()
    )
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(source_config_path): _sha256(root / source_config_path),
        paper_spec["path"]: _sha256(paper_path),
    }
    dependencies.update(
        {
            spec["path"]: _sha256(path)
            for spec, path in zip(config["author_code"].values(), code_paths.values(), strict=True)
        }
    )
    dependencies.update(
        {
            spec["path"]: _sha256(root / spec["path"])
            for spec in (source_config["white_noise_files"].values())
        }
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "author_method": {
            "measurement": "whole_cell_current_clamp_membrane_potential",
            "filter_estimator": "regularized_reverse_correlation_plus_spatial_regression",
            "temporal_filter_formula": "real(linear_filter @ x_weights)",
            "author_population_rescaling": (
                "max_abs_complete_filter * temporal_filter[:499] / max_abs_temporal_filter"
            ),
            "raw_temporal_filter_declared_correct_scale": False,
            "kernel_sample_interval_seconds": float(expected["kernel_sample_interval_seconds"]),
            "filter_duration_seconds": float(expected["filter_duration_seconds"]),
            "author_reported_LN_score_independent_validation_verified": False,
            "paper_reports_cross_stimulus_flash_mismatch": True,
        },
        "source_results": source_results,
        "aggregate": {
            "source_count": len(source_results),
            "saline_record_count": sum(item["record_count"] for item in source_results.values()),
            "saline_unique_recording_id_count_sum": sum(
                item["unique_recording_id_count"] for item in source_results.values()
            ),
            "orientation_counts": dict(sorted(orientation_counts.items())),
            "kernel_lengths": sorted(all_kernel_lengths),
            "maximum_temporal_formula_error": max(all_temporal_errors),
            "maximum_complete_formula_error": max(all_complete_errors),
        },
        "gates": {
            "all_four_voltage_derived_temporal_kernel_sets_verified": all_four,
            "physical_kernel_time_axis_verified": True,
            "author_population_rescaling_reproduced": True,
            "raw_temporal_filter_absolute_gain_transferable": False,
            "stimulus_invariant_source_kernel_verified": False,
            "biological_individual_identity_verified": False,
            "independent_dynamic_validation_available": False,
            "CT1_voltage_derived_temporal_kernel_available": False,
            "v7_normalized_state_mapping_available": False,
        },
        "Tm1_Tm2_Tm4_Tm9_voltage_derived_temporal_kernels_verified": all_four,
        "source_kernel_transfer_authorized": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "voltage_derived_source_kernels_exist_but_gain_stimulus_identity_"
            "CT1_and_independent_validation_gates_remain_incomplete"
        ),
        "boundary": config["boundary"],
    }
