"""Audit Kohn-Portes source-filter frequency tuning without transferring gain."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)

CONFIG = Path("configs/driving-v7-kohn-portes-t5-frequency-tuning-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_kohn_portes_t5_frequency_tuning_audit.py"
)


def _frequency_curve(filt: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.arange(
        config["start_hz"], config["stop_exclusive_hz"], config["step_hz"]
    )
    dt = float(config["convolution_dt_seconds"] )
    duration = float(config["sine_cycle_count_at_minimum_frequency"]) / frequencies.min()
    time = np.arange(0.0, duration, dt)
    start = int(float(config["analysis_start_seconds"]) / dt)
    end = int(float(config["analysis_end_fraction"]) * duration / dt)
    amplitudes = []
    for frequency in frequencies:
        sine = np.sin(2.0 * np.pi * frequency * time)
        response = np.convolve(np.asarray(filt, dtype=np.float64).reshape(-1), sine)
        amplitudes.append(float(np.max(response[start:end])))
    amplitudes = np.asarray(amplitudes)
    if not np.isfinite(amplitudes).all() or float(np.max(amplitudes)) <= 0.0:
        raise ValueError("invalid Kohn-Portes sine-convolution response")
    return frequencies, amplitudes / np.max(amplitudes)


def _rounded_frequency(value: float) -> float:
    return float(round(float(value), 12))


def evaluate_v7_kohn_portes_t5_frequency_tuning_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_config_path = Path(config["source_config"])
    source_kernel_path = Path(config["source_kernel_evidence"])
    peak_latency_path = Path(config["peak_latency_evidence"])
    v7_config_path = Path(config["v7_config"])
    source_config = yaml.safe_load((root / source_config_path).read_text(encoding="utf-8"))
    source_kernel = json.loads((root / source_kernel_path).read_text(encoding="utf-8"))
    peak_latency = json.loads((root / peak_latency_path).read_text(encoding="utf-8"))
    v7 = yaml.safe_load((root / v7_config_path).read_text(encoding="utf-8"))
    expected = config["expected"]
    notebook_path = _verify_file(root, config["author_notebook"])
    notebook = notebook_path.read_text(encoding="utf-8")
    fragments = (
        "freqs = np.arange(0.1,10,0.05)",
        "stimulus = sine_wave1d(tt,t_period=1/freq,phase=0)",
        "response = np.convolve(filt,stimulus)",
        "resp_all.append(resp['max']/np.max(resp['max']))",
        "Note that Tm9 is tuned to lower frequencies",
    )
    if any(fragment not in notebook for fragment in fragments):
        raise ValueError("Kohn-Portes frequency-tuning notebook evidence changed")
    if list(source_kernel["source_results"]) != expected["source_order"]:
        raise ValueError("Kohn-Portes source order changed")
    configured = v7["controlled_vision"]["columnar_correlator_v1"]
    if configured["T5_fast_sources"] != expected["source_order"][:3] or (
        configured["T5_delayed_sources"] != ["Tm9", "CT1"]
    ):
        raise ValueError("v7 T5 fast/delayed source declaration changed")

    state_results = {}
    for state, key in (("saline", "white_noise_files"), ("OA", "white_noise_OA_files")):
        sources = {}
        for source in expected["source_order"]:
            spec = source_config[key][source]
            payload = _load_restricted(_verify_file(root, spec))
            curves = []
            grouped: dict[str, list[np.ndarray]] = defaultdict(list)
            records = []
            for record in payload:
                frequencies, curve = _frequency_curve(
                    record["temporal_filter"], config["author_frequency_grid"]
                )
                preferred = _rounded_frequency(frequencies[int(np.argmax(curve))])
                records.append(
                    {
                        "recording_id": str(record["recording_id"]),
                        "subrecording_number": str(record["subrecording_number"]),
                        "preferred_frequency_hz": preferred,
                    }
                )
                curves.append(curve)
                grouped[str(record["recording_id"])].append(curve)
            record_frequencies = [record["preferred_frequency_hz"] for record in records]
            if not np.allclose(
                record_frequencies,
                expected["record_preferred_frequencies_hz"][state][source],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"Kohn-Portes preferred frequencies changed: {state}/{source}")
            collapsed = [
                _rounded_frequency(frequencies[int(np.argmax(np.mean(values, axis=0)))])
                for values in grouped.values()
            ]
            population_peak = _rounded_frequency(
                frequencies[int(np.argmax(np.mean(curves, axis=0)))]
            )
            median = _rounded_frequency(np.median(record_frequencies))
            if not np.isclose(
                median,
                expected["record_median_preferred_frequency_hz"][state][source],
            ):
                raise ValueError(
                    f"Kohn-Portes median preferred frequency changed: {state}/{source}"
                )
            if not np.isclose(
                population_peak,
                expected["author_population_curve_peak_frequency_hz"][state][source],
            ):
                raise ValueError(f"Kohn-Portes population curve peak changed: {state}/{source}")
            sources[source] = {
                "records": records,
                "record_median_preferred_frequency_hz": median,
                "recording_id_collapsed_preferred_frequencies_hz": collapsed,
                "author_normalized_population_curve_peak_frequency_hz": population_peak,
            }
        fast = expected["source_order"][:3]
        state_results[state] = {
            "sources": sources,
            "Tm9_record_median_lower_than_each_fast_source": all(
                sources["Tm9"]["record_median_preferred_frequency_hz"]
                < sources[source]["record_median_preferred_frequency_hz"]
                for source in fast
            ),
            "Tm9_author_population_curve_peak_lower_than_each_fast_source": all(
                sources["Tm9"]["author_normalized_population_curve_peak_frequency_hz"]
                < sources[source][
                    "author_normalized_population_curve_peak_frequency_hz"
                ]
                for source in fast
            ),
        }

    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(source_config_path): _sha256(root / source_config_path),
        str(source_kernel_path): _sha256(root / source_kernel_path),
        str(peak_latency_path): _sha256(root / peak_latency_path),
        str(v7_config_path): _sha256(root / v7_config_path),
        config["author_notebook"]["path"]: _sha256(notebook_path),
        **{
            spec["path"]: _sha256(root / spec["path"])
            for key in ("white_noise_files", "white_noise_OA_files")
            for spec in source_config[key].values()
        },
    }
    gates = {
        "author_frequency_response_method_reproduced": True,
        "Tm9_record_median_lower_than_each_fast_source_in_both_states": all(
            state_results[state]["Tm9_record_median_lower_than_each_fast_source"]
            for state in state_results
        ),
        "Tm9_author_population_curve_peak_lower_than_each_fast_source_in_both_states": all(
            state_results[state][
                "Tm9_author_population_curve_peak_lower_than_each_fast_source"
            ]
            for state in state_results
        ),
        "preferred_frequency_summary_invariant": False,
        "absolute_gain_available": False,
        "CT1_frequency_tuning_available": False,
        "independent_frequency_tuning_validation_available": False,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "author_frequency_grid": config["author_frequency_grid"],
        "state_results": state_results,
        "gates": gates,
        "relative_low_frequency_Tm9_shape_evidence_available": True,
        "authorize_source_frequency_transfer_to_v7": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "relative_frequency_shape_depends_on_summary_and_lacks_gain_CT1_and_independent_validation"
        ),
        "cross_check": {
            "peak_latency_transfer_authorized": peak_latency[
                "authorize_source_delay_transfer_to_v7"
            ]
        },
        "boundary": config["boundary"],
    }
