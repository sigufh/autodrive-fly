"""Audit condition-specific T5 source-filter peak latency."""

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

CONFIG = Path("configs/driving-v7-kohn-portes-t5-peak-latency-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_kohn_portes_t5_peak_latency_audit.py"
)


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _peak_index(trace: np.ndarray, common_length: int) -> int:
    values = np.asarray(trace, dtype=np.float64).reshape(-1)[:common_length]
    if values.size != common_length or not np.isfinite(values).all():
        raise ValueError("invalid Kohn-Portes temporal filter")
    return int(np.argmin(values))


def evaluate_v7_kohn_portes_t5_peak_latency_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_config_path = Path(config["source_config"])
    source_kernel_path = Path(config["source_kernel_evidence"])
    v7_config_path = Path(config["v7_config"])
    source_config = yaml.safe_load((root / source_config_path).read_text(encoding="utf-8"))
    source_kernel = json.loads((root / source_kernel_path).read_text(encoding="utf-8"))
    expected = config["expected"]
    v7 = yaml.safe_load((root / v7_config_path).read_text(encoding="utf-8"))
    configured_sources = v7["controlled_vision"]["columnar_correlator_v1"]
    if configured_sources["T5_fast_sources"] != expected["source_order"][:3] or (
        configured_sources["T5_delayed_sources"] != ["Tm9", "CT1"]
    ):
        raise ValueError("v7 T5 fast/delayed source declaration changed")
    notebook_path = _verify_file(root, config["author_notebook"])
    notebook = notebook_path.read_text(encoding="utf-8")
    required_fragments = (
        "peak = np.argmin(wn['temporal_filter'])*wn['sampling_rate']*wn['downsampling_factor']",
        "Note that Tm9 is slower than Tm1, Tm2, and Tm4",
        "## 1. Baseline (Saline)",
        "## 2. Neuromodulator (Octopamine)",
    )
    if any(fragment not in notebook for fragment in required_fragments):
        raise ValueError("Kohn-Portes peak-latency notebook evidence changed")
    if list(source_kernel["source_results"]) != expected["source_order"]:
        raise ValueError("Kohn-Portes source order changed")
    dt = float(source_kernel["author_method"]["kernel_sample_interval_seconds"])
    if not np.isclose(dt, expected["sample_interval_seconds"]):
        raise ValueError("Kohn-Portes kernel time step changed")

    state_results = {}
    for state, key in (("saline", "white_noise_files"), ("OA", "white_noise_OA_files")):
        source_results = {}
        for source in expected["source_order"]:
            spec = source_config[key][source]
            payload = _load_restricted(_verify_file(root, spec))
            grouped: dict[str, list[np.ndarray]] = defaultdict(list)
            records = []
            for record in payload:
                index = _peak_index(
                    record["temporal_filter"], int(expected["common_kernel_length"])
                )
                record_dt = float(record["sampling_rate"]) * int(
                    record["downsampling_factor"]
                )
                if not np.isclose(record_dt, dt):
                    raise ValueError(f"Kohn-Portes kernel time step changed: {source}")
                recording_id = str(record["recording_id"])
                grouped[recording_id].append(
                    np.asarray(record["temporal_filter"], dtype=np.float64)
                    .reshape(-1)[: int(expected["common_kernel_length"])]
                )
                records.append(
                    {
                        "recording_id": recording_id,
                        "subrecording_number": str(record["subrecording_number"]),
                        "peak_index": index,
                        "peak_latency_milliseconds": float(index * dt * 1000.0),
                    }
                )
            record_indices = [record["peak_index"] for record in records]
            collapsed = [
                _peak_index(np.mean(traces, axis=0), int(expected["common_kernel_length"]))
                for traces in grouped.values()
            ]
            if record_indices != expected["record_peak_indices"][state][source]:
                raise ValueError(f"Kohn-Portes record peak indices changed: {state}/{source}")
            if collapsed != expected["recording_id_collapsed_peak_indices"][state][source]:
                raise ValueError(
                    f"Kohn-Portes recording-ID collapsed peaks changed: {state}/{source}"
                )
            record_latencies = [float(index * dt * 1000.0) for index in record_indices]
            collapsed_latencies = [float(index * dt * 1000.0) for index in collapsed]
            record_summary = _summary(record_latencies)
            if not np.isclose(
                record_summary["median"],
                expected["source_record_median_peak_milliseconds"][state][source],
            ):
                raise ValueError(f"Kohn-Portes median peak latency changed: {state}/{source}")
            source_results[source] = {
                "records": records,
                "record_peak_latency_milliseconds": record_summary,
                "recording_id_collapsed_peak_latency_milliseconds": _summary(
                    collapsed_latencies
                ),
                "recording_id_collapsed_records": [
                    {
                        "recording_id": recording_id,
                        "peak_index": index,
                        "peak_latency_milliseconds": float(index * dt * 1000.0),
                    }
                    for recording_id, index in zip(grouped, collapsed, strict=True)
                ],
                "recording_id_count": len(grouped),
            }

        fast_sources = expected["source_order"][:3]
        fast_values = np.concatenate(
            [
                [
                    record["peak_latency_milliseconds"]
                    for record in source_results[source]["records"]
                ]
                for source in fast_sources
            ]
        )
        tm9_values = np.asarray(
            [
                record["peak_latency_milliseconds"]
                for record in source_results["Tm9"]["records"]
            ]
        )
        state_results[state] = {
            "sources": source_results,
            "Tm9_median_minus_each_fast_source_milliseconds": {
                source: float(
                    source_results["Tm9"]["record_peak_latency_milliseconds"]["median"]
                    - source_results[source]["record_peak_latency_milliseconds"]["median"]
                )
                for source in fast_sources
            },
            "pooled_fast_median_milliseconds": float(np.median(fast_values)),
            "Tm9_median_milliseconds": float(np.median(tm9_values)),
            "Tm9_strictly_slower_pair_fraction": float(
                np.mean(tm9_values[:, None] > fast_values[None, :])
            ),
            "all_three_fast_source_medians_precede_Tm9": all(
                source_results[source]["record_peak_latency_milliseconds"]["median"]
                < source_results["Tm9"]["record_peak_latency_milliseconds"]["median"]
                for source in fast_sources
            ),
        }

    common_id_deltas = {}
    for source in expected["source_order"]:
        by_state = {}
        for state in expected["states"]:
            by_state[state] = {
                record["recording_id"]: record["peak_latency_milliseconds"]
                for record in state_results[state]["sources"][source][
                    "recording_id_collapsed_records"
                ]
            }
        common = sorted(set(by_state["saline"]) & set(by_state["OA"]))
        deltas = [
            float(round(by_state["OA"][identifier] - by_state["saline"][identifier], 12))
            for identifier in common
        ]
        if not np.allclose(
            deltas,
            expected["common_recording_id_OA_minus_saline_milliseconds"][source],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Kohn-Portes common recording-ID latency deltas changed: {source}")
        common_id_deltas[source] = {
            "common_recording_ids": common,
            "OA_minus_saline_peak_latency_milliseconds": deltas,
            "median_delta_milliseconds": float(np.median(deltas)),
            "biological_individual_interpretation_authorized": False,
        }

    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(source_config_path): _sha256(root / source_config_path),
        str(source_kernel_path): _sha256(root / source_kernel_path),
        str(v7_config_path): _sha256(root / v7_config_path),
        config["author_notebook"]["path"]: _sha256(notebook_path),
        **{
            spec["path"]: _sha256(root / spec["path"])
            for key in ("white_noise_files", "white_noise_OA_files")
            for spec in source_config[key].values()
        },
    }
    gates = {
        "author_peak_latency_formula_verified": True,
        "saline_Tm9_median_slower_than_each_Tm1_Tm2_Tm4_median": state_results[
            "saline"
        ]["all_three_fast_source_medians_precede_Tm9"],
        "OA_Tm9_median_slower_than_each_Tm1_Tm2_Tm4_median": state_results[
            "OA"
        ]["all_three_fast_source_medians_precede_Tm9"],
        "Tm9_delay_ordering_state_invariant": all(
            state_results[state]["all_three_fast_source_medians_precede_Tm9"]
            for state in expected["states"]
        ),
        "CT1_peak_latency_available": False,
        "v7_source_delay_substeps_biologically_calibrated": False,
        "independent_peak_latency_validation_available": False,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "author_method": {
            "peak_definition": "argmin(temporal_filter) * sampling_rate * downsampling_factor",
            "sample_interval_seconds": dt,
            "notebook_claim": "Tm9 is slower than Tm1, Tm2, and Tm4",
            "v7_declared_fast_sources": configured_sources["T5_fast_sources"],
            "v7_declared_delayed_sources": configured_sources["T5_delayed_sources"],
            "v7_correlator_history_substeps": configured_sources["history_substeps"],
        },
        "state_results": state_results,
        "common_recording_id_state_deltas": common_id_deltas,
        "gates": gates,
        "relative_Tm9_delay_candidate_supported_in_saline": gates[
            "saline_Tm9_median_slower_than_each_Tm1_Tm2_Tm4_median"
        ],
        "relative_Tm9_delay_candidate_supported_across_states": gates[
            "Tm9_delay_ordering_state_invariant"
        ],
        "authorize_source_delay_transfer_to_v7": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "saline_relative_Tm9_delay_is_descriptive_but_not_state_invariant_or_transfer_ready"
        ),
        "boundary": config["boundary"],
    }
