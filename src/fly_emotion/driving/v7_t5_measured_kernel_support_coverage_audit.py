"""Audit how much measured T5 kernel support enters the local-edge precheck."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_t5_measured_kernel_identifiability import (
    _population_kernel,
)
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path(
    "configs/driving-v7-t5-measured-kernel-support-coverage-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_measured_kernel_support_coverage_audit.py"
)


def evaluate_v7_t5_measured_kernel_support_coverage_audit(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_path = Path(audit["measured_kernel_evidence"])
    measured_config_path = Path(audit["measured_kernel_config"])
    measured_implementation_path = Path(audit["measured_kernel_implementation"])
    evidence = json.loads((root / evidence_path).read_text(encoding="utf-8"))
    measured = yaml.safe_load((root / measured_config_path).read_text(encoding="utf-8"))
    source_config_path = Path(measured["source_config"])
    stage1_path = Path(measured["stage1_protocol"])
    local_path = Path(measured["local_edge_config"])
    source_config = yaml.safe_load((root / source_config_path).read_text(encoding="utf-8"))
    stage1 = yaml.safe_load((root / stage1_path).read_text(encoding="utf-8"))
    local = yaml.safe_load((root / local_path).read_text(encoding="utf-8"))

    source_order = audit["required_source_order"]
    if measured["source_types"] != source_order:
        raise ValueError("measured-kernel source order changed")
    if evidence["kernel_summary"]["source_order"] != source_order:
        raise ValueError("measured-kernel evidence source order changed")
    kernel_length = int(evidence["kernel_summary"]["kernel_length"] )
    sample_ms = float(evidence["kernel_summary"]["sample_interval_milliseconds"] )

    condition = next(
        row for row in stage1["conditions"] if row["condition_id"] == measured["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("kernel-support audit may consume tuning only")
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    stimuli = [
        _local_step_edge(
            float(center_x),
            float(center_y),
            float(local["stimulus"]["aperture_radius_pixels"]),
            speed,
            polarity,
            direction,
            int(local["common_background_frames"]),
        )
        for center_x in local["centers"]["x"]
        for center_y in local["centers"]["y"]
        for polarity in ("on", "off")
        for direction in ("left", "right", "up", "down")
    ]
    baseline_frames = int(
        yaml.safe_load((root / measured["lamina_split_protocol"]).read_text())[
            "baseline_frames"
        ]
    )
    trace_lengths = [len(stimulus.frames) - baseline_frames for stimulus in stimuli]
    if len(stimuli) != int(audit["expected_stimulus_count"]):
        raise ValueError("measured-kernel stimulus count changed")
    if set(trace_lengths) != {int(audit["expected_post_baseline_trace_samples"])}:
        raise ValueError("measured-kernel post-baseline trace length changed")
    trace_samples = trace_lengths[0]

    author_spec = audit["author_convolution_code"]
    author_path = _verify_file(root, author_spec)
    author_text = author_path.read_text(encoding="utf-8")
    causal_calls = (
        "lfilter(ifilter, 1, iX)",
        "lfilter(self.linear_filter, 1, X, axis=0)",
    )
    causal_call_verified = all(fragment in author_text for fragment in causal_calls)
    explicit_import_verified = bool(
        "from scipy.signal import lfilter" in author_text
        or "from scipy.signal import *" in author_text
    )
    if not causal_call_verified:
        raise ValueError("author causal lfilter calls changed")

    source_results = {}
    source_paths = {}
    minimum_fraction = float(
        audit["coverage_gate"]["minimum_population_kernel_L1_mass_fraction"]
    )
    for source in source_order:
        spec = source_config["white_noise_files"][source]
        source_path = _verify_file(root, spec)
        source_paths[source] = source_path
        kernel, recording_count = _population_kernel(
            _load_restricted(source_path), kernel_length
        )
        absolute = np.abs(kernel)
        peak_index = int(np.argmax(absolute))
        covered = float(np.sum(absolute[:trace_samples]) / np.sum(absolute))
        source_results[source] = {
            "recording_id_count": recording_count,
            "population_kernel_absolute_peak_index": peak_index,
            "population_kernel_absolute_peak_lag_milliseconds": peak_index * sample_ms,
            "absolute_peak_within_scored_trace": peak_index < trace_samples,
            "scored_prefix_L1_mass_fraction": covered,
            "unscored_tail_L1_mass_fraction": 1.0 - covered,
            "full_support_coverage_passed": covered >= minimum_fraction,
        }
    all_peaks_covered = all(
        item["absolute_peak_within_scored_trace"] for item in source_results.values()
    )
    full_support_covered = all(
        item["full_support_coverage_passed"] for item in source_results.values()
    )
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(evidence_path): _sha256(root / evidence_path),
        str(measured_config_path): _sha256(root / measured_config_path),
        str(measured_implementation_path): _sha256(root / measured_implementation_path),
        str(source_config_path): _sha256(root / source_config_path),
        str(stage1_path): _sha256(root / stage1_path),
        str(local_path): _sha256(root / local_path),
        str(Path(measured["lamina_split_protocol"])): _sha256(
            root / measured["lamina_split_protocol"]
        ),
        author_spec["path"]: _sha256(author_path),
        **{
            source_config["white_noise_files"][source]["path"]: _sha256(path)
            for source, path in source_paths.items()
        },
    }
    return {
        "protocol": {
            "name": audit["name"],
            "observed_on": audit["observed_on"],
            "dependencies_sha256": dependencies,
            "condition_id": measured["condition_id"],
            "stimulus_count": len(stimuli),
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "author_convolution_semantics": {
            "stored_kernel_passed_to_lfilter_without_reversal": causal_call_verified,
            "lfilter_explicitly_imported_in_exported_module": explicit_import_verified,
            "exported_module_self_contained_execution_verified": False,
            "causal_index_zero_interpretation_supported": causal_call_verified,
        },
        "window": {
            "kernel_length_samples": kernel_length,
            "kernel_sample_interval_milliseconds": sample_ms,
            "kernel_nominal_support_milliseconds": kernel_length * sample_ms,
            "kernel_maximum_discrete_lag_milliseconds": (kernel_length - 1) * sample_ms,
            "post_baseline_trace_samples": trace_samples,
            "post_baseline_trace_duration_milliseconds": trace_samples * sample_ms,
            "maximum_kernel_lag_used_milliseconds": (trace_samples - 1) * sample_ms,
            "kernel_coefficient_count_used": trace_samples,
            "kernel_coefficient_fraction_used": trace_samples / kernel_length,
        },
        "coverage_gate": {
            "minimum_population_kernel_L1_mass_fraction": minimum_fraction,
            "all_population_absolute_peaks_within_scored_trace": all_peaks_covered,
            "all_population_kernel_L1_mass_coverage_passed": full_support_covered,
        },
        "source_results": source_results,
        "early_support_negative_result_available": bool(
            all_peaks_covered and not evidence["temporal_identifiability_passed"]
        ),
        "full_kernel_support_evaluated": False,
        "full_support_negative_conclusion_authorized": False,
        "direction_scoring_authorized": False,
        "direction_scoring_performed": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "local_edge_trace_does_not_cover_full_measured_kernel_support",
        "boundary": audit["boundary"],
    }
