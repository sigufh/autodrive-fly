import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_emotion.driving.v7_t5_measured_kernel_identifiability import (
    _validate_timebase,
)

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-measured-kernel-identifiability.json"


def test_measured_kernel_precheck_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_id"] == "S1-T01"
    assert protocol["position_count"] == 15
    assert protocol["stimulus_count_per_mode"] == 120
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["runtime_modified"] is False


def test_measured_kernel_inventory_and_target_denominator_are_fixed() -> None:
    report = json.loads(REPORT.read_text())
    kernels = report["kernel_summary"]
    assert kernels["source_order"] == ["Tm1", "Tm2", "Tm4", "Tm9"]
    assert kernels["recording_id_counts"] == {
        "Tm1": 7,
        "Tm2": 5,
        "Tm4": 6,
        "Tm9": 6,
    }
    assert kernels["kernel_length"] == 499
    assert kernels["sample_interval_milliseconds"] == 10.0
    assert kernels["normalization"] == "unit_L1_per_source"
    assert report["fixed_target_denominator"] == 6719
    assert report["valid_target_count"] == 6715
    assert report["invalid_source_or_axis_target_count"] == 4


def test_measured_kernel_timebase_is_explicit_and_solver_alignment_stays_closed() -> None:
    report = json.loads(REPORT.read_text())
    timebase = report["timebase_contract"]
    assert timebase["kernel_sample_interval_milliseconds"] == 10.0
    assert timebase["stimulus_frame_interval_milliseconds"] == 10.0
    assert timebase["convolution_samples_per_stimulus_frame"] == 1
    assert timebase["probe_brain_updates_per_frame"] == 1
    assert timebase["kernel_to_stimulus_frame_alignment_verified"] is True
    assert timebase["probe_brain_update_interval_milliseconds"] is None
    assert timebase["probe_brain_update_interval_biologically_calibrated"] is False
    assert timebase["external_recording_to_probe_solver_alignment_verified"] is False
    assert set(timebase["gates"].values()) == {True}
    assert report["authorize_physical_source_dynamics_transfer"] is False


def test_measured_kernel_timebase_fails_closed_on_frame_mismatch() -> None:
    config = {
        "kernel": {
            "sample_interval_milliseconds": 10.0,
            "convolution_samples_per_stimulus_frame": 1,
        }
    }
    coordinates = {
        "offline_time_coordinate_contract_complete": True,
        "time_coordinates": {
            "frame_interval_milliseconds": 20.0,
            "applied_to_offline_v7_controlled_visual_evaluation": True,
        },
    }
    with pytest.raises(ValueError, match="kernel_sample_interval"):
        _validate_timebase(config, coordinates, {"brain_substeps_per_frame": 1})


def test_all_readouts_fail_temporal_controls_before_direction_scoring() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "summed_filtered_source_centroid_projection": (
            31.50743933011604,
            0.9996084681893145,
        ),
        "fast_pool_vs_Tm9_centroid_difference": (
            2.557479297173785,
            1.0992633306058064,
        ),
        "temporal_difference_filtered_Tm_pair_reichardt": (
            1.9212814396370625,
            0.9962811391546738,
        ),
    }
    assert set(report["candidate_results"]) == set(expected)
    for name, ratios in expected.items():
        result = report["candidate_results"][name]
        assert np.isclose(result["shuffle_to_ordered_residual_energy_ratio"], ratios[0])
        assert np.isclose(result["static_to_ordered_energy_ratio"], ratios[1])
        assert set(result["gates"].values()) == {False}
        assert result["passed"] is False
    assert report["temporal_identifiability_passed"] is False
    assert report["direction_scoring_performed"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["authorize_physical_source_dynamics_transfer"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False


def test_measured_kernel_boundary_keeps_source_and_target_roles_separate() -> None:
    boundary = json.loads(REPORT.read_text())["boundary"]
    assert boundary["measured_kernel_shape_only_not_absolute_gain"] is True
    assert boundary["kernel_timebase_bound_to_offline_stimulus_frames"] is True
    assert boundary["probe_brain_update_interval_not_biologically_calibrated"] is True
    assert boundary["source_activity_driven_only_from_R1_R6_propagation"] is True
    assert boundary["subtype_and_direction_labels_not_read_by_dynamics"] is True
    assert boundary["target_activity_injection"] is False
    assert boundary["calibration_evaluated"] is False
    assert boundary["final_evaluated"] is False
