import hashlib
import json
from pathlib import Path

import numpy as np

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


def test_both_readouts_fail_temporal_controls_before_direction_scoring() -> None:
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
    }
    for name, ratios in expected.items():
        result = report["candidate_results"][name]
        assert np.isclose(result["shuffle_to_ordered_residual_energy_ratio"], ratios[0])
        assert np.isclose(result["static_to_ordered_energy_ratio"], ratios[1])
        assert set(result["gates"].values()) == {False}
        assert result["passed"] is False
    assert report["temporal_identifiability_passed"] is False
    assert report["direction_scoring_performed"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False


def test_measured_kernel_boundary_keeps_source_and_target_roles_separate() -> None:
    boundary = json.loads(REPORT.read_text())["boundary"]
    assert boundary["measured_kernel_shape_only_not_absolute_gain"] is True
    assert boundary["source_activity_driven_only_from_R1_R6_propagation"] is True
    assert boundary["subtype_and_direction_labels_not_read_by_dynamics"] is True
    assert boundary["target_activity_injection"] is False
    assert boundary["calibration_evaluated"] is False
    assert boundary["final_evaluated"] is False
