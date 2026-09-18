import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_t5_ct1_axis_aware_precheck import _axis_pair_component

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-ct1-axis-aware-antisymmetric-precheck.json"


def test_axis_pair_component_reduces_to_antisymmetric_temporal_term() -> None:
    fast_position = 3.0
    ct1_position = 1.0
    current = {
        "fast_mass": np.asarray([5.0]),
        "CT1_mass": np.asarray([7.0]),
        "fast_x": np.asarray([fast_position * 5.0]),
        "CT1_x": np.asarray([ct1_position * 7.0]),
    }
    previous = {
        "fast_mass": np.asarray([2.0]),
        "CT1_mass": np.asarray([11.0]),
        "fast_x": np.asarray([fast_position * 2.0]),
        "CT1_x": np.asarray([ct1_position * 11.0]),
    }
    value, _ = _axis_pair_component(current, previous, "x", 1.0)
    expected = (fast_position - ct1_position) * (5.0 * 11.0 - 7.0 * 2.0)
    assert value[0] == expected


def test_t5_ct1_axis_aware_antisymmetric_is_hash_bound_and_obeys_gate() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["parameter_fit_to_neural_response"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["fixed_target_denominator"] == 6719
    assert report["joint_source_and_axis_present_count"] == 6713
    assert [item["direction_pass_count"] for item in report["ordered_candidates"]] == [
        0,
        0,
        0,
        0,
    ]
    assert [item["polarity_pass_count"] for item in report["ordered_candidates"]] == [
        8,
        8,
        8,
        8,
    ]
    assert report["control_eligible_gains"] == []
    assert report["control_results"] == {}
    assert report["candidate_passed"] == bool(report["control_passing_strict_gains"])
    assert report["controls_evaluated"] == bool(report["control_eligible_gains"])
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["boundary"]["corrects_prior_spatial_reverse_order_sign"]
    assert report["boundary"]["every_valid_target_axis_is_out_of_fold"]
    assert report["boundary"]["dynamics_do_not_read_subtype_or_direction_labels"]
