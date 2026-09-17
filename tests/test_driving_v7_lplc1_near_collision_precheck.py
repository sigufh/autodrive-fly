import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-lplc1-near-collision-precheck.json"


def test_lplc1_precheck_has_matched_mirrored_stimuli_and_fixed_boundaries() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "LPLC-T01"
    assert report["protocol"]["stimulus_count"] == 10
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["stimulus_geometry_gate_passed"] is True
    assert report["boundary"]["LPLC1_specific_stimuli"] is True
    assert report["boundary"]["shared_generic_looming_rule"] is False
    assert report["boundary"]["pixel_sizes_not_claimed_as_visual_degrees"] is True


def test_lplc1_has_direct_input_but_no_near_collision_response_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["direct_source_coverage"]["L"][
        "targets_with_direct_T4_T5_count"
    ] == 67
    assert report["direct_source_coverage"]["R"][
        "targets_with_direct_T4_T5_count"
    ] == 66
    assert report["direct_source_coverage"]["L"][
        "located_direct_T4_T5_weight_fraction"
    ] == 1.0
    assert report["direct_source_coverage"]["R"][
        "located_direct_T4_T5_weight_fraction"
    ] == 1.0
    for result in report["per_side"].values():
        assert not any(score["passed"] for score in result.values())
    assert report["per_side"]["L"]["near_vs_miss"]["median_signed_contrast"] < 0.001
    assert report["per_side"]["R"]["near_vs_miss"]["median_signed_contrast"] < 0.002
    assert report["per_side"]["L"]["stationary_vs_rotating_background"][
        "median_signed_contrast"
    ] < -0.48
    assert report["per_side"]["R"]["stationary_vs_rotating_background"][
        "median_signed_contrast"
    ] < -0.48
    assert report["mirror_gate_passed"] is True
    assert report["response_gate_passed"] is False
    assert report["LPLC1_near_collision_precheck_passed"] is False
    assert report["expand_to_three_conditions"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
