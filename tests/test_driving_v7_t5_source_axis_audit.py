import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-source-axis-audit.json"


def test_t5_source_axis_audit_is_structure_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["target_count"] == 6719
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["sign_search"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["boundary"]["source_axis_uses_labels"] is False
    assert report["boundary"]["labels_used_for_posthoc_scoring_only"] is True


def test_t5_source_axis_is_vertical_only_and_not_authorized() -> None:
    report = json.loads(REPORT.read_text())
    summaries = report["population_summaries"]
    assert all(item["valid_axis_fraction"] > 0.998 for item in summaries.values())
    assert [name for name, item in summaries.items() if item["passed"]] == [
        "T5c_L",
        "T5c_R",
        "T5d_L",
        "T5d_R",
    ]
    assert summaries["T5a_L"]["median_expected_cosine"] < -0.7
    assert summaries["T5a_R"]["median_expected_cosine"] < -0.7
    assert summaries["T5b_L"]["median_expected_cosine"] < -0.7
    assert summaries["T5b_R"]["median_expected_cosine"] < -0.7
    assert [name for name, item in report["population_mirror"].items() if item["passed"]] == [
        "c",
        "d",
    ]
    assert report["within_eye_opponent_axis_pairs"]["T5a_L<->T5b_L"][
        "opposite_angle_degrees"
    ] < 3.0
    assert report["within_eye_opponent_axis_pairs"]["T5c_R<->T5d_R"][
        "opposite_angle_degrees"
    ] < 7.0
    assert report["strict_source_axis_gate_passed"] is False
    assert report["authorize_dynamic_axis_candidate"] is False
    calibration = report["independent_T4_axis_calibration"]
    assert calibration["T5_used_for_fit_or_model_selection"] is False
    assert calibration["zero_shot_T5_accuracy"] > 0.83
    assert calibration["held_out_T4_accuracy"] < 0.47
    assert calibration["cross_eye_maximum_error_degrees"] > 36.0
    assert calibration["axis_calibration_passed"] is False
    assert calibration["transform_application_authorized"] is False
