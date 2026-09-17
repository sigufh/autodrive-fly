import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-continuous-moment-precheck.json"


def test_continuous_moment_precheck_is_tuning_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["position_count"] == 15
    assert report["protocol"]["stimulus_count"] == 60
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_label_free_structural_axis_passes_all_T5_populations_and_mirrors() -> None:
    report = json.loads(REPORT.read_text())
    axis = report["structural_axis"]
    assert axis["all_population_and_mirror_gates_passed"] is True
    assert all(item["passed"] for item in axis["population_results"].values())
    assert all(item["valid_axis_fraction"] > 0.998 for item in axis["population_results"].values())
    assert all(
        item["median_expected_cosine"] > 0.72
        for item in axis["population_results"].values()
    )
    assert all(item["passed"] for item in axis["mirror"].values())


def test_continuous_moment_activity_fails_and_stops_before_controls() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["ordered_activity_moment"]) == {
        "signed_mean",
        "maximum",
        "positive_mean",
        "terminal_sum",
    }
    assert all(
        result["direction_pass_count"] == 0
        for result in report["ordered_activity_moment"].values()
    )
    assert all(
        result["bilateral_direction_subtypes"] == []
        for result in report["ordered_activity_moment"].values()
    )
    assert report["control_eligibility_gate_passed"] is False
    assert report["controls_evaluated"] is False
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_three_tuning_conditions"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
