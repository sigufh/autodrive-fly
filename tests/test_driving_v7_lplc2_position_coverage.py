import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-lplc2-position-coverage.json"


def test_position_coverage_is_fixed_matched_and_tuning_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_ids"] == ["LPLC-T01", "LPLC-T02", "LPLC-T03"]
    assert protocol["position_count"] == 15
    assert protocol["stimulus_count"] == 141
    assert protocol["matched_noise_within_position"] is True
    assert protocol["common_background_frames"] == 2
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["runtime_modified"] is False
    candidate = report["candidate"]
    assert candidate["position_grid_selection_uses_target_or_response"] is False
    assert candidate["observability_is_functional_preference"] is False


def test_position_coverage_restores_observability_but_not_lplc2_preference() -> None:
    report = json.loads(REPORT.read_text())
    assert report["anatomy_target_counts"] == {"L": 94, "R": 91}
    observable = report["observability_consistency"]
    assert observable["passing_condition_count"] == 3
    assert observable["joint_valid_fraction"] == 1.0
    assert observable["all_condition_separated_count"] == 1902
    assert observable["all_condition_separated_fraction"] > 0.99
    assert observable["passed"] is True
    assert all(item["passed"] for item in report["mirror_summary"].values())
    radial = report["radial_population_consistency"]
    assert radial["R"]["wide_field_translation"]["passing_condition_count"] == 2
    assert all(
        not result["passed"] for mechanisms in radial.values() for result in mechanisms.values()
    )
    assert report["observability_gate_passed"] is True
    assert report["LPLC2_position_coverage_gates_passed"] is False
    anatomy = report["anatomy_assignment_control"]
    assert anatomy["post_failure_control_only"] is True
    assert anatomy["assignment"]["target_response_used_for_assignment"] is False
    assert anatomy["assignment"]["by_side"]["L"]["target_count"] == 94
    assert anatomy["assignment"]["by_side"]["R"]["target_count"] == 91
    assert anatomy["passed"] is False
    assert all(
        not result["passed"]
        for mechanisms in anatomy["population_consistency"].values()
        for result in mechanisms.values()
    )
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
