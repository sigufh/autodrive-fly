import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-danger-throttle.json"


def test_danger_throttle_is_tuning_only_and_keeps_steering_fixed() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["tuning_only"] is True
    assert protocol["calibration_evaluated"] is False
    assert protocol["external_final_evaluated"] is False
    assert protocol["steering_policy_changed"] is False
    assert protocol["runtime_modified"] is False


def test_danger_slowing_does_not_fix_nested_navigation_limit() -> None:
    report = json.loads(REPORT.read_text())
    summaries = {item["candidate"]["name"]: item for item in report["candidate_summaries"]}
    assert [
        summaries["constant"]["success_count"],
        summaries["constant"]["total_obstacles_passed"],
    ] == [4, 50]
    for name in ("linear_moderate", "linear_strong", "quadratic", "threshold_stop"):
        assert summaries[name]["success_count"] == 0
        assert summaries[name]["timeout_count"] == 6
    assert report["selected_candidate"] == {"name": "constant", "formula": "base"}
    assert report["tuning_passed"] is False
    assert report["negative_control_reduces_success"] is False
    assert report["advance_to_neural_cv"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_navigation_release"] is False
