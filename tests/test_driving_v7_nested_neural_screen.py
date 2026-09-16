import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-nested-neural-screen.json"


def test_nested_neural_screen_is_tuning_only_hash_bound_and_nonadvancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_ids"] == ["S1-T01", "S1-T02", "S1-T03"]
    assert report["protocol"]["stimulus_count"] == 69
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["candidate"]["receptor_count"] == 1914
    assert report["strict_neural_gates_passed"] is False
    assert report["calibration_evaluated"] is False
    assert report["final_evaluated"] is False
    assert report["advance_to_visual_gate"] is False


def test_nested_neural_screen_preserves_mirror_but_exposes_response_failures() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "S1-T01": (3, 3, 0),
        "S1-T02": (3, 2, 0),
        "S1-T03": (3, 0, 0),
    }
    for condition, counts in expected.items():
        item = report["summaries"][condition]
        assert (
            item["direction_pass_count"],
            item["polarity_pass_count"],
            item["looming_pass_count"],
        ) == counts
        assert item["mirror_pass_count"] == item["mirror_total"] == 18
        assert item["passed"] is False
    aggregate = report["summaries"]["aggregate"]
    assert aggregate["direction_pass_count"] == 3
    assert aggregate["polarity_pass_count"] == 3
    assert aggregate["looming_pass_count"] == 0
    assert aggregate["mirror_pass_count"] == aggregate["mirror_total"] == 54
