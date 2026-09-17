import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-lamina-split.json"


def test_t5_lamina_split_is_tuning_only_and_non_advancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_ids"] == ["S1-T01", "S1-T02", "S1-T03"]
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["boundary"]["tuning_only"] is True
    assert report["boundary"]["labels_changed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False


def test_lamina_split_recovers_off_polarity_but_not_t5_direction() -> None:
    report = json.loads(REPORT.read_text())
    consistency = report["population_consistency"]
    assert list(consistency) == [
        "T5a_L",
        "T5a_R",
        "T5b_L",
        "T5b_R",
        "T5c_L",
        "T5c_R",
        "T5d_L",
        "T5d_R",
    ]
    assert [item["polarity"]["passing_condition_count"] for item in consistency.values()] == [
        3, 3, 3, 3, 3, 3, 3, 3
    ]
    assert [item["direction"]["passing_condition_count"] for item in consistency.values()] == [
        1, 0, 0, 0, 2, 2, 0, 0
    ]
    assert report["mirror_gate_passed"] is True
    assert report["strict_T5_gates_passed"] is False
