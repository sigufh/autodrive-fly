import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-normalized-correlator.json"


def test_normalized_t4_correlator_is_tuning_only_and_shared() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_ids"] == ["S1-T01", "S1-T02", "S1-T03"]
    assert protocol["candidate_count"] == 12
    assert protocol["global_gain_shared_by_all_T4_targets"] is True
    assert protocol["subtype_labels_used_by_dynamics"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["external_final_evaluated"] is False


def test_normalized_t4_correlator_improves_but_does_not_pass() -> None:
    report = json.loads(REPORT.read_text())
    assert report["selected_candidate"] == {
        "mode": "additive",
        "gain": 0.5,
        "passed_gate_count": 16,
    }
    assert max(item["passed_gate_count"] for item in report["candidates"]) == 16
    assert [
        (item["mode"], item["gain"], item["lag_substeps"], item["passed_gate_count"])
        for item in report["lag_followup_candidates"]
    ] == [
        ("additive", 0.5, 1, 16),
        ("additive", 0.5, 2, 14),
        ("additive", 0.5, 3, 13),
        ("additive", 0.5, 4, 14),
        ("multiplicative", 4.0, 1, 16),
        ("multiplicative", 4.0, 2, 15),
        ("multiplicative", 4.0, 3, 15),
        ("multiplicative", 4.0, 4, 16),
    ]
    assert report["selected_lag_followup"] == {
        "mode": "additive",
        "gain": 0.5,
        "lag_substeps": 1,
        "passed_gate_count": 16,
    }
    assert report["tuning_passed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_navigation_release"] is False
