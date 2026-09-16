import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4t5-local-edge-backends.json"


def test_existing_backend_ab_is_frozen_single_condition_and_non_advancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_id"] == "LPLC-T01"
    assert protocol["candidate_count"] == 6
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["runtime_modified"] is False


def test_no_existing_backend_recovers_local_direction_selectivity() -> None:
    report = json.loads(REPORT.read_text())
    assert {result["direction_pass_count"] for result in report["candidates"].values()} == {0}
    assert {result["polarity_pass_count"] for result in report["candidates"].values()} == {0, 8}
    assert report["advancing_candidates"] == []
    assert report["existing_backend_reuse_gate_passed"] is False
    assert report["expand_to_three_conditions"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
