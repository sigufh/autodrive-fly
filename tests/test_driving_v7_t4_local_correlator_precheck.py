import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-local-correlator-precheck.json"


def test_local_correlator_precheck_reuses_frozen_candidates() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["candidate_count"] == 2
    assert report["protocol"]["condition_id"] == "LPLC-T01"
    assert report["protocol"]["parameter_fit"] is False
    assert set(report["candidates"]) == {
        "additive:gain=0.5:lag=1",
        "multiplicative:gain=4:lag=1",
    }


def test_local_correlator_stops_after_zero_t4_populations_pass() -> None:
    report = json.loads(REPORT.read_text())
    assert all(result["T4_direction_pass_count"] == 0 for result in report["candidates"].values())
    assert all(result["T4_polarity_pass_count"] == 0 for result in report["candidates"].values())
    assert report["advancing_candidates"] == []
    assert report["local_correlator_gate_passed"] is False
    assert report["expand_to_three_conditions"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
