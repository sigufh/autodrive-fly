import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-pair-lag-audit.json"


def test_T4_pair_lag_audit_is_frozen_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["lag_count"] == 4
    assert report["protocol"]["candidate_count"] == 12
    assert report["protocol"]["stimulus_count"] == 120
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_no_frozen_lag_recovers_T4_direction_or_polarity() -> None:
    report = json.loads(REPORT.read_text())
    assert report["lags_frames"] == [1, 2, 3, 4]
    assert set(report["temporal_reductions"]) == {"signed_mean", "maximum", "positive_mean"}
    assert all(item["direction_pass_count"] == 0 for item in report["candidates"].values())
    assert all(item["polarity_pass_count"] == 0 for item in report["candidates"].values())
    assert all(
        item["bilateral_direction_subtypes"] == []
        for item in report["candidates"].values()
    )
    assert report["advancing_candidates"] == []
    assert report["lag_audit_passed"] is False


def test_T4_pair_lag_stop_rule_prevents_expansion() -> None:
    report = json.loads(REPORT.read_text())
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["stop_reason"] == (
        "no_frozen_lag_produced_any_bilateral_direction_selective_T4_subtype"
    )
    assert report["boundary"]["lag_range_previously_frozen"] is True
    assert report["boundary"]["no_additional_formula_or_gain_search"] is True
