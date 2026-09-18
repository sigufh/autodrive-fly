import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-ct1-multiplicative-precheck.json"


def test_t5_ct1_multiplicative_is_hash_bound_and_source_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["source_roles"]["enhancer_base"] == ["Tm9"]
    assert report["source_roles"]["slow_inhibition"] == ["CT1"]
    assert report["boundary"]["T5_axis_calibration_is_authorization_only"] is True
    assert report["boundary"]["T5_axis_transform_consumed_by_scalar_dynamics"] is False


def test_t5_ct1_multiplicative_obeys_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["fixed_target_denominator"] == 6719
    assert [item["direction_pass_count"] for item in report["ordered_candidates"]] == [
        0,
        0,
        0,
        2,
    ]
    assert [item["polarity_pass_count"] for item in report["ordered_candidates"]] == [
        8,
        8,
        8,
        8,
    ]
    assert report["ordered_precheck_passed"] is False
    assert report["candidate_passed"] is False
    assert report["strict_ordered_gains"] == []
    assert report["control_eligible_gains"] == [8.0]
    assert report["controls_evaluated"] is True
    assert set(report["control_results"]) == {"temporal_shuffle", "static_sham"}
    assert all(set(results) == {"8.0"} for results in report["control_results"].values())
    assert all(
        results["8.0"]["direction_pass_count"] == 0
        for results in report["control_results"].values()
    )
    assert report["control_passing_strict_gains"] == []
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["boundary"]["Tm9_and_CT1_not_merged"] is True
