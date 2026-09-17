import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-synapse-antisymmetric-precheck.json"


def test_T4_synapse_antisymmetric_candidate_is_frozen_and_T4_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["initial_condition_id"] == "S1-T01"
    assert report["protocol"]["candidate_count"] == 8
    assert report["protocol"]["T5_evaluated"] is False
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_T4_synapse_antisymmetric_main_gate_fails_without_changing_denominator() -> None:
    report = json.loads(REPORT.read_text())
    assert report["fixed_T4_population_denominator"] == 6861
    assert report["diagnostic_target_count"] == 6749
    assert report["diagnostic_valid_target_count"] == 6749
    assert [item["direction_pass_count"] for item in report["ordered_candidates"]] == [
        2
    ] * 8
    assert all(item["polarity_pass_count"] == 8 for item in report["ordered_candidates"])
    assert all(
        item["bilateral_direction_subtypes"] == []
        for item in report["ordered_candidates"]
    )
    assert report["control_eligible_candidates"] == []
    assert report["strict_ordered_candidates"] == []


def test_T4_synapse_antisymmetric_stop_rule_prevents_expansion() -> None:
    report = json.loads(REPORT.read_text())
    assert report["controls_evaluated"] is False
    assert report["control_results"] == {}
    assert report["three_condition_evaluation_performed"] is False
    assert list(report["tuning_condition_results"]) == ["S1-T01"]
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["stop_reason"] == (
        "no_bilateral_T4_direction_subtype_passed_antisymmetric_main_gate"
    )
    assert report["boundary"]["T5_mapping_application_forbidden"] is True
