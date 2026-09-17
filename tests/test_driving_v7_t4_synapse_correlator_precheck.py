import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-synapse-correlator-precheck.json"


def test_T4_synapse_correlator_is_tuning_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["stimulus_count"] == 8
    assert report["protocol"]["candidate_count"] == 8
    assert report["protocol"]["T5_evaluated"] is False
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_T4_synapse_correlator_preserves_ON_but_has_no_bilateral_direction_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["fixed_T4_population_denominator"] == 6861
    assert report["diagnostic_target_count"] == 6749
    assert report["diagnostic_valid_target_count"] == 6749
    assert [item["direction_pass_count"] for item in report["candidates"]] == [2, 2, 2, 1] * 2
    assert all(item["polarity_pass_count"] == 8 for item in report["candidates"])
    assert all(item["bilateral_direction_subtypes"] == [] for item in report["candidates"])
    assert report["control_eligible_candidates"] == []
    assert report["advancing_candidates"] == []
    assert report["main_gate_passed"] is False


def test_T4_synapse_correlator_stop_rule_prevents_expansion() -> None:
    report = json.loads(REPORT.read_text())
    assert report["controls_evaluated"] is False
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["stop_reason"] == (
        "no_bilateral_T4_direction_subtype_passed_synapse_axis_main_gate"
    )
    assert report["boundary"]["T5_mapping_application_forbidden"] is True
    assert report["boundary"]["missing_conductance_targets_count_as_invalid"] is True
