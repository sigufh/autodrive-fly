import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-source-pair-precheck.json"


def test_t5_source_pair_precheck_is_tuning_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["position_count"] == 15
    assert report["protocol"]["stimulus_count"] == 120
    assert report["protocol"]["candidate_count"] == 6
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_t5_source_pair_preserves_off_polarity_but_has_no_direction_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["source_coverage"]["target_count"] == 6719
    assert report["source_coverage"]["joint_present_count"] == 6718
    assert all(item["direction_pass_count"] == 0 for item in report["candidates"])
    assert all(item["polarity_pass_count"] == 8 for item in report["candidates"])
    assert all(item["bilateral_direction_subtypes"] == [] for item in report["candidates"])
    assert report["advancing_gains"] == []
    assert report["ordered_precheck_passed"] is False
    assert report["control_eligibility_gate_passed"] is False


def test_t5_source_pair_stop_rule_prevents_expansion() -> None:
    report = json.loads(REPORT.read_text())
    assert report["controls_evaluated"] is False
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_three_tuning_conditions"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["stop_reason"] == (
        "no_global_gain_produced_any_bilateral_direction_selective_T5_subtype"
    )
    boundary = report["boundary"]
    assert boundary["source_pair_screen_may_not_authorize_validation"] is True
    assert boundary["dynamics_do_not_read_subtype_or_direction_labels"] is True
    assert boundary["fixed_whole_population_denominators"] is True
    assert boundary["labels_changed"] is False
    assert boundary["thresholds_changed"] is False
