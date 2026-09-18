import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-synapse-crossfit-precheck.json"


def test_t4_crossfit_precheck_is_hash_bound_and_source_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["parameter_fit_to_neural_response"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["fixed_T4_population_denominator"] == 6861


def test_t4_crossfit_precheck_obeys_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["diagnostic_target_count"] == 6749
    assert report["diagnostic_valid_target_count"] == 6749
    assert [item["direction_pass_count"] for item in report["ordered_candidates"]] == [2] * 8
    assert [item["polarity_pass_count"] for item in report["ordered_candidates"]] == [8] * 8
    assert report["control_eligible_candidates"] == []
    assert report["strict_ordered_candidates"] == []
    assert report["control_results"] == {}
    assert report["candidate_passed"] == bool(report["control_passing_candidates"])
    assert report["controls_evaluated"] == bool(report["control_eligible_candidates"])
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["boundary"]["every_valid_target_axis_is_out_of_fold"]
    assert report["boundary"]["dynamics_do_not_read_subtype_or_direction_labels"]
