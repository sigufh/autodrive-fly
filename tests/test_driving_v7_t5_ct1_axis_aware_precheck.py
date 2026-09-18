import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-ct1-axis-aware-precheck.json"


def test_t5_ct1_axis_aware_precheck_is_hash_bound_and_causal() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["parameter_fit_to_neural_response"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["fixed_target_denominator"] == 6719
    assert report["crossfit_axes_sha256"] == json.loads(
        (ROOT / "artifacts/v7-t5-ct1-crossfit-axis-audit.json").read_text()
    )["out_of_fold_predictions_sha256"]
    assert report["source_roles"]["enhancer_base"] == ["Tm9"]
    assert report["source_roles"]["slow_inhibition"] == ["CT1"]


def test_t5_ct1_axis_aware_precheck_obeys_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["joint_source_and_axis_present_count"] == 6713
    assert [item["direction_pass_count"] for item in report["ordered_candidates"]] == [
        0,
        0,
        0,
        0,
    ]
    assert [item["polarity_pass_count"] for item in report["ordered_candidates"]] == [
        8,
        8,
        8,
        8,
    ]
    assert report["control_eligible_gains"] == []
    assert report["strict_ordered_gains"] == []
    assert report["control_results"] == {}
    assert report["candidate_passed"] == bool(report["control_passing_strict_gains"])
    assert report["controls_evaluated"] == bool(report["control_eligible_gains"])
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["boundary"]["every_valid_target_axis_is_out_of_fold"]
    assert report["boundary"]["dynamics_do_not_read_subtype_or_direction_labels"]
    assert report["boundary"]["LPLC_and_vehicle_experiments_remain_frozen"]
