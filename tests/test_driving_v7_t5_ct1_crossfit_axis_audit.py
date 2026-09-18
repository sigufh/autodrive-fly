import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-ct1-crossfit-axis-audit.json"


def test_t5_ct1_crossfit_axis_is_hash_bound_and_leakage_free() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["fit_uses_neural_response"] is False
    assert report["protocol"]["functional_stimulus_evaluated"] is False
    assert report["fixed_target_denominator"] == 6719
    assert report["valid_target_count"] == 6713
    assert sum(report["fold_counts"].values()) == report["valid_target_count"]
    assert all(
        item["body_id_overlap"] == 0
        for eye in report["fit_application_counts"].values()
        for item in eye.values()
    )
    assert report["out_of_fold_overall"]["accuracy"] > 0.88
    assert report["out_of_fold_overall"]["median_angle_error_degrees"] < 15.0


def test_t5_ct1_crossfit_axis_obeys_structural_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert len(report["out_of_fold_by_population"]) == 8
    assert report["T5_CT1_crossfit_axis_passed"] == all(report["gates"].values())
    assert report["T5_CT1_crossfit_axis_passed"] is True
    assert report["authorize_axis_aware_single_condition_precheck"] == report[
        "T5_CT1_crossfit_axis_passed"
    ]
    assert report["advance_to_functional_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["boundary"]["transform_selection_does_not_read_subtype_or_direction"]
    assert report["boundary"]["LPLC_and_vehicle_experiments_remain_frozen"]
