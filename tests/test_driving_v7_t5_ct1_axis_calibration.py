import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-ct1-axis-calibration.json"


def test_t5_ct1_axis_calibration_is_hash_bound_and_disjoint() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["fit_uses_neural_response"] is False
    assert report["dataset"]["fit_held_out_body_id_overlap"] == 0
    assert report["dataset"]["fit_count"] + report["dataset"]["held_out_count"] == report[
        "dataset"
    ]["valid_target_count"]


def test_t5_ct1_axis_calibration_passes_held_out_structure_only() -> None:
    report = json.loads(REPORT.read_text())
    assert report["held_out_accuracy_mean"] > 0.87
    assert report["held_out_median_angle_error_mean_degrees"] < 16.0
    assert report["held_out_accuracy_mean"] > report["random_orthogonal_baseline"][
        "held_out_accuracy_p95"
    ]
    assert set(report["gates"].values()) == {True}
    assert report["T5_CT1_axis_calibration_passed"] is True
    assert report["authorize_T5_single_condition_precheck"] is True
    assert report["advance_to_calibration_stimulus"] is False
    assert report["boundary"]["no_T4_mapping_reused"] is True
    assert report["boundary"]["LPLC_and_vehicle_experiments_remain_frozen"] is True
