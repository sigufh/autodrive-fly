import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-synapse-axis-calibration.json"


def test_synapse_axis_calibration_is_hash_bound_and_split_safe() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["T5_used_for_fit_or_model_selection"] is False
    assert report["protocol"]["random_orthogonal_baseline_count"] == 256
    assert report["dataset"]["fit_held_out_body_id_overlap"] == 0
    assert report["dataset"]["T4_fit_count"] > 3400
    assert report["dataset"]["T4_held_out_count"] > 3400
    assert report["dataset"]["T5_zero_shot_count"] == 6718


def test_synapse_axis_passes_held_out_T4_but_not_zero_shot_T5() -> None:
    report = json.loads(REPORT.read_text())
    assert report["held_out_T4"]["accuracy"] > 0.96
    assert report["held_out_T4"]["median_angle_error_degrees"] < 14.0
    assert report["cross_eye_mirror"]["maximum_T4_degrees"] < 2.0
    assert all(report["preregistered_T4_gates"].values())
    assert report["T4_synapse_axis_calibration_passed"] is True
    assert report["zero_shot_T5"]["accuracy"] < 0.11
    assert report["zero_shot_T5"]["median_angle_error_degrees"] > 139.0
    assert report["zero_shot_T5_mapping_passed"] is False


def test_synapse_axis_only_authorizes_T4_tuning_precheck() -> None:
    report = json.loads(REPORT.read_text())
    assert report["authorize_T4_single_condition_precheck"] is True
    assert report["authorize_T5_mapping_application"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["boundary"]["failed_zero_shot_T5_forbids_T5_mapping_application"] is True
