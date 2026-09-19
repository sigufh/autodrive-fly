import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-physical-time-transfer-audit.json"


def test_t5_physical_time_transfer_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["functional_stimulus_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False
    scheduler = report["time_layers"]["stage1_scheduler"]
    assert scheduler["frame_interval_milliseconds"] == 10.0
    assert scheduler["applied_to_current_runtime"] is False
    assert scheduler["biologically_calibrated"] is False


def test_t5_physical_time_transfer_stays_closed_on_all_missing_fields() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["missing_fields"]) == {
        "v7_camera_angular_calibration",
        "Tm1_Tm2_Tm4_Tm9_membrane_like_kernels",
        "CT1_membrane_like_kernel",
        "stable_source_or_target_cell_to_MaleCNS_mapping",
        "independent_dynamic_validation_cohort",
    }
    assert report["required_fields"]["v7_physical_frame_interval"] is True
    assert report["required_fields"]["v7_physical_solver_interval"] is True
    assert report["required_fields"]["v7_camera_angular_calibration"] is False
    coordinates = report["time_layers"]["offline_stimulus_coordinate_contract"]
    assert coordinates["time_coordinates_complete"] is True
    assert coordinates["horizontal_coordinates_complete"] is True
    assert coordinates["two_dimensional_angular_calibration_complete"] is False
    assert coordinates["biologically_calibrated"] is False
    assert coordinates["external_recording_alignment_verified"] is False
    assert report["time_layers"]["T5_target_voltage"]["native_time_vectors_verified"]
    assert report["time_layers"]["T5_target_voltage"]["measures_source_types"] is False
    assert report["time_layers"]["T5_source_filters"]["raw_calcium_contract_complete"]
    assert not report["time_layers"]["T5_source_filters"]["deconvolved_contract_complete"]
    assert report["time_layers"]["T5_source_filters"]["missing_CT1"]
    flyvis = report["time_layers"]["FlyVis_source_time_constants"]
    assert flyvis["required_types_present"] is True
    assert flyvis["all_above_solver_dt"] is False
    assert flyvis["all_cross_model_IQR_stable"] is False
    assert flyvis["transferable"] is False
    assert report["current_source_temporal_identifiability"] == {
        "passing_T5_source_population_count": 0,
        "T5_source_population_denominator": 32,
        "passed": False,
    }
    assert report["T5_physical_source_transfer_ready"] is False
    assert report["authorize_physical_source_dynamics_candidate"] is False
    assert report["advance_to_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
