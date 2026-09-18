import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-recording-field-audit.json"


def test_T4_field_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_verified_payload_supports_ten_of_fifteen_fields() -> None:
    report = json.loads(REPORT.read_text())
    assert report["required_field_count"] == 15
    assert report["available_field_count"] == 10
    assert report["missing_fields"] == [
        "cohort_role",
        "stimulus_id",
        "stimulus_direction",
        "stimulus_angular_position_degrees",
        "baseline_window_seconds",
    ]
    assert report["all_required_recording_fields_available"] is False


def test_PD_ND_synthesis_and_analysis_baseline_do_not_count_as_source_fields() -> None:
    report = json.loads(REPORT.read_text())
    assert report["array_has_direction_axis"] is False
    assert report["notebook_evidence"][
        "PD_ND_labels_apply_after_source_average_and_synthetic_shift"
    ] is True
    assert report["field_status"]["stimulus_direction"]["available"] is False
    assert report["field_status"]["baseline_window_seconds"]["available"] is False
    assert report["boundary"]["baseline_window_is_post_hoc_analysis_protocol"] is True


def test_incomplete_fields_keep_all_downstream_gates_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["complete_stimulus_and_baseline_fields_on_allowed_payload"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
