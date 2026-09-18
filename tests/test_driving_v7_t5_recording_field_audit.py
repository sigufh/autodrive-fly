import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-recording-field-audit.json"


def test_T5_field_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_fields_are_not_joined_across_modalities() -> None:
    report = json.loads(REPORT.read_text())
    modalities = report["modalities"]
    assert modalities["aggregated_full_field_OFF_flash"]["available_field_count"] == 7
    assert modalities["raw_white_noise"]["available_field_count"] == 8
    assert modalities["raw_drifting_grating"]["available_field_count"] == 9
    assert all(not item["all_required_fields_coexist"] for item in modalities.values())
    assert report["cross_modality_union"]["accepted_as_single_recording_payload"] is False


def test_recording_keys_do_not_become_biological_individuals() -> None:
    report = json.loads(REPORT.read_text())
    expected = {"Tm1": 8, "Tm2": 7, "Tm4": 7, "Tm9": 13}
    for source, upper_bound in expected.items():
        row = report["source_checks"][source]
        assert row["full_repository_recording_id_upper_bound"] == upper_bound
        assert row["biological_individual_semantics_verified"] is False
        assert row["all_required_fields_coexist_in_any_allowed_voltage_modality"] is False


def test_CT1_and_downstream_gates_remain_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["source_checks"]["CT1"]["full_field_flash_all_conditions_have_voltage"] is False
    assert report["all_five_T5_sources_have_complete_coexisting_recording_fields"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
