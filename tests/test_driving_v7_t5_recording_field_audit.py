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
    provenance = report["stimulus_provenance"]
    assert provenance["fixed_generator_commit"] == (
        "b589a224493cb66bda4c55f632b213cacb082b24"
    )
    assert provenance["record_specific_stimulus_logs_locally_available"] is False
    assert provenance["generator_defaults_accepted_as_record_fields"] is False
    assert provenance["stimulus_provenance_contract_complete"] is False
    assert provenance["external_successful_indexes_linked_log_found"] is False
    assert provenance["PMC_supplement_content_retrieved_and_inspected"] is False
    assert provenance["publisher_supplements_retrieved_and_inspected"] is True
    assert provenance["publisher_supplements_contain_record_specific_log"] is False
    assert provenance["Figshare_search_endpoint_accessible"] is False
    assert provenance["global_absence_claimed"] is False
    assert provenance["historical_flash_payload_path_count"] == 8
    assert provenance["historical_flash_payloads_with_alternate_blob_versions"] == []
    assert provenance["historical_flash_payloads_restore_record_metadata"] is False


def test_recording_keys_do_not_become_biological_individuals() -> None:
    report = json.loads(REPORT.read_text())
    expected = {"Tm1": 8, "Tm2": 7, "Tm4": 7, "Tm9": 13}
    for source, upper_bound in expected.items():
        row = report["source_checks"][source]
        assert row["full_repository_recording_id_upper_bound"] == upper_bound
        assert row["biological_individual_semantics_verified"] is False
        assert row["all_required_fields_coexist_in_any_allowed_voltage_modality"] is False


def test_braun_calcium_fields_are_visible_but_not_joined_to_voltage() -> None:
    report = json.loads(REPORT.read_text())
    calcium = report["independent_incompatible_calcium_evidence"]
    assert calcium["dataset_doi"] == "10.17617/3.QE3MFT"
    assert calcium["sources"] == ["CT1", "Tm2", "Tm9"]
    assert calcium["sources_with_stable_fly_IDs"] == ["CT1", "Tm2", "Tm9"]
    assert calcium["sources_with_complete_condition_grid"] == ["CT1", "Tm2", "Tm9"]
    assert calcium["sources_with_allowed_response_unit"] == []
    assert calcium["accepted_as_voltage_recording_field_completion"] is False
    assert report["source_checks"]["Tm2"]["independent_calcium_fly_count"] == 9
    assert report["source_checks"]["Tm9"]["independent_calcium_fly_count"] == 11
    assert report["source_checks"]["CT1"]["independent_calcium_fly_count"] == 9


def test_CT1_and_downstream_gates_remain_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["source_checks"]["CT1"]["full_field_flash_all_conditions_have_voltage"] is False
    assert report["all_five_T5_sources_have_complete_coexisting_recording_fields"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
