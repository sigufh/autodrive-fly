import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-stimulus-provenance-audit.json"


def test_stimulus_provenance_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_fixed_generator_does_not_replace_record_specific_logs() -> None:
    report = json.loads(REPORT.read_text())
    repository = report["stimulus_repository"]
    assert repository["commit"] == "b589a224493cb66bda4c55f632b213cacb082b24"
    assert repository["archive"]["actual_file_count"] == 85
    assert repository["white_noise_uses_unseeded_random_generation_and_shuffle"] is True
    assert (
        repository[
            "drifting_grating_generator_exposes_direction_center_radius_and_frequency"
        ]
        is True
    )
    raw = report["raw_record_summary"]
    assert raw["white_noise_record_count"] == 25
    assert raw["drifting_grating_record_count"] == 26
    assert raw["record_specific_stimulus_logs_locally_available"] is False
    assert raw["record_specific_stimulus_log_found_in_audited_public_history"] is False
    assert raw["audited_public_history_branch_count"] == 22
    assert raw["audited_public_history_commit_count"] == 447
    assert raw["external_successful_indexes_linked_log_found"] is False
    assert raw["PMC_supplement_content_retrieved_and_inspected"] is False
    assert raw["publisher_supplements_retrieved_and_inspected"] is True
    assert raw["publisher_supplements_contain_record_specific_log"] is False
    assert raw["Figshare_search_endpoint_accessible"] is False
    assert raw["global_absence_claimed"] is False
    assert raw["historical_flash_payload_path_count"] == 8
    assert raw["historical_flash_payloads_with_alternate_blob_versions"] == []
    assert raw["historical_flash_payloads_restore_record_metadata"] is False
    assert all(
        item["unique_blob_count"] == 1
        and item["commit_presence_count"] == 10
        and item["current_blob_is_only_historical_version"] is True
        for item in raw["historical_flash_payloads"].values()
    )


def test_missing_fields_are_not_filled_from_generator_defaults() -> None:
    report = json.loads(REPORT.read_text())
    status = report["recording_field_status"]
    assert status["raw_white_noise"]["stimulus_angular_position_degrees"]["available"] is False
    assert status["raw_drifting_grating"]["stimulus_direction"]["available"] is False
    assert status["raw_drifting_grating"]["stimulus_angular_position_degrees"]["available"] is False
    assert (
        status["raw_drifting_grating"][
            "stimulus_angular_speed_degrees_per_second"
        ]["available"]
        is True
    )
    assert all(not status[name]["baseline_window_seconds"]["available"] for name in status)


def test_stimulus_provenance_and_downstream_gates_remain_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["stimulus_provenance_contract_complete"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"]["generator_capability_is_not_record_specific_provenance"] is True
