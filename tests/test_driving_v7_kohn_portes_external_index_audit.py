import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-external-index-audit.json"


def test_external_index_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_successful_indexes_have_no_linked_numeric_payload() -> None:
    indexes = json.loads(REPORT.read_text())["audited_indexes"]
    assert indexes["Crossref"]["preprint_relation"] == [
        "10.1101/2021.04.17.440267"
    ]
    assert indexes["Crossref"]["numeric_data_links"] == []
    assert indexes["DataCite"]["related_DOI_result_count"] == 0
    assert indexes["DataCite"]["exact_title_result_count"] == 0
    assert indexes["Europe_PMC"]["has_database_cross_references"] is False


def test_inaccessible_endpoints_are_not_treated_as_absence() -> None:
    report = json.loads(REPORT.read_text())
    indexes = report["audited_indexes"]
    assert indexes["Europe_PMC"]["has_supplementary_file"] is True
    assert indexes["PMC"]["supplement_content_type"] == "pdf"
    assert indexes["PMC"]["PMC_endpoint_content_retrieved"] is False
    assert indexes["PMC"]["download_response_is_proof_of_work_HTML"] is True
    assert indexes["PMC"]["publisher_equivalent_supplements_retrieved_and_inspected"] is True
    supplements = indexes["PMC"]["publisher_supplements"]
    assert supplements["publisher_supplement_1"]["pages"] == 11
    assert supplements["publisher_supplement_2"]["pages"] == 29
    assert all(item["recording_token_count_checked"] == 35 for item in supplements.values())
    assert all(item["embedded_attachment_names"] == [] for item in supplements.values())
    assert all(not any(item["exact_record_token_counts"].values()) for item in supplements.values())
    assert indexes["Figshare"]["search_endpoint_status"] == 403
    assert indexes["Figshare"]["search_result_interpretable"] is False
    assert report["record_specific_stimulus_log_global_absence_claimed"] is False


def test_external_index_boundary_keeps_transfer_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report[
        "record_specific_stimulus_log_found_in_successfully_audited_external_indexes"
    ] is False
    assert report["authorize_recording_field_recovery"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["boundary"][
        "successful_zero_result_and_inaccessible_endpoint_are_separate"
    ] is True
