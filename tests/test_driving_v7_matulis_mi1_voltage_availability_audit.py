import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-matulis-mi1-voltage-availability-audit.json"


def test_matulis_Mi1_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_independent_Mi1_current_clamp_protocol_is_preserved() -> None:
    report = json.loads(REPORT.read_text())
    measurement = report["measurement"]
    assert measurement["source_type"] == "Mi1"
    assert measurement["modality"] == "in_vivo_whole_cell_current_clamp"
    assert measurement["sample_rate_hz"] == 10000
    assert measurement["stochastic_stimulus_update_rate_hz"] == 120
    assert measurement["epoch_duration_seconds"] == 5
    assert measurement["reported_cell_count"] == 3
    assert measurement["reported_fly_count"] == 3
    assert report["transfer_gates"]["allowed_response_unit_published"] is True


def test_public_attachments_are_documents_not_numeric_voltage() -> None:
    report = json.loads(REPORT.read_text())
    attachments = report["public_attachment_inventory"]
    assert attachments["BioStudies_accession"] == "S-EPMC7003801"
    assert attachments["numeric_data_file_count"] == 0
    assert attachments["embedded_object_names"] == [
        "word/embeddings/oleObject1.bin"
    ]
    assert attachments["embedded_object_classification"] == "Adobe_Photoshop_image"
    assert attachments["supplement_PDF_page_count"] == 14
    assert attachments["supplement_PDF_attachment_names"] == []
    availability = report["data_and_code_availability"]
    assert availability["public_repository_deposit_declared"] is False
    assert availability["available_upon_request_only"] is True
    assert availability["local_numeric_voltage_payload_verified"] is False


def test_upon_request_Mi1_data_does_not_open_any_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["transfer_gates"]["public_numeric_voltage_payload_available"] is False
    assert report["transfer_gates"][
        "stable_pseudonymous_biological_individual_ids_available"
    ] is False
    assert report["independent_Mi1_voltage_transfer_authorized"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False


def test_successful_external_indexes_have_no_linked_numeric_payload() -> None:
    report = json.loads(REPORT.read_text())
    indexes = report["audited_external_indexes"]
    assert indexes["Crossref_relation_count"] == 0
    assert indexes["Crossref_numeric_data_links"] == []
    assert indexes["DataCite_related_DOI_count"] == 0
    assert indexes["DataCite_exact_title_count"] == 0
    assert indexes["Dryad_DOI_count"] == 0
    assert indexes["Zenodo_exact_title_count"] == 0
    assert indexes["Zenodo_related_DOI_count"] == 0
    assert indexes["public_numeric_voltage_payload_found"] is False
    assert indexes["claim_scope"] == (
        "bounded_successful_indexes_not_global_nonexistence"
    )
