import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-gonzalez-suarez-mi4-evidence-audit.json"


def test_gonzalez_suarez_audit_is_hash_and_revision_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["paper"]["doi"] == "10.1016/j.cub.2022.06.075"
    assert report["repository_evidence"]["local_and_remote_head"] == (
        "100bb2f52cb9628477c3883e4b17774b0b244e67"
    )


def test_independent_Mi4_evidence_is_calcium_type_average_not_voltage() -> None:
    report = json.loads(REPORT.read_text())
    paper = report["paper_measurement_evidence"]
    repository = report["repository_evidence"]
    assert paper["Mi4_GCaMP6f_fly_count"] == 15
    assert paper["Mi4_measurement_modality"] == "two_photon_GCaMP6f_calcium_imaging"
    assert paper["ArcLight_voltage_source_types"] == ["Mi1", "Tm3"]
    assert paper["Mi4_experimental_membrane_voltage_measured"] is False
    assert paper["C3_measured"] is False
    assert repository["Mi4_type_average_filter_available"] is True
    assert repository["C3_filter_available"] is False
    assert repository["individual_cell_axis_available"] is False
    assert repository["stable_biological_individual_ID_available"] is False
    assert repository["filter_sample_interval_seconds"] == 1 / 30


def test_unretrieved_supplement_is_not_treated_as_absence_and_gates_stay_closed() -> None:
    report = json.loads(REPORT.read_text())
    indexes = report["public_indexes"]
    assert indexes["PMC_supplement_content_retrieved"] is False
    assert indexes["PMC_supplement_response_is_proof_of_work_HTML"] is True
    assert indexes["bioRxiv_full_document_with_supplement_retrieved"] is True
    preprint = report["preprint_evidence"]
    assert preprint["pages"] == 61
    assert preprint["embedded_attachment_names"] == []
    assert preprint["data_availability"] == "upon_request"
    assert preprint["individual_flies_are_statistical_units"] is True
    assert preprint["public_individual_fly_numeric_payload_attached"] is False
    assert preprint["Mi4_GCaMP6f_fly_count"] == 15
    assert preprint["ArcLight_voltage_source_types"] == ["Mi1", "Tm3"]
    assert report["independent_Mi4_calcium_type_average_available"] is True
    assert report["independent_Mi4_experimental_membrane_voltage_available"] is False
    assert report["independent_Mi4_individual_numeric_dynamics_available"] is False
    assert report["independent_C3_source_dynamics_available"] is False
    assert report["authorize_Mi4_C3_voltage_transfer"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["boundary"]["inaccessible_supplement_content_is_not_classified_as_absent"] is True
