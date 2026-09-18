import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-behnia-t4-fast-source-availability-audit.json"


def test_availability_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_machine_readable_indexes_expose_no_numeric_payload() -> None:
    report = json.loads(REPORT.read_text())
    indexes = report["audited_public_indexes"]
    assert indexes["europe_PMC"] == {
        "exact_record_count": 1,
        "is_open_access": False,
        "has_supplementary_files": False,
        "has_database_cross_references": False,
    }
    assert indexes["Crossref"]["relation_count"] == 0
    assert indexes["Crossref"]["numeric_data_links"] == []
    assert indexes["DataCite"]["related_dataset_count"] == 0
    assert indexes["Zenodo"]["exact_title_result_count"] == 0
    assert indexes["GitHub"]["exact_title_numeric_file_hits"] == []
    assert indexes["GitHub"]["exact_DOI_neuronal_payload_hits"] == []
    assert report["local_numeric_trace_payload_found_in_audited_indexes"] is False


def test_absence_in_audited_indexes_does_not_claim_global_nonexistence() -> None:
    report = json.loads(REPORT.read_text())
    assert report["boundary"][
        "audited_public_indexes_only_not_proof_of_global_nonexistence"
    ] is True
    assert report["boundary"]["no_curve_digitization"] is True
    assert report["independent_Mi1_Tm3_numeric_transfer_authorized"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
