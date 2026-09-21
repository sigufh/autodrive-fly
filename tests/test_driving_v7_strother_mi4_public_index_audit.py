import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-strother-mi4-public-index-audit.json"


def test_strother_index_audit_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["paper"]["doi"] == "10.1073/pnas.1703090115"


def test_successful_indexes_have_no_linked_Mi4_numeric_trace() -> None:
    report = json.loads(REPORT.read_text())
    indexes = report["audited_indexes"]
    assert indexes["Crossref_relation_count"] == 0
    assert indexes["Crossref_numeric_data_links"] == []
    assert indexes["DataCite_related_result_count"] == 0
    assert indexes["DataCite_exact_title_result_count"] == 0
    assert indexes["Zenodo_exact_DOI_result_count"] == 0
    assert indexes["Zenodo_exact_title_result_count"] == 0
    assert indexes["GitHub_exact_DOI_repository_count"] == 0
    assert indexes["GitHub_exact_title_repository_count"] == 0
    assert report["local_numeric_Mi4_trace_payload_found_in_successful_indexes"] is False


def test_inaccessible_figshare_is_not_absence_and_gates_stay_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["audited_indexes"]["Figshare_search_status"] == 403
    assert report["audited_indexes"]["Figshare_search_interpretable"] is False
    assert report["global_absence_claimed"] is False
    assert report["Mi4_measurement_modality"] == "deltaF_over_F"
    assert report["authorize_Mi4_source_dynamics_transfer"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["boundary"]["figshare_403_is_not_proof_of_absence"] is True
