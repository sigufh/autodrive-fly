import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-ketkar-2019-source-data-attachment-audit.json"


def test_ketkar_attachment_audit_is_code_hash_and_raw_payload_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_attachment_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["paper"]["doi"] == "10.7554/eLife.49373"
    assert report["attachment_count"] == 11
    assert all(
        item["official_url"].startswith(
            "https://cdn.elifesciences.org/articles/49373/elife-49373-"
        )
        for item in report["attachments"].values()
    )


def test_ketkar_attachments_are_valid_summary_tables_without_embedded_payloads() -> None:
    report = json.loads(REPORT.read_text())
    assert report["all_attachments_valid_docx_zip"] is True
    assert report["all_attachments_describe_mean_plus_minus_sem_tables"] is True
    assert report["embedded_object_count"] == 0
    assert report["Mi1_Tm3_GCaMP_summary_evidence_found"] is True
    assert report["aggregate_term_hits"]["Mi1"] == 33
    assert report["aggregate_term_hits"]["Tm3"] == 32
    assert report["aggregate_term_hits"]["GCaMP"] == 38


def test_ketkar_attachments_do_not_supply_required_source_voltage_or_ids() -> None:
    report = json.loads(REPORT.read_text())
    assert report["required_source_term_hits"] == {"Mi4": 0, "C3": 0}
    assert not any(report["individual_identifier_term_hits"].values())
    assert report["Mi4_or_C3_attachment_payload_found"] is False
    assert report["individual_source_dynamics_payload_found"] is False
    assert report["experimental_membrane_voltage_payload_found"] is False
    assert report["source_dynamics_transfer_gate_changed"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["boundary"]["bounded_attachment_audit_not_global_nonexistence"] is True
