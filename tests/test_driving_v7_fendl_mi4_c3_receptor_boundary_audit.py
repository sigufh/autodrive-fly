import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-fendl-mi4-c3-receptor-boundary-audit.json"


def test_fendl_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_fendl_evidence_is_target_receptor_not_source_dynamics() -> None:
    report = json.loads(REPORT.read_text())
    assert report["thesis"]["page_count"] == 133
    assert report["thesis"]["embedded_attachment_count"] == 0
    assert report["required_source_term_evidence"]["Mi4"] == {
        "exact_term_count": 9,
        "exact_term_pages": [31, 34, 35, 70, 73, 75, 103, 106],
    }
    assert report["required_source_term_evidence"]["C3"] == {
        "exact_term_count": 9,
        "exact_term_pages": [31, 34, 35, 70, 73, 75, 103, 106],
    }
    assert report["direct_functional_imaging_cell_types"] == ["L1", "Mi9", "LPi4-3"]
    assert report["target_receptor_localization_cell_types"] == ["T4", "T5"]
    assert report["target_receptor_evidence"]["Rdl_localized_on_T4_T5_dendrites"] is True
    assert report["Mi4_direct_neural_recording_verified"] is False
    assert report["C3_direct_neural_recording_verified"] is False
    assert report["Mi4_C3_source_numeric_payload_verified"] is False
    assert report["experimental_Mi4_C3_membrane_voltage_verified"] is False
    assert report["authorize_Mi4_C3_source_dynamics_transfer"] is False
