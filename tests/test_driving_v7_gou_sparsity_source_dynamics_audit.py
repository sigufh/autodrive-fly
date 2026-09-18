import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-gou-sparsity-source-dynamics-audit.json"


def test_gou_audit_is_hash_bound_read_only_and_repository_specific() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["repositories"]["Dryad"]["version_id"] == 402168
    assert report["repositories"]["DANDI"]["version"] == "0.250602.0251"
    assert report["repositories"]["DANDI"]["asset_count"] == 282
    assert report["protocol"]["external_payload_evaluated"] is False
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_gou_data_cover_only_four_of_nine_required_sources() -> None:
    report = json.loads(REPORT.read_text())
    assert report["covered_required_sources_by_family"] == {
        "T4": ["Mi1", "Tm3"],
        "T5": ["Tm1", "Tm2"],
    }
    assert report["missing_required_sources_by_family"] == {
        "T4": ["Mi4", "C3"],
        "T5": ["Tm4", "Tm9", "CT1"],
    }
    assert report["covered_required_source_count"] == 4
    assert report["required_source_count"] == 9
    assert report["coverage_fraction"] == 4 / 9
    assert report["partial_source_dynamics_evidence_present"] is True


def test_partial_fluorescence_evidence_does_not_open_any_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["source_response_unit_gates"].values()) == {False}
    assert report["complete_external_source_dynamics_evidence"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["advance_to_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["download_disposition"]["Dryad_archive_download_authorized"] is False
    assert report["download_disposition"]["DANDI_bulk_download_authorized"] is False
    assert report["boundary"]["partial_source_coverage_does_not_satisfy_the_unified_contract"] is True
