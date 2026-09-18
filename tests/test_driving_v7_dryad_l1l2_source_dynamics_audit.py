import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-dryad-l1l2-source-dynamics-audit.json"


def test_dryad_l1l2_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["dataset"]["identifier"] == "doi:10.5061/dryad.ngf1vhj4c"
    assert report["dataset"]["file_count"] == 75
    assert report["dataset"]["total_bytes"] == 49_600_989_798
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["external_payload_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_publisher_hashes_are_not_misreported_as_locally_verified() -> None:
    integrity = json.loads(REPORT.read_text())["small_file_integrity"]
    assert integrity["publisher_reported_hashes_present"] is True
    assert integrity["local_payload_hashes_verified"] is False
    assert all(
        item["local_payload_downloaded"] is False
        and item["local_sha256_verified"] is False
        for item in integrity["files"].values()
    )


def test_l1_l2_scope_rejects_dataset_without_using_download_failure_as_reason() -> None:
    report = json.loads(REPORT.read_text())
    assert report["official_scope"]["measured_cell_types"] == ["L1", "L2"]
    assert report["covered_required_sources"] == []
    assert report["covered_required_source_count"] == 0
    assert report["required_source_count"] == 9
    assert report["missing_required_sources_by_family"]["T4"] == [
        "Mi1", "Tm3", "Mi4", "C3"
    ]
    assert report["missing_required_sources_by_family"]["T5"] == [
        "Tm1", "Tm2", "Tm4", "Tm9", "CT1"
    ]
    assert report["suitable_external_source_dynamics_evidence"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["boundary"]["download_failure_is_not_the_scientific_rejection_reason"] is True
    assert report["download_disposition"]["large_dataset_download_authorized"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
