import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-behnia-t4-fast-source-audit.json"


def test_behnia_fast_source_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["paper"]["doi"] == "10.1038/nature13427"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_independent_fast_source_phenotype_is_preserved() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["source_evidence"]) == {"Mi1", "Tm3"}
    assert report["source_evidence"]["Mi1"]["Gaussian_noise_cell_count"] == 7
    assert report["source_evidence"]["Tm3"]["Gaussian_noise_cell_count"] == 11
    assert report["reported_protocol"]["Mi1_minus_Tm3_peak_latency_milliseconds"] == 18
    assert report["transfer_gates"]["independent_Mi1_later_than_Tm3_phenotype_published"] is True


def test_missing_numeric_payload_and_sources_keep_transfer_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["public_resource_inventory"]["numeric_payload_links_found"] == []
    assert report["transfer_gates"]["local_numeric_trace_payload_verified"] is False
    assert report["transfer_gates"]["all_four_T4_sources_covered"] is False
    assert report["independent_T4_source_dynamics_transfer_authorized"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
