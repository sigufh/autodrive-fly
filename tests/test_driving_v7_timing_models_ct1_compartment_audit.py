import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-timing-models-ct1-compartment-audit.json"


def test_CT1_compartment_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["protocol"]["pypdf_version"].startswith("6.")
    assert len(report["protocol"]["supplements"]) == 2
    assert all(
        item["Figure_S5_caption_verified"] is True
        and item["embedded_attachment_count"] == 0
        and item["actual_sha256"] == item["sha256"]
        for item in report["protocol"]["supplements"]
    )
    assert report["CT1_filter_evidence"]["sample_count"] == 60
    assert report["CT1_filter_evidence"]["sample_interval_seconds"] == 1 / 30
    assert report["CT1_filter_evidence"]["control_fly_count_reported_in_paper"] == 17


def test_CT1_filter_is_stable_and_not_a_Mi4_or_Tm9_copy() -> None:
    report = json.loads(REPORT.read_text())
    gates = report["transfer_gates"]
    assert gates["CT1_column_present_with_finite_SEM"] is True
    assert gates["CT1_column_distinct_from_Mi4_and_Tm9"] is True
    assert gates["CT1_stable_across_deconvolution_assumptions"] is True
    assert report["CT1_filter_evidence"]["minimum_cross_deconvolution_correlation"] > 0.98
    for by_source in report["CT1_filter_evidence"]["distinction_from_comparison_sources"].values():
        assert all(item["exactly_equal"] is False for item in by_source.values())


def test_unlabeled_CT1_compartment_cannot_satisfy_T5_contract() -> None:
    report = json.loads(REPORT.read_text())
    assert report["CT1_type_average_dynamics_verified"] is True
    assert report["T5_lobula_CT1_dynamic_phenotype_published"] is True
    assert report["T5_lobula_CT1_numerical_payload_verified"] is False
    assert report["T5_lobula_CT1_dynamics_verified"] is False
    assert report["transfer_gates"]["CT1_lobula_L1_Lo1_provenance_available"] is False
    assert report["transfer_gates"]["allowed_response_unit_available"] is False
    assert report["supplement_S5_evidence"]["wild_type_lobula_CT1_fly_count"] == 17
    assert report["supplement_S5_evidence"]["cac_RNAi_lobula_CT1_fly_count"] == 10
    assert report["supplement_S5_evidence"]["frame_interval_seconds"] == 1 / 30
    assert report["T5_CT1_source_dynamics_transfer_authorized"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"]["medulla_M10_evidence_must_not_substitute_for_lobula_L1_Lo1"] is True
