import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-timing-models-source-filter-audit.json"


def test_timing_models_audit_is_commit_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["repository_commit"] == (
        "100bb2f52cb9628477c3883e4b17774b0b244e67"
    )
    assert len(report["protocol"]["verified_files"]) == 7
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_measured_filters_are_stable_across_deconvolution_assumptions() -> None:
    report = json.loads(REPORT.read_text())
    data = report["filter_data"]
    assert data["calcium_deconvolution_milliseconds"] == [200, 250, 300]
    assert data["filter_shape"] == [60, 12]
    assert data["sample_interval_seconds"] == 1 / 30
    assert data["time_start_seconds"] == 0.0
    assert abs(data["time_stop_seconds"] - 59 / 30) < 1e-12
    assert set(data["minimum_pairwise_correlation_by_current_source"]) == {
        "Mi1",
        "Tm3",
        "Mi4",
    }
    assert min(data["minimum_pairwise_correlation_by_current_source"].values()) > 0.984
    assert set(data["stability_gates"].values()) == {True}
    assert report["covered_source_filter_stability_verified"] is True


def test_timing_models_does_not_fill_C3_or_identity_gates() -> None:
    report = json.loads(REPORT.read_text())
    contract = report["current_v7_source_contract"]
    assert contract["required_sources"] == ["Mi1", "Tm3", "Mi4", "C3"]
    assert contract["covered_sources"] == ["Mi1", "Mi4", "Tm3"]
    assert contract["missing_sources"] == ["C3"]
    assert contract["coverage_fraction"] == 0.75
    assert {"CT1", "Mi9"} <= set(contract["noncontract_model_sources"])
    assert report["transfer_gates"]["every_current_source_type_covered"] is False
    assert report["transfer_gates"]["independent_cell_holdout_available"] is False
    assert report["complete_source_filter_candidate_authorized"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["boundary"]["Mi4_or_CT1_must_not_substitute_for_C3"] is True
    assert report["boundary"]["T5_and_LPLC_remain_frozen"] is True
