import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-c3-analytic-filter-precheck.json"


def test_c3_filter_precheck_is_hash_bound_and_source_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is True
    assert report["protocol"]["fit_scope"] == "C3_source_STRF_only"
    assert report["protocol"]["T4_response_used"] is False
    assert report["protocol"]["direction_label_used"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_band_pass_family_is_better_but_not_robust_for_every_held_out_fly_axis() -> None:
    report = json.loads(REPORT.read_text())
    families = report["model_families"]
    assert report["band_pass_BIC_below_low_pass_BIC"] is True
    assert families["band_pass"]["full_data_BIC"] < families["low_pass"]["full_data_BIC"]
    assert families["band_pass"]["cross_validation"]["held_out_unit_count"] == 7
    assert families["band_pass"]["cross_validation"]["median_held_out_correlation"] > 0.84
    assert families["band_pass"]["cross_validation"]["minimum_held_out_correlation"] < 0.67
    assert families["band_pass"]["gates"]["median_held_out_correlation"] is True
    assert families["band_pass"]["gates"]["minimum_held_out_correlation"] is False
    assert families["band_pass"]["gates"]["every_held_out_fly_axis"] is False
    assert families["band_pass"]["passed"] is False
    assert families["low_pass"]["passed"] is False


def test_cross_paper_calcium_assumption_and_failed_loo_keep_transfer_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["model_contract"]["calcium_low_pass_seconds"] == 0.35
    assert report["C3_analytic_filter_family_precheck_passed"] is False
    assert set(report["transfer_gates"].values()) == {False}
    assert report["C3_analytic_filter_transfer_authorized"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["boundary"]["inherited_threshold_not_changed"] is True
    assert report["boundary"]["cross_paper_calcium_constant_is_unvalidated_for_C3_dataset"] is True
    assert report["boundary"]["no_T4_response_or_direction_label_used"] is True
    assert report["boundary"]["T5_and_LPLC_remain_frozen"] is True
