import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-c3-measured-filter-robustness.json"


def test_C3_measured_filter_audit_is_hash_bound_and_source_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["T4_response_used"] is False
    assert report["protocol"]["direction_label_used"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_C3_processing_is_stable_to_calcium_assumption_but_not_fly_holdout() -> None:
    report = json.loads(REPORT.read_text())
    cross = report["minimum_cross_deconvolution_correlation_by_unit"]
    assert len(cross) == 7
    assert min(cross.values()) > 0.984
    assert report["cross_deconvolution_stability_passed"] is True
    results = report["leave_one_fly_out_by_assumption"]
    assert set(results) == {"200ms", "250ms", "300ms", "350ms"}
    assert all(item["median_held_out_correlation"] > 0.85 for item in results.values())
    assert all(item["minimum_held_out_correlation"] < 0.58 for item in results.values())
    assert all(item["gates"]["median_held_out_fly_axis_correlation"] for item in results.values())
    assert all(not item["gates"]["every_held_out_fly_axis"] for item in results.values())
    assert report["every_deconvolution_assumption_passed"] is False


def test_C3_failed_cross_fly_gate_keeps_functional_candidate_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["C3_measured_filter_robustness_passed"] is False
    assert report["C3_type_shared_kernel_authorized"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["boundary"]["inherited_threshold_not_changed"] is True
    assert report["boundary"]["no_T4_response_or_direction_label_used"] is True
    assert report["boundary"]["T5_and_LPLC_remain_frozen"] is True
