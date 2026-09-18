import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-fig3-source-kernel-audit.json"


def test_fig3_source_kernel_audit_is_hash_bound_and_source_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    source = report["protocol"]["source_file"]
    assert source["bytes"] == 1_906_405
    assert source["sha256"] == (
        "dd41cd3d9a0895c68c8e528a21ee9cde8d4898d34ab7dded842f7641b9997cf3"
    )
    assert source["workbook_sheet_count"] == 6
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_fig3_ON_source_kernels_expose_stable_and_unstable_types() -> None:
    report = json.loads(REPORT.read_text())
    on = report["conditions"]["on"]["source_results"]
    assert {name: item["cell_count"] for name, item in on.items()} == {
        "Tm3": 12,
        "Mi1": 24,
        "Mi4": 19,
        "C3": 16,
    }
    assert on["Tm3"]["ready"] is True
    assert on["Mi4"]["ready"] is True
    assert on["Mi1"]["alternating_split_kernel_correlation"] < 0.66
    assert on["Mi1"]["ready"] is False
    assert on["C3"]["cell_peak_latency_IQR_milliseconds"] > 760.0
    assert on["C3"]["ready"] is False
    assert report["conditions"]["on"]["all_source_types_ready"] is False
    assert report["all_condition_source_kernels_ready"] is False


def test_fig3_prior_fast_delayed_pool_ordering_is_rejected() -> None:
    report = json.loads(REPORT.read_text())
    grouping = report["prior_fast_delayed_grouping"]
    assert grouping["fast_sources"] == ["Tm3", "Mi1"]
    assert grouping["delayed_sources"] == ["Mi4", "C3"]
    differences = grouping["pairwise_ON_median_peak_latency_differences_milliseconds"]
    assert differences["Mi4_minus_Tm3"] < 0.0
    assert differences["Mi4_minus_Mi1"] < 0.0
    assert differences["C3_minus_Tm3"] > 300.0
    assert differences["C3_minus_Mi1"] > 300.0
    assert grouping["every_delayed_source_later_than_every_fast_source"] is False
    assert report["prior_two_pool_kernel_authorized"] is False
    assert report["source_specific_kernel_candidate_authorized"] is False
    assert report["boundary"]["no_T4_response_used_for_kernel_selection"] is True
    assert report["boundary"]["no_PD_ND_label_used_for_kernel_extraction"] is True
