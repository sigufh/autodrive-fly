import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-fig3-source-kernel-robustness.json"


def test_fig3_kernel_robustness_is_hash_bound_and_label_blind() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["bootstrap_seed"] == 20260918
    assert report["protocol"]["bootstrap_replicates"] == 256
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["boundary"]["every_recorded_cell_retained"] is True
    assert report["boundary"]["no_outlier_removal"] is True
    assert report["boundary"]["no_T4_response_used"] is True
    assert report["boundary"]["no_direction_label_used"] is True


def test_fig3_ON_kernels_fail_cross_cell_robustness_except_Tm3() -> None:
    sources = json.loads(REPORT.read_text())["conditions"]["on"]["sources"]
    assert sources["Tm3"]["passed"] is True
    assert sources["Mi1"]["median_leave_one_cell_out_correlation"] < 0.61
    assert sources["Mi1"]["negative_leave_one_cell_out_fraction"] > 0.29
    assert sources["Mi1"]["bootstrap_split_correlation_p05"] < 0.66
    assert sources["Mi1"]["passed"] is False
    assert sources["Mi4"]["minimum_leave_one_cell_out_correlation"] < -0.64
    assert sources["Mi4"]["passed"] is False
    assert sources["C3"]["mean_median_kernel_correlation"] < 0.72
    assert sources["C3"]["bootstrap_split_correlation_p05"] < 0.71
    assert sources["C3"]["passed"] is False


def test_fig3_robustness_stop_rule_prevents_functional_candidate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["conditions"]["on"]["all_sources_passed"] is False
    assert report["conditions"]["off"]["all_sources_passed"] is False
    assert report["all_source_kernel_robustness_gates_passed"] is False
    assert report["source_specific_kernel_candidate_authorized"] is False
    assert report["advance_to_functional_precheck"] is False
    assert report["stop_reason"] == (
        "Fig3_source_kernels_not_robust_across_recorded_cells"
    )
