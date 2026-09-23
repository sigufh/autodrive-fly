import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-fig3-source-kernel-alignment-failure-audit.json"


def test_alignment_failure_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["maximum_oracle_lag_milliseconds"] == 250
    assert report["protocol"]["reference_minimum_cell_SNR"] == 3.0
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_alignment_diagnostic_reproduces_zero_lag_failure() -> None:
    report = json.loads(REPORT.read_text())
    assert report["preexisting_normalization"] == {
        "baseline_offset_removed": True,
        "L2_gain_removed": True,
    }
    assert report["conditions"]["on"]["sources"]["Mi1"][
        "zero_lag_minimum_correlation"
    ] < -0.70
    assert report["conditions"]["on"]["sources"]["Mi4"][
        "zero_lag_minimum_correlation"
    ] < -0.64
    assert report["conditions"]["off"]["sources"]["Mi4"][
        "zero_lag_minimum_correlation"
    ] < -0.40


def test_bounded_oracle_alignment_does_not_rescue_every_negative_cell() -> None:
    report = json.loads(REPORT.read_text())
    assert report["original_negative_leave_one_out_cell_count"] > 0
    assert report["oracle_shift_negative_leave_one_out_cell_count"] > 0
    assert (
        report[
            "bounded_oracle_shift_eliminates_all_negative_leave_one_out_cells"
        ]
        is False
    )
    assert report["bounded_latency_jitter_explains_every_negative_cell"] is False
    assert report["negative_cells_at_or_above_reference_SNR_count"] == 11
    assert report["low_SNR_explains_every_negative_cell"] is False
    assert report["ON_OFF_negative_cell_id_intersection_count"] == 0
    assert report["one_stable_bad_cell_set_explains_ON_OFF_failures"] is False
    assert report["authorize_post_hoc_cell_exclusion"] is False
    assert report["conditions"]["on"]["sources"]["Mi4"][
        "maximum_negative_cell_SNR"
    ] > 25.0
    assert report["conditions"]["on"]["sources"]["Mi1"][
        "maximum_negative_cell_SNR"
    ] > 70.0
    assert report["cross_condition_cell_consistency"]["Mi4"][
        "negative_cell_id_intersection"
    ] == []
    assert report["original_source_kernel_robustness_failure_retained"] is True
    assert report["authorize_aligned_source_kernel"] is False
    assert report["authorize_T4_functional_precheck"] is False
