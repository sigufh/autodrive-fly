import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-population-kernel-alignment-failure-audit.json"


def test_T5_alignment_failure_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["kernel_sample_interval_milliseconds"] == 10.0
    assert report["protocol"]["maximum_oracle_lag_milliseconds"] == 250
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_oracle_alignment_does_not_rescue_Tm2_partition_robustness() -> None:
    report = json.loads(REPORT.read_text())
    assert report["Tm2_zero_lag_LOO_median"] < 0.8
    assert report["Tm2_oracle_shift_LOO_median"] > 0.8
    assert report["Tm2_zero_lag_partition_p05"] < 0.77
    assert report["Tm2_oracle_shift_partition_p05"] < 0.78
    assert report["Tm2_oracle_shift_partition_p05"] < report["thresholds"][
        "minimum_partition_correlation_p05"
    ]
    assert report["source_results"]["Tm2"]["oracle_diagnostic_gates"][
        "partition_correlation_p05"
    ] is False
    assert report["Tm2_oracle_all_shape_gates_passed"] is False


def test_oracle_alignment_retains_the_original_negative_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["bounded_latency_jitter_explains_Tm2_robustness_failure"] is False
    assert report["original_population_kernel_robustness_failure_retained"] is True
    assert report["authorize_oracle_aligned_population_kernel"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["boundary"]["oracle_lag_uses_both_compared_waveforms"] is True
    assert report["boundary"]["oracle_lag_is_not_deployable"] is True
