import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-cross-state-recording-pool-audit.json"


def test_cross_state_pool_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["kernel_length"] == 499
    assert report["protocol"]["kernel_sample_interval_milliseconds"] == 10.0
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_Tm2_OA_only_ids_do_not_expand_the_saline_kernel_cohort() -> None:
    report = json.loads(REPORT.read_text())
    tm2 = report["source_results"]["Tm2"]
    assert tm2["shared_recording_ids"] == ["201013", "201119", "210721"]
    assert tm2["saline_only_recording_ids"] == ["200927", "201125"]
    assert tm2["OA_only_recording_ids"] == ["210718", "210719"]
    assert tm2["paired_shape_correlation_summary"]["minimum"] < 0.43
    assert tm2["paired_shape_correlation_summary"]["maximum"] < 0.77
    assert tm2["paired_OA_to_saline_absolute_peak_ratio_summary"]["minimum"] < 0.39
    assert report["Tm2_OA_only_ids_expand_saline_cohort"] is False


def test_cross_state_pooling_does_not_unlock_T5() -> None:
    report = json.loads(REPORT.read_text())
    assert report["recording_id_is_biological_individual_id"] is False
    assert report["saline_and_OA_kernels_exchangeable"] is False
    assert report["authorize_cross_state_recording_pool"] is False
    assert report["authorize_T5_source_dynamics_fit"] is False
    assert report["authorize_T5_functional_precheck"] is False
