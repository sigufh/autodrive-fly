import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-c3-strf-flash-transfer.json"


def test_c3_strf_flash_transfer_is_hash_bound_and_zero_fit() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["T4_response_used"] is False
    assert report["protocol"]["direction_label_used"] is False


def test_c3_strf_and_external_flash_cohorts_are_disjoint() -> None:
    cohorts = json.loads(REPORT.read_text())["cohorts"]
    assert cohorts["source_fly_axis_unit_count"] == 7
    assert cohorts["flash_fly_count_before_exclusion"] == 27
    assert cohorts["external_flash_fly_count"] == 22
    assert set(cohorts["source_fly_ids"]).isdisjoint(cohorts["external_flash_fly_ids"])
    assert cohorts["overlapping_fly_ids_excluded"] == cohorts["source_fly_ids"]


def test_c3_strf_step_prediction_obeys_preregistered_robustness_gate() -> None:
    report = json.loads(REPORT.read_text())
    validation = report["external_validation"]
    assert validation["mean_waveform_correlation"] > 0.80
    assert validation["per_fly_correlation_summary"]["count"] == 22
    assert validation["bootstrap"]["source_resampling_unit"] == (
        "fly_with_all_available_axes_retained"
    )
    assert validation["bootstrap"]["correlation_p05"] < 0.12
    assert report["C3_source_kernel_candidate_authorized"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["boundary"]["source_kernel_not_yet_mapped_to_MaleCNS_bodies"] is True
    assert report["boundary"]["T5_and_LPLC_remain_frozen"] is True
