import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-c3-flash-preregistration.json"


def test_c3_flash_preregistration_is_revision_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["archived_revision"] == (
        "745c6114b72a61dabc9389a1f024dbe22f679edf"
    )
    source = report["protocol"]["source_data"]
    assert source["bytes"] == 18_437_984
    assert source["git_blob_sha1"] == "7e22b9dfed9bdf65d8e77b6118ef6e235ab3b124"
    assert source["sha256"] == (
        "4ba864b0df6e68d101ddee61836ba7bbb34c6b74116be95d7e69f97ef9ff445a"
    )
    assert report["protocol"]["pilot_model_response_observed"] is True
    assert report["protocol"]["ensemble_outputs_observed"] is False


def test_author_c3_flash_analysis_is_reproduced() -> None:
    report = json.loads(REPORT.read_text())
    empirical = report["empirical_C3_flash"]
    assert empirical["record_count"] == 354
    assert empirical["selected_ROI_count"] == 332
    assert empirical["fly_count"] == 27
    assert empirical["sample_interval_seconds"] == 0.1
    assert empirical["epoch_samples"] == 130
    assert empirical["ON_onset_sample"] == 50
    assert empirical["OFF_onset_sample"] == 100
    assert empirical["ON_duration_seconds"] == 5.0
    assert empirical["onset_peak_latency_seconds"]["minimum"] == 0.2
    assert abs(empirical["onset_peak_latency_seconds"]["maximum"] - 0.7) < 1e-12
    assert empirical["absolute_recovery_to_peak_ratio"]["q975"] < 0.12


def test_external_comparison_was_frozen_without_model_output() -> None:
    report = json.loads(REPORT.read_text())
    contract = report["comparison_contract"]
    assert report["comparison_metrics_frozen"] is True
    assert contract["model_sample_intervals_seconds"] == [0.005, 0.02]
    assert contract["permitted_external_input_cell_types"] == [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
    ]
    assert contract["calcium_forward_time_constants_seconds"] == [0.2, 0.25, 0.3, 0.35]
    assert contract["inherited_minimum_correlation"] == 0.8
    assert contract["no_threshold_selection_after_model_observation"] is True
    assert report["boundary"]["no_target_activity_injection"] is True
    assert report["boundary"]["R7_R8_external_input_excluded"] is True
    assert report["boundary"]["T5_and_LPLC_remain_frozen"] is True
