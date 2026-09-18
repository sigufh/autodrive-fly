import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-ct1-axis-sequence-identifiability.json"


def test_t5_ct1_axis_sequence_diagnostic_is_hash_bound_and_non_authorizing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["fixed_target_denominator"] == 6719
    assert report["valid_target_count"] == 6713
    assert report["candidate_selected"] is False
    assert report["authorize_new_functional_candidate"] is False
    assert report["advance_to_three_tuning_conditions"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False


def test_t5_ct1_axis_sequence_diagnostic_preserves_all_modes_and_reductions() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["mode_reduction_results"]) == {
        "ordered",
        "temporal_shuffle",
        "static_sham",
    }
    reductions = {
        "positive_axis_sequence_peak",
        "positive_axis_sequence_mean",
        "signed_axis_sequence_mean",
        "signed_axis_sequence_terminal",
    }
    assert all(set(items) == reductions for items in report["mode_reduction_results"].values())
    assert report["boundary"]["post_failure_diagnostic_only"]
    assert report["boundary"]["no_gate_authorization"]
