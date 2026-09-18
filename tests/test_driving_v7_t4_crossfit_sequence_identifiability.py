import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-crossfit-sequence-identifiability.json"


def test_t4_crossfit_sequence_audit_is_hash_bound_and_non_authorizing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["fixed_T4_population_denominator"] == 6861
    assert report["valid_target_count"] == 6749
    assert report["candidate_selected"] is False
    assert report["authorize_new_functional_candidate"] is False
    assert report["advance_to_three_tuning_conditions"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["maximum_ordered_direction_pass_count"] == 2
    assert all(
        not values
        for values in report["ordered_bilateral_direction_subtypes_by_reduction"].values()
    )


def test_t4_crossfit_sequence_audit_keeps_all_modes_and_reductions() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["mode_reduction_results"]) == {
        "ordered",
        "temporal_shuffle",
        "static_sham",
    }
    assert all(
        set(items) == {"positive_peak", "signed_mean"}
        for items in report["mode_reduction_results"].values()
    )
    assert report["mode_reduction_results"]["temporal_shuffle"]["signed_mean"][
        "bilateral_direction_subtypes"
    ] == ["a"]
    assert report["boundary"]["post_failure_diagnostic_only"]
    assert report["boundary"]["no_gate_authorization"]
