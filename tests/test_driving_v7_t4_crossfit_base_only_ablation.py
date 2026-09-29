import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-crossfit-base-only-ablation.json"


def test_base_only_ablation_is_hash_bound_and_diagnostic_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["source_sequence_gain"] == 0.0
    assert report["base_only_is_new_candidate"] is False


def test_base_only_results_preserve_denominators_and_reduction_invariance() -> None:
    report = json.loads(REPORT.read_text())
    assert report["fixed_T4_population_denominator"] == 6861
    assert report["valid_target_count"] == 6749
    assert set(report["mode_results"]) == {
        "ordered",
        "temporal_shuffle",
        "static_sham",
    }
    assert all(result["reduction_invariant"] for result in report["mode_results"].values())
    assert report["ordered_direction_passing_populations"] == ["T4a_R", "T4c_R"]
    assert report["ordered_polarity_passing_populations"] == [
        "T4a_L",
        "T4a_R",
        "T4b_L",
        "T4b_R",
        "T4c_L",
        "T4c_R",
        "T4d_L",
        "T4d_R",
    ]
    assert report["ordered_reproduces_dominant_frozen_family_pass_set"] is True
    assert report["base_only_ordered_direction_pass_count"] == 2
    assert report["base_only_control_maximum_direction_pass_count"] == 0
    assert report["base_only_control_specificity_at_direction_gate"] is True
    assert report["base_only_controls_have_direction_passes"] is False
    assert report["source_sequence_only_ordered_direction_passing_populations"] == [
        "T4c_R",
        "T4d_L",
    ]
    assert report["source_sequence_only_control_maximum_direction_pass_count"] == 3
    assert report["source_sequence_controls_have_direction_passes"] is True
    assert report["gate_level_unilateral_pass_pattern_present_in_conductance_base"] is True
    assert (
        report["current_source_sequence_overlay_expands_bilateral_direction_passes"]
        is False
    )


def test_base_only_ablation_never_authorizes_a_candidate_or_downstream_stage() -> None:
    report = json.loads(REPORT.read_text())
    assert report["authorize_base_only_candidate"] is False
    assert report["authorize_source_sequence_removal"] is False
    assert report["authorize_direction_or_subtype_label_change"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
