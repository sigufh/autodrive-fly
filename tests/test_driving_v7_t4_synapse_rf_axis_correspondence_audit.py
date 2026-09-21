import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-synapse-rf-axis-correspondence-audit.json"


def test_synapse_rf_axis_audit_is_hash_bound_and_structural_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit_to_neural_response"] is False
    assert report["protocol"]["functional_stimulus_evaluated"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_target_synapse_and_source_rf_axes_are_not_interchangeable() -> None:
    report = json.loads(REPORT.read_text())
    assert report["fixed_T4_population_denominator"] == 6861
    assert report["conductance_target_count"] == 6749
    assert report["joint_valid_target_count"] == 6749
    identity = report["comparison_results"]["identity"]
    assert identity["median_angle_degrees"] == 68.18171979285492
    assert identity["positive_cosine_fraction"] == 0.5476366869165802
    assert identity["same_cardinal_direction_fraction"] == 0.4951844717735961
    assert report["raw_source_RF_axis_exactly_matches_target_synapse_axis"] is False
    assert report["source_RF_axis_interchangeability_verified"] is False


def test_horizontal_reflection_is_descriptive_and_non_advancing() -> None:
    report = json.loads(REPORT.read_text())
    assert report["descriptive_best_signed_permutation"] == "reflect_horizontal"
    reflected = report["comparison_results"]["reflect_horizontal"]
    assert reflected["median_angle_degrees"] == 21.112026999265463
    assert reflected["positive_cosine_fraction"] == 0.9973329382130686
    assert reflected["same_cardinal_direction_fraction"] == 0.8749444362127723
    identity = report["population_comparison_results"]["identity"]
    assert identity["T4a_L"]["median_angle_degrees"] > 140
    assert identity["T4b_R"]["same_cardinal_direction_fraction"] < 0.03
    assert identity["T4c_L"]["same_cardinal_direction_fraction"] > 0.96
    assert report["authorize_RF_axis_replacement"] is False
    assert report["authorize_new_T4_functional_candidate"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
