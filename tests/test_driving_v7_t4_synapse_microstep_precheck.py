import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-synapse-microstep-precheck.json"


def test_T4_microstep_precheck_reuses_frozen_leaks_and_correct_baseline_prefix() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["brain_substeps_per_frame"] == 4
    assert report["protocol"]["baseline_prefix_frames_in_controls"] == 8
    assert report["protocol"]["frozen_source_leaks_per_substep"] == {
        "Mi1": 0.24,
        "Tm3": 0.62,
        "Mi4": 0.14,
        "C3": 0.16,
    }
    assert report["protocol"]["T5_evaluated"] is False
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_T4_microstep_temporal_identifiability_fails_before_direction_scoring() -> None:
    report = json.loads(REPORT.read_text())
    temporal = report["temporal_identifiability"]
    assert report["fixed_T4_population_denominator"] == 6861
    assert report["diagnostic_target_count"] == 6749
    assert report["diagnostic_valid_target_count"] == 6749
    assert temporal["shuffle_to_ordered_residual_energy_ratio"] > 1.9
    assert temporal["static_to_ordered_energy_ratio"] > 0.53
    assert temporal["gates"] == {
        "shuffle_residual_attenuated": False,
        "static_energy_attenuated": False,
    }
    assert temporal["passed"] is False
    assert report["direction_scoring_performed"] is False
    assert report["ordered_candidates"] == []


def test_T4_microstep_stop_rule_keeps_downstream_frozen() -> None:
    report = json.loads(REPORT.read_text())
    assert report["advance_to_three_tuning_conditions"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["stop_reason"] == "microstep_temporal_identifiability_gate_failed"
    assert report["boundary"]["physical_time_constant_claimed"] is False
    assert report["boundary"]["T5_mapping_application_forbidden"] is True
