import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-upstream-latency-audit.json"


def test_upstream_latency_audit_is_tuning_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["position_count"] == 15
    assert report["protocol"]["direction_count"] == 4
    assert report["protocol"]["mode_count"] == 3
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_ordered_latency_passes_but_is_not_motion_specific() -> None:
    report = json.loads(REPORT.read_text())
    ordered = report["modes"]["ordered"]
    assert ordered["T4"]["passing_population_count"] == 8
    assert ordered["T4"]["all_population_latency_gate_passed"] is True
    assert ordered["T5"]["passing_population_count"] == 8
    assert ordered["T5"]["all_population_latency_gate_passed"] is True
    assert report["ordered_source_latency_gate_passed"] is True
    for mode in ("temporal_shuffle", "static_sham"):
        assert report["modes"][mode]["T4"]["all_population_latency_gate_passed"] is True
        assert report["modes"][mode]["T5"]["all_population_latency_gate_passed"] is True
    assert report["failure_controls_rejected"] is False
    assert report["source_latency_temporal_identifiability_passed"] is False


def test_upstream_latency_failure_does_not_authorize_target_or_LPLC_work() -> None:
    report = json.loads(REPORT.read_text())
    assert report["authorize_new_target_dynamics_candidate"] is False
    assert report["advance_to_three_tuning_conditions"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["stop_reason"] == (
        "fast_delayed_latency_persists_under_temporal_shuffle_and_static_sham"
    )
    boundary = report["boundary"]
    assert boundary["all_four_directions_used_without_preferred_direction_selection"] is True
    assert boundary["fixed_whole_population_times_four_directions_denominator"] is True
    assert boundary["source_latency_is_not_target_direction_selectivity"] is True
