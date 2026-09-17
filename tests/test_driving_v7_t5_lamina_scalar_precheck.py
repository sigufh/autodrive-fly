import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-lamina-scalar-precheck.json"


def test_t5_lamina_scalar_precheck_is_frozen_and_non_advancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["position_count"] == 15
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["boundary"]["direct_real_T5_sources_only"] is True
    assert report["boundary"]["labels_changed"] is False


def test_t5_lamina_scalar_has_source_coverage_but_fails_direction_and_order() -> None:
    report = json.loads(REPORT.read_text())
    assert report["source_coverage"]["target_count"] == 6719
    assert report["source_coverage"]["joint_present_count"] == 6718
    ordered = report["modes"]["ordered"]
    assert ordered["fast_state_peak"]["direction_pass_count"] == 0
    assert ordered["delayed_state_peak"]["direction_pass_count"] == 0
    correlation = ordered["fast_delayed_correlation"]
    assert correlation["direction_pass_count"] == 0
    assert correlation["polarity_pass_count"] == 8
    assert correlation["bilateral_direction_subtypes"] == []
    shuffled = report["modes"]["temporal_shuffle"]["fast_delayed_correlation"]
    assert shuffled["bilateral_direction_subtypes"] == ["a"]
    assert report["modes"]["static_sham"]["fast_delayed_correlation"][
        "bilateral_direction_subtypes"
    ] == []
    assert report["ordered_correlation_mirror_gate_passed"] is True
    assert report["candidate_passed"] is False
    assert report["expand_to_three_conditions"] is False
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
