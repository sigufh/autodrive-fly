import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROTOCOL = ROOT / "artifacts/v7-navigation-nested.json"
EVALUATION = ROOT / "artifacts/v7-navigation-nested-eval.json"


def test_navigation_nested_protocol_has_three_tuning_one_calibration_external_final() -> None:
    report = json.loads(PROTOCOL.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["role_counts"] == {"tuning": 3, "calibration": 1, "external_final": 1}
    assert [item["condition_id"] for item in report["local_conditions"]] == [
        "NAV-T01",
        "NAV-T02",
        "NAV-T03",
        "NAV-C01",
    ]
    assert report["historical_regression"]["fit_weight"] == 0
    assert report["historical_regression"]["selection_weight"] == 0
    assert report["historical_regression"]["calibration_weight"] == 0
    final = report["external_final"]
    assert final["committed"] is False
    assert final["local_seed_available"] is False
    assert final["local_generator_available"] is False
    assert report["advance_to_tuning"] is True
    assert report["advance_to_calibration"] is False
    assert report["advance_to_final"] is False


def test_navigation_nested_candidate_stops_before_calibration() -> None:
    report = json.loads(EVALUATION.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["architecture_changed_after_protocol_freeze"] is False
    assert report["protocol"]["historical_regression_used_for_fit_or_selection"] is False
    assert report["cross_validation"]["success_count"] == 4
    assert report["cross_validation"]["total_obstacles_passed"] == 48
    assert report["cross_validation"]["passed"] is False
    assert report["tuning_passed"] is False
    assert report["calibration_episodes"] == []
    assert report["calibration_receipt"]["attempt_count"] == 0
    assert report["calibration_passed"] is False
    assert report["advance_to_final"] is False
    assert report["advance_to_navigation_release"] is False


def test_navigation_nested_failure_includes_action_policy_limit() -> None:
    attribution = json.loads(EVALUATION.read_text())["tuning_failure_attribution"]
    assert attribution["role"] == (
        "post_tuning_failure_attribution_only_not_parameter_selection"
    )
    assert attribution["success_count"] == 4
    assert attribution["total_obstacles_passed"] == 50
    assert attribution["diagnosis"] == "fixed_local_action_policy_limit_present"
    by_seed = {item["seed"]: item for item in attribution["r1r6_local_upper_bound"]}
    assert by_seed[9104]["obstacles_passed"] == 7
    assert by_seed[9105]["obstacles_passed"] == 7
