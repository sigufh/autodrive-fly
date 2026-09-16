import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROTOCOL = ROOT / "artifacts/v7-fusion-nested.json"
EVALUATION = ROOT / "artifacts/v7-fusion-nested-eval.json"


def test_fusion_nested_protocol_freezes_architecture_and_external_final() -> None:
    report = json.loads(PROTOCOL.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["frozen_architecture"] == {
        "global_features": "six_bin_current_mean_18",
        "local_features": "lamina_24_bin_odd_plus_paired_even_times_odd",
        "lamina_weight": 0.75,
        "ridge_alpha": 1.0,
        "heading": "EPG_PEN_PEG_yaw_integrator",
        "descending": "FC2_PFL3_PFL2_DNa02_transparent",
    }
    assert report["role_counts"] == {"tuning": 3, "calibration": 1, "external_final": 1}
    assert report["external_final"]["committed"] is False
    assert report["external_final"]["local_seed_available"] is False
    assert report["external_final"]["local_generator_available"] is False
    assert report["advance_to_tuning"] is True
    assert report["advance_to_calibration"] is False


def test_fusion_nested_stops_at_cross_validation_without_consuming_calibration() -> None:
    report = json.loads(EVALUATION.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["architecture_changed_after_protocol_freeze"] is False
    assert report["protocol"]["historical_regression_used_for_fit_or_selection"] is False
    assert report["cross_validation"]["success_count"] == 2
    assert report["cross_validation"]["total_obstacles_passed"] == 34
    assert report["cross_validation"]["passed"] is False
    assert report["tuning_passed"] is True
    assert all(item["obstacles_passed"] == 9 for item in report["tuning_episodes"])
    assert report["calibration_episodes"] == []
    assert report["calibration_receipt"]["attempt_count"] == 0
    assert report["calibration_passed"] is False
    assert report["advance_to_final"] is False
    assert report["advance_to_navigation_release"] is False
