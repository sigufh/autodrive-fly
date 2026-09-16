import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
UPPER = ROOT / "artifacts/v7-visual-corridor-goal.json"
NEURAL = ROOT / "artifacts/v7-neural-corridor.json"


def test_visual_corridor_upper_bound_is_tuning_only_and_causal() -> None:
    report = json.loads(UPPER.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["tuning_only"] is True
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["environment_geometry_used"] is False
    assert report["selected_candidate"] == {
        "corridor_width_columns": 12,
        "goal_gain": 8.0,
        "switch_margin": 0.25,
        "active_danger_threshold": 0.5,
    }
    assert report["selected_summary"]["success_count"] == 6
    assert report["selected_summary"]["total_obstacles_passed"] == 54
    assert report["controls"]["no_visual_goal"]["success_count"] == 0
    assert report["controls"]["reversed_visual_goal"]["success_count"] == 0
    assert report["tuning_passed"] is True
    assert report["causal_controls_passed"] is True
    assert report["advance_to_neural_cv"] is True
    assert report["advance_to_calibration"] is False


def test_neural_corridor_cannot_recover_receptor_level_upper_bound() -> None:
    report = json.loads(NEURAL.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["mirror_pair_is_indivisible_cv_unit"] is True
    assert report["protocol"]["tuning_only"] is True
    assert report["protocol"]["calibration_evaluated"] is False
    assert [fold["success_count"] for fold in report["folds"]] == [0, 0, 0]
    assert [fold["total_obstacles_passed"] for fold in report["folds"]] == [14, 8, 2]
    assert report["success_count"] == 0
    assert report["total_obstacles_passed"] == 24
    assert report["cross_validation_passed"] is False
    assert report["advance_to_full_tuning"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_navigation_release"] is False
