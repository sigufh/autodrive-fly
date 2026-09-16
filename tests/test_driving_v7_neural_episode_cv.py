import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-neural-episode-cv.json"


def test_neural_episode_cv_is_pair_grouped_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["mirror_pair_is_indivisible_cv_unit"] is True
    assert protocol["training_r2_used_for_selection"] is False
    assert protocol["raw_heading_read_by_controller"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_run_once_after_candidate_freeze"] is True
    assert protocol["external_final_evaluated"] is False
    for candidate in report["candidate_summaries"]:
        assert [fold["held_out_seeds"] for fold in candidate["folds"]] == [
            [8200, 8201],
            [8202, 8203],
            [8204, 8205],
        ]
        for fold in candidate["folds"]:
            assert not set(fold["training_seeds"]) & set(fold["held_out_seeds"])


def test_ridge_strength_alone_does_not_fix_episode_generalization() -> None:
    report = json.loads(REPORT.read_text())
    candidates = report["candidate_summaries"]
    assert [item["alpha"] for item in candidates] == [0.1, 1.0, 10.0, 100.0, 1000.0]
    assert {item["held_out_success_count"] for item in candidates} == {4}
    assert {item["held_out_obstacles_passed"] for item in candidates} == {52}
    for candidate in candidates:
        last_fold = candidate["folds"][2]
        assert last_fold["held_out_seeds"] == [8204, 8205]
        assert [item["obstacles_passed"] for item in last_fold["held_out_episodes"]] == [8, 8]
    assert report["selected_alpha"] == 1000.0
    assert report["ridge_only_cross_validation_passed"] is False


def test_current_mean_wins_episode_cv_and_fresh_calibration() -> None:
    report = json.loads(REPORT.read_text())
    summaries = {item["name"]: item for item in report["feature_complexity_summaries"]}
    assert summaries["current_mean"]["retained_feature_count"] == 18
    assert summaries["current_mean"]["held_out_success_count"] == 6
    assert summaries["current_mean"]["held_out_obstacles_passed"] == 54
    assert summaries["full"]["retained_feature_count"] == 450
    assert summaries["full"]["held_out_success_count"] == 4
    assert summaries["full"]["held_out_obstacles_passed"] == 52
    assert report["selected_feature_variant"] == "current_mean"
    assert report["cross_validation_passed"] is True
    assert report["tuning_passed"] is True
    assert report["calibration_passed"] is True
    assert [item["seed"] for item in report["calibration_episodes"]] == [8600, 8601]
    assert [item["obstacles_passed"] for item in report["calibration_episodes"]] == [9, 9]
    assert report["advance_to_topology_controls"] is True
    assert report["advance_to_fc2_pfl_comparison"] is True
    assert report["advance_to_navigation_release"] is False
