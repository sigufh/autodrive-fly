import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-neural-goal-fusion.json"


def test_goal_fusion_is_tuning_only_pair_isolated_and_equivariant() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["tuning_only"] is True
    assert protocol["one_global_fusion_weight_across_folds"] is True
    assert protocol["mirror_pair_is_indivisible_cv_unit"] is True
    assert protocol["calibration_evaluated"] is False
    assert protocol["external_final_evaluated"] is False
    for summary in report["candidate_summaries"]:
        for fold in summary["folds"]:
            assert not set(fold["training_seeds"]) & set(fold["held_out_seeds"])


def test_goal_fusion_passes_all_three_held_out_pairs() -> None:
    report = json.loads(REPORT.read_text())
    summaries = {item["lamina_weight"]: item for item in report["candidate_summaries"]}
    assert [summaries[value]["held_out_obstacles_passed"] for value in (0.0, 0.25, 0.5)] == [
        32,
        48,
        40,
    ]
    assert summaries[0.75]["held_out_success_count"] == 6
    assert summaries[0.75]["held_out_obstacles_passed"] == 54
    assert summaries[1.0]["held_out_success_count"] == 6
    assert summaries[1.0]["held_out_obstacles_passed"] == 54
    assert summaries[0.75]["maximum_mirror_goal_error"] < 1e-8
    assert report["selected_lamina_weight"] == 0.75
    assert report["cross_validation_passed"] is True
    assert report["advance_to_full_tuning"] is True
    assert report["advance_to_calibration"] is False
    assert report["advance_to_navigation_release"] is False
