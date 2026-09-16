import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
LAYER = ROOT / "artifacts/v7-visual-layer-locality.json"
GOAL = ROOT / "artifacts/v7-lamina-goal.json"


def test_visual_layer_locality_selects_lamina_without_using_r1r6_readout() -> None:
    report = json.loads(LAYER.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["direct_R1_R6_readout_used"] is False
    assert report["selected_family"] == "lamina"
    results = {item["family"]: item for item in report["family_results"]}
    assert results["lamina"]["held_out_safety_mae"] < results["T5_upstream"][
        "held_out_safety_mae"
    ]
    assert results["T5_upstream"]["held_out_safety_mae"] < results["T4_upstream"][
        "held_out_safety_mae"
    ]
    assert results["T4_upstream"]["held_out_safety_mae"] < results[
        "T4_T5_output"
    ]["held_out_safety_mae"]
    assert report["closed_loop_success_count"] == 0
    assert report["closed_loop_obstacles_passed"] == 16
    assert report["cross_validation_passed"] is False


def test_lamina_direct_goal_improves_but_does_not_pass_pair_cv() -> None:
    report = json.loads(GOAL.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["tuning_only"] is True
    assert report["protocol"]["held_out_pairs_never_labeled_for_fit"] is True
    assert report["protocol"]["teacher_actions_injected"] is False
    assert report["structure"] == {
        "family": "lamina",
        "declared_target_count": 7114,
        "unmapped_target_count": 915,
        "minimum_bin_count": 54,
        "feature_dimension": 48,
    }
    assert [fold["success_count"] for fold in report["folds"]] == [0, 0, 2]
    assert [fold["total_obstacles_passed"] for fold in report["folds"]] == [14, 8, 18]
    assert report["success_count"] == 2
    assert report["total_obstacles_passed"] == 40
    assert report["cross_validation_passed"] is False
    assert report["advance_to_full_tuning"] is False
    assert report["advance_to_calibration"] is False
