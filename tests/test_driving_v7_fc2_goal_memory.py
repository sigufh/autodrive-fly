import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-fc2-goal-memory.json"


def test_fc2_goal_memory_is_tuning_only_pair_isolated_and_equivariant() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["tuning_only"] is True
    assert protocol["mirror_pair_is_indivisible_cv_unit"] is True
    assert protocol["odd_heads_have_zero_intercept"] is True
    assert protocol["calibration_evaluated"] is False
    assert protocol["external_final_evaluated"] is False
    for summary in report["candidate_summaries"]:
        for fold in summary["folds"]:
            assert not set(fold["training_seeds"]) & set(fold["held_out_seeds"])


def test_fc2_goal_memory_improves_but_does_not_pass() -> None:
    report = json.loads(REPORT.read_text())
    results = {
        (item["candidate"]["feature_parity"], item["candidate"]["memory_rho"]): item
        for item in report["candidate_summaries"]
    }
    assert results[("even_and_odd", 0.0)]["held_out_obstacles_passed"] == 40
    assert results[("even_and_odd", 0.5)]["held_out_obstacles_passed"] == 44
    assert results[("odd_plus_paired_even_times_odd", 0.0)][
        "held_out_obstacles_passed"
    ] == 46
    assert results[("odd_plus_paired_even_times_odd", 0.8)][
        "held_out_obstacles_passed"
    ] == 46
    assert all(
        item["maximum_mirror_goal_error"] < 1e-8
        for key, item in results.items()
        if key[0] == "odd_plus_paired_even_times_odd"
    )
    assert report["selected_candidate"] == {
        "feature_parity": "odd_plus_paired_even_times_odd",
        "memory_rho": 0.8,
    }
    assert report["cross_validation_passed"] is False
    assert report["advance_to_full_tuning"] is False
    assert report["advance_to_calibration"] is False
