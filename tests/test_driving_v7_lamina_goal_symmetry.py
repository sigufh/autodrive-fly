import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-lamina-goal-symmetry.json"


def test_lamina_goal_symmetry_is_tuning_only_and_pair_isolated() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["tuning_only"] is True
    assert protocol["mirror_pair_is_indivisible_cv_unit"] is True
    assert protocol["calibration_evaluated"] is False
    assert protocol["external_final_evaluated"] is False


def test_odd_only_and_discrete_goals_do_not_improve_closed_loop() -> None:
    report = json.loads(REPORT.read_text())
    results = {item["candidate"]["name"]: item for item in report["candidate_summaries"]}
    assert [
        results[name]["held_out_obstacles_passed"]
        for name in ("full_continuous", "odd_continuous", "odd_hold_010", "odd_ternary")
    ] == [40, 20, 20, 16]
    assert results["full_continuous"]["held_out_success_count"] == 2
    assert results["odd_continuous"]["held_out_success_count"] == 0
    assert results["odd_hold_010"]["held_out_success_count"] == 0
    assert results["odd_ternary"]["held_out_success_count"] == 0
    assert report["selected_candidate"]["name"] == "full_continuous"
    assert report["cross_validation_passed"] is False
    assert report["advance_to_full_tuning"] is False
    assert report["advance_to_calibration"] is False
