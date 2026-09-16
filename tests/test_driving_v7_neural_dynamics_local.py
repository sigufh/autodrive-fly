import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-neural-dynamics-local.json"


def test_dynamics_screen_is_tuning_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["tuning_only"] is True
    assert protocol["diagnostic_two_state_views"] is True
    assert protocol["calibration_evaluated"] is False
    assert protocol["external_final_evaluated"] is False
    assert protocol["runtime_modified"] is False


def test_best_safety_dynamics_still_fails_closed_loop() -> None:
    report = json.loads(REPORT.read_text())
    results = {item["backend"]: item for item in report["backend_results"]}
    assert set(results) == {
        "typed_visual_leak_v1",
        "typed_visual_subgraph_v1",
        "columnar_delay_v1",
        "columnar_correlator_v1",
    }
    assert report["selected_backend"] == "typed_visual_leak_v1"
    assert results["typed_visual_leak_v1"]["held_out_safety_mae"] < results[
        "typed_visual_subgraph_v1"
    ]["held_out_safety_mae"]
    assert results["typed_visual_leak_v1"]["held_out_safety_mean_r2"] > 0.85
    assert [fold["success_count"] for fold in report["closed_loop_folds"]] == [0, 0, 0]
    assert [fold["total_obstacles_passed"] for fold in report["closed_loop_folds"]] == [
        8,
        4,
        6,
    ]
    assert report["closed_loop_success_count"] == 0
    assert report["closed_loop_obstacles_passed"] == 18
    assert report["cross_validation_passed"] is False
    assert report["advance_to_single_state_refactor"] is False
    assert report["advance_to_calibration"] is False
