import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-visual-layer-locality.json"


def test_visual_layer_locality_is_tuning_only_and_neural() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["tuning_only"] is True
    assert protocol["direct_R1_R6_readout_used"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["external_final_evaluated"] is False
    assert protocol["runtime_modified"] is False


def test_lamina_retains_most_local_information_but_fails_closed_loop() -> None:
    report = json.loads(REPORT.read_text())
    results = {item["family"]: item for item in report["family_results"]}
    assert set(results) == {"lamina", "T4_upstream", "T5_upstream", "T4_T5_output"}
    assert report["selected_family"] == "lamina"
    assert results["lamina"]["held_out_safety_mae"] < results["T5_upstream"][
        "held_out_safety_mae"
    ]
    assert results["T5_upstream"]["held_out_safety_mae"] < results["T4_upstream"][
        "held_out_safety_mae"
    ]
    assert results["T4_upstream"]["held_out_safety_mae"] < results[
        "T4_T5_output"
    ]["held_out_safety_mae"]
    assert results["lamina"]["held_out_safety_mean_r2"] > 0.91
    assert [fold["success_count"] for fold in report["closed_loop_folds"]] == [0, 0, 0]
    assert [fold["total_obstacles_passed"] for fold in report["closed_loop_folds"]] == [
        14,
        2,
        0,
    ]
    assert report["closed_loop_success_count"] == 0
    assert report["closed_loop_obstacles_passed"] == 16
    assert report["cross_validation_passed"] is False
    assert report["advance_to_full_tuning"] is False
    assert report["advance_to_calibration"] is False
