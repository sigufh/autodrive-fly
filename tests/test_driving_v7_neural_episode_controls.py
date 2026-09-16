import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-neural-episode-controls.json"


def test_episode_controls_are_post_calibration_and_frozen() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["phase"] == "post_calibration_frozen_controls"
    assert protocol["model_changed_after_calibration"] is False
    assert protocol["controls_used_for_selection"] is False
    assert protocol["external_final_evaluated"] is False


def test_current_mean_candidate_requires_visual_and_heading_channels() -> None:
    report = json.loads(REPORT.read_text())
    tuning = report["ablations"]
    calibration = report["calibration_ablations"]
    assert [tuning["full"]["success_count"], tuning["full"]["total_obstacles_passed"]] == [6, 54]
    assert [
        calibration["full"]["success_count"],
        calibration["full"]["total_obstacles_passed"],
    ] == [2, 18]
    assert tuning["no_T4_T5_spatial"]["success_count"] == 0
    assert calibration["no_T4_T5_spatial"]["total_obstacles_passed"] == 6
    assert calibration["no_LPLC1"]["total_obstacles_passed"] == 6
    assert calibration["no_LC4"]["total_obstacles_passed"] == 8
    assert calibration["no_all_LPLC_LC"]["total_obstacles_passed"] == 8
    assert tuning["frozen_heading"]["success_count"] == 0
    assert tuning["reversed_yaw_update"]["success_count"] == 0
    assert report["all_LPLC_LC_collectively_redundant_under_frozen_candidate"] is False
    assert all(report["causal_gates"].values())


def test_current_mean_candidate_requires_real_malecns_topology() -> None:
    report = json.loads(REPORT.read_text())
    tuning = report["topology_controls"]
    calibration = report["calibration_topology_controls"]
    assert [
        tuning["real_malecns"]["success_count"],
        tuning["real_malecns"]["total_obstacles_passed"],
    ] == [6, 54]
    assert [
        calibration["real_malecns"]["success_count"],
        calibration["real_malecns"]["total_obstacles_passed"],
    ] == [2, 18]
    assert calibration["shuffled_retina_coordinates"]["total_obstacles_passed"] == 4
    assert calibration["source_preserving_target_shuffle"]["total_obstacles_passed"] == 0
    assert calibration["shuffled_transmitter_signs"]["total_obstacles_passed"] == 14
    assert all(report["topology_gates"].values())
    assert report["real_topology_advantage_passed"] is True
    assert report["advance_to_fc2_pfl_comparison"] is True
    assert report["advance_to_navigation_release"] is False
