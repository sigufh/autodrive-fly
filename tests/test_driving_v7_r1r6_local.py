import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_r1r6_local import (
    CONFIG,
    TUNING_ARTIFACT,
    local_visual_channels,
    reconstruct_image,
)
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina

ROOT = Path(__file__).parents[1]
CALIBRATION = ROOT / "artifacts/v7-r1r6-local-calibration.json"
CONTROLS = ROOT / "artifacts/v7-r1r6-local-controls.json"


def test_mass_balanced_retina_exactly_reconstructs_every_pixel() -> None:
    retina = build_mass_balanced_retina(ROOT)
    assert len(np.unique(retina.pixel_id)) == 24 * 24
    pixel_mass = np.bincount(retina.pixel_id, weights=retina.mass, minlength=24 * 24)
    np.testing.assert_allclose(pixel_mass, 2.0, atol=1e-12)
    image = np.arange(24 * 48, dtype=np.float64).reshape(24, 48) / (24 * 48)
    np.testing.assert_allclose(reconstruct_image(retina, retina.encode(image)), image, atol=1e-12)


def test_local_channels_separate_near_danger_and_road_center() -> None:
    config = json.loads(json.dumps(__import__("yaml").safe_load((ROOT / CONFIG).read_text())))
    image = np.full((24, 48), 0.12, dtype=np.float64)
    image[8:, 23] = 0.30
    image[18:, 14:17] = 0.98
    result = local_visual_channels(image, config)
    assert result["danger"] > 0
    assert result["obstacle_asymmetry"] > 0
    assert result["road_centroid"] < 0
    mirrored = local_visual_channels(image[:, ::-1], config)
    assert mirrored["danger"] == result["danger"]
    np.testing.assert_allclose(mirrored["obstacle_asymmetry"], -result["obstacle_asymmetry"])
    np.testing.assert_allclose(mirrored["road_centroid"], -result["road_centroid"])


def test_saved_local_tuning_and_one_time_calibration_pass_without_release() -> None:
    tuning = json.loads((ROOT / TUNING_ARTIFACT).read_text())
    calibration = json.loads(CALIBRATION.read_text())
    for report in (tuning, calibration):
        for path, digest in report["protocol"]["dependencies_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert tuning["maximum_synthetic_reconstruction_error"] <= 1e-12
    assert tuning["tuning_passed"] is True
    winner = next(
        item
        for item in tuning["candidate_summaries"]
        if item["candidate"]["throttle"] == tuning["selected_candidate"]["throttle"]
        and item["candidate"]["obstacle_gain"] == tuning["selected_candidate"]["obstacle_gain"]
        and item["candidate"]["road_gain"] == tuning["selected_candidate"]["road_gain"]
        and item["candidate"]["heading_gain"] == tuning["selected_candidate"]["heading_gain"]
    )
    assert winner["success_count"] == 6
    assert winner["total_obstacles_passed"] == 54
    assert calibration["seeds"] == [8100, 8101]
    assert calibration["calibration_passed"] is True
    assert [item["obstacles_passed"] for item in calibration["episodes"]] == [9, 9]
    assert calibration["engineering_input_upper_bound_only"] is True
    assert calibration["final_evaluated"] is False
    assert calibration["advance_to_navigation_release"] is False


def test_local_channel_controls_require_obstacle_road_and_heading_feedback() -> None:
    report = json.loads(CONTROLS.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    arms = report["arms"]
    assert arms["full"]["success_count"] == 6
    assert arms["full"]["total_obstacles_passed"] == 54
    assert arms["no_obstacle"]["success_count"] == 0
    assert arms["no_road"]["success_count"] == 4
    assert arms["no_heading"]["success_count"] == 0
    assert arms["zero_action"]["success_count"] == 0
    assert report["causal_controls_passed"] is True
    assert report["engineering_input_upper_bound_only"] is True
    assert report["final_evaluated"] is False
    assert report["advance_to_navigation_release"] is False
