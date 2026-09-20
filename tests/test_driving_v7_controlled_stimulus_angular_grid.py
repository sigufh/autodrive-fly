import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-controlled-stimulus-angular-grid.json"


def test_angular_grid_is_hash_bound_and_does_not_modify_runtime() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["stimulus_arrays_modified"] is False
    assert report["protocol"]["default_camera_modified"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_declared_grid_has_square_angular_pitch_and_centered_elevation() -> None:
    grid = json.loads(REPORT.read_text())["engineering_angular_grid"]
    assert grid["azimuth_grid_shape"] == [24, 48]
    assert grid["elevation_grid_shape"] == [24, 48]
    elevation = np.asarray(grid["elevation_radians"])
    assert len(elevation) == 24
    assert np.isclose(elevation.mean(), 0.0)
    assert np.allclose(
        np.diff(elevation),
        -np.radians(grid["angular_pixel_pitch_degrees"]),
    )
    assert np.isclose(grid["vertical_fov_degrees"], 70.09590046813252)


def test_engineering_angles_do_not_claim_camera_or_biological_calibration() -> None:
    report = json.loads(REPORT.read_text())
    units = report["controlled_stimulus_units"]
    assert units["horizontal_edge_mean_speed_degrees_per_second"] > 873
    assert units["vertical_edge_mean_speed_degrees_per_second"] > 386
    assert units["up_is_increasing_elevation"] is True
    assert units["down_is_decreasing_elevation"] is True
    assert units["looming_radius_end_degrees"] > 33
    assert units["rotation_phase_is_not_camera_yaw_or_roll"] is True
    gates = report["gates"]
    assert gates["offline_two_dimensional_engineering_angular_grid_complete"] is True
    assert gates["default_camera_two_dimensional_ray_calibration_complete"] is False
    assert gates["biological_angular_calibration_complete"] is False
    assert gates["external_recording_alignment_verified"] is False
    assert report["authorize_offline_angular_stimulus_reporting"] is True
    assert report["authorize_source_dynamics_transfer"] is False
