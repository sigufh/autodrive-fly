"""Define an offline 2-D engineering angular grid without camera claims."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import V7VisualProbe, _moving_edge, build_controlled_stimuli
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-controlled-stimulus-angular-grid.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_controlled_stimulus_angular_grid.py")


def _edge_centres(frames: np.ndarray, axis: str) -> np.ndarray:
    centres = []
    for frame in frames:
        line = frame[0] if axis == "x" else frame[:, 0]
        centres.append(int(np.flatnonzero(line == np.max(frame))[-1]))
    return np.asarray(centres, dtype=np.int32)


def evaluate_v7_controlled_stimulus_angular_grid(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "v7_contract",
            "stimulus_coordinate_evidence",
            "vertical_coordinate_evidence",
            "v7_implementation",
        )
    }
    v7 = yaml.safe_load((root / paths["v7_contract"]).read_text())
    coordinate = json.loads((root / paths["stimulus_coordinate_evidence"]).read_text())
    negative = json.loads((root / paths["vertical_coordinate_evidence"]).read_text())
    grid = config["grid"]
    expected = config["expected"]
    width = int(v7["controlled_vision"]["width"])
    height = int(v7["controlled_vision"]["height"])
    frames = int(v7["controlled_vision"]["frames_per_stimulus"])
    if (width, height) != (int(grid["width_pixels"]), int(grid["height_pixels"])):
        raise ValueError("controlled-stimulus image dimensions changed")

    horizontal = coordinate["camera_coordinates"]
    azimuth = np.asarray(horizontal["ray_angles_radians"], dtype=np.float64)
    pitch = float(horizontal["angular_sample_spacing_radians"])
    elevation = ((height - 1) / 2 - np.arange(height, dtype=np.float64)) * pitch
    azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation)
    if azimuth_grid.shape != (height, width) or elevation_grid.shape != (height, width):
        raise ValueError("engineering angular grid shape changed")
    if not np.allclose(np.diff(elevation), -pitch):
        raise ValueError("vertical angular pitch is not uniform")
    if not np.isclose(float(elevation.mean()), 0.0, atol=1e-15):
        raise ValueError("vertical angular grid is not centered")

    pitch_degrees = float(np.degrees(pitch))
    vertical_fov_degrees = float(np.degrees(elevation[0] - elevation[-1]))
    values = {
        "horizontal_fov_degrees": float(horizontal["horizontal_fov_degrees"]),
        "angular_pixel_pitch_degrees": pitch_degrees,
        "vertical_fov_degrees": vertical_fov_degrees,
        "vertical_minimum_degrees": float(np.degrees(elevation[-1])),
        "vertical_maximum_degrees": float(np.degrees(elevation[0])),
    }
    for name, value in values.items():
        if not np.isclose(value, float(expected[name]), atol=1e-12):
            raise ValueError(f"controlled angular grid changed: {name}")

    frame_seconds = float(grid["frame_interval_seconds"])
    right = _moving_edge(
        width=width, height=height, frames=frames, axis="x", direction=1, bright=True
    )
    down = _moving_edge(
        width=width, height=height, frames=frames, axis="y", direction=1, bright=True
    )
    horizontal_centres = _edge_centres(right, "x")
    vertical_centres = _edge_centres(down, "y")
    horizontal_speed = float(
        (horizontal_centres[-1] - horizontal_centres[0])
        * pitch_degrees
        / ((frames - 1) * frame_seconds)
    )
    vertical_speed = float(
        (vertical_centres[-1] - vertical_centres[0])
        * pitch_degrees
        / ((frames - 1) * frame_seconds)
    )
    if not np.isclose(horizontal_speed, expected["horizontal_edge_mean_speed_degrees_per_second"]):
        raise ValueError("horizontal edge angular speed changed")
    if not np.isclose(vertical_speed, expected["vertical_edge_mean_speed_degrees_per_second"]):
        raise ValueError("vertical edge angular speed changed")

    stimuli = {
        stimulus.name: stimulus
        for stimulus in build_controlled_stimuli(width=width, height=height, frames=frames)
    }
    looming = stimuli["on_looming"].frames
    centre_x, centre_y = (width - 1) / 2, (height - 1) / 2
    yy, xx = np.mgrid[:height, :width]
    radii = []
    for frame in looming:
        foreground = frame > 0.5
        squared = (xx - centre_x) ** 2 + (yy - centre_y) ** 2
        radii.append(float(np.sqrt(squared[foreground].max())))
    radius_start = radii[0] * pitch_degrees
    radius_end = radii[-1] * pitch_degrees
    if not np.isclose(radius_start, expected["looming_radius_start_degrees"]):
        raise ValueError("looming start radius changed")
    if not np.isclose(radius_end, expected["looming_radius_end_degrees"]):
        raise ValueError("looming end radius changed")

    retinal_source = inspect.getsource(V7VisualProbe._retinal_coordinates)
    if "v[mask] = 1.0 - local_y" not in retinal_source:
        raise ValueError("retinal-v normalization changed")
    if not negative["boundary"]["image_rows_are_not_vertical_camera_rays"]:
        raise ValueError("vertical camera-ray boundary changed")

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in paths.values()},
            },
            "parameter_fit": False,
            "stimulus_arrays_modified": False,
            "default_camera_modified": False,
            "runtime_modified": False,
        },
        "engineering_angular_grid": {
            "projection": "equiangular_raster",
            "width_pixels": width,
            "height_pixels": height,
            "azimuth_radians": azimuth.tolist(),
            "elevation_radians": elevation.tolist(),
            "azimuth_grid_shape": list(azimuth_grid.shape),
            "elevation_grid_shape": list(elevation_grid.shape),
            **values,
            "elevation_sign": grid["elevation_sign"],
            "center_convention": grid["center_convention"],
        },
        "controlled_stimulus_units": {
            "horizontal_edge_mean_speed_degrees_per_second": horizontal_speed,
            "vertical_edge_mean_speed_degrees_per_second": vertical_speed,
            "up_is_increasing_elevation": True,
            "down_is_decreasing_elevation": True,
            "looming_radius_start_degrees": radius_start,
            "looming_radius_end_degrees": radius_end,
            "rotation_phase_is_not_camera_yaw_or_roll": True,
        },
        "gates": {
            "offline_two_dimensional_engineering_angular_grid_complete": True,
            "default_camera_two_dimensional_ray_calibration_complete": False,
            "biological_angular_calibration_complete": False,
            "external_recording_alignment_verified": False,
        },
        "authorize_offline_angular_stimulus_reporting": True,
        "authorize_source_dynamics_transfer": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "offline_engineering_grid_is_complete_but_camera_biological_and_"
            "external_recording_calibration_remain_unverified"
        ),
        "boundary": config["boundary"],
    }
