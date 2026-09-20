"""Audit why the v7 image row axis lacks a physical angular calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import build_controlled_stimuli
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_stimulus_coordinate_contract import implemented_front_rays

CONFIG = Path("configs/driving-v7-vertical-angular-coordinate-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_vertical_angular_coordinate_audit.py")


def evaluate_v7_vertical_angular_coordinate_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "v7_contract",
            "stimulus_coordinate_config",
            "stimulus_coordinate_evidence",
            "environment_implementation",
            "v7_implementation",
        )
    }
    v7 = yaml.safe_load((root / paths["v7_contract"]).read_text())
    coordinate = json.loads((root / paths["stimulus_coordinate_evidence"]).read_text())
    environment_source = (root / paths["environment_implementation"]).read_text()
    v7_source = (root / paths["v7_implementation"]).read_text()
    expected = config["expected"]
    width = int(v7["controlled_vision"]["width"])
    height = int(v7["controlled_vision"]["height"])
    frames = int(v7["controlled_vision"]["frames_per_stimulus"])
    if (width, height, frames) != (
        int(expected["width_pixels"]),
        int(expected["height_pixels"]),
        int(expected["frame_count"]),
    ):
        raise ValueError("v7 controlled-stimulus dimensions changed")

    rays = implemented_front_rays(width)
    if not np.array_equal(
        rays,
        np.linspace(
            float(expected["horizontal_ray_min_radians"]),
            float(expected["horizontal_ray_max_radians"]),
            width,
        ),
    ):
        raise ValueError("front-camera horizontal rays changed")
    horizontal_fov_degrees = float(np.degrees(rays[-1] - rays[0]))
    horizontal_spacing_degrees = float(np.degrees(np.diff(rays)[0]))
    if not np.isclose(horizontal_fov_degrees, expected["horizontal_fov_degrees"]):
        raise ValueError("horizontal FOV changed")
    if not np.isclose(horizontal_spacing_degrees, expected["horizontal_sample_spacing_degrees"]):
        raise ValueError("horizontal angular sample spacing changed")

    environment_fragments = (
        "def _render_view(",
        "relative_angles: np.ndarray",
        "horizon = self.image_height // 3",
        "image[-height:, column] = value",
        "relative_angles=np.linspace(-1.25, 1.25, self.image_width)",
    )
    if any(fragment not in environment_source for fragment in environment_fragments):
        raise ValueError("road-camera row construction changed")
    if "vertical_angles" in environment_source or "vertical_fov" in environment_source:
        raise ValueError("road camera gained an unreviewed vertical-angle implementation")

    stimulus_fragments = (
        'extent = width if axis == "x" else height',
        "positions = np.linspace(2, extent - 3, frames)",
        "cx, cy = (width - 1) / 2, (height - 1) / 2",
        "x = (xx - (width - 1) / 2) / width",
        "y = (yy - (height - 1) / 2) / height",
        "v[mask] = 1.0 - local_y",
    )
    if any(fragment not in v7_source for fragment in stimulus_fragments):
        raise ValueError("v7 row or retinal-v semantics changed")
    if "vertical_fov" in v7_source or "vertical_angle" in v7_source:
        raise ValueError("v7 gained an unreviewed vertical-angle implementation")

    stimuli = build_controlled_stimuli(width=width, height=height, frames=frames)
    names = [stimulus.name for stimulus in stimuli]
    if names != expected["controlled_stimulus_names"]:
        raise ValueError("controlled stimulus inventory changed")
    if any(stimulus.frames.shape != (frames, height, width) for stimulus in stimuli):
        raise ValueError("controlled stimulus shape changed")
    up = next(stimulus for stimulus in stimuli if stimulus.name == "on_edge_up")
    down = next(stimulus for stimulus in stimuli if stimulus.name == "on_edge_down")
    if not np.array_equal(up.frames, down.frames[:, ::-1, :]):
        raise ValueError("vertical edge reversal no longer uses row-axis mirroring")
    horizontal = coordinate["camera_coordinates"]
    if not horizontal["horizontal_angular_coordinates_complete"]:
        raise ValueError("horizontal coordinate prerequisite is no longer complete")
    if horizontal["vertical_angular_coordinates_complete"]:
        raise ValueError("vertical coordinates unexpectedly marked complete upstream")

    gates = {
        "horizontal_camera_ray_angles_verified": True,
        "vertical_camera_ray_angles_declared": False,
        "vertical_FOV_declared": False,
        "vertical_pixel_to_angle_formula_declared": False,
        "controlled_vertical_motion_has_physical_angular_units": False,
        "looming_radius_has_physical_angular_units": False,
        "rotation_has_physical_camera_angular_units": False,
        "MaleCNS_retinal_v_has_physical_angular_units": False,
        "external_recording_vertical_alignment_verified": False,
    }
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
            "functional_stimulus_evaluated": False,
            "runtime_modified": False,
        },
        "image_geometry": {
            "width_pixels": width,
            "height_pixels": height,
            "frame_count": frames,
            "road_horizon_row": height // 3,
            "controlled_stimulus_names": names,
        },
        "coordinate_semantics": {
            "road_camera_columns": "explicit_horizontal_ray_angles_radians",
            "road_camera_rows": "synthetic_horizon_and_distance_dependent_object_height",
            "controlled_stimulus_columns": (
                "array_indices_with_declared_horizontal_ray_angle_mapping"
            ),
            "controlled_stimulus_rows": "dimensionless_array_indices",
            "MaleCNS_retinal_u_v": "normalized_optic_hex_topology_coordinates",
            "horizontal_fov_degrees": horizontal_fov_degrees,
            "horizontal_sample_spacing_degrees": horizontal_spacing_degrees,
            "vertical_fov_degrees": None,
            "vertical_sample_spacing_degrees": None,
        },
        "gates": gates,
        "offline_horizontal_angular_coordinate_complete": True,
        "offline_two_dimensional_angular_calibration_complete": False,
        "biological_angular_calibration_complete": False,
        "authorize_vertical_angular_speed_conversion": False,
        "authorize_looming_angular_size_conversion": False,
        "authorize_source_dynamics_transfer": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "horizontal_rays_exist_but_image_rows_and_retinal_v_have_no_"
            "physical_vertical_angle_calibration"
        ),
        "boundary": config["boundary"],
    }
