"""Audit explicit physical coordinates for offline v7 visual stimuli."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-stimulus-coordinate-contract.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_stimulus_coordinate_contract.py")


class _RayCapture:
    def __init__(self, width: int):
        self.image_width = width
        self.relative_angles: np.ndarray | None = None

    def _render_view(
        self, *, relative_angles: np.ndarray, update_front_sensors: bool
    ) -> np.ndarray:
        if not update_front_sensors:
            raise ValueError("front camera must update front sensors")
        self.relative_angles = np.asarray(relative_angles, dtype=np.float64)
        return self.relative_angles


def implemented_front_rays(width: int) -> np.ndarray:
    capture = _RayCapture(width)
    DrivingEnvironment.observe(capture)  # type: ignore[arg-type]
    if capture.relative_angles is None:
        raise ValueError("front camera did not expose its ray coordinates")
    return capture.relative_angles


def evaluate_v7_stimulus_coordinate_contract(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        key: Path(config[key])
        for key in (
            "v7_contract",
            "stage1_split_protocol",
            "stage1_split_evidence",
            "environment_implementation",
        )
    }
    v7 = yaml.safe_load((root / paths["v7_contract"]).read_text(encoding="utf-8"))
    split_config = yaml.safe_load(
        (root / paths["stage1_split_protocol"]).read_text(encoding="utf-8")
    )
    split = json.loads((root / paths["stage1_split_evidence"]).read_text(encoding="utf-8"))
    time = config["time_coordinates"]
    camera = config["camera_coordinates"]
    scheduler = split_config["engineering_timebase"]
    width = int(v7["controlled_vision"]["width"])
    frame_ms = float(time["frame_interval_milliseconds"])
    substep_ms = float(time["substep_interval_milliseconds"])
    substeps = int(time["neural_substeps_per_frame"])
    rays = implemented_front_rays(width)
    expected_rays = np.linspace(
        float(camera["horizontal_ray_min_radians"]),
        float(camera["horizontal_ray_max_radians"]),
        int(camera["horizontal_ray_count"]),
    )
    spacing_radians = float(np.diff(rays)[0])
    spacing_degrees = float(np.degrees(spacing_radians))
    scheduler_matches = bool(
        frame_ms == float(scheduler["frame_interval_milliseconds"])
        and substep_ms == float(scheduler["nominal_substep_interval_milliseconds"])
        and substeps == int(scheduler["neural_substeps_per_frame"])
        and scheduler == split["engineering_timebase"]
    )
    horizontal_matches = bool(np.array_equal(rays, expected_rays))
    uniform_and_inclusive = bool(
        np.allclose(np.diff(rays), spacing_radians)
        and rays[0] == expected_rays[0]
        and rays[-1] == expected_rays[-1]
    )
    gates = {
        "frame_interval_explicit_and_positive": frame_ms > 0,
        "substep_interval_explicit_and_positive": substep_ms > 0,
        "frame_and_substep_intervals_consistent": bool(np.isclose(frame_ms, substep_ms * substeps)),
        "stage1_scheduler_matches_contract": scheduler_matches,
        "camera_width_matches_controlled_vision": int(camera["horizontal_ray_count"]) == width,
        "horizontal_camera_rays_match_implementation": horizontal_matches,
        "horizontal_camera_rays_are_uniform_and_inclusive": uniform_and_inclusive,
    }
    split_speeds = {
        name: [float(value) for value in values["edge_speeds_pixels_per_frame"]]
        for name, values in split_config["splits"].items()
    }
    angular_speeds = {
        name: [speed * spacing_degrees / (frame_ms / 1000.0) for speed in speeds]
        for name, speeds in split_speeds.items()
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "scope": config["scope"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in paths.values()},
            },
            "parameter_fit": False,
            "functional_stimulus_evaluated": False,
            "runtime_modified": False,
        },
        "time_coordinates": {
            "frame_interval_milliseconds": frame_ms,
            "frame_interval_seconds": frame_ms / 1000.0,
            "neural_substeps_per_frame": substeps,
            "substep_interval_milliseconds": substep_ms,
            "substep_interval_seconds": substep_ms / 1000.0,
            "frame_timestamp_convention": time["frame_timestamp_convention"],
            "applied_to_offline_v7_controlled_visual_evaluation": True,
            "applied_to_default_runtime": False,
            "biologically_calibrated": False,
        },
        "camera_coordinates": {
            "projection": camera["projection"],
            "horizontal_ray_count": len(rays),
            "horizontal_ray_min_radians": float(rays[0]),
            "horizontal_ray_max_radians": float(rays[-1]),
            "horizontal_fov_radians": float(rays[-1] - rays[0]),
            "horizontal_fov_degrees": float(np.degrees(rays[-1] - rays[0])),
            "angular_sample_spacing_radians": spacing_radians,
            "angular_sample_spacing_degrees": spacing_degrees,
            "ray_endpoint_convention": camera["ray_endpoint_convention"],
            "ray_angles_radians": rays.tolist(),
            "horizontal_angular_coordinates_complete": horizontal_matches and uniform_and_inclusive,
            "vertical_angular_coordinates_complete": False,
            "two_dimensional_angular_calibration_complete": False,
        },
        "stage1_moving_edge_coordinates": {
            "pixels_per_frame_by_split": split_speeds,
            "degrees_per_second_by_split": angular_speeds,
            "conversion": (
                "pixels_per_frame * angular_sample_spacing_degrees / frame_interval_seconds"
            ),
        },
        "contract_gates": gates,
        "offline_time_coordinate_contract_complete": all(
            gates[name]
            for name in (
                "frame_interval_explicit_and_positive",
                "substep_interval_explicit_and_positive",
                "frame_and_substep_intervals_consistent",
                "stage1_scheduler_matches_contract",
            )
        ),
        "offline_horizontal_stimulus_coordinate_contract_complete": all(gates.values()),
        "offline_two_dimensional_stimulus_coordinate_contract_complete": False,
        "biological_timebase_calibrated": False,
        "external_recording_alignment_verified": False,
        "authorize_source_dynamics_transfer": False,
        "authorize_T4_T5_functional_candidate": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "engineering_coordinates_defined_but_vertical_angle_and_biological_"
            "source_transfer_unresolved"
        ),
        "boundary": config["boundary"],
    }
