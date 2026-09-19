import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_stimulus_coordinate_contract import implemented_front_rays

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-stimulus-coordinate-contract.json"


def test_coordinate_contract_is_hash_bound_and_non_advancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["authorize_source_dynamics_transfer"] is False
    assert report["authorize_T4_T5_functional_candidate"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False


def test_offline_time_coordinates_are_explicit_but_not_biological() -> None:
    report = json.loads(REPORT.read_text())
    time = report["time_coordinates"]
    assert time["frame_interval_milliseconds"] == 10.0
    assert time["substep_interval_milliseconds"] == 2.5
    assert time["neural_substeps_per_frame"] == 4
    assert time["frame_timestamp_convention"] == "frame_start"
    assert time["applied_to_default_runtime"] is False
    assert time["biologically_calibrated"] is False
    assert report["offline_time_coordinate_contract_complete"] is True
    assert report["biological_timebase_calibrated"] is False


def test_horizontal_camera_coordinates_match_implementation_without_filling_vertical() -> None:
    report = json.loads(REPORT.read_text())
    camera = report["camera_coordinates"]
    assert np.array_equal(implemented_front_rays(48), np.asarray(camera["ray_angles_radians"]))
    assert camera["horizontal_ray_count"] == 48
    assert camera["horizontal_ray_min_radians"] == -1.25
    assert camera["horizontal_ray_max_radians"] == 1.25
    assert np.isclose(camera["horizontal_fov_degrees"], np.degrees(2.5))
    assert np.isclose(camera["angular_sample_spacing_degrees"], np.degrees(2.5 / 47))
    assert report["offline_horizontal_stimulus_coordinate_contract_complete"] is True
    assert report["offline_two_dimensional_stimulus_coordinate_contract_complete"] is False
    assert camera["vertical_angular_coordinates_complete"] is False


def test_stage1_edge_speeds_have_reproducible_horizontal_angular_units() -> None:
    report = json.loads(REPORT.read_text())
    values = report["stage1_moving_edge_coordinates"]
    assert values["pixels_per_frame_by_split"] == {
        "development": [1.0, 2.0],
        "validation": [0.5, 1.5],
        "ood": [0.25, 3.0],
        "final": [0.75, 2.5],
    }
    scale = report["camera_coordinates"]["angular_sample_spacing_degrees"] / 0.01
    for split, speeds in values["pixels_per_frame_by_split"].items():
        assert np.allclose(
            values["degrees_per_second_by_split"][split],
            np.asarray(speeds) * scale,
        )
    assert report["external_recording_alignment_verified"] is False
    assert report["boundary"]["does_not_establish_source_membrane_dynamics"] is True
