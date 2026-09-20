import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-vertical-angular-coordinate-audit.json"


def test_vertical_coordinate_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["functional_stimulus_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_only_horizontal_camera_axis_has_physical_angles() -> None:
    report = json.loads(REPORT.read_text())
    geometry = report["coordinate_semantics"]
    assert geometry["horizontal_fov_degrees"] > 143
    assert geometry["vertical_fov_degrees"] is None
    assert geometry["vertical_sample_spacing_degrees"] is None
    assert geometry["road_camera_rows"] == (
        "synthetic_horizon_and_distance_dependent_object_height"
    )
    assert geometry["MaleCNS_retinal_u_v"] == ("normalized_optic_hex_topology_coordinates")
    gates = report["gates"]
    assert gates["horizontal_camera_ray_angles_verified"] is True
    assert gates["vertical_camera_ray_angles_declared"] is False
    assert gates["vertical_FOV_declared"] is False
    assert gates["vertical_pixel_to_angle_formula_declared"] is False


def test_missing_vertical_calibration_keeps_all_downstream_gates_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["offline_horizontal_angular_coordinate_complete"] is True
    assert report["offline_two_dimensional_angular_calibration_complete"] is False
    assert report["biological_angular_calibration_complete"] is False
    assert report["authorize_vertical_angular_speed_conversion"] is False
    assert report["authorize_looming_angular_size_conversion"] is False
    assert report["authorize_source_dynamics_transfer"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
