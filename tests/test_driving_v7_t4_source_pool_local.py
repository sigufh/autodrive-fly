import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-pool-local.json"


def test_source_pool_local_is_read_only_tuning_diagnostic() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_id"] == "LPLC-T01"
    assert protocol["target_count"] == 6861
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["runtime_modified"] is False


def test_source_pool_has_only_unilateral_signal_and_does_not_authorize_formula() -> None:
    report = json.loads(REPORT.read_text())
    assert report["direction_pass_counts"] == {
        "center": 0,
        "proximal": 1,
        "distal": 1,
        "center_proximal_correlation": 0,
        "pairwise_proximal_vector": 1,
        "pairwise_distal_vector": 0,
        "rectified_pairwise_proximal_vector": 0,
        "rectified_pairwise_distal_vector": 0,
        "center_centroid_velocity": 0,
        "proximal_centroid_velocity": 0,
        "distal_centroid_velocity": 0,
        "anatomy_axis_motion_drive": 0,
    }
    assert all(not values for values in report["bilateral_passing_subtypes"].values())
    assert report["maximum_bilateral_direction_pair_count"] == 0
    assert report["authorize_new_target_formula"] is False
    left = report["population_scores"]["T4d_L"]["pairwise_proximal_vector"]
    right = report["population_scores"]["T4d_R"]["pairwise_proximal_vector"]
    assert left["median_signed_contrast"] > 0.10
    assert left["positive_cell_fraction"] < 0.60
    assert left["passed"] is False
    assert right["passed"] is True
    assert all(
        not report["population_scores"][population][readout]["passed"]
        for population in report["population_scores"]
        for readout in (
            "rectified_pairwise_proximal_vector",
            "rectified_pairwise_distal_vector",
        )
    )
    center_velocity = report["population_scores"]["T4d_R"]["center_centroid_velocity"]
    assert center_velocity["median_signed_contrast"] > 0.10
    assert center_velocity["positive_cell_fraction"] < 0.60
    assert center_velocity["passed"] is False
    assert report["anatomy_axis"] == {
        "valid_target_count": 6861,
        "valid_target_fraction": 1.0,
        "invalid_target_body_ids": [],
        "subtype_labels_used_by_axis": False,
        "sign_search": False,
    }
    assert all(
        not score["anatomy_axis_motion_drive"]["passed"]
        for score in report["population_scores"].values()
    )
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
