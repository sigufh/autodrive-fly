import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-heading-ring.json"


def test_heading_ring_is_hash_bound_and_does_not_read_raw_heading() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["visual_model_changed"] is False
    assert protocol["action_adapter_changed"] is False
    assert protocol["raw_heading_read_by_controller"] is False
    assert protocol["yaw_rate_is_proprioceptive_input"] is True
    assert protocol["development_regression_seeds_not_counted_as_calibration"] is True
    assert protocol["calibration_run_once_after_candidate_freeze"] is True
    assert protocol["external_final_evaluated"] is False


def test_heading_ring_tracks_holds_and_mirrors() -> None:
    assays = json.loads(REPORT.read_text())["heading_assays"]
    assert assays["passed"] is True
    assert all(assays["gates"].values())
    assert assays["maximum_tracking_error"] < 1e-12
    assert assays["occlusion_drift"] < 1e-12
    assert assays["maximum_mirror_error"] < 1e-12


def test_heading_ring_has_malecns_structural_support_without_overclaiming() -> None:
    structure = json.loads(REPORT.read_text())["malecns_structure"]
    assert structure["population_counts"] == {
        "EPG": 46,
        "PEN": 42,
        "PEG": 18,
        "FC2": 92,
        "PFL3": 24,
        "PFL2": 12,
        "DNa": 4,
        "DNp20": 2,
    }
    assert structure["selected_edges"]["EPG->PEN"]["edge_count"] == 664
    assert structure["selected_edges"]["PEN->EPG"]["edge_count"] == 735
    assert structure["selected_edges"]["EPG->PEG"]["edge_count"] == 379
    assert structure["selected_edges"]["PEG->EPG"]["edge_count"] == 290
    assert structure["selected_edges"]["FC2->PFL3"]["edge_count"] == 576
    assert structure["selected_edges"]["PFL3->DNa"]["edge_count"] == 24
    assert structure["structure_is_not_functional_validation"] is True


def test_neural_heading_passes_tuning_and_controls_but_fails_fresh_calibration() -> None:
    report = json.loads(REPORT.read_text())
    arms = report["navigation_arms"]
    neural = arms["neural_heading"]
    assert neural["tuning_success_count"] == 6
    assert neural["tuning_obstacles_passed"] == 54
    assert neural["calibration_success_count"] == 0
    assert neural["calibration_obstacles_passed"] == 6
    assert neural["maximum_heading_decode_error"] < 1e-12
    assert [item["seed"] for item in neural["development_regression"]] == [8300, 8301]
    assert all(item["success"] for item in neural["development_regression"])
    assert [item["seed"] for item in neural["calibration"]] == [8400, 8401]
    assert all(not item["success"] for item in neural["calibration"])
    assert arms["frozen_heading"]["tuning_success_count"] == 0
    assert arms["reversed_yaw_update"]["tuning_success_count"] == 0
    assert report["causal_controls_passed"] is True
    assert report["navigation_passed"] is False
    assert report["advance_to_fc2_pfl_comparison"] is False
    assert report["advance_to_navigation_release"] is False


def test_calibration_failure_is_visual_readout_not_heading_ring() -> None:
    attribution = json.loads(REPORT.read_text())["calibration_failure_attribution"]
    assert attribution["role"] == "post_failure_attribution_only_not_parameter_selection"
    assert attribution["raw_heading_matches_neural_heading_failure"] is True
    assert attribution["r1r6_local_upper_bound_passes_both"] is True
    assert attribution["diagnosis"] == "frozen_neural_visual_readout_generalization_gap"
    assert [item["obstacles_passed"] for item in attribution["raw_heading_reference"]] == [3, 3]
    assert [
        item["obstacles_passed"]
        for item in attribution["r1r6_local_visual_upper_bound"]
    ] == [9, 9]
