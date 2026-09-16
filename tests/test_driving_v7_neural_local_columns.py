import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-neural-local-columns.json"


def test_local_column_candidate_is_hash_bound_and_isolated() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["visual_input_only_via_R1_R6"] is True
    assert protocol["target_activity_injection"] is False
    assert protocol["raw_heading_read_by_controller"] is False
    assert protocol["runtime_modified"] is False
    assert protocol["external_final_evaluated"] is False


def test_local_column_structure_has_dense_real_neural_coverage() -> None:
    structure = json.loads(REPORT.read_text())["structure"]
    assert structure["motion_spatial_bins"] == 24
    assert len(structure["motion_bin_counts"]) == 48
    assert structure["minimum_motion_bin_count"] == 44
    assert structure["unmapped_flow_target_count"] == 1
    assert structure["looming_populations"] == [
        "LPLC1_L",
        "LPLC1_R",
        "LPLC2_L",
        "LPLC2_R",
        "LC4_L",
        "LC4_R",
    ]


def test_local_column_candidate_overfits_and_does_not_advance() -> None:
    report = json.loads(REPORT.read_text())
    assert report["model"]["fit_r2"] == [
        0.9640424540538781,
        0.9865763556319248,
        0.977917601052383,
    ]
    assert report["tuning_passed"] is True
    assert all(item["obstacles_passed"] == 9 for item in report["tuning_episodes"])
    assert report["calibration_passed"] is False
    assert [item["obstacles_passed"] for item in report["calibration_episodes"]] == [8, 8]
    attribution = report["calibration_failure_attribution"]
    assert attribution["role"] == "post_failure_attribution_only_not_parameter_selection"
    assert attribution["six_bin_reference_passes_both"] is True
    assert attribution["r1r6_local_upper_bound_passes_both"] is True
    assert attribution["diagnosis"] == "24_bin_tuning_overfit"
    assert report["advance_to_topology_controls"] is False
    assert report["advance_to_navigation_release"] is False
