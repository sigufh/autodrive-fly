import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-fc2-pfl-dna.json"


def test_fc2_pfl_dna_is_frozen_transparent_and_nonreleasing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["transparent_fixed_mapping"] is True
    assert protocol["environment_geometry_read_by_controller"] is False
    assert protocol["reward_read_by_controller"] is False
    assert protocol["raw_heading_read_by_controller"] is False
    assert protocol["external_final_evaluated"] is False
    assert report["advance_to_navigation_release"] is False
    assert report["advance_to_mushroom_body"] is False


def test_fc2_pfl_dna_reports_supported_and_unsupported_connectome_claims() -> None:
    structure = json.loads(REPORT.read_text())["malecns_structure"]
    assert structure["population_counts"] == {
        "FC2": 92,
        "EPG": 46,
        "PFL3": 24,
        "PFL2": 12,
        "DNa01": 2,
        "DNa02": 2,
        "DNp20": 2,
    }
    assert structure["edges"]["FC2->PFL3"]["edge_count"] == 576
    assert structure["edges"]["EPG->PFL3"]["edge_count"] == 113
    assert structure["edges"]["PFL3->DNa02"]["edge_count"] == 24
    assert structure["edges"]["PFL3->DNa01"]["edge_count"] == 0
    assert structure["edges"]["PFL3->DNp20"]["edge_count"] == 0
    assert structure["unsupported_direct_claims"] == ["PFL3->DNa01", "PFL3->DNp20"]
    assert structure["cross_side_pfl3_dna02_edges"] == {
        "PFL3_L->DNa02_R": {"edge_count": 12, "synapse_weight": 380.0},
        "PFL3_R->DNa02_L": {"edge_count": 12, "synapse_weight": 356.0},
    }


def test_heading_goal_comparison_terminates_and_readouts_are_action_equivalent() -> None:
    report = json.loads(REPORT.read_text())
    assay = report["turn_termination_assay"]
    assert assay["passed"] is True
    assert assay["maximum_final_error"] <= 0.012
    assert assay["maximum_mirror_error"] < 1e-12
    assert all(item["monotone_error_reduction"] for item in assay["records"])
    assert all(item["pfl2_termination"] for item in assay["records"])
    assert report["readout_action_equivalence"] is True
    hierarchy = report["readout_arms"]["hierarchical_fc2_pfl3_dna02"]["tuning"]
    assert [hierarchy["success_count"], hierarchy["total_obstacles_passed"]] == [6, 54]
    for name in ("no_fc2_goal", "no_epg_heading", "no_pfl3", "no_dna02"):
        assert report["readout_arms"][name]["tuning"]["success_count"] == 0
    assert report["causal_controls_passed"] is True


def test_fc2_pfl_dna_fails_fresh_calibration_due_to_visual_readout() -> None:
    report = json.loads(REPORT.read_text())
    assert report["tuning_passed"] is True
    assert report["calibration_passed"] is False
    assert [item["seed"] for item in report["calibration_episodes"]] == [8700, 8701]
    assert [item["obstacles_passed"] for item in report["calibration_episodes"]] == [2, 2]
    attribution = report["calibration_failure_attribution"]
    assert attribution["role"] == "post_failure_attribution_only_not_parameter_selection"
    assert attribution["direct_reference_matches_hierarchical_failure"] is True
    assert attribution["r1r6_local_upper_bound_passes_both"] is True
    assert attribution["diagnosis"] == "frozen_neural_visual_readout_generalization_gap"
