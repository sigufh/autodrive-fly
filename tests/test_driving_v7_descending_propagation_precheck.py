import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-descending-propagation-precheck.json"


def test_descending_precheck_is_fixed_environment_blind_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["stimulus_generated_without_environment"] is True
    assert protocol["vehicle_not_run"] is True
    assert protocol["parameter_fit"] is False
    assert protocol["threshold_search"] is False
    assert protocol["calibration_consumed"] is False
    assert protocol["final_consumed"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["runtime_modified"] is False
    assert report["thresholds"] == {
        "minimum_absolute_readout": 0.000001,
        "maximum_normalized_mirror_error": 0.20,
    }
    assert report["threshold_provenance"] == {
        "minimum_absolute_readout": "stage1_scoring.minimum_valid_denominator",
        "maximum_normalized_mirror_error": (
            "stage1_scoring.maximum_energy_weighted_mirror_error"
        ),
    }


def test_real_graph_dna02_state_passes_but_dna01_and_dnp20_stop() -> None:
    report = json.loads(REPORT.read_text())
    results = report["target_readout_results"]
    assert results["DNa02"]["selected_hop_from_maximum_bilateral_shortest_depth"] == 1
    assert results["DNa01"]["selected_hop_from_maximum_bilateral_shortest_depth"] == 2
    assert results["DNp20"]["selected_hop_from_maximum_bilateral_shortest_depth"] == 3
    assert results["DNa02"]["precheck_passed"] is True
    assert results["DNa01"]["magnitude_gate_passed"] is True
    assert results["DNa01"]["opponent_sign_gate_passed"] is True
    assert results["DNa01"]["mirror_gate_passed"] is False
    assert results["DNa01"]["precheck_passed"] is False
    assert results["DNp20"]["precheck_passed"] is False
    assert results["DNp20"]["magnitude_gate_passed"] is False
    assert results["DNp20"]["opponent_sign_gate_passed"] is False
    assert results["DNp20"]["mirror_gate_passed"] is False
    assert report["passing_target_populations"] == ["DNa02"]
    assert report["failed_target_populations"] == ["DNa01", "DNp20"]


def test_precheck_reads_real_simulated_states_without_authorizing_release() -> None:
    report = json.loads(REPORT.read_text())
    assert report["real_graph_propagated_target_states_were_read"] is True
    assert report["biological_neural_states_measured"] is False
    assert report["physiological_neural_states_claimed"] is False
    assert report["algebraic_action_equivalence_used_as_evidence"] is False
    assert report["all_descending_readouts_precheck_passed"] is False
    assert report["functional_neural_readout_validated"] is False
    assert report["advance_to_vehicle_assay"] is False
    assert report["advance_to_navigation_release"] is False
    assert report["advance_to_mushroom_body"] is False
    assert report["stop_reason"] == (
        "one_or_more_real_descending_state_readouts_failed_fixed_precheck"
    )
