import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from fly_emotion.driving.v7_t5_lamina_only_measured_kernel_replacement import (
    _lamina_only_source_matrices,
)

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-lamina-only-measured-kernel-replacement.json"


def test_lamina_source_matrix_keeps_weights_unsigned_for_one_sign_application() -> None:
    class Probe:
        node_types = np.asarray(["L1", "L2", "Tm1"], dtype=object)
        source_sign = np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
        adjacency = sparse.csr_matrix(
            np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.25, 0.75, 0.0]]
            )
        )

    matrices, coverage = _lamina_only_source_matrices(
        Probe(), ["Tm1"], ["L1", "L2"]
    )
    matrix = matrices["Tm1"]
    assert np.allclose(matrix.toarray(), [[0.25, 0.75, 0.0]])
    state = np.asarray([2.0, 3.0, 0.0])
    assert np.allclose(matrix @ (state * Probe.source_sign), [1.75])
    assert coverage["Tm1"]["source_node_with_lamina_input_count"] == 1


def test_lamina_only_replacement_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["stimulus_count_per_mode"] == 120
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_replacement_contract_removes_tm_dynamics_and_preserves_fixed_denominator() -> None:
    contract = json.loads(REPORT.read_text())["replacement_contract"]
    assert contract["retained_source_types"] == ["L1", "L2", "L3"]
    assert contract["replaced_source_types"] == ["Tm1", "Tm2", "Tm4", "Tm9"]
    assert contract["preserve_synchronous_prior_lamina_state_order"] is True
    assert contract["remove_Tm_leak"] is True
    assert contract["remove_Tm_tanh"] is True
    assert contract["remove_non_lamina_visual_inputs"] is True
    assert contract["remove_Tm_T4_T5_recurrent_or_feedback_inputs"] is True
    assert contract["remove_CT1_input"] is True
    assert contract["source_input_samples"] == 27
    assert contract["kernel_samples"] == 499
    assert contract["output_samples"] == 525
    assert contract["tested_brain_updates_per_frame"] == [1, 4]
    assert contract["fixed_target_denominator"] == 6719
    assert contract["valid_target_count"] == 6715
    assert {
        source: item["source_node_with_lamina_input_count"]
        for source, item in contract["source_input_coverage"].items()
    } == {"Tm1": 1774, "Tm2": 1765, "Tm4": 1670, "Tm9": 1770}


def test_lamina_only_replacement_obeys_temporal_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "summed_filtered_source_centroid_projection": {
            "1": (1.5698382891422213, 0.5056834500354587),
            "4": (1.6164007523973414, 0.4988382782934325),
        },
        "fast_pool_vs_Tm9_centroid_difference": {
            "1": (2.3605189049473343, 0.44503033278039983),
            "4": (2.2398989473633173, 0.42111038663450145),
        },
        "temporal_difference_filtered_Tm_pair_reichardt": {
            "1": (1.6853864631000266, 0.3191557479999038),
            "4": (1.6277322953083266, 0.3327291401605626),
        },
    }
    assert set(report["candidate_results"]) == set(expected)
    for name, by_update in expected.items():
        result = report["candidate_results"][name]
        for update, ratios in by_update.items():
            scored = result["by_brain_updates_per_frame"][update]
            assert np.isclose(scored["shuffle_to_ordered_residual_energy_ratio"], ratios[0])
            assert np.isclose(scored["static_to_ordered_energy_ratio"], ratios[1])
            assert scored["shuffle_to_ordered_residual_energy_ratio"] > 0.50
            assert scored["passed"] is False
        assert result["passed_every_update_count"] is False
    assert report["all_candidates_failed_every_update_count"] is True
    assert report["temporal_identifiability_passed"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["direction_scoring_performed"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["authorize_physical_source_dynamics_transfer"] is False


def test_replacement_boundary_does_not_open_downstream_stages() -> None:
    report = json.loads(REPORT.read_text())
    boundary = report["boundary"]
    assert boundary["structural_source_replacement_diagnostic_only"] is True
    assert boundary["no_external_state_or_gain_mapping_claimed"] is True
    assert boundary["source_activity_driven_only_from_R1_R6_through_L1_L2_L3"] is True
    assert boundary["subtype_and_direction_labels_not_read_by_dynamics"] is True
    assert boundary["target_activity_injection"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
