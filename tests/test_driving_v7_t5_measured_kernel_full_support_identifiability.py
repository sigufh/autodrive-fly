import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_t5_measured_kernel_full_support_identifiability import (
    _full_convolve,
)

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-measured-kernel-full-support-identifiability.json"


def test_fft_full_convolution_matches_numpy_reference() -> None:
    values = np.asarray([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]])
    kernel = np.asarray([0.25, -0.5, 1.0, 0.125])
    expected = np.stack(
        [np.convolve(values[:, column], kernel, mode="full") for column in range(2)],
        axis=1,
    )
    assert np.allclose(_full_convolve(values, kernel), expected, atol=1e-12, rtol=0.0)


def test_full_support_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["stimulus_count_per_mode"] == 120
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_full_support_contract_is_fixed() -> None:
    contract = json.loads(REPORT.read_text())["full_support_contract"]
    assert contract["source_input_samples"] == 27
    assert contract["kernel_samples"] == 499
    assert contract["zero_tail_samples"] == 498
    assert contract["output_samples"] == 525
    assert contract["source_drive_extension"] == (
        "zero_after_observed_post_baseline_trace"
    )
    assert contract["neural_network_advanced_during_zero_tail"] is False
    assert contract["tested_brain_updates_per_frame"] == [1, 4]
    assert contract["fixed_target_denominator"] == 6719
    assert contract["valid_target_count"] == 6715


def test_full_support_results_obey_cross_substep_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "summed_filtered_source_centroid_projection": {
            "1": (2.235240927555609, 0.9998745971065794),
            "4": (2.1870407530168094, 0.9994703008655053),
        },
        "fast_pool_vs_Tm9_centroid_difference": {
            "1": (1.4427593685871942, 0.7917100247917601),
            "4": (1.5106499053307265, 0.7626229899532233),
        },
        "temporal_difference_filtered_Tm_pair_reichardt": {
            "1": (0.9916077454956942, 0.5963309213687981),
            "4": (1.1968601719437424, 0.6614761950279847),
        },
    }
    assert set(report["candidate_results"]) == set(expected)
    for name, by_update in expected.items():
        result = report["candidate_results"][name]
        for update, ratios in by_update.items():
            scored = result["by_brain_updates_per_frame"][update]
            assert np.isclose(scored["shuffle_to_ordered_residual_energy_ratio"], ratios[0])
            assert np.isclose(scored["static_to_ordered_energy_ratio"], ratios[1])
            assert set(scored["gates"].values()) == {False}
            assert scored["passed"] is False
        assert result["passed_every_update_count"] is False
    assert report["full_kernel_support_evaluated"] is True
    assert report["all_candidates_failed_every_update_count"] is True
    assert report["cross_substep_temporal_identifiability_passed"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["direction_scoring_performed"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["authorize_physical_source_dynamics_transfer"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False


def test_zero_tail_boundary_is_not_new_neural_or_visual_input() -> None:
    boundary = json.loads(REPORT.read_text())["boundary"]
    assert boundary["zero_tail_is_not_additional_visual_stimulus"] is True
    assert boundary["neural_network_not_advanced_during_zero_tail"] is True
    assert boundary["source_activity_driven_only_from_R1_R6_propagation"] is True
    assert boundary["subtype_and_direction_labels_not_read_by_dynamics"] is True
    assert boundary["target_activity_injection"] is False
