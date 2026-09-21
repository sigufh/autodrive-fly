import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-tm2-loo-full-support-sensitivity.json"


def test_tm2_loo_sensitivity_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["stimulus_count_per_mode"] == 120
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_tm2_loo_contract_is_fixed_without_claiming_independent_flies() -> None:
    report = json.loads(REPORT.read_text())
    contract = report["sensitivity_contract"]
    assert contract["source"] == "Tm2"
    assert contract["held_out_recording_ids"] == [
        "200927",
        "201013",
        "201119",
        "201125",
        "210721",
    ]
    assert contract["fold_count"] == 5
    assert contract["tested_brain_updates_per_frame"] == [1, 4]
    assert contract["source_input_samples"] == 27
    assert contract["kernel_samples"] == 499
    assert contract["output_samples"] == 525
    assert contract["fixed_target_denominator"] == 6719
    assert contract["valid_target_count"] == 6715
    assert report["independent_biological_validation_performed"] is False
    assert report["boundary"]["recording_id_is_not_biological_individual_id"] is True


def test_tm2_loo_full_support_results_obey_robust_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "summed_filtered_source_centroid_projection": (
            (2.101569895483, 2.353461648721),
            (0.99941709653, 0.999891852471),
        ),
        "fast_pool_vs_Tm9_centroid_difference": (
            (1.364844832451, 1.506357303968),
            (0.746691826155, 0.7940622246),
        ),
        "temporal_difference_filtered_Tm_pair_reichardt": (
            (0.986089142503, 1.228792848632),
            (0.592141593503, 0.674159523867),
        ),
    }
    assert set(report["candidate_results"]) == set(expected)
    for candidate, ranges in expected.items():
        result = report["candidate_results"][candidate]
        assert set(result["folds"]) == {
            "200927",
            "201013",
            "201119",
            "201125",
            "210721",
        }
        assert result["evaluation_count"] == 10
        assert result["passed_evaluation_count"] == 0
        assert np.allclose(result["shuffle_ratio_range"], ranges[0], atol=1e-12)
        assert np.allclose(result["static_ratio_range"], ranges[1], atol=1e-12)
        assert result["passed_every_fold_and_update_count"] is False
        for fold in result["folds"].values():
            for scored in fold["by_brain_updates_per_frame"].values():
                assert set(scored["gates"].values()) == {False}
                assert scored["passed"] is False
    assert report["evaluation_count"] == 30
    assert report["passed_evaluation_count"] == 0
    assert report["all_candidates_failed_every_fold_and_update_count"] is True
    assert report["robust_temporal_identifiability_passed"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["direction_scoring_performed"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["authorize_physical_source_dynamics_transfer"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
