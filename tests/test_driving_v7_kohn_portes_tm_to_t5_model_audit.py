import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-tm-to-t5-model-audit.json"


def test_tm_to_t5_model_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["repository_commit"] == (
        "ecc7703e506d7491d01532e9e339f2246648447f"
    )
    assert report["protocol"]["parameter_fit_for_v7"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_recovered_author_inputs_match_fixed_git_blobs_and_expected_schemas() -> None:
    report = json.loads(REPORT.read_text())
    recovered = report["recovered_model_inputs"]
    assert recovered["stationary_flash"]["condition_count"] == 6
    assert recovered["stationary_flash"]["location_trace_count"] == 84
    assert recovered["stationary_flash"]["samples_per_trace"] == 45000
    assert recovered["stationary_flash"]["full_trace_nan_count"] == 898883
    assert recovered["stationary_flash"]["response_unit"] == "millivolts"
    assert recovered["moving_bar"]["condition_count"] == 12
    assert recovered["moving_bar"]["recorded_cell_count"] == 17
    assert recovered["Shinomiya_fib19_T5_input"]["T5_count"] == 20
    assert recovered["Shinomiya_fib19_T5_input"]["source_count_sums"] == {
        "Tm1": 485,
        "Tm2": 648,
        "Tm4": 341,
        "Tm9": 1181,
        "CT1": 557,
    }
    assert recovered["Shinomiya_fib19_T5_input"]["CT1_removed_before_four_source_ratio"]
    assert recovered["Shinomiya_fib19_T5_input"]["MaleCNS_mapping"] is False


def test_figure5_training_fit_is_reproduced_but_not_independent_validation() -> None:
    report = json.loads(REPORT.read_text())
    fit = report["Figure5_Tm1_Tm9_static_flash_regression"]
    assert fit["same_samples_used_for_fit_and_score"] is True
    assert fit["source_input_normalization"] == (
        "each_source_trace_divided_by_own_positive_maximum"
    )
    representative = fit["condition_results"]["saline"]["0.16:9"]["R2"]
    assert len(representative["values"]) == 14
    assert representative["minimum"] < 0.0
    assert fit["aggregate_training_R2"]["saline"]["count"] == 84
    assert fit["aggregate_training_R2"]["OA"]["count"] == 84
    assert fit["condition_results"]["saline"]["0.16:9"][
        "fit_window_nan_count_before_author_zero_fill"
    ] == 0
    gates = report["gates"]
    assert gates["Figure5_representative_regression_reproduced"] is True
    assert gates["Figure5_all_six_conditions_recomputed"] is True
    assert gates["Figure5_fit_and_score_samples_disjoint"] is False
    assert gates["Figure5_independent_cell_validation_available"] is False
    assert gates["Figure5_absolute_input_gain_preserved"] is False


def test_figure6_limitations_and_transfer_stop_are_explicit() -> None:
    report = json.loads(REPORT.read_text())
    boundary = report["Figure6_model_boundary"]
    assert boundary["external_axolotl_tmodel_source_in_fixed_repository_tree"] is False
    assert boundary["preset_weights_declared_irrelevant_after_numeric_scaling"] is True
    assert boundary["per_source_simulation_output_peak_normalized"] is True
    assert boundary["connectome_weight_source"] == "Shinomiya_2019_fib19_twenty_T5_cells"
    assert boundary["CT1_removed_before_four_source_weight_ratios"] is True
    assert boundary["Tm2_ND_assignment_matches_Tm2_source"] is False
    assert report["author_Tm_to_T5_model_independently_validated"] is False
    assert report["authorize_Tm_to_T5_model_transfer_to_v7"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
