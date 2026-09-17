import hashlib
import json
from pathlib import Path

from fly_emotion.driving.v7_goal_audit import evaluate_v7_goal_coverage

ROOT = Path(__file__).parents[1]


def test_goal_audit_maps_every_numbered_item_without_unlocking_later_stages() -> None:
    report = evaluate_v7_goal_coverage(ROOT)
    assert [item["item"] for item in report["checks"]] == list(range(9))
    assert report["summary"]["complete_items"] == [0]
    assert report["summary"]["boundary_only_items"] == [8]
    assert report["summary"]["component_pass_items"] == [2, 3]
    assert report["summary"]["failed_items"] == [1]
    assert report["summary"]["incomplete_items"] == [6]
    assert report["summary"]["partially_validated_items"] == [2, 3, 4]
    assert report["summary"]["not_authorized_items"] == [5, 7]
    assert report["summary"]["objective_complete"] is False
    assert report["summary"]["current_stage"] == "controlled_vision"
    disclosure = report["checks"][8]["observations"]
    assert disclosure["three_way_contribution_report_available"] is True
    assert disclosure["contribution_layers"] == [
        "upper_planner",
        "fly_local_core",
        "engineering_executor",
    ]
    assert disclosure["v7_status_is_offline_only"] is True
    checklist = {item["requirement"]: item for item in report["requirement_checklist"]}
    assert checklist["8.separate_planner_fly_core_executor_contributions"]["status"] == "passed"
    topology = report["checks"][6]["observations"]
    assert topology["strict_degree_preserving_control_complete"] is False
    assert topology["strict_visual_target_degree_preserving_control_constructed"] is True
    assert topology["strict_visual_target_degree_preserving_rewired_fraction"] > 0.80
    assert topology["strict_visual_target_control_response_evaluated"] is False
    assert topology["parameter_matched_baseline_protocol_ready"] is True
    assert topology["parameter_matched_baseline_budget"] == 1353
    assert topology["parameter_matched_baseline_counts"] == {
        "linear": 1353,
        "one_hidden_layer_mlp": 1365,
        "single_layer_gru": 1365,
    }
    assert topology["parameter_matched_baseline_evaluation_performed"] is False


def test_saved_goal_audit_is_hash_bound_and_matches_recalculation() -> None:
    saved = json.loads((ROOT / "artifacts/v7-goal-audit.json").read_text())
    for path, digest in saved["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    current = evaluate_v7_goal_coverage(ROOT)
    assert current == saved
    visual = saved["checks"][1]
    assert visual["observations"]["visual_inputs_only_at_receptors"] is True
    assert visual["observations"]["controlled_response_gates_pass"] is False
    assert visual["observations"]["T5_biological_PD_code_assigned"] is None
    assert visual["observations"]["T5_external_direction_label_map_verified"] is False
    assert visual["observations"]["T5_model_scoring_allowed"] is False
    assert visual["observations"]["T5_published_fitted_parameter_vectors_available"] is False
    assert visual["observations"]["T5_processed_cells_with_both_direction_codes"] == 17
    assert visual["observations"]["T5_processed_direction_code_mapping_verified"] is False
    assert visual["observations"]["T5_processed_stable_biological_ids_available"] is False
    assert visual["observations"]["T5_processed_independent_cell_holdout_available"] is False
    assert visual["observations"]["T5_unified_model_files_verified"] is False
    assert visual["observations"]["physical_timebase_identified"] is False
    assert visual["observations"]["strict_scoring_contract_frozen"] is True
    assert visual["observations"]["development_R1_R6_input_gates_pass"] is True
    assert visual["observations"]["development_neural_evaluation_allowed"] is True
    assert visual["observations"]["development_neural_screen_performed"] is True
    assert visual["observations"]["development_response_gates_pass"] is False
    assert visual["observations"]["development_passing_retinal_backends"] == []
    assert visual["observations"]["LPLC2_radial_opponency_gates_passed"] is False
    assert visual["observations"]["LPLC2_radial_fixed_target_denominators"] == {
        "L": 94,
        "R": 91,
    }
    assert visual["observations"]["LPLC2_radial_direction_labels_used"] is True
    assert visual["observations"]["LPLC2_radial_may_authorize_strict_visual_gate"] is False
    assert visual["observations"]["LPLC2_radial_temporal_identifiability_passed"] is False
    assert visual["observations"]["LPLC2_radial_all_condition_separated_fraction"] == {
        "L": 4 / 94,
        "R": 9 / 91,
    }
    assert visual["observations"]["radial_layer_localization_passing_populations"] == []
    assert visual["observations"]["mapped_R1_R6_all_condition_separated_fraction"] == (168 / 1914)
    assert visual["observations"]["visual_graph_unselected_R1_R6_count"] == 1463
    assert visual["observations"]["selected_R1_R6_renormalization_AB_gate_restored"] is False
    assert visual["observations"]["full_retina_projection_AB_passing_populations"] == []
    assert visual["observations"]["full_retina_mapping_is_exact_mirror"] is False
    assert visual["observations"]["full_retina_lamina_all_condition_separated_fraction"] == {
        "L1": 290 / 1776,
        "L2": 243 / 1779,
        "L3": 304 / 1772,
    }
    assert visual["observations"]["unmatched_noise_layer_localization_confounded"] is True
    assert visual["observations"]["matched_noise_passing_populations"] == []
    assert visual["observations"][
        "matched_noise_mapped_R1_R6_all_condition_separated_fraction"
    ] == (168 / 1914)
    assert visual["observations"]["LPLC2_position_observability_gate_passed"] is True
    assert visual["observations"]["LPLC2_position_observability_all_condition_fraction"] == (
        1902 / 1914
    )
    assert visual["observations"]["LPLC2_position_coverage_gates_passed"] is False
    assert visual["observations"]["local_edge_precheck_direction_pass_count"] == 0
    assert visual["observations"]["local_edge_precheck_polarity_pass_count"] == 0
    assert visual["observations"]["local_edge_precheck_expand_to_three_conditions"] is False
    assert set(visual["observations"]["local_edge_backend_direction_pass_counts"].values()) == {0}
    assert set(visual["observations"]["local_edge_backend_polarity_pass_counts"].values()) == {0, 8}
    assert visual["observations"]["existing_backend_reuse_gate_passed"] is False
    assert set(visual["observations"]["T4_local_correlator_direction_pass_counts"].values()) == {0}
    assert visual["observations"]["T4_local_correlator_gate_passed"] is False
    assert visual["observations"]["T4_source_pool_direction_pass_counts"] == {
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
    assert visual["observations"]["T4_source_pool_maximum_bilateral_pair_count"] == 0
    assert visual["observations"]["T4_source_pool_authorizes_target_formula"] is False
    assert visual["observations"]["three_hop_main_direction_bilateral_populations"] == [
        "T4a",
        "T4b",
        "T4c",
        "T4d",
        "T5a",
        "T5b",
        "T5c",
        "T5d",
    ]
    assert visual["observations"]["three_hop_temporal_shuffle_control_passed"] is False
    assert visual["observations"]["three_hop_static_sham_control_passed"] is True
    assert visual["observations"]["three_hop_temporal_shuffle_attenuated_population_count"] == 0
    assert visual["observations"]["three_hop_temporal_shuffle_minimum_energy_ratio"] > 3.0
    assert visual["observations"]["three_hop_strict_gates_passed"] is False
    assert visual["observations"]["three_hop_independent_channel_coverage_passed"] is False
    assert visual["observations"]["three_hop_scalar_reichardt_authorized"] is False
    assert visual["observations"]["four_hop_scalar_ordered_direction_pass_counts"] == {
        "T4_proximal": 0,
        "T4_distal": 1,
        "T5_tm9": 2,
    }
    assert visual["observations"]["four_hop_scalar_shuffle_bilateral_subtypes"] == {
        "T4_proximal": [],
        "T4_distal": ["a", "c"],
        "T5_tm9": ["a"],
    }
    assert visual["observations"]["four_hop_scalar_gate_passed"] is False
    assert visual["observations"]["LC4_position_speed_precheck_passed"] is False
    assert visual["observations"]["LC4_position_speed_positive_slope_fraction"] == {
        "L": 0.0,
        "R": 1 / 13,
    }
    assert visual["observations"]["LC4_position_speed_primary_readout"] == (
        "peak_positive_state_derivative"
    )
    assert visual["observations"]["LC4_input_speed_valid_fraction"] == {
        "L": 1 / 71,
        "R": 3 / 55,
    }
    assert visual["observations"]["LC4_input_speed_positive_slope_fraction"] == {
        "L": 1 / 71,
        "R": 3 / 55,
    }
    assert visual["observations"]["LC4_input_speed_mirror_passed"] is True
    assert visual["observations"]["LC4_input_speed_precheck_passed"] is False
    assert visual["observations"]["LC4_excitatory_input_derivative_valid_counts"] == {
        "L": 0,
        "R": 0,
    }
    assert visual["observations"]["LC4_inhibitory_input_derivative_monotonic_fractions"] == {
        "L": 63 / 71,
        "R": 47 / 55,
    }
    assert visual["observations"]["T5_lamina_split_polarity_passing_condition_counts"] == {
        f"T5{subtype}_{side}": 3 for subtype in "abcd" for side in "LR"
    }
    assert list(
        visual["observations"]["T5_lamina_split_direction_passing_condition_counts"].values()
    ) == [1, 0, 0, 0, 2, 2, 0, 0]
    assert visual["observations"]["T5_lamina_split_mirror_gate_passed"] is True
    assert visual["observations"]["T5_lamina_split_strict_gates_passed"] is False
    assert visual["observations"]["T5_lamina_scalar_joint_source_fraction"] == 6718 / 6719
    assert visual["observations"]["T5_lamina_scalar_ordered_direction_pass_count"] == 0
    assert visual["observations"]["T5_lamina_scalar_ordered_polarity_pass_count"] == 8
    assert visual["observations"]["T5_lamina_scalar_shuffle_bilateral_subtypes"] == ["a"]
    assert visual["observations"]["T5_lamina_scalar_precheck_passed"] is False
    assert visual["observations"]["T5_source_axis_passing_populations"] == [
        "T5c_L",
        "T5c_R",
        "T5d_L",
        "T5d_R",
    ]
    assert visual["observations"]["T5_source_axis_mirror_passing_subtypes"] == ["c", "d"]
    assert visual["observations"]["T5_source_axis_strict_gate_passed"] is False
    assert visual["observations"]["T5_source_axis_dynamic_candidate_authorized"] is False
    assert visual["observations"]["T5_zero_shot_T4_axis_calibration_accuracy"] > 0.83
    assert visual["observations"]["T5_source_axis_transform_application_authorized"] is False
    assert visual["observations"]["LPLC1_near_collision_direct_input_fraction"] == {
        "L": 67 / 68,
        "R": 1.0,
    }
    assert visual["observations"]["LPLC1_near_collision_response_pass_counts"] == {
        "near_vs_miss": 0,
        "approach_vs_recede": 0,
        "stationary_vs_rotating_background": 0,
    }
    assert visual["observations"]["LPLC1_near_collision_stimulus_geometry_passed"] is True
    assert visual["observations"]["LPLC1_near_collision_mirror_passed"] is True
    assert visual["observations"]["LPLC1_near_collision_precheck_passed"] is False
    assert visual["observations"]["LPLC1_input_group_target_fractions"]["L"][
        "object_detectors"
    ] == 1.0
    assert visual["observations"]["LPLC1_input_group_target_fractions"]["R"][
        "motion_detectors"
    ] == 1.0
    assert visual["observations"]["LPLC1_input_group_located_weight_fractions"]["L"][
        "glutamatergic"
    ] < 0.61
    assert visual["observations"]["LPLC1_input_group_located_weight_fractions"]["R"][
        "glutamatergic"
    ] < 0.78
    assert visual["observations"]["LPLC1_input_structure_gate_passed"] is False
    assert visual["observations"]["LPLC1_spatial_inhibition_mechanism_authorized"] is False
    assert visual["observations"]["development_geometry_ab_performed"] is True
    assert visual["observations"]["reversed_geometry_development_gates_pass"] is False
    assert visual["observations"]["reversed_geometry_passing_retinal_backends"] == []
    assert visual["observations"]["T5_targets_with_every_fast_and_any_delayed_source"] == 6717
    assert visual["observations"]["T5_target_count_in_structure_audit"] == 6719
    assert visual["observations"]["looming_targets_with_any_T4_and_any_T5"] == 319
    assert visual["observations"]["looming_target_count_in_structure_audit"] == 445
    assert visual["observations"]["existing_artifacts_rescored_under_new_contract"] is False
    release = saved["checks"][7]
    assert release["observations"]["stage1_stimulus_splits_disjoint"] is True
    assert release["observations"]["reserved_final_evaluated"] is False
    assert release["observations"]["blinded_one_time_final_available"] is False
