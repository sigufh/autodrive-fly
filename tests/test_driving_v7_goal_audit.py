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
    assert topology["strict_visual_target_control_response_evaluated"] is True
    assert topology["parameter_matched_baseline_protocol_ready"] is True
    assert topology["parameter_matched_baseline_budget"] == 1353
    assert topology["parameter_matched_baseline_counts"] == {
        "linear": 1353,
        "one_hidden_layer_mlp": 1365,
        "single_layer_gru": 1365,
    }
    assert topology["parameter_matched_baseline_evaluation_performed"] is False
    descending = report["checks"][3]["observations"]
    assert descending["bilateral_shortest_paths_supported"] is True
    assert descending["shortest_hops_by_target"] == {
        "DNa02": {"L": 1, "R": 1},
        "DNa01": {"L": 2, "R": 2},
        "DNp20": {"L": 2, "R": 3},
    }
    assert descending["action_equivalence_is_formula_level_only"] is True
    assert descending["real_neural_state_readout_performed"] is False
    assert descending["fixed_input_propagation_precheck_evaluated"] is True
    assert descending["functional_neural_readout_validated"] is False
    assert descending["real_state_precheck_passing_populations"] == ["DNa02"]
    assert descending["real_state_precheck_failed_populations"] == ["DNa01", "DNp20"]
    assert descending["all_descending_readouts_precheck_passed"] is False


def test_saved_goal_audit_is_hash_bound_and_matches_recalculation() -> None:
    saved = json.loads((ROOT / "artifacts/v7-goal-audit.json").read_text())
    for path, digest in saved["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    current = evaluate_v7_goal_coverage(ROOT)
    assert current == saved
    visual = saved["checks"][1]
    assert visual["observations"]["visual_inputs_only_at_receptors"] is True
    assert visual["observations"]["controlled_response_gates_pass"] is False
    assert visual["observations"]["T5_biological_PD_code_assigned"] == 1
    assert visual["observations"]["T5_external_direction_label_map_verified"] is True
    assert visual["observations"]["T5_model_scoring_allowed"] is True
    assert visual["observations"]["T5_published_fitted_parameter_vectors_available"] is False
    assert visual["observations"]["T5_processed_cells_with_both_direction_codes"] == 17
    assert visual["observations"]["T5_processed_direction_code_mapping_verified"] is False
    assert visual["observations"]["T5_processed_stable_biological_ids_available"] is False
    assert visual["observations"]["T5_typed_spatial_pair_joint_source_fraction"] > 0.999
    assert visual["observations"]["T5_typed_spatial_pair_direction_pass_counts"] == [
        0,
        0,
        0,
        0,
    ]
    assert visual["observations"]["T5_typed_spatial_pair_polarity_pass_counts"] == [
        8,
        8,
        8,
        8,
    ]
    assert visual["observations"]["T5_typed_spatial_pair_control_eligible"] is False
    assert visual["observations"]["T5_native_waveform_condition_count"] == 12
    assert visual["observations"]["T5_native_waveform_passing_condition_count"] == 1
    assert visual["observations"]["T5_native_waveform_template_authorized"] is False
    for values in visual["observations"]["T5_CT1_terminal_mapping_R2_by_eye"].values():
        assert min(values) > 0.98
    assert visual["observations"]["T5_CT1_terminal_axis_passing_populations"] == [
        "T5c_L",
        "T5c_R",
        "T5d_L",
        "T5d_R",
    ]
    assert visual["observations"]["T5_CT1_terminal_axis_candidate_authorized"] is False
    assert visual["observations"]["T5_CT1_axis_held_out_accuracy"] > 0.88
    assert visual["observations"]["T5_CT1_axis_held_out_median_angle_degrees"] < 15
    assert visual["observations"]["T5_CT1_axis_calibration_passed"] is True
    assert visual["observations"]["T5_CT1_crossfit_axis_accuracy"] > 0.88
    assert visual["observations"]["T5_CT1_crossfit_axis_median_angle_degrees"] < 15
    assert visual["observations"]["T5_CT1_crossfit_axis_passed"] is True
    assert visual["observations"]["T5_CT1_dynamics_ordered_direction_pass_counts"] == [
        2,
        2,
        2,
        2,
    ]
    assert visual["observations"]["T5_CT1_dynamics_ordered_polarity_pass_counts"] == [
        0,
        0,
        0,
        0,
    ]
    assert set(
        value
        for counts in visual["observations"][
            "T5_CT1_dynamics_control_direction_pass_counts"
        ].values()
        for value in counts
    ) == {0}
    assert visual["observations"]["T5_CT1_dynamics_strict_candidate_passed"] is False
    assert visual["observations"]["T5_CT1_multiplicative_ordered_direction_pass_counts"] == [
        0,
        0,
        0,
        2,
    ]
    assert visual["observations"]["T5_CT1_multiplicative_ordered_polarity_pass_counts"] == [
        8,
        8,
        8,
        8,
    ]
    assert visual["observations"]["T5_CT1_multiplicative_control_eligible_gains"] == [
        8.0
    ]
    assert visual["observations"]["T5_CT1_multiplicative_controls_evaluated"] is True
    assert visual["observations"]["T5_CT1_multiplicative_strict_candidate_passed"] is False
    assert (
        visual["observations"][
            "T5_CT1_multiplicative_three_condition_evaluation_performed"
        ]
        is False
    )
    assert visual["observations"]["T5_CT1_axis_aware_ordered_direction_pass_counts"] == [
        0,
        0,
        0,
        0,
    ]
    assert visual["observations"]["T5_CT1_axis_aware_ordered_polarity_pass_counts"] == [
        8,
        8,
        8,
        8,
    ]
    assert visual["observations"]["T5_CT1_axis_aware_controls_evaluated"] is False
    assert visual["observations"]["T5_CT1_axis_aware_strict_candidate_passed"] is False
    assert (
        visual["observations"]["T5_CT1_axis_aware_three_condition_evaluation_performed"]
        is False
    )
    assert visual["observations"][
        "T5_CT1_axis_aware_antisymmetric_direction_pass_counts"
    ] == [0, 0, 0, 0]
    assert visual["observations"][
        "T5_CT1_axis_aware_antisymmetric_polarity_pass_counts"
    ] == [8, 8, 8, 8]
    assert visual["observations"][
        "T5_CT1_axis_aware_antisymmetric_controls_evaluated"
    ] is False
    assert visual["observations"][
        "T5_CT1_axis_aware_antisymmetric_candidate_passed"
    ] is False
    assert visual["observations"]["T5_CT1_axis_sequence_maximum_direction_pass_count"] == 2
    assert set(
        subtype
        for values in visual["observations"][
            "T5_CT1_axis_sequence_ordered_bilateral_subtypes"
        ].values()
        for subtype in values
    ) == {"d"}
    assert visual["observations"]["T5_CT1_axis_sequence_candidate_selected"] is False
    assert (
        visual["observations"]["T5_CT1_axis_sequence_authorizes_functional_candidate"]
        is False
    )
    assert set(visual["observations"]["T5_physical_source_transfer_missing_fields"]) == {
        "v7_physical_frame_interval",
        "v7_physical_solver_interval",
        "v7_camera_angular_calibration",
        "Tm1_Tm2_Tm4_Tm9_membrane_like_kernels",
        "CT1_membrane_like_kernel",
        "stable_source_or_target_cell_to_MaleCNS_mapping",
        "independent_dynamic_validation_cohort",
    }
    assert not any(
        visual["observations"]["T5_physical_source_transfer_required_fields"].values()
    )
    assert visual["observations"]["T5_physical_source_transfer_ready"] is False
    assert visual["observations"]["T4_T5_source_dynamics_ready"] is False
    assert visual["observations"]["T4_T5_source_dynamics_passing_gates"] == [
        "T4_crossfit_structure_axis",
        "T5_crossfit_structure_axis",
    ]
    assert set(visual["observations"]["T4_T5_source_dynamics_failing_gates"]) == {
        "T4_source_dynamics_transfer",
        "T5_source_dynamics_transfer",
        "shared_physical_v7_timebase",
        "independent_dynamic_validation",
    }
    assert visual["observations"]["T4_T5_source_dynamics_authorizes_new_candidate"] is False
    assert visual["observations"]["T4_millivolt_source_types_complete"] is True
    assert visual["observations"]["T4_source_biological_individual_ID_available"] is False
    assert visual["observations"]["T4_individual_level_source_validation_ready"] is False
    assert visual["observations"]["source_dynamics_external_contract_satisfied"] is False
    assert visual["observations"]["source_dynamics_external_payload_present"] is False
    assert visual["observations"]["Dryad_L1L2_required_source_coverage_count"] == 0
    assert visual["observations"]["Dryad_L1L2_source_dynamics_transfer_authorized"] is False
    assert visual["observations"]["Dryad_L1L2_large_download_authorized"] is False
    assert visual["observations"]["Gou_sparsity_required_source_coverage_count"] == 4
    assert visual["observations"]["Gou_sparsity_partial_source_evidence_present"] is True
    assert visual["observations"]["Gou_sparsity_source_dynamics_transfer_authorized"] is False
    assert visual["observations"]["Gou_sparsity_bulk_download_authorized"] is False
    assert visual["observations"]["T5_contrast_opponency_temporal_sources"] == [
        "Tm4",
        "Tm9",
    ]
    assert visual["observations"]["T5_contrast_opponency_spatial_only_sources"] == [
        "CT1"
    ]
    assert visual["observations"]["T5_contrast_opponency_transfer_authorized"] is False
    assert visual["observations"]["Yang_T5_optical_voltage_phenotype_sources"] == [
        "Tm1",
        "Tm2",
    ]
    assert visual["observations"]["Yang_T5_numerical_voltage_payload_verified"] is False
    assert visual["observations"]["Yang_T5_voltage_transfer_authorized"] is False
    assert visual["observations"]["Kohn_Portes_T5_numeric_voltage_sources"] == [
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
    ]
    assert visual["observations"]["Kohn_Portes_T5_missing_voltage_sources"] == ["CT1"]
    assert visual["observations"]["Kohn_Portes_T5_identity_retained"] is False
    assert visual["observations"]["Kohn_Portes_T5_transfer_authorized"] is False
    assert visual["observations"]["MaleCNS_nine_source_type_body_sets_available"] is True
    assert visual["observations"]["MaleCNS_external_source_mapping_contract_satisfied"] is False
    assert visual["observations"]["MaleCNS_Tm9_unlocated_source_body_count"] == 1
    assert visual["observations"]["MaleCNS_CT1_columnar_Lo1_retinotopy_available"] is False
    assert visual["observations"]["source_evidence_matrix_all_nine_complete"] is False
    assert visual["observations"]["source_evidence_matrix_authorizes_fit"] is False
    assert visual["observations"]["source_evidence_matrix_T4_local_millivolt_count"] == 4
    assert visual["observations"]["source_evidence_matrix_T5_local_calcium_count"] == 3
    assert visual["observations"]["source_evidence_matrix_T5_local_millivolt_count"] == 4
    assert (
        visual["observations"]["source_evidence_matrix_C3_has_any_numerical_fly_IDs"]
        is True
    )
    assert (
        visual["observations"]["source_evidence_matrix_C3_fixed_external_robustness_passed"]
        is False
    )
    assert visual["observations"]["TimingModels_CT1_type_average_dynamics_verified"] is True
    assert visual["observations"]["TimingModels_T5_lobula_CT1_phenotype_published"] is True
    assert visual["observations"]["TimingModels_T5_lobula_CT1_numerical_payload_verified"] is False
    assert visual["observations"]["TimingModels_T5_lobula_CT1_dynamics_verified"] is False
    assert visual["observations"]["TimingModels_T5_CT1_transfer_authorized"] is False
    assert visual["observations"]["source_dynamics_external_required_split_roles"] == [
        "training",
        "validation",
        "external_final",
    ]
    assert visual["observations"]["source_type_temporal_passing_counts"] == {
        "T4": {"passed": 0, "denominator": 32},
        "T5": {"passed": 0, "denominator": 32},
    }
    assert visual["observations"]["source_type_temporal_identifiability_passed"] is False
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
    assert visual["observations"]["T4_continuous_pair_structural_axis_passed"] is True
    assert visual["observations"]["T4_continuous_pair_direction_pass_counts"] == {
        "signed_mean": 0,
        "maximum": 0,
        "positive_mean": 0,
        "terminal_sum": 0,
    }
    assert visual["observations"]["T4_continuous_pair_polarity_pass_counts"] == {
        "signed_mean": 0,
        "maximum": 0,
        "positive_mean": 0,
        "terminal_sum": 0,
    }
    assert visual["observations"]["T4_continuous_pair_controls_evaluated"] is False
    assert visual["observations"][
        "T4_continuous_pair_three_condition_evaluation_performed"
    ] is False
    assert visual["observations"]["T4_pair_lag_candidate_count"] == 12
    assert set(visual["observations"]["T4_pair_lag_direction_pass_counts"].values()) == {0}
    assert set(visual["observations"]["T4_pair_lag_polarity_pass_counts"].values()) == {0}
    assert visual["observations"]["T4_pair_lag_audit_passed"] is False
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
    assert visual["observations"][
        "three_hop_temporal_consistency_passing_family_metrics"
    ] == []
    temporal_ratios = visual["observations"][
        "three_hop_temporal_consistency_minimum_ratios"
    ]
    assert all(ratio > 1.0 for family in temporal_ratios.values() for ratio in family.values())
    for family in temporal_ratios.values():
        for name in (
            "mean_vector_magnitude",
            "coherence_weighted_magnitude",
            "squared_coherence_weighted_magnitude",
            "sign_persistence_weighted_magnitude",
        ):
            assert family[name] > 2.0
    assert visual["observations"]["three_hop_temporal_consistency_gate_passed"] is False
    assert visual["observations"]["upstream_ordered_latency_passing_populations"] == {
        "T4": 8,
        "T5": 8,
    }
    assert visual["observations"]["upstream_shuffle_latency_passing_populations"] == {
        "T4": 8,
        "T5": 8,
    }
    assert visual["observations"]["upstream_static_latency_passing_populations"] == {
        "T4": 8,
        "T5": 8,
    }
    assert visual["observations"][
        "upstream_source_latency_temporal_identifiability_passed"
    ] is False
    assert visual["observations"]["upstream_paired_residual_passing_populations"] == {
        "T4": 0,
        "T5": 0,
    }
    assert all(
        value > 4.0
        for value in visual["observations"][
            "upstream_paired_residual_minimum_shuffle_ratios"
        ].values()
    )
    assert visual["observations"][
        "upstream_paired_residual_temporal_identifiability_passed"
    ] is False
    assert visual["observations"]["synapse_spatial_selected_row_count"] == 1_535_378
    assert visual["observations"]["synapse_spatial_complete_target_counts"] == {
        "T4": 6860,
        "T5": 6718,
    }
    assert visual["observations"]["synapse_spatial_aggregate_weight_matches"] == {
        "T4": True,
        "T5": True,
    }
    assert visual["observations"]["synapse_spatial_structure_gate_passed"] is True
    assert visual["observations"]["synapse_axis_held_out_T4_accuracy"] > 0.96
    assert visual["observations"]["synapse_axis_held_out_T4_median_angle_degrees"] < 14.0
    assert visual["observations"]["synapse_axis_T4_calibration_passed"] is True
    assert visual["observations"]["T4_synapse_crossfit_axis_accuracy"] > 0.95
    assert visual["observations"]["T4_synapse_crossfit_axis_median_angle_degrees"] < 14
    assert visual["observations"]["T4_synapse_crossfit_axis_passed"] is True
    assert visual["observations"]["synapse_axis_zero_shot_T5_accuracy"] < 0.11
    assert visual["observations"]["synapse_axis_zero_shot_T5_mapping_passed"] is False
    assert visual["observations"]["synapse_axis_T5_mapping_application_authorized"] is False
    assert visual["observations"]["T4_synapse_correlator_fixed_population_denominator"] == 6861
    assert visual["observations"]["T4_synapse_correlator_valid_target_count"] == 6749
    assert visual["observations"]["T4_synapse_correlator_direction_pass_counts"] == [
        2, 2, 2, 1, 2, 2, 2, 1
    ]
    assert visual["observations"]["T4_synapse_correlator_polarity_pass_counts"] == [8] * 8
    assert visual["observations"]["T4_synapse_correlator_bilateral_direction_subtypes"] == [
        [] for _ in range(8)
    ]
    assert visual["observations"]["T4_synapse_correlator_main_gate_passed"] is False
    assert visual["observations"]["T4_synapse_correlator_controls_evaluated"] is False
    assert visual["observations"][
        "T4_synapse_correlator_three_condition_evaluation_performed"
    ] is False
    assert visual["observations"]["T4_synapse_antisymmetric_direction_pass_counts"] == [
        2
    ] * 8
    assert visual["observations"]["T4_synapse_antisymmetric_polarity_pass_counts"] == [8] * 8
    assert visual["observations"]["T4_synapse_antisymmetric_bilateral_direction_subtypes"] == [
        [] for _ in range(8)
    ]
    assert visual["observations"]["T4_synapse_antisymmetric_controls_evaluated"] is False
    assert visual["observations"]["T4_synapse_crossfit_direction_pass_counts"] == [2] * 8
    assert visual["observations"]["T4_synapse_crossfit_polarity_pass_counts"] == [8] * 8
    assert visual["observations"]["T4_synapse_crossfit_controls_evaluated"] is False
    assert visual["observations"]["T4_synapse_crossfit_candidate_passed"] is False
    assert visual["observations"]["T4_crossfit_sequence_maximum_direction_pass_count"] == 2
    assert all(
        not values
        for values in visual["observations"][
            "T4_crossfit_sequence_ordered_bilateral_subtypes"
        ].values()
    )
    assert visual["observations"][
        "T4_crossfit_sequence_shuffle_signed_mean_bilateral_subtypes"
    ] == ["a"]
    assert visual["observations"]["T4_crossfit_sequence_authorizes_functional_candidate"] is False
    assert visual["observations"]["T4_balanced_retina_ordered_direction_pass_count"] == 0
    assert (
        visual["observations"]["T4_retinal_sampling_imbalance_explains_direction_failure"]
        is False
    )
    assert visual["observations"][
        "T4_synapse_antisymmetric_three_condition_evaluation_performed"
    ] is False
    assert visual["observations"]["T4_synapse_centered_direction_pass_counts"] == [
        2, 2, 2, 1, 2, 2, 2, 2
    ]
    assert visual["observations"]["T4_synapse_centered_polarity_pass_counts"] == [8] * 8
    assert visual["observations"]["T4_synapse_centered_bilateral_direction_subtypes"] == [
        [] for _ in range(8)
    ]
    assert visual["observations"]["T4_synapse_centered_controls_evaluated"] is False
    assert visual["observations"][
        "T4_synapse_centered_three_condition_evaluation_performed"
    ] is False
    assert visual["observations"]["T4_synapse_microstep_shuffle_residual_ratio"] > 1.9
    assert visual["observations"]["T4_synapse_microstep_static_energy_ratio"] > 0.53
    assert visual["observations"]["T4_synapse_microstep_temporal_identifiability_passed"] is False
    assert visual["observations"]["T4_synapse_microstep_direction_scoring_performed"] is False
    assert set(
        visual["observations"]["T4_source_dynamics_transfer_fields_available"].values()
    ) == {False}
    assert visual["observations"]["T4_source_dynamics_transfer_authorized"] is False
    assert visual["observations"]["T4_source_dynamics_next_candidate_authorized"] is False
    assert visual["observations"]["unified_model_package_files_verified"] is True
    assert visual["observations"]["unified_model_T4_target_parameters_available"] is True
    assert visual["observations"]["unified_model_MaleCNS_source_transfer_authorized"] is False
    assert visual["observations"]["fig3_ON_source_specific_kernels_ready"] is False
    assert visual["observations"]["fig3_prior_two_pool_kernel_authorized"] is False
    assert visual["observations"]["fig3_source_specific_kernel_candidate_authorized"] is False
    assert visual["observations"]["fig3_source_kernel_cross_cell_robustness_passed"] is False
    assert visual["observations"]["Arenz_source_filter_parameters_verified"] is True
    assert visual["observations"]["Arenz_current_source_coverage_fraction"] == 0.75
    assert visual["observations"]["Arenz_missing_current_sources"] == ["C3"]
    assert visual["observations"]["Arenz_source_filter_candidate_authorized"] is False
    assert visual["observations"]["Arenz_T5_source_coverage_fraction"] == 1.0
    assert visual["observations"]["Arenz_T5_raw_filter_contract_complete"] is True
    assert visual["observations"]["Arenz_T5_deconvolved_filter_contract_complete"] is False
    assert visual["observations"]["Arenz_T5_Tm9_deconvolved_R2"] == 0.273
    assert visual["observations"]["Arenz_T5_source_filter_transfer_authorized"] is False
    assert visual["observations"]["C3_STRF_numerical_data_verified"] is True
    assert visual["observations"]["C3_direct_temporal_measurement_available"] is True
    assert visual["observations"]["combined_source_temporal_evidence_complete"] is True
    assert visual["observations"]["combined_source_parameterization_transferable"] is False
    assert visual["observations"]["C3_source_filter_candidate_authorized"] is False
    assert visual["observations"]["Fig1_source_spatial_tables_verified"] is True
    assert visual["observations"]["Fig1_source_average_tables_have_time_axis"] is False
    assert visual["observations"]["Fig1_source_object_payload_safely_inspected"] is False
    assert visual["observations"]["Fig1_source_temporal_kernel_transfer_authorized"] is False
    assert visual["observations"]["C3_analytic_filter_band_pass_preferred"] is True
    assert visual["observations"]["C3_analytic_filter_minimum_LOFO_correlation"] < 0.67
    assert visual["observations"]["C3_analytic_filter_family_precheck_passed"] is False
    assert visual["observations"]["C3_analytic_filter_transfer_authorized"] is False
    assert visual["observations"]["TimingModels_covered_source_filters_stable"] is True
    assert visual["observations"]["TimingModels_current_source_coverage_fraction"] == 0.75
    assert visual["observations"]["TimingModels_missing_current_sources"] == ["C3"]
    assert visual["observations"]["TimingModels_complete_source_candidate_authorized"] is False
    assert visual["observations"]["FlyVis_C3_pretrained_model_count"] == 50
    assert 0.066 < visual["observations"]["FlyVis_C3_median_time_constant_seconds"] < 0.068
    assert visual["observations"]["FlyVis_C3_models_at_or_below_solver_dt"] == 21
    assert visual["observations"]["FlyVis_C3_time_constant_transfer_authorized"] is False
    assert visual["observations"]["FlyVis_visual_source_coverage_complete"] is True
    assert visual["observations"]["FlyVis_visual_source_time_constants_transferable"] is False
    assert visual["observations"]["FlyVis_C3_effective_cross_dt_minimum_correlation"] < 0.67
    assert min(
        visual["observations"]["FlyVis_C3_effective_LOMO_minimum_by_dt"].values()
    ) < -0.81
    assert visual["observations"]["FlyVis_C3_effective_external_maximum_correlation"] < 0.72
    assert visual["observations"][
        "FlyVis_C3_effective_dynamics_transfer_authorized"
    ] is False
    assert visual["observations"]["C3_measured_filter_cross_deconvolution_stable"] is True
    assert max(
        visual["observations"]["C3_measured_filter_minimum_LOFO_by_assumption"].values()
    ) < 0.58
    assert visual["observations"]["C3_measured_filter_type_shared_kernel_authorized"] is False
    assert visual["observations"]["C3_STRF_flash_external_fly_count"] == 22
    assert visual["observations"]["C3_STRF_flash_mean_correlation"] > 0.97
    assert visual["observations"]["C3_STRF_flash_bootstrap_p05"] < 0.12
    assert visual["observations"]["C3_STRF_flash_transfer_passed"] is False
    assert visual["observations"]["public_T4_models_C3_specific_parameters_available"] is False
    assert visual["observations"]["public_T4_models_complete_source_transfer_authorized"] is False
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
    assert visual["observations"]["T5_source_pair_joint_source_fraction"] == 6718 / 6719
    assert visual["observations"]["T5_source_pair_direction_pass_counts"] == [0] * 6
    assert visual["observations"]["T5_source_pair_polarity_pass_counts"] == [8] * 6
    assert visual["observations"]["T5_source_pair_bilateral_direction_subtypes"] == [
        []
    ] * 6
    assert visual["observations"]["T5_source_pair_precheck_passed"] is False
    assert visual["observations"]["T5_source_pair_controls_evaluated"] is False
    assert visual["observations"][
        "T5_source_pair_three_condition_evaluation_performed"
    ] is False
    assert visual["observations"][
        "LPLC_mechanism_repair_authorized_by_T4_T5_gate"
    ] is False
    assert visual["observations"]["T5_continuous_moment_structural_axis_passed"] is True
    assert visual["observations"]["T5_continuous_moment_direction_pass_counts"] == {
        "signed_mean": 0,
        "maximum": 0,
        "positive_mean": 0,
        "terminal_sum": 0,
    }
    assert visual["observations"]["T5_continuous_moment_controls_evaluated"] is False
    assert visual["observations"][
        "T5_continuous_moment_three_condition_evaluation_performed"
    ] is False
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
