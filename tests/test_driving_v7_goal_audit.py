import hashlib
import json
from pathlib import Path

import numpy as np

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
    assert disclosure["controlled_vision_evidence_boundaries_disclosed"] is True
    assert disclosure["disclosed_boundary_fields"] == [
        "nine_source_contract_complete",
        "Mi4_C3_independent_numeric_voltage_candidate_count",
        "T5_voltage_field_counts",
        "T5_voltage_derived_kernel_sources",
        "T5_voltage_derived_kernel_record_count",
        "T5_kernel_sample_interval_seconds",
        "T5_raw_kernel_gain_transferable",
        "T5_kernel_stimulus_invariant",
        "T5_source_kernel_transfer_authorized",
        "T5_state_unit_record_count",
        "T5_exact_volts_to_millivolts_scale_verified",
        "T5_millivolts_or_filter_output_to_state_mapping_available",
        "T5_candidate_normalization_formula",
        "T5_candidate_clipping_rule",
        "T5_saline_fast_pooled_median_latency_ms",
        "T5_saline_Tm9_median_latency_ms",
        "T5_OA_fast_pooled_median_latency_ms",
        "T5_OA_Tm9_median_latency_ms",
        "T5_saline_Tm9_relative_delay_supported",
        "T5_Tm9_delay_ordering_state_invariant",
        "T5_source_delay_transfer_authorized",
        "T5_saline_record_median_preferred_frequency_hz",
        "T5_OA_record_median_preferred_frequency_hz",
        "T5_preferred_frequency_summary_invariant",
        "T5_source_frequency_transfer_authorized",
        "T5_Tm_to_T5_model_inputs_git_blob_verified",
        "T5_Tm_to_T5_Figure5_training_fit_count",
        "T5_Tm_to_T5_fit_score_samples_disjoint",
        "T5_Tm_to_T5_independent_validation_available",
        "T5_Figure6_Tm2_ND_source_reference_correct",
        "T5_Tm_to_T5_model_transfer_authorized",
        "T5_FIB19_MaleCNS_population_source_rank_matches",
        "T5_FIB19_MaleCNS_maximum_population_ratio_difference",
        "T5_MaleCNS_Tm9_fraction_outside_FIB19_range",
        "T5_FIB19_to_MaleCNS_weight_transfer_authorized",
        "T5_moving_bar_condition_count",
        "T5_moving_bar_gain_free_DSI_available",
        "T5_moving_bar_gain_fit_score_samples_disjoint",
        "T5_moving_bar_validation_available",
        "T5_moving_bar_transfer_authorized",
        "T5_four_Tm_exact_type_average_mapping_complete",
        "T5_all_five_source_mapping_complete",
        "T5_CT1_mapping_Lo1_column_count_by_body",
        "Braun_calcium_fly_counts",
        "Braun_calcium_condition_grids_complete",
        "Braun_calcium_allowed_voltage_sources",
        "Gou_Dryad_archive_hash_locally_verified",
        "Gou_Dryad_local_processed_calcium_sources",
        "Gou_Dryad_flash_fly_axis_sizes",
        "Gou_Dryad_moving_bar_fly_axis_sizes",
        "Gou_Dryad_Mi1_Tm3_direction_axis_identifiable",
        "Gou_Dryad_direction_invariance_evaluated",
        "Gou_DANDI_all_assets_stimulus_metadata_indexed",
        "Gou_DANDI_Mi1_Tm3_asset_counts",
        "Gou_DANDI_Mi1_Tm3_stimulus_metadata_available",
        "Gou_Dryad_stable_biological_individual_IDs_verified",
        "Gou_DANDI_asset_level_stable_participant_IDs_verified",
        "Gou_DANDI_unique_subject_ID_count",
        "Gou_Dryad_distinct_fliesUsed_label_count",
        "Gou_Dryad_DANDI_identity_crosswalk_match_count",
        "Gou_Dryad_rows_to_DANDI_subject_crosswalk_verified",
        "Gou_Dryad_experimental_membrane_voltage",
        "v7_offline_time_coordinate_contract_complete",
        "T5_transfer_synthesis_kernel_shape_source_count",
        "T5_transfer_synthesis_mapping_source_count",
        "T5_transfer_synthesis_positive_shape_evidence_count",
        "T5_transfer_synthesis_absolute_gain_available",
        "T5_transfer_synthesis_official_target_model_available",
        "T5_transfer_synthesis_target_to_source_mapping_available",
        "T5_transfer_synthesis_Figure6_axolotl_source_available",
        "T5_transfer_synthesis_related_axolotl_project_identified",
        "T5_transfer_synthesis_axolotl_repository_readable",
        "T5_measured_kernel_temporal_identifiability_passed",
        "T5_measured_kernel_direction_scoring_performed",
        "T5_measured_kernel_frame_alignment_verified",
        "T5_measured_kernel_probe_solver_alignment_verified",
        "T5_measured_kernel_physical_transfer_authorized",
        "T5_measured_kernel_standard_substep_negative_reproduced",
        "T5_measured_kernel_cross_substep_identifiability_passed",
        "T5_measured_kernel_cross_substep_direction_scoring_performed",
        "T5_measured_kernel_cross_substep_direction_scoring_authorized",
        "T5_measured_kernel_causal_index_zero_supported",
        "T5_measured_kernel_all_population_peaks_covered",
        "T5_measured_kernel_full_L1_support_covered",
        "T5_measured_kernel_prefix_alone_full_support_negative_authorized",
        "T5_measured_kernel_trace_samples",
        "T5_measured_kernel_minimum_prefix_L1_mass_fraction",
        "T5_measured_kernel_maximum_prefix_L1_mass_fraction",
        "T5_measured_kernel_zero_tail_full_support_evaluated",
        "T5_measured_kernel_zero_tail_cross_substep_identifiability_passed",
        "T5_measured_kernel_zero_tail_all_candidates_failed",
        "T5_measured_kernel_zero_tail_direction_scoring_authorized",
        "T5_measured_kernel_zero_tail_direction_scoring_performed",
        "T5_measured_kernel_zero_tail_samples",
        "T5_measured_kernel_full_support_output_samples",
        "T5_population_kernel_recording_id_robustness_passed",
        "T5_Tm2_population_kernel_recording_id_robustness_passed",
        "T5_population_kernel_independent_validation_available",
        "T5_population_kernel_transfer_authorized",
        "T5_population_kernel_robustness_passing_source_count",
        "T5_population_kernel_robustness_failing_sources",
        "T5_Tm2_recording_id_vs_rest_median_correlation",
        "T5_Tm2_partition_correlation_p05",
        "T5_population_kernel_author_default_baseline_verified",
        "T5_population_kernel_no_baseline_matches_author_default",
        "T5_population_kernel_author_row_weighting_exactly_reproduced",
        "T5_population_kernel_tail_baseline_variant_authorized",
        "T5_population_kernel_author_call_count",
        "T5_population_kernel_explicit_baseline_call_count",
        "T5_Tm1_row_vs_recording_id_weighted_correlation",
        "T5_author_row_weighted_changed_sources",
        "T5_author_row_weighted_full_support_all_candidates_failed",
        "T5_author_row_weighted_cross_substep_identifiability_passed",
        "T5_author_row_weighted_direction_scoring_authorized",
        "T5_author_row_weighted_direction_scoring_performed",
        "T5_Tm2_LOO_full_support_all_evaluations_failed",
        "T5_Tm2_LOO_robust_temporal_identifiability_passed",
        "T5_Tm2_LOO_direction_scoring_authorized",
        "T5_Tm2_LOO_direction_scoring_performed",
        "T5_Tm2_LOO_independent_biological_validation_performed",
        "T5_Tm2_LOO_fold_count",
        "T5_Tm2_LOO_evaluation_count",
        "T5_Tm2_LOO_passed_evaluation_count",
        "T5_Tm2_LOO_candidate_ratio_ranges",
        "T5_measured_kernel_typed_recurrent_cascade_verified",
        "T5_measured_kernel_replaces_existing_source_dynamics",
        "T5_measured_kernel_single_stage_biological_interpretation_authorized",
        "T5_measured_kernel_external_state_mapping_available",
        "T5_measured_kernel_source_leaks",
        "T5_measured_kernel_source_node_counts",
        "T5_measured_kernel_source_inputs_partitioned_exactly_once",
        "T5_measured_kernel_every_source_has_recurrent_or_feedback_input",
        "T5_measured_kernel_feedforward_only_source_drive_available",
        "T5_measured_kernel_source_dynamics_replacement_evaluated",
        "T5_measured_kernel_source_dynamics_replacement_authorized",
        "T5_measured_kernel_recurrent_or_feedback_fraction_by_source",
        "T5_measured_kernel_Tm9_CT1_input_fraction",
        "T5_lamina_only_replacement_evaluated",
        "T5_lamina_only_replacement_all_candidates_failed",
        "T5_lamina_only_replacement_temporal_identifiability_passed",
        "T5_lamina_only_replacement_direction_scoring_authorized",
        "T5_lamina_only_replacement_direction_scoring_performed",
        "T5_lamina_only_replacement_physical_transfer_authorized",
        "T5_lamina_only_source_coverage",
        "T5_lamina_only_candidate_ratios_by_update",
        "T5_temporal_shuffle_frame_multiset_preserved",
        "T5_temporal_shuffle_retinal_energy_preserved",
        "T5_temporal_shuffle_lamina_source_energy_preserved",
        "T5_temporal_shuffle_energy_matched_control_verified",
        "T5_equal_energy_temporal_selectivity_interpretation_authorized",
        "T5_new_energy_normalized_gate_authorized",
        "T5_temporal_shuffle_input_energy_ratios_by_update",
        "T5_static_sham_input_energy_ratios_by_update",
        "T5_increment_order_control_images_valid",
        "T5_increment_order_control_terminal_frame_preserved",
        "T5_increment_order_control_R1_R6_drive_multiset_preserved",
        "T5_increment_order_control_R1_R6_energy_matched",
        "T5_increment_order_control_independent_condition_evaluated",
        "T5_increment_order_control_replacement_authorized",
        "T5_increment_order_control_R1_R6_energy_ratio_summary",
        "T5_increment_order_control_lamina_energy_ratios_by_update",
        "T5_increment_order_control_output_ratios_by_update",
        "T5_increment_order_replication_input_validity_passed",
        "T5_increment_order_replication_same_candidate_passed",
        "T5_increment_order_replication_gate_passed",
        "T5_increment_order_replication_direction_scoring_authorized",
        "T5_increment_order_replication_candidate_passes",
        "T5_increment_order_replication_output_ratios",
        "T5_transfer_synthesis_CT1_complete",
        "T5_transfer_synthesis_ready",
        "v7_offline_horizontal_coordinate_contract_complete",
        "v7_offline_two_dimensional_angular_calibration_complete",
        "v7_vertical_camera_ray_angles_declared",
        "v7_vertical_FOV_declared",
        "v7_vertical_pixel_to_angle_formula_declared",
        "v7_vertical_motion_has_physical_angular_units",
        "v7_looming_radius_has_physical_angular_units",
        "v7_retinal_v_has_physical_angular_units",
        "v7_offline_2D_engineering_angular_grid_complete",
        "v7_offline_engineering_vertical_FOV_degrees",
        "v7_offline_engineering_angular_pixel_pitch_degrees",
        "v7_offline_engineering_grid_biologically_calibrated",
        "v7_controlled_stimulus_and_R1_R6_input_boundary_complete",
        "v7_controlled_base_stimulus_count",
        "v7_controlled_stimuli_per_independent_split",
        "v7_typed_LPLC_LC4_stimulus_count",
        "v7_external_drive_non_R1_R6_node_count",
        "v7_target_direct_external_drive_overlap",
        "v7_offline_frame_interval_milliseconds",
        "v7_offline_substep_interval_milliseconds",
        "v7_horizontal_fov_degrees",
        "T4_source_mapping_mode",
        "T4_source_recording_level_body_assignment",
        "T4_exact_type_average_mapping_complete",
        "T4_author_minmax_formula_reproduced",
        "T4_state_mapping_held_out_outside_fraction_by_source",
        "T4_author_minmax_semantics_match_v7_state",
        "T4_millivolts_to_v7_state_mapping_available",
        "Ketkar_2019_official_source_data_attachment_count",
        "Ketkar_2019_attachments_are_mean_SEM_tables",
        "Ketkar_2019_Mi1_Tm3_GCaMP_summary_found",
        "Ketkar_2019_Mi4_C3_payload_found",
        "Ketkar_2019_individual_source_dynamics_found",
        "Ketkar_2019_membrane_voltage_found",
        "Gonzalez_Suarez_2022_Mi4_GCaMP6f_fly_count",
        "Gonzalez_Suarez_2022_Mi4_type_average_filter_available",
        "Gonzalez_Suarez_2022_individual_cell_axis_available",
        "Gonzalez_Suarez_2022_Mi4_voltage_available",
        "Gonzalez_Suarez_2022_C3_dynamics_available",
        "Gonzalez_Suarez_2022_bioRxiv_supplement_retrieved",
        "Gonzalez_Suarez_2022_individual_flies_are_statistical_units",
        "Gonzalez_Suarez_2022_public_individual_numeric_payload",
        "Yuan_2020_C3_intervention_candidate_verified",
        "Yuan_2020_C3_direct_recording_verified",
        "Yuan_2020_C3_numeric_source_dynamics_verified",
        "Yuan_2020_supplement_retrieved",
        "Strother_2018_successful_index_Mi4_trace_found",
        "Strother_2018_Figshare_search_interpretable",
        "Strother_2018_global_absence_claimed",
        "C3_citation_graph_union_unique_work_count",
        "C3_citation_graph_unresolved_reference_ID_count",
        "Pang_2025_directly_recorded_neuron_types",
        "Pang_2025_Dryad_file_count",
        "Pang_2025_new_C3_direct_recording_found",
        "Hao_2026_ASAP7y_candidate_classification",
        "Hao_2026_ASAP7y_paper_source_experimental_cell_types",
        "Hao_2026_ASAP7y_public_author_figure_named_examples",
        "Hao_2026_ASAP7y_complete_cell_type_set_resolved",
        "Hao_2026_ASAP7y_Drosophila_voltage_verified",
        "Hao_2026_ASAP7y_public_numeric_payload_verified",
        "Hao_2025_dissertation_restricted_until",
        "Hao_2025_dissertation_public_abstract_names_cell_types",
        "Hao_2026_ASAP7y_successful_index_numeric_payload_found",
        "Hao_2026_ASAP7y_global_payload_absence_claimed",
            "Gur_2024_directly_recorded_neuron_types",
            "Gur_2024_Mi4_proofreading_row_count",
            "Gur_2024_C3_proofreading_row_count",
            "Gur_2024_Mi4_C3_files_are_proofreading_only",
            "Gur_2024_Mi4_C3_direct_physiology_found",
            "Tanaka_2023_Mi4_fly_count",
            "Tanaka_2023_Mi4_selected_ROI_count",
            "Tanaka_2023_Mi4_individual_fly_axis_available",
            "Tanaka_2023_Mi4_response_unit",
            "Tanaka_2023_Mi4_experimental_membrane_voltage",
            "Tanaka_2023_Mi4_source_dynamics_transfer_authorized",
            "Tanaka_2023_Figure_6_named_Mi4_C3_member_count",
            "Tanaka_2023_Figure_6_Mi4_C3_neural_activity_recording",
            "Tanaka_2023_Figure_6_Mi4_C3_source_dynamics_payload",
            "Wu_2026_afterimages_Mi4_fly_count",
            "Wu_2026_afterimages_Mi4_ROI_count",
            "Wu_2026_afterimages_Mi4_response_unit",
            "Wu_2026_afterimages_public_numeric_Mi4_payload_verified",
            "Wu_2026_afterimages_Mi4_source_dynamics_transfer_authorized",
            "Henning_C3_directional_payload_fly_count",
            "Henning_C3_directional_caption_fly_count",
            "Henning_C3_directional_ROI_count",
            "Henning_C3_directional_edge_speed_degrees_per_second",
            "Henning_C3_directional_cohort_independent",
            "Henning_C3_direction_specific_source_kernel_verified",
            "Henning_C3_directional_source_transfer_authorized",
            "C2C3_version_of_record_DOI",
            "C2C3_version_of_record_repository_revision",
            "C2C3_version_of_record_new_payload_modality",
            "C2C3_version_of_record_C3_fly_count",
            "C2C3_version_of_record_Mi1_control_fly_count",
            "C2C3_version_of_record_new_C3_Mi4_voltage_found",
            "C2C3_version_of_record_new_Mi4_payload_found",
            "C2C3_version_of_record_MaleCNS_crosswalk_found",
            "C2C3_version_of_record_changed_T4_transfer_gate",
        "T5_record_specific_stimulus_logs_available",
        "Motyxia2_public_history_branch_count",
        "Motyxia2_public_history_commit_count",
        "T5_record_log_found_in_Motyxia2_public_history",
        "T5_external_successful_indexes_linked_log_found",
        "T5_PMC_supplement_content_inspected",
        "T5_publisher_supplements_inspected",
        "T5_publisher_supplements_contain_record_log",
        "T5_Figshare_search_accessible",
        "T5_stimulus_log_global_absence_claimed",
        "T5_generator_defaults_used_as_record_fields",
        "T5_Figure4_relative_PD_ND_mapping_verified",
        "T5_Figure4_native_coordinate_motion_mapping_verified",
        "T5_Figure4_absolute_physical_direction_mapping_verified",
        "T5_Kohn_Portes_record_level_direction_code_available",
        "T5_Kohn_Portes_record_level_physical_direction_available",
        "T5_cross_dataset_direction_mapping_authorized",
            "T4_source_pool_camera_frame_bilateral_readouts",
            "T4_source_pool_camera_frame_bilateral_subtypes",
            "T4_source_pool_camera_frame_post_hoc_candidate_discovered",
            "T4_source_pool_camera_frame_replication_preregistered",
            "T4_source_pool_camera_frame_replication_evaluated",
            "T4_source_pool_camera_frame_target_formula_authorized",
            "T4_source_pool_camera_frame_ordered_replication_passed",
            "T4_source_pool_camera_frame_controls_evaluated",
            "T4_source_pool_camera_frame_replication_gate_passed",
            "T4_source_pool_camera_frame_replication_results",
            "T4_synapse_RF_axis_joint_valid_target_count",
            "T4_synapse_RF_axis_identity_median_angle_degrees",
            "T4_synapse_RF_axis_identity_cardinal_match_fraction",
            "T4_synapse_RF_axis_descriptive_best_transform",
            "T4_synapse_RF_axis_best_median_angle_degrees",
            "T4_source_RF_axis_interchangeability_verified",
            "T4_RF_axis_replacement_authorized",
    ]
    checklist = {item["requirement"]: item for item in report["requirement_checklist"]}
    assert checklist["8.separate_planner_fly_core_executor_contributions"]["status"] == "passed"
    assert checklist["8.disclose_controlled_vision_evidence_boundaries"]["status"] == (
        "passed"
    )
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
    assert visual["observations"]["Borst_2025_parameterized_target_verified"] is True
    assert (
        visual["observations"]["Borst_2025_target_is_experimental_membrane_voltage"]
        is False
    )
    assert visual["observations"]["Borst_2025_source_transfer_authorized"] is False
    assert visual["observations"]["Pirogova_numeric_calcium_sources"] == [
        "Mi1",
        "Tm1",
        "Tm2",
        "Tm3",
    ]
    assert visual["observations"]["Pirogova_biological_individual_ids_available"] is False
    assert visual["observations"]["Pirogova_source_transfer_authorized"] is False
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
        "v7_camera_angular_calibration",
        "Tm1_Tm2_Tm4_Tm9_membrane_like_kernels",
        "CT1_membrane_like_kernel",
        "stable_source_or_target_cell_to_MaleCNS_mapping",
        "independent_dynamic_validation_cohort",
    }
    assert visual["observations"]["T5_physical_source_transfer_required_fields"] == {
        "v7_physical_frame_interval": True,
        "v7_physical_solver_interval": True,
        "v7_camera_angular_calibration": False,
        "Tm1_Tm2_Tm4_Tm9_membrane_like_kernels": False,
        "CT1_membrane_like_kernel": False,
        "stable_source_or_target_cell_to_MaleCNS_mapping": False,
        "independent_dynamic_validation_cohort": False,
    }
    assert visual["observations"]["v7_offline_time_coordinate_contract_complete"] is True
    assert (
        visual["observations"]["v7_offline_horizontal_coordinate_contract_complete"]
        is True
    )
    assert (
        visual["observations"][
            "v7_offline_two_dimensional_angular_calibration_complete"
        ]
        is False
    )
    assert visual["observations"]["v7_vertical_camera_ray_angles_declared"] is False
    assert visual["observations"]["v7_vertical_FOV_declared"] is False
    assert (
        visual["observations"]["v7_vertical_pixel_to_angle_formula_declared"]
        is False
    )
    assert (
        visual["observations"]["v7_vertical_motion_has_physical_angular_units"]
        is False
    )
    assert (
        visual["observations"]["v7_looming_radius_has_physical_angular_units"]
        is False
    )
    assert visual["observations"]["v7_retinal_v_has_physical_angular_units"] is False
    assert visual["observations"]["v7_offline_2D_engineering_angular_grid_complete"] is True
    assert np.isclose(
        visual["observations"]["v7_offline_engineering_vertical_FOV_degrees"],
        70.09590046813252,
    )
    assert np.isclose(
        visual["observations"][
            "v7_offline_engineering_angular_pixel_pitch_degrees"
        ],
        3.047647846440544,
    )
    assert visual["observations"]["v7_offline_engineering_grid_biologically_calibrated"] is False
    assert (
        visual["observations"][
            "v7_controlled_stimulus_and_R1_R6_input_boundary_complete"
        ]
        is True
    )
    assert visual["observations"]["v7_controlled_base_stimulus_count"] == 20
    assert visual["observations"]["v7_controlled_stimuli_per_independent_split"] == 172
    assert visual["observations"]["v7_typed_LPLC_LC4_stimulus_count"] == 33
    assert visual["observations"]["v7_external_drive_non_R1_R6_node_count"] == 0
    assert visual["observations"]["v7_target_direct_external_drive_overlap"] == 0
    assert visual["observations"]["v7_offline_frame_interval_milliseconds"] == 10.0
    assert visual["observations"]["v7_offline_substep_interval_milliseconds"] == 2.5
    assert visual["observations"]["T5_physical_source_transfer_ready"] is False
    assert visual["observations"]["T5_transfer_synthesis_kernel_shape_source_count"] == 4
    assert visual["observations"]["T5_transfer_synthesis_mapping_source_count"] == 4
    assert visual["observations"]["T5_transfer_synthesis_positive_shape_evidence_count"] == 4
    assert visual["observations"]["T5_transfer_synthesis_absolute_gain_available"] is False
    assert (
        visual["observations"][
            "T5_transfer_synthesis_official_target_model_available"
        ]
        is True
    )
    assert (
        visual["observations"][
            "T5_transfer_synthesis_target_to_source_mapping_available"
        ]
        is False
    )
    assert (
        visual["observations"][
            "T5_transfer_synthesis_Figure6_axolotl_source_available"
        ]
        is False
    )
    assert (
        visual["observations"][
            "T5_transfer_synthesis_related_axolotl_project_identified"
        ]
        is True
    )
    assert (
        visual["observations"][
            "T5_transfer_synthesis_axolotl_repository_readable"
        ]
        is False
    )
    assert (
        visual["observations"]["T5_measured_kernel_temporal_identifiability_passed"]
        is False
    )
    assert (
        visual["observations"]["T5_measured_kernel_direction_scoring_performed"]
        is False
    )
    assert visual["observations"]["T5_measured_kernel_frame_alignment_verified"] is True
    assert (
        visual["observations"]["T5_measured_kernel_probe_solver_alignment_verified"]
        is False
    )
    assert (
        visual["observations"]["T5_measured_kernel_physical_transfer_authorized"]
        is False
    )
    assert (
        visual["observations"][
            "T5_measured_kernel_standard_substep_negative_reproduced"
        ]
        is True
    )
    assert (
        visual["observations"][
            "T5_measured_kernel_cross_substep_identifiability_passed"
        ]
        is False
    )
    assert (
        visual["observations"][
            "T5_measured_kernel_cross_substep_direction_scoring_performed"
        ]
        is False
    )
    assert (
        visual["observations"][
            "T5_measured_kernel_cross_substep_direction_scoring_authorized"
        ]
        is False
    )
    assert visual["observations"]["T5_measured_kernel_causal_index_zero_supported"] is True
    assert visual["observations"]["T5_measured_kernel_all_population_peaks_covered"] is True
    assert visual["observations"]["T5_measured_kernel_full_L1_support_covered"] is False
    assert (
        visual["observations"][
            "T5_measured_kernel_prefix_alone_full_support_negative_authorized"
        ]
        is False
    )
    assert visual["observations"]["T5_measured_kernel_trace_samples"] == 27
    assert np.isclose(
        visual["observations"]["T5_measured_kernel_minimum_prefix_L1_mass_fraction"],
        0.4082240394891599,
    )
    assert np.isclose(
        visual["observations"]["T5_measured_kernel_maximum_prefix_L1_mass_fraction"],
        0.47168529082482163,
    )
    assert visual["observations"]["T5_measured_kernel_zero_tail_full_support_evaluated"] is True
    assert (
        visual["observations"][
            "T5_measured_kernel_zero_tail_cross_substep_identifiability_passed"
        ]
        is False
    )
    assert (
        visual["observations"]["T5_measured_kernel_zero_tail_all_candidates_failed"]
        is True
    )
    assert (
        visual["observations"]["T5_measured_kernel_zero_tail_direction_scoring_authorized"]
        is False
    )
    assert (
        visual["observations"]["T5_measured_kernel_zero_tail_direction_scoring_performed"]
        is False
    )
    assert visual["observations"]["T5_measured_kernel_zero_tail_samples"] == 498
    assert visual["observations"]["T5_measured_kernel_full_support_output_samples"] == 525
    assert (
        visual["observations"]["T5_population_kernel_recording_id_robustness_passed"]
        is False
    )
    assert (
        visual["observations"][
            "T5_Tm2_population_kernel_recording_id_robustness_passed"
        ]
        is False
    )
    assert (
        visual["observations"]["T5_population_kernel_independent_validation_available"]
        is False
    )
    assert visual["observations"]["T5_population_kernel_transfer_authorized"] is False
    assert visual["observations"]["T5_population_kernel_robustness_passing_source_count"] == 3
    assert visual["observations"]["T5_population_kernel_robustness_failing_sources"] == [
        "Tm2"
    ]
    assert np.isclose(
        visual["observations"]["T5_Tm2_recording_id_vs_rest_median_correlation"],
        0.785694273562359,
    )
    assert np.isclose(
        visual["observations"]["T5_Tm2_partition_correlation_p05"],
        0.76950266256901,
    )
    assert (
        visual["observations"]["T5_population_kernel_author_default_baseline_verified"]
        is True
    )
    assert (
        visual["observations"]["T5_population_kernel_no_baseline_matches_author_default"]
        is True
    )
    assert (
        visual["observations"][
            "T5_population_kernel_author_row_weighting_exactly_reproduced"
        ]
        is False
    )
    assert (
        visual["observations"]["T5_population_kernel_tail_baseline_variant_authorized"]
        is False
    )
    assert visual["observations"]["T5_population_kernel_author_call_count"] == 9
    assert visual["observations"]["T5_population_kernel_explicit_baseline_call_count"] == 0
    assert np.isclose(
        visual["observations"]["T5_Tm1_row_vs_recording_id_weighted_correlation"],
        0.9995598341098436,
    )
    assert visual["observations"]["T5_author_row_weighted_changed_sources"] == ["Tm1"]
    assert (
        visual["observations"][
            "T5_author_row_weighted_full_support_all_candidates_failed"
        ]
        is True
    )
    assert (
        visual["observations"][
            "T5_author_row_weighted_cross_substep_identifiability_passed"
        ]
        is False
    )
    assert (
        visual["observations"]["T5_author_row_weighted_direction_scoring_authorized"]
        is False
    )
    assert (
        visual["observations"]["T5_author_row_weighted_direction_scoring_performed"]
        is False
    )
    assert visual["observations"]["T5_Tm2_LOO_full_support_all_evaluations_failed"] is True
    assert visual["observations"]["T5_Tm2_LOO_robust_temporal_identifiability_passed"] is False
    assert visual["observations"]["T5_Tm2_LOO_direction_scoring_authorized"] is False
    assert visual["observations"]["T5_Tm2_LOO_direction_scoring_performed"] is False
    assert (
        visual["observations"]["T5_Tm2_LOO_independent_biological_validation_performed"]
        is False
    )
    assert visual["observations"]["T5_Tm2_LOO_fold_count"] == 5
    assert visual["observations"]["T5_Tm2_LOO_evaluation_count"] == 30
    assert visual["observations"]["T5_Tm2_LOO_passed_evaluation_count"] == 0
    ranges = visual["observations"]["T5_Tm2_LOO_candidate_ratio_ranges"]
    assert np.allclose(
        ranges["temporal_difference_filtered_Tm_pair_reichardt"]["shuffle"],
        [0.9860891425025265, 1.2287928486318291],
    )
    assert np.allclose(
        ranges["temporal_difference_filtered_Tm_pair_reichardt"]["static"],
        [0.5921415935025431, 0.6741595238667628],
    )
    assert (
        visual["observations"]["T5_measured_kernel_typed_recurrent_cascade_verified"]
        is True
    )
    assert visual["observations"]["T5_measured_kernel_replaces_existing_source_dynamics"] is False
    assert (
        visual["observations"][
            "T5_measured_kernel_single_stage_biological_interpretation_authorized"
        ]
        is False
    )
    assert visual["observations"]["T5_measured_kernel_external_state_mapping_available"] is False
    assert visual["observations"]["T5_measured_kernel_source_leaks"] == {
        "Tm1": 0.28,
        "Tm2": 0.55,
        "Tm4": 0.34,
        "Tm9": 0.16,
    }
    assert visual["observations"]["T5_measured_kernel_source_node_counts"] == {
        "Tm1": 1777,
        "Tm2": 1766,
        "Tm4": 1670,
        "Tm9": 1771,
    }
    assert (
        visual["observations"]["T5_measured_kernel_source_inputs_partitioned_exactly_once"]
        is True
    )
    assert (
        visual["observations"][
            "T5_measured_kernel_every_source_has_recurrent_or_feedback_input"
        ]
        is True
    )
    assert (
        visual["observations"]["T5_measured_kernel_feedforward_only_source_drive_available"]
        is False
    )
    assert (
        visual["observations"]["T5_measured_kernel_source_dynamics_replacement_evaluated"]
        is False
    )
    assert (
        visual["observations"]["T5_measured_kernel_source_dynamics_replacement_authorized"]
        is False
    )
    assert visual["observations"][
        "T5_measured_kernel_recurrent_or_feedback_fraction_by_source"
    ] == {
        "Tm1": 0.06242381162002561,
        "Tm2": 0.04116725419043349,
        "Tm4": 0.19606226563946827,
        "Tm9": 0.23058754793814995,
    }
    assert np.isclose(
        visual["observations"]["T5_measured_kernel_Tm9_CT1_input_fraction"],
        0.09073247310668342,
    )
    assert visual["observations"]["T5_lamina_only_replacement_evaluated"] is True
    assert (
        visual["observations"]["T5_lamina_only_replacement_all_candidates_failed"]
        is True
    )
    assert (
        visual["observations"]["T5_lamina_only_replacement_temporal_identifiability_passed"]
        is False
    )
    assert (
        visual["observations"]["T5_lamina_only_replacement_direction_scoring_authorized"]
        is False
    )
    assert visual["observations"]["T5_lamina_only_replacement_direction_scoring_performed"] is False
    assert (
        visual["observations"]["T5_lamina_only_replacement_physical_transfer_authorized"]
        is False
    )
    coverage = visual["observations"]["T5_lamina_only_source_coverage"]
    missing = {
        source: item["source_node_without_lamina_input_count"]
        for source, item in coverage.items()
    }
    assert missing == {
        "Tm1": 3,
        "Tm2": 1,
        "Tm4": 0,
        "Tm9": 1,
    }
    replacement_ratios = visual["observations"][
        "T5_lamina_only_candidate_ratios_by_update"
    ]
    assert np.isclose(
        replacement_ratios["temporal_difference_filtered_Tm_pair_reichardt"]["4"][
            "shuffle"
        ],
        1.6277322953083266,
    )
    assert visual["observations"]["T5_temporal_shuffle_frame_multiset_preserved"] is True
    assert visual["observations"]["T5_temporal_shuffle_retinal_energy_preserved"] is False
    assert visual["observations"]["T5_temporal_shuffle_lamina_source_energy_preserved"] is False
    assert visual["observations"]["T5_temporal_shuffle_energy_matched_control_verified"] is False
    assert (
        visual["observations"][
            "T5_equal_energy_temporal_selectivity_interpretation_authorized"
        ]
        is False
    )
    assert visual["observations"]["T5_new_energy_normalized_gate_authorized"] is False
    input_ratios = visual["observations"]["T5_temporal_shuffle_input_energy_ratios_by_update"]
    assert np.isclose(input_ratios["1"]["pixel_temporal_difference"], 6.93426194129865)
    assert np.isclose(input_ratios["1"]["R1_R6_signed_frame_difference"], 5.837713478478663)
    assert np.isclose(
        input_ratios["4"]["lamina_only_Tm_preactivation"]["Tm9"],
        5.127327256460109,
    )
    assert visual["observations"]["T5_increment_order_control_images_valid"] is True
    assert (
        visual["observations"]["T5_increment_order_control_terminal_frame_preserved"]
        is True
    )
    assert (
        visual["observations"]["T5_increment_order_control_R1_R6_energy_matched"]
        is True
    )
    assert (
        visual["observations"][
            "T5_increment_order_control_independent_condition_evaluated"
        ]
        is False
    )
    assert (
        visual["observations"]["T5_increment_order_control_replacement_authorized"]
        is False
    )
    assert np.isclose(
        visual["observations"]["T5_increment_order_control_output_ratios_by_update"][
            "1"
        ]["summed_filtered_source_centroid_projection"],
        1.0087448156400647,
    )
    assert (
        visual["observations"]["T5_increment_order_replication_input_validity_passed"]
        is True
    )
    assert (
        visual["observations"]["T5_increment_order_replication_same_candidate_passed"]
        is False
    )
    assert visual["observations"]["T5_increment_order_replication_gate_passed"] is False
    assert (
        visual["observations"][
            "T5_increment_order_replication_direction_scoring_authorized"
        ]
        is False
    )
    assert np.isclose(
        visual["observations"]["T5_increment_order_replication_output_ratios"][
            "S1-T02"
        ]["1"]["fast_pool_vs_Tm9_centroid_difference"],
        1.0892274724774895,
    )
    assert visual["observations"]["T5_transfer_synthesis_CT1_complete"] is False
    assert visual["observations"]["T5_transfer_synthesis_ready"] is False
    assert visual["observations"]["T4_T5_source_dynamics_ready"] is False
    assert visual["observations"]["T4_T5_source_dynamics_passing_gates"] == [
        "T4_crossfit_structure_axis",
        "T5_crossfit_structure_axis",
        "shared_physical_v7_timebase",
    ]
    assert set(visual["observations"]["T4_T5_source_dynamics_failing_gates"]) == {
        "T4_source_dynamics_transfer",
        "T5_source_dynamics_transfer",
        "independent_dynamic_validation",
    }
    assert visual["observations"]["T4_T5_source_dynamics_authorizes_new_candidate"] is False
    assert visual["observations"]["T4_millivolt_source_types_complete"] is True
    assert visual["observations"]["T4_source_biological_individual_ID_available"] is True
    assert visual["observations"]["T4_individual_level_source_validation_ready"] is False
    assert visual["observations"]["T4_individual_split_passing_source_conditions"] == [
        "on:Tm3"
    ]
    assert visual["observations"]["T4_all_individual_split_gates_passed"] is False
    assert visual["observations"]["T4_individual_split_sample_interval_milliseconds"] == 1.0
    assert (
        visual["observations"]["T4_1khz_split_pass_fail_pattern_changed_from_10ms"]
        is False
    )
    assert visual["observations"]["Edmond_Fig3_frozen_four_file_manifest_complete"] is True
    assert visual["observations"]["Edmond_Fig3_1khz_payload_currently_verified"] is True
    assert (
        visual["observations"]["Edmond_Fig3_full_resolution_recompute_authorized"]
        is True
    )
    assert visual["observations"]["Edmond_network_failure_is_scientific_rejection"] is False
    assert visual["observations"]["T4_recording_field_count_available"] == 10
    assert visual["observations"]["T4_recording_field_count_required"] == 15
    assert visual["observations"]["T4_missing_recording_fields"] == [
        "cohort_role",
        "stimulus_id",
        "stimulus_direction",
        "stimulus_angular_position_degrees",
        "baseline_window_seconds",
    ]
    assert visual["observations"]["T4_complete_stimulus_and_baseline_fields"] is False
    assert visual["observations"]["T4_Edmond_complete_dataset_file_count"] == 74
    assert visual["observations"]["T4_Edmond_Fig3_directory_file_count"] == 9
    assert visual["observations"]["T4_Edmond_Fig3_identity_sidecars"] == []
    assert visual["observations"]["T4_Edmond_workbook_recording_metadata_recovered"] is False
    assert visual["observations"]["T4_Edmond_Fig1_Fig3_cohort_relation_identifiable"] is False
    assert visual["observations"]["T4_Edmond_Fig3_edge_baseline_window_declared"] is False
    assert visual["observations"]["T5_voltage_modality_field_counts"] == {
        "aggregated_full_field_OFF_flash": 7,
        "raw_white_noise": 8,
        "raw_drifting_grating": 9,
    }
    assert visual["observations"]["T5_cross_modality_field_union_accepted"] is False
    assert visual["observations"]["T5_all_five_complete_recording_fields"] is False
    assert visual["observations"]["Kohn_Portes_stimulus_generator_commit"] == (
        "b589a224493cb66bda4c55f632b213cacb082b24"
    )
    assert visual["observations"]["Kohn_Portes_record_specific_stimulus_logs_available"] is False
    assert visual["observations"]["Kohn_Portes_generator_defaults_used_as_record_fields"] is False
    assert visual["observations"]["Kohn_Portes_stimulus_provenance_complete"] is False
    assert visual["observations"]["T5_Figure4_relative_PD_ND_mapping_verified"] is True
    assert (
        visual["observations"][
            "T5_Figure4_native_coordinate_motion_mapping_verified"
        ]
        is True
    )
    assert (
        visual["observations"][
            "T5_Figure4_absolute_physical_direction_mapping_verified"
        ]
        is False
    )
    assert (
        visual["observations"]["Kohn_Portes_record_level_direction_code_available"]
        is False
    )
    assert (
        visual["observations"][
            "Kohn_Portes_record_level_physical_direction_available"
        ]
        is False
    )
    assert visual["observations"]["T5_cross_dataset_direction_mapping_authorized"] is False
    assert visual["observations"]["Motyxia2_public_history_branch_count"] == 22
    assert visual["observations"]["Motyxia2_public_history_commit_count"] == 447
    assert (
        visual["observations"][
            "Kohn_Portes_record_log_found_in_Motyxia2_public_history"
        ]
        is False
    )
    assert (
        visual["observations"][
            "Kohn_Portes_external_successful_indexes_linked_log_found"
        ]
        is False
    )
    assert visual["observations"]["Kohn_Portes_PMC_supplement_content_inspected"] is False
    assert visual["observations"]["Kohn_Portes_publisher_supplements_inspected"] is True
    assert (
        visual["observations"]["Kohn_Portes_publisher_supplements_contain_record_log"]
        is False
    )
    assert visual["observations"]["Kohn_Portes_Figshare_search_accessible"] is False
    assert visual["observations"]["Kohn_Portes_stimulus_log_global_absence_claimed"] is False
    assert visual["observations"]["Behnia_T4_independent_fast_source_phenotypes"] == [
        "Mi1",
        "Tm3",
    ]
    assert visual["observations"]["Behnia_T4_independent_numeric_payload_verified"] is False
    assert (
        visual["observations"][
            "Behnia_T4_numeric_payload_found_in_audited_public_indexes"
        ]
        is False
    )
    assert visual["observations"]["Behnia_T4_independent_transfer_authorized"] is False
    assert visual["observations"]["T4_inhibitory_external_Mi4_allowed_unit"] is False
    assert visual["observations"]["T4_inhibitory_external_C3_allowed_unit"] is False
    assert (
        visual["observations"]["T4_inhibitory_external_C3_fixed_robustness_passed"]
        is False
    )
    assert visual["observations"]["T4_inhibitory_external_transfer_authorized"] is False
    assert visual["observations"]["Mi4_C3_candidate_count_audited"] == 13
    assert visual["observations"]["Ketkar_2019_official_source_data_attachment_count"] == 11
    assert visual["observations"]["Ketkar_2019_attachments_are_mean_SEM_tables"] is True
    assert visual["observations"]["Ketkar_2019_Mi1_Tm3_GCaMP_summary_found"] is True
    assert visual["observations"]["Ketkar_2019_Mi4_C3_payload_found"] is False
    assert visual["observations"]["Ketkar_2019_individual_source_dynamics_found"] is False
    assert visual["observations"]["Ketkar_2019_membrane_voltage_found"] is False
    assert visual["observations"]["Gonzalez_Suarez_2022_Mi4_GCaMP6f_fly_count"] == 15
    assert visual["observations"]["Gonzalez_Suarez_2022_Mi4_type_average_filter_available"] is True
    assert visual["observations"]["Gonzalez_Suarez_2022_individual_cell_axis_available"] is False
    assert visual["observations"]["Gonzalez_Suarez_2022_Mi4_voltage_available"] is False
    assert visual["observations"]["Gonzalez_Suarez_2022_C3_dynamics_available"] is False
    assert visual["observations"]["Gonzalez_Suarez_2022_bioRxiv_supplement_retrieved"] is True
    assert (
        visual["observations"][
            "Gonzalez_Suarez_2022_individual_flies_are_statistical_units"
        ]
        is True
    )
    assert visual["observations"]["Gonzalez_Suarez_2022_public_individual_numeric_payload"] is False
    assert visual["observations"]["Yuan_2020_C3_intervention_candidate_verified"] is True
    assert visual["observations"]["Yuan_2020_C3_direct_recording_verified"] is False
    assert visual["observations"]["Yuan_2020_C3_numeric_source_dynamics_verified"] is False
    assert visual["observations"]["Yuan_2020_supplement_retrieved"] is False
    assert visual["observations"]["Strother_2018_successful_index_Mi4_trace_found"] is False
    assert visual["observations"]["Strother_2018_Figshare_search_interpretable"] is False
    assert visual["observations"]["Strother_2018_global_absence_claimed"] is False
    assert visual["observations"]["C3_citation_graph_union_unique_work_count"] == 240
    assert visual["observations"]["C3_citation_graph_unresolved_reference_ID_count"] == 2
    assert visual["observations"]["Pang_2025_directly_recorded_neuron_types"] == ["L1", "L2"]
    assert visual["observations"]["Pang_2025_Dryad_file_count"] == 75
    assert visual["observations"]["Pang_2025_new_C3_direct_recording_found"] is False
    assert visual["observations"]["Hao_2026_ASAP7y_candidate_classification"] == (
        "unresolved_high_value_candidate"
    )
    assert visual["observations"]["Hao_2026_ASAP7y_paper_source_experimental_cell_types"] == []
    assert visual["observations"]["Hao_2026_ASAP7y_public_author_figure_named_examples"] == [
        "Dm9",
        "MeLo13",
    ]
    assert visual["observations"]["Hao_2026_ASAP7y_complete_cell_type_set_resolved"] is False
    assert visual["observations"]["Hao_2026_ASAP7y_Drosophila_voltage_verified"] is True
    assert visual["observations"]["Hao_2026_ASAP7y_public_numeric_payload_verified"] is False
    assert visual["observations"]["Hao_2025_dissertation_restricted_until"] == (
        "2027-03-14"
    )
    assert (
        visual["observations"][
            "Hao_2025_dissertation_public_abstract_names_cell_types"
        ]
        is False
    )
    assert (
        visual["observations"][
            "Hao_2026_ASAP7y_successful_index_numeric_payload_found"
        ]
        is False
    )
    assert (
        visual["observations"]["Hao_2026_ASAP7y_global_payload_absence_claimed"]
        is False
    )
    assert {"Tm1", "Tm2", "Tm4", "Tm9", "Dm12"} <= set(
        visual["observations"]["Gur_2024_directly_recorded_neuron_types"]
    )
    assert visual["observations"]["Gur_2024_Mi4_proofreading_row_count"] == 723
    assert visual["observations"]["Gur_2024_C3_proofreading_row_count"] == 678
    assert visual["observations"]["Gur_2024_Mi4_C3_files_are_proofreading_only"] is True
    assert visual["observations"]["Gur_2024_Mi4_C3_direct_physiology_found"] is False
    assert visual["observations"]["Tanaka_2023_Mi4_fly_count"] == 10
    assert visual["observations"]["Tanaka_2023_Mi4_selected_ROI_count"] == 201
    assert visual["observations"]["Tanaka_2023_Mi4_individual_fly_axis_available"] is True
    assert visual["observations"]["Tanaka_2023_Mi4_response_unit"] == "deltaF_over_F"
    assert visual["observations"]["Tanaka_2023_Mi4_experimental_membrane_voltage"] is False
    assert (
        visual["observations"]["Tanaka_2023_Mi4_source_dynamics_transfer_authorized"]
        is False
    )
    assert visual["observations"]["Tanaka_2023_Figure_6_named_Mi4_C3_member_count"] == 4
    assert (
        visual["observations"][
            "Tanaka_2023_Figure_6_Mi4_C3_neural_activity_recording"
        ]
        is False
    )
    assert (
        visual["observations"][
            "Tanaka_2023_Figure_6_Mi4_C3_source_dynamics_payload"
        ]
        is False
    )
    assert visual["observations"]["Wu_2026_afterimages_Mi4_fly_count"] == 6
    assert visual["observations"]["Wu_2026_afterimages_Mi4_ROI_count"] == 113
    assert visual["observations"]["Wu_2026_afterimages_Mi4_response_unit"] == (
        "deltaF_over_F"
    )
    assert (
        visual["observations"][
            "Wu_2026_afterimages_public_numeric_Mi4_payload_verified"
        ]
        is False
    )
    assert (
        visual["observations"][
            "Wu_2026_afterimages_Mi4_source_dynamics_transfer_authorized"
        ]
        is False
    )
    assert visual["observations"]["Henning_C3_directional_payload_fly_count"] == 6
    assert visual["observations"]["Henning_C3_directional_caption_fly_count"] == 8
    assert visual["observations"]["Henning_C3_directional_ROI_count"] == 77
    assert (
        visual["observations"][
            "Henning_C3_directional_edge_speed_degrees_per_second"
        ]
        == 20.0
    )
    assert visual["observations"]["Henning_C3_directional_cohort_independent"] is False
    assert (
        visual["observations"][
            "Henning_C3_direction_specific_source_kernel_verified"
        ]
        is False
    )
    assert (
        visual["observations"]["Henning_C3_directional_source_transfer_authorized"]
        is False
    )
    assert visual["observations"]["Mi4_C3_direct_numeric_voltage_candidates"] == [
        "Groschner_2022"
    ]
    assert visual["observations"]["Mi4_C3_independent_numeric_voltage_candidates"] == []
    assert (
        visual["observations"]["Mi4_C3_independent_voltage_transfer_authorized"]
        is False
    )
    assert visual["observations"]["source_dynamics_external_contract_satisfied"] is False
    assert visual["observations"]["source_dynamics_external_payload_present"] is False
    assert visual["observations"]["Dryad_L1L2_required_source_coverage_count"] == 0
    assert visual["observations"]["Dryad_L1L2_source_dynamics_transfer_authorized"] is False
    assert visual["observations"]["Dryad_L1L2_large_download_authorized"] is False
    assert visual["observations"]["Gou_sparsity_required_source_coverage_count"] == 4
    assert visual["observations"]["Gou_sparsity_partial_source_evidence_present"] is True
    assert visual["observations"]["Gou_sparsity_source_dynamics_transfer_authorized"] is False
    assert visual["observations"]["Gou_sparsity_bulk_download_authorized"] is False
    assert visual["observations"]["Gou_Dryad_archive_hash_locally_verified"] is True
    assert visual["observations"]["Gou_Dryad_local_processed_calcium_sources"] == [
        "Mi1",
        "Tm3",
        "Tm1",
        "Tm2",
    ]
    assert visual["observations"]["Gou_Dryad_flash_fly_axis_sizes"] == {
        "Mi1": 16,
        "Tm3": 11,
        "Tm1": 9,
        "Tm2": 8,
    }
    assert visual["observations"]["Gou_Dryad_moving_bar_fly_axis_sizes"] == {
        "Mi1": 10,
        "Tm3": 6,
        "Tm1": 8,
        "Tm2": 12,
    }
    assert visual["observations"]["Gou_Dryad_Mi1_Tm3_direction_axis_identifiable"] is False
    assert visual["observations"]["Gou_Dryad_direction_invariance_evaluated"] is False
    assert visual["observations"]["Gou_DANDI_all_assets_stimulus_metadata_indexed"] is True
    assert visual["observations"]["Gou_DANDI_Mi1_Tm3_asset_counts"] == {
        "Mi1": 144,
        "Tm3": 20,
    }
    assert visual["observations"]["Gou_DANDI_Mi1_Tm3_stimulus_metadata_available"] is False
    assert (
        visual["observations"][
            "Gou_Dryad_stable_biological_individual_IDs_verified"
        ]
        is False
    )
    assert visual["observations"]["Gou_Dryad_experimental_membrane_voltage"] is False
    assert visual["observations"]["Gou_DANDI_asset_level_stable_participant_IDs_verified"] is True
    assert visual["observations"]["Gou_DANDI_unique_subject_ID_count"] == 282
    assert visual["observations"]["Gou_Dryad_distinct_fliesUsed_label_count"] == 66
    assert visual["observations"]["Gou_Dryad_DANDI_identity_crosswalk_match_count"] == 0
    assert visual["observations"]["Gou_Dryad_rows_to_DANDI_subject_crosswalk_verified"] is False
    assert visual["observations"]["T5_contrast_opponency_temporal_sources"] == [
        "Tm4",
        "Tm9",
    ]
    assert visual["observations"]["T5_contrast_opponency_spatial_only_sources"] == [
        "CT1"
    ]
    assert visual["observations"]["T5_contrast_opponency_transfer_authorized"] is False
    assert visual["observations"]["Braun_T5_local_calcium_sources"] == [
        "CT1",
        "Tm2",
        "Tm9",
    ]
    assert visual["observations"]["Braun_T5_sources_with_stable_fly_IDs"] == [
        "CT1",
        "Tm2",
        "Tm9",
    ]
    assert visual["observations"]["Braun_T5_payload_fly_counts"] == {
        "Tm2": 9,
        "Tm9": 11,
        "CT1": 9,
    }
    assert visual["observations"]["Braun_T5_all_condition_grids_complete"] is True
    assert visual["observations"]["Braun_T5_allowed_voltage_sources"] == []
    assert visual["observations"]["Braun_T5_baseline_window_declared"] is False
    assert visual["observations"]["Braun_T5_transfer_authorized"] is False
    assert visual["observations"]["Yang_T5_optical_voltage_phenotype_sources"] == [
        "Tm1",
        "Tm2",
    ]
    assert visual["observations"]["Yang_T5_numerical_voltage_payload_verified"] is False
    assert visual["observations"]["Yang_T5_voltage_transfer_authorized"] is False
    assert visual["observations"]["Yang_PMC_attachment_count"] == 9
    assert visual["observations"]["Yang_PMC_numeric_attachment_count"] == 0
    assert visual["observations"]["Yang_successful_public_indexes_numeric_payload_found"] is False
    assert visual["observations"]["Yang_Figshare_search_accessible"] is False
    assert visual["observations"]["Kohn_Portes_T5_numeric_voltage_sources"] == [
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
    ]
    assert visual["observations"]["Kohn_Portes_T5_voltage_derived_kernel_sources"] == [
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
    ]
    assert visual["observations"]["Kohn_Portes_T5_voltage_derived_kernel_record_count"] == 25
    assert visual["observations"]["Kohn_Portes_T5_kernel_sample_interval_seconds"] == 0.01
    assert visual["observations"]["Kohn_Portes_T5_raw_kernel_gain_transferable"] is False
    assert visual["observations"]["Kohn_Portes_T5_kernel_stimulus_invariant"] is False
    assert visual["observations"]["Kohn_Portes_T5_source_kernel_transfer_authorized"] is False
    assert visual["observations"]["Kohn_Portes_T5_state_unit_record_count"] == 41
    assert (
        visual["observations"][
            "Kohn_Portes_T5_exact_volts_to_millivolts_scale_verified"
        ]
        is True
    )
    assert visual["observations"]["Kohn_Portes_T5_state_mapping_available"] is False
    assert visual["observations"]["Kohn_Portes_T5_candidate_normalization_formula"] is None
    assert visual["observations"]["Kohn_Portes_T5_candidate_clipping_rule"] is None
    assert (
        visual["observations"][
            "Kohn_Portes_T5_saline_fast_pooled_median_latency_ms"
        ]
        == 50.0
    )
    assert visual["observations"]["Kohn_Portes_T5_saline_Tm9_median_latency_ms"] == 80.0
    assert visual["observations"]["Kohn_Portes_T5_OA_fast_pooled_median_latency_ms"] == 40.0
    assert visual["observations"]["Kohn_Portes_T5_OA_Tm9_median_latency_ms"] == 50.0
    assert visual["observations"]["Kohn_Portes_T5_saline_Tm9_relative_delay_supported"]
    assert (
        visual["observations"]["Kohn_Portes_T5_Tm9_delay_ordering_state_invariant"]
        is False
    )
    assert visual["observations"]["Kohn_Portes_T5_source_delay_transfer_authorized"] is False
    assert visual["observations"][
        "Kohn_Portes_T5_saline_record_median_preferred_frequency_hz"
    ] == {"Tm1": 0.95, "Tm2": 0.95, "Tm4": 0.675, "Tm9": 0.475}
    assert visual["observations"][
        "Kohn_Portes_T5_OA_record_median_preferred_frequency_hz"
    ] == {"Tm1": 2.325, "Tm2": 2.8, "Tm4": 1.45, "Tm9": 0.25}
    assert (
        visual["observations"]["Kohn_Portes_T5_preferred_frequency_summary_invariant"]
        is False
    )
    assert (
        visual["observations"]["Kohn_Portes_T5_source_frequency_transfer_authorized"]
        is False
    )
    assert visual["observations"]["Kohn_Portes_Tm_to_T5_model_inputs_git_blob_verified"]
    assert visual["observations"]["Kohn_Portes_Tm_to_T5_Figure5_training_fit_count"] == 168
    assert visual["observations"]["Kohn_Portes_Tm_to_T5_fit_score_samples_disjoint"] is False
    assert (
        visual["observations"]["Kohn_Portes_Tm_to_T5_independent_validation_available"]
        is False
    )
    assert visual["observations"]["Kohn_Portes_Figure6_Tm2_ND_source_reference_correct"] is False
    assert visual["observations"]["Kohn_Portes_Tm_to_T5_model_transfer_authorized"] is False
    assert visual["observations"]["FIB19_MaleCNS_T5_population_source_rank_matches"] is True
    assert np.isclose(
        visual["observations"]["FIB19_MaleCNS_T5_maximum_population_ratio_difference"],
        0.126798822451821,
    )
    assert np.isclose(
        visual["observations"]["MaleCNS_T5_Tm9_fraction_outside_FIB19_range"],
        0.7307234295921405,
    )
    assert visual["observations"]["FIB19_to_MaleCNS_T5_weight_transfer_authorized"] is False
    assert visual["observations"]["Kohn_Portes_T5_moving_bar_condition_count"] == 24
    assert visual["observations"]["Kohn_Portes_T5_moving_bar_gain_free_DSI_available"]
    assert (
        visual["observations"][
            "Kohn_Portes_T5_moving_bar_gain_fit_score_samples_disjoint"
        ]
        is False
    )
    assert visual["observations"]["Kohn_Portes_T5_moving_bar_validation_available"] is False
    assert visual["observations"]["Kohn_Portes_T5_moving_bar_transfer_authorized"] is False
    assert visual["observations"]["T5_four_Tm_exact_type_average_mapping_complete"] is True
    assert visual["observations"]["T5_all_five_source_mapping_complete"] is False
    assert visual["observations"]["T5_CT1_mapping_Lo1_column_count_by_body"] == {
        "10009": 868,
        "10157": 802,
    }
    assert visual["observations"]["Kohn_Portes_T5_missing_voltage_sources"] == ["CT1"]
    assert visual["observations"]["Kohn_Portes_T5_identity_retained"] is False
    assert visual["observations"]["Kohn_Portes_T5_recording_ID_field_retained"] is True
    assert (
        visual["observations"][
            "Kohn_Portes_T5_minimum_train_validation_capacity_ready"
        ]
        is False
    )
    assert visual["observations"]["Kohn_Portes_T5_transfer_authorized"] is False
    assert visual["observations"]["Kohn_Portes_full_repository_recording_ID_upper_bounds"] == {
        "Tm1": 8,
        "Tm2": 7,
        "Tm4": 7,
        "Tm9": 13,
    }
    assert visual["observations"][
        "Kohn_Portes_sources_meeting_numeric_5_plus_3_recording_ID_upper_bound"
    ] == ["Tm1", "Tm9"]
    assert visual["observations"]["Kohn_Portes_biological_individual_semantics_verified"] is False
    assert visual["observations"]["Kohn_Portes_biological_individual_split_authorized"] is False
    assert visual["observations"]["CT1_extreme_experimental_calcium_verified"] is True
    assert visual["observations"]["CT1_extreme_model_voltage_is_simulated"] is True
    assert visual["observations"]["CT1_extreme_experimental_voltage_available"] is False
    assert visual["observations"]["CT1_extreme_transfer_authorized"] is False
    assert visual["observations"]["CT1_PuRe_official_archive_candidate_count"] == 2
    assert visual["observations"]["CT1_PuRe_archive_contents_verified"] is True
    assert visual["observations"]["CT1_PuRe_new_numerical_payload_verified"] is False
    assert visual["observations"]["CT1_voltage_audited_candidate_count"] == 11
    assert visual["observations"]["CT1_direct_experimental_voltage_candidate_found"] is False
    assert (
        visual["observations"]["CT1_direct_Lo1_experimental_voltage_candidate_found"]
        is False
    )
    assert visual["observations"]["CT1_experimental_voltage_transfer_authorized"] is False
    assert visual["observations"]["CT1_incremental_2025_2026_relevant_candidates"] == [
        "Samara_Borst_2025",
        "Henning_2026",
        "Okuno_2026",
    ]
    assert visual["observations"][
        "CT1_incremental_2025_2026_direct_voltage_candidates"
    ] == []
    assert visual["observations"]["MaleCNS_nine_source_type_body_sets_available"] is True
    assert visual["observations"]["MaleCNS_external_source_mapping_contract_satisfied"] is False
    assert visual["observations"]["MaleCNS_Tm9_unlocated_source_body_count"] == 0
    assert visual["observations"]["MaleCNS_Tm9_532266_existing_rule_identifiable"] is False
    assert visual["observations"]["MaleCNS_Tm9_532266_post_hoc_repair_authorized"] is False
    assert (
        visual["observations"][
            "MaleCNS_Tm9_532266_official_synapse_coordinate_identifiable"
        ]
        is True
    )
    assert visual["observations"]["MaleCNS_Tm9_532266_official_synapse_coordinate"] == [15, 2]
    assert visual["observations"]["MaleCNS_Tm9_complete_columnar_retinotopy_after_audit"] is True
    assert visual["observations"]["MaleCNS_latest_public_release"] == "male-cns:v1.0"
    assert visual["observations"]["MaleCNS_current_annotation_matches_frozen_v1_0"] is True
    assert visual["observations"][
        "MaleCNS_Tm9_532266_native_coordinate_still_missing_current_release"
    ] is True
    assert visual["observations"]["MaleCNS_CT1_columnar_Lo1_retinotopy_available"] is True
    assert visual["observations"]["MaleCNS_CT1_complete_official_LO_column_coverage"] is False
    assert visual["observations"]["source_evidence_matrix_all_nine_complete"] is False
    assert visual["observations"]["source_evidence_matrix_authorizes_fit"] is False
    assert visual["observations"]["source_evidence_matrix_T4_local_millivolt_count"] == 4
    assert visual["observations"]["source_evidence_matrix_T5_local_calcium_count"] == 5
    assert visual["observations"]["source_evidence_matrix_T5_local_millivolt_count"] == 4
    assert (
        visual["observations"]["source_evidence_matrix_C3_has_any_numerical_fly_IDs"]
        is True
    )
    assert (
        visual["observations"]["source_evidence_matrix_C3_fixed_external_robustness_passed"]
        is False
    )
    assert visual["observations"]["source_type_average_mapped_sources"] == [
        "Mi1",
        "Tm3",
        "Mi4",
        "C3",
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
    ]
    assert visual["observations"]["source_type_average_all_nine_mapping_complete"] is False
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
    assert visual["observations"]["T4_source_pool_camera_frame_bilateral_readouts"] == [
        "pairwise_distal_vector"
    ]
    assert visual["observations"]["T4_source_pool_camera_frame_bilateral_subtypes"] == {
        "pairwise_distal_vector": ["d"]
    }
    assert (
        visual["observations"]["T4_source_pool_camera_frame_post_hoc_candidate_discovered"]
        is True
    )
    assert (
        visual["observations"]["T4_source_pool_camera_frame_replication_preregistered"]
        is False
    )
    assert visual["observations"]["T4_source_pool_camera_frame_replication_evaluated"] is True
    assert (
        visual["observations"]["T4_source_pool_camera_frame_target_formula_authorized"]
        is False
    )
    assert (
        visual["observations"]["T4_source_pool_camera_frame_ordered_replication_passed"]
        is False
    )
    assert visual["observations"]["T4_source_pool_camera_frame_controls_evaluated"] is False
    assert (
        visual["observations"]["T4_source_pool_camera_frame_replication_gate_passed"]
        is False
    )
    assert visual["observations"]["T4_source_pool_camera_frame_replication_results"][
        "LPLC-T03"
    ]["population_scores"]["T4d_R"]["median_signed_contrast"] == (
        -0.7844614740115432
    )
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
    assert visual["observations"]["T4_synapse_RF_axis_joint_valid_target_count"] == 6749
    assert visual["observations"]["T4_synapse_RF_axis_identity_median_angle_degrees"] == (
        68.18171979285492
    )
    assert visual["observations"]["T4_synapse_RF_axis_identity_cardinal_match_fraction"] == (
        0.4951844717735961
    )
    assert visual["observations"]["T4_synapse_RF_axis_descriptive_best_transform"] == (
        "reflect_horizontal"
    )
    assert visual["observations"]["T4_synapse_RF_axis_best_median_angle_degrees"] == (
        21.112026999265463
    )
    assert visual["observations"]["T4_source_RF_axis_interchangeability_verified"] is False
    assert visual["observations"]["T4_RF_axis_replacement_authorized"] is False
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
    assert visual["observations"]["T4_source_dynamics_transfer_fields_available"] == {
        "direction_independent_source_kernel": False,
        "source_to_MaleCNS_identity_mapping": True,
        "physical_v7_sample_interval": True,
        "millivolts_to_v7_normalized_state_mapping": False,
        "ordered_source_sequence_identifiability": False,
    }
    assert visual["observations"]["T4_source_dynamics_transfer_authorized"] is False
    assert visual["observations"]["T4_source_mapping_mode"] == "exact_type_average"
    assert visual["observations"]["T4_source_recording_level_body_assignment"] is False
    assert visual["observations"]["T4_exact_type_average_mapping_complete"] is True
    assert visual["observations"]["T4_author_minmax_formula_reproduced"] is True
    assert visual["observations"]["T4_state_mapping_held_out_outside_fraction_by_source"] == {
        "Mi1": 0.1295625,
        "Tm3": 0.17478125,
        "Mi4": 0.2870446428571429,
        "C3": 0.18847916666666667,
    }
    assert visual["observations"]["T4_author_minmax_semantics_match_v7_state"] is False
    assert visual["observations"]["T4_millivolts_to_v7_state_mapping_available"] is False
    assert visual["observations"]["T4_source_dynamics_next_candidate_authorized"] is False
    assert visual["observations"]["C2C3_version_of_record_DOI"] == (
        "10.7554/eLife.108529.3"
    )
    assert visual["observations"]["C2C3_version_of_record_repository_revision"] == (
        "745c6114b72a61dabc9389a1f024dbe22f679edf"
    )
    assert visual["observations"]["C2C3_version_of_record_new_payload_modality"] == (
        "calcium_fluorescence_STRF"
    )
    assert visual["observations"]["C2C3_version_of_record_C3_fly_count"] == 8
    assert visual["observations"]["C2C3_version_of_record_Mi1_control_fly_count"] == 7
    assert visual["observations"]["C2C3_version_of_record_new_C3_Mi4_voltage_found"] is False
    assert visual["observations"]["C2C3_version_of_record_new_Mi4_payload_found"] is False
    assert visual["observations"]["C2C3_version_of_record_MaleCNS_crosswalk_found"] is False
    assert visual["observations"]["C2C3_version_of_record_changed_T4_transfer_gate"] is False
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
    assert visual["observations"]["Fig1_source_object_payload_safely_inspected"] is True
    assert visual["observations"]["Fig1_individual_source_temporal_RF_arrays_available"] is True
    assert visual["observations"]["Fig1_source_ordinals_match_extended_workbook"] is True
    assert visual["observations"]["Fig1_source_temporal_RF_has_allowed_voltage_unit"] is False
    assert visual["observations"]["Fig1_source_temporal_RF_independent_from_Fig3"] is False
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
