import fly_emotion_api.main as api_module
from fastapi.testclient import TestClient
from fly_emotion_api.main import app


def test_health_reports_driving_task() -> None:
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["connectome"] == "male-cns:v1.0"
    assert response.json()["task"] == "visual-obstacle-and-city-driving"
    assert response.json()["language_model"] == "retired"


def test_health_does_not_mark_obsolete_checkpoint_ready(tmp_path, monkeypatch) -> None:
    import numpy as np

    monkeypatch.setattr(api_module, "ROOT", tmp_path)
    target = tmp_path / "artifacts/checkpoints/driving-policy.npz"
    target.parent.mkdir(parents=True)
    np.savez(target, format_version=np.asarray([2]))
    response = TestClient(app).get("/api/health")
    assert response.json()["policy_checkpoint_ready"] is False
    assert response.json()["policy_checkpoint_version"] == 2
    assert response.json()["required_policy_version"] == 5


def test_v7_status_is_hash_verified_and_explicitly_not_deployed() -> None:
    response = TestClient(app).get("/api/v7/status")
    assert response.status_code == 200
    status = response.json()
    assert status["version"] == "v7-experimental"
    assert status["source"] == "hash-verified-offline-goal-audit"
    assert status["current_stage"] == "controlled_vision"
    assert status["objective_complete"] is False
    assert status["deployment_enabled"] is False
    assert status["default_runtime_changed"] is False
    assert status["gates"]["T4_T5_direction_and_ON_OFF"] is False
    assert status["gates"]["LPLC1_near_collision"] is False
    assert status["gates"]["LPLC2_radial_opponency"] is False
    assert status["gates"]["LC4_angular_speed"] is False
    assert status["gates"]["EPG_PEN_PEG_heading"] is True
    assert status["gates"]["PFL3_DNa_transparent_mapping"] is True
    assert status["contributions"]["upper_planner"]["status"] == "paused"
    assert status["contributions"]["fly_local_core"][
        "active_v7_in_default_runtime"
    ] is False
    boundaries = status["evidence_boundaries"]
    assert boundaries["nine_source_contract_complete"] is False
    assert boundaries["Mi4_C3_direct_numeric_voltage_candidates"] == [
        "Groschner_2022"
    ]
    assert boundaries["Mi4_C3_independent_numeric_voltage_candidate_count"] == 0
    assert boundaries["Ketkar_2019_official_source_data_attachment_count"] == 11
    assert boundaries["Ketkar_2019_attachments_are_mean_SEM_tables"] is True
    assert boundaries["Ketkar_2019_Mi1_Tm3_GCaMP_summary_found"] is True
    assert boundaries["Ketkar_2019_Mi4_C3_payload_found"] is False
    assert boundaries["Ketkar_2019_individual_source_dynamics_found"] is False
    assert boundaries["Ketkar_2019_membrane_voltage_found"] is False
    assert boundaries["T5_voltage_field_counts"] == {
        "aggregated_full_field_OFF_flash": 7,
        "raw_white_noise": 8,
        "raw_drifting_grating": 9,
    }
    assert boundaries["T5_voltage_derived_kernel_sources"] == [
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
    ]
    assert boundaries["T5_voltage_derived_kernel_record_count"] == 25
    assert boundaries["T5_kernel_sample_interval_seconds"] == 0.01
    assert boundaries["T5_raw_kernel_gain_transferable"] is False
    assert boundaries["T5_kernel_stimulus_invariant"] is False
    assert boundaries["T5_source_kernel_transfer_authorized"] is False
    assert boundaries["T5_state_unit_record_count"] == 41
    assert boundaries["T5_exact_volts_to_millivolts_scale_verified"] is True
    assert (
        boundaries["T5_millivolts_or_filter_output_to_state_mapping_available"]
        is False
    )
    assert (
        boundaries["T5_measured_kernel_cross_substep_direction_scoring_authorized"]
        is False
    )
    assert boundaries["T5_measured_kernel_causal_index_zero_supported"] is True
    assert boundaries["T5_measured_kernel_all_population_peaks_covered"] is True
    assert boundaries["T5_measured_kernel_full_L1_support_covered"] is False
    assert (
        boundaries["T5_measured_kernel_prefix_alone_full_support_negative_authorized"]
        is False
    )
    assert boundaries["T5_measured_kernel_trace_samples"] == 27
    assert boundaries["T5_measured_kernel_minimum_prefix_L1_mass_fraction"] == (
        0.4082240394891599
    )
    assert boundaries["T5_measured_kernel_maximum_prefix_L1_mass_fraction"] == (
        0.47168529082482163
    )
    assert boundaries["T5_measured_kernel_zero_tail_full_support_evaluated"] is True
    assert (
        boundaries["T5_measured_kernel_zero_tail_cross_substep_identifiability_passed"]
        is False
    )
    assert boundaries["T5_measured_kernel_zero_tail_all_candidates_failed"] is True
    assert boundaries["T5_measured_kernel_zero_tail_direction_scoring_authorized"] is False
    assert boundaries["T5_measured_kernel_zero_tail_direction_scoring_performed"] is False
    assert boundaries["T5_measured_kernel_zero_tail_samples"] == 498
    assert boundaries["T5_measured_kernel_full_support_output_samples"] == 525
    assert boundaries["T5_population_kernel_recording_id_robustness_passed"] is False
    assert boundaries["T5_Tm2_population_kernel_recording_id_robustness_passed"] is False
    assert boundaries["T5_population_kernel_independent_validation_available"] is False
    assert boundaries["T5_population_kernel_transfer_authorized"] is False
    assert boundaries["T5_population_kernel_robustness_passing_source_count"] == 3
    assert boundaries["T5_population_kernel_robustness_failing_sources"] == ["Tm2"]
    assert boundaries["T5_Tm2_recording_id_vs_rest_median_correlation"] == (
        0.785694273562359
    )
    assert boundaries["T5_Tm2_partition_correlation_p05"] == 0.76950266256901
    assert boundaries["T5_population_kernel_author_default_baseline_verified"] is True
    assert boundaries["T5_population_kernel_no_baseline_matches_author_default"] is True
    assert (
        boundaries["T5_population_kernel_author_row_weighting_exactly_reproduced"]
        is False
    )
    assert boundaries["T5_population_kernel_tail_baseline_variant_authorized"] is False
    assert boundaries["T5_population_kernel_author_call_count"] == 9
    assert boundaries["T5_population_kernel_explicit_baseline_call_count"] == 0
    assert boundaries["T5_Tm1_row_vs_recording_id_weighted_correlation"] == (
        0.9995598341098436
    )
    assert boundaries["T5_author_row_weighted_changed_sources"] == ["Tm1"]
    assert boundaries["T5_author_row_weighted_full_support_all_candidates_failed"] is True
    assert boundaries["T5_author_row_weighted_cross_substep_identifiability_passed"] is False
    assert boundaries["T5_author_row_weighted_direction_scoring_authorized"] is False
    assert boundaries["T5_author_row_weighted_direction_scoring_performed"] is False
    assert boundaries["T5_Tm2_LOO_full_support_all_evaluations_failed"] is True
    assert boundaries["T5_Tm2_LOO_robust_temporal_identifiability_passed"] is False
    assert boundaries["T5_Tm2_LOO_direction_scoring_authorized"] is False
    assert boundaries["T5_Tm2_LOO_direction_scoring_performed"] is False
    assert boundaries["T5_Tm2_LOO_independent_biological_validation_performed"] is False
    assert boundaries["T5_Tm2_LOO_fold_count"] == 5
    assert boundaries["T5_Tm2_LOO_evaluation_count"] == 30
    assert boundaries["T5_Tm2_LOO_passed_evaluation_count"] == 0
    assert boundaries["T5_Tm2_LOO_candidate_ratio_ranges"][
        "temporal_difference_filtered_Tm_pair_reichardt"
    ]["static"] == [0.5921415935025431, 0.6741595238667628]
    assert boundaries["T5_measured_kernel_typed_recurrent_cascade_verified"] is True
    assert boundaries["T5_measured_kernel_replaces_existing_source_dynamics"] is False
    assert (
        boundaries["T5_measured_kernel_single_stage_biological_interpretation_authorized"]
        is False
    )
    assert boundaries["T5_measured_kernel_external_state_mapping_available"] is False
    assert boundaries["T5_measured_kernel_source_leaks"] == {
        "Tm1": 0.28,
        "Tm2": 0.55,
        "Tm4": 0.34,
        "Tm9": 0.16,
    }
    assert boundaries["T5_measured_kernel_source_node_counts"] == {
        "Tm1": 1777,
        "Tm2": 1766,
        "Tm4": 1670,
        "Tm9": 1771,
    }
    assert boundaries["T5_measured_kernel_source_inputs_partitioned_exactly_once"] is True
    assert boundaries["T5_measured_kernel_every_source_has_recurrent_or_feedback_input"] is True
    assert boundaries["T5_measured_kernel_feedforward_only_source_drive_available"] is False
    assert boundaries["T5_measured_kernel_source_dynamics_replacement_evaluated"] is False
    assert boundaries["T5_measured_kernel_source_dynamics_replacement_authorized"] is False
    assert boundaries["T5_measured_kernel_recurrent_or_feedback_fraction_by_source"] == {
        "Tm1": 0.06242381162002561,
        "Tm2": 0.04116725419043349,
        "Tm4": 0.19606226563946827,
        "Tm9": 0.23058754793814995,
    }
    assert boundaries["T5_measured_kernel_Tm9_CT1_input_fraction"] == (
        0.09073247310668342
    )
    assert boundaries["T5_lamina_only_replacement_evaluated"] is True
    assert boundaries["T5_lamina_only_replacement_all_candidates_failed"] is True
    assert boundaries["T5_lamina_only_replacement_temporal_identifiability_passed"] is False
    assert boundaries["T5_lamina_only_replacement_direction_scoring_authorized"] is False
    assert boundaries["T5_lamina_only_replacement_direction_scoring_performed"] is False
    assert boundaries["T5_lamina_only_replacement_physical_transfer_authorized"] is False
    assert boundaries["T5_lamina_only_source_coverage"]["Tm1"][
        "source_node_without_lamina_input_count"
    ] == 3
    assert boundaries["T5_lamina_only_candidate_ratios_by_update"][
        "temporal_difference_filtered_Tm_pair_reichardt"
    ]["4"]["shuffle"] == 1.6277322953083266
    assert boundaries["T5_temporal_shuffle_frame_multiset_preserved"] is True
    assert boundaries["T5_temporal_shuffle_retinal_energy_preserved"] is False
    assert boundaries["T5_temporal_shuffle_lamina_source_energy_preserved"] is False
    assert boundaries["T5_temporal_shuffle_energy_matched_control_verified"] is False
    assert (
        boundaries[
            "T5_equal_energy_temporal_selectivity_interpretation_authorized"
        ]
        is False
    )
    assert boundaries["T5_new_energy_normalized_gate_authorized"] is False
    assert boundaries["T5_temporal_shuffle_input_energy_ratios_by_update"][
        "1"
    ]["R1_R6_signed_frame_difference"] == 5.837713478478663
    assert boundaries["T5_static_sham_input_energy_ratios_by_update"]["4"][
        "lamina_only_Tm_preactivation"
    ]["Tm9"] == 0.4363820143500369
    assert boundaries["T5_increment_order_control_images_valid"] is True
    assert boundaries["T5_increment_order_control_terminal_frame_preserved"] is True
    assert boundaries["T5_increment_order_control_R1_R6_drive_multiset_preserved"] is True
    assert boundaries["T5_increment_order_control_R1_R6_energy_matched"] is True
    assert boundaries["T5_increment_order_control_independent_condition_evaluated"] is False
    assert boundaries["T5_increment_order_control_replacement_authorized"] is False
    assert boundaries["T5_increment_order_control_R1_R6_energy_ratio_summary"][
        "median"
    ] == 1.0
    assert boundaries["T5_increment_order_control_output_ratios_by_update"]["4"][
        "temporal_difference_filtered_Tm_pair_reichardt"
    ] == 1.3045208104632329
    assert boundaries["T5_increment_order_replication_input_validity_passed"] is True
    assert boundaries["T5_increment_order_replication_same_candidate_passed"] is False
    assert boundaries["T5_increment_order_replication_gate_passed"] is False
    assert (
        boundaries["T5_increment_order_replication_direction_scoring_authorized"]
        is False
    )
    assert boundaries["T5_increment_order_replication_candidate_passes"] == {
        "summed_filtered_source_centroid_projection": False,
        "fast_pool_vs_Tm9_centroid_difference": False,
        "temporal_difference_filtered_Tm_pair_reichardt": False,
    }
    assert boundaries["T5_increment_order_replication_output_ratios"]["S1-T03"][
        "4"
    ]["temporal_difference_filtered_Tm_pair_reichardt"] == 1.2259874086367528
    assert boundaries["T5_candidate_normalization_formula"] is None
    assert boundaries["T5_candidate_clipping_rule"] is None
    assert boundaries["T5_saline_fast_pooled_median_latency_ms"] == 50.0
    assert boundaries["T5_saline_Tm9_median_latency_ms"] == 80.0
    assert boundaries["T5_OA_fast_pooled_median_latency_ms"] == 40.0
    assert boundaries["T5_OA_Tm9_median_latency_ms"] == 50.0
    assert boundaries["T5_saline_Tm9_relative_delay_supported"] is True
    assert boundaries["T5_Tm9_delay_ordering_state_invariant"] is False
    assert boundaries["T5_source_delay_transfer_authorized"] is False
    assert boundaries["T5_saline_record_median_preferred_frequency_hz"] == {
        "Tm1": 0.95,
        "Tm2": 0.95,
        "Tm4": 0.675,
        "Tm9": 0.475,
    }
    assert boundaries["T5_OA_record_median_preferred_frequency_hz"] == {
        "Tm1": 2.325,
        "Tm2": 2.8,
        "Tm4": 1.45,
        "Tm9": 0.25,
    }
    assert boundaries["T5_preferred_frequency_summary_invariant"] is False
    assert boundaries["T5_source_frequency_transfer_authorized"] is False
    assert boundaries["T5_Tm_to_T5_model_inputs_git_blob_verified"] is True
    assert boundaries["T5_Tm_to_T5_Figure5_training_fit_count"] == 168
    assert boundaries["T5_Tm_to_T5_fit_score_samples_disjoint"] is False
    assert boundaries["T5_Tm_to_T5_independent_validation_available"] is False
    assert boundaries["T5_Figure6_Tm2_ND_source_reference_correct"] is False
    assert boundaries["T5_Tm_to_T5_model_transfer_authorized"] is False
    assert boundaries["T5_FIB19_MaleCNS_population_source_rank_matches"] is True
    assert boundaries["T5_FIB19_MaleCNS_maximum_population_ratio_difference"] > 0.12
    assert abs(
        boundaries["T5_MaleCNS_Tm9_fraction_outside_FIB19_range"]
        - 0.7307234295921405
    ) < 1e-15
    assert boundaries["T5_FIB19_to_MaleCNS_weight_transfer_authorized"] is False
    assert boundaries["T5_moving_bar_condition_count"] == 24
    assert boundaries["T5_moving_bar_gain_free_DSI_available"] is True
    assert boundaries["T5_moving_bar_gain_fit_score_samples_disjoint"] is False
    assert boundaries["T5_moving_bar_validation_available"] is False
    assert boundaries["T5_moving_bar_transfer_authorized"] is False
    assert boundaries["T5_four_Tm_exact_type_average_mapping_complete"] is True
    assert boundaries["T5_all_five_source_mapping_complete"] is False
    assert boundaries["T5_CT1_mapping_Lo1_column_count_by_body"] == {
        "10009": 868,
        "10157": 802,
    }
    assert boundaries["Braun_calcium_fly_counts"] == {
        "Tm2": 9,
        "Tm9": 11,
        "CT1": 9,
    }
    assert boundaries["Braun_calcium_condition_grids_complete"] is True
    assert boundaries["Braun_calcium_allowed_voltage_sources"] == []
    assert boundaries["Gou_Dryad_archive_hash_locally_verified"] is True
    assert boundaries["Gou_Dryad_local_processed_calcium_sources"] == [
        "Mi1",
        "Tm3",
        "Tm1",
        "Tm2",
    ]
    assert boundaries["Gou_Dryad_flash_fly_axis_sizes"] == {
        "Mi1": 16,
        "Tm3": 11,
        "Tm1": 9,
        "Tm2": 8,
    }
    assert boundaries["Gou_Dryad_moving_bar_fly_axis_sizes"] == {
        "Mi1": 10,
        "Tm3": 6,
        "Tm1": 8,
        "Tm2": 12,
    }
    assert boundaries["Gou_Dryad_Mi1_Tm3_direction_axis_identifiable"] is False
    assert boundaries["Gou_Dryad_direction_invariance_evaluated"] is False
    assert boundaries["Gou_DANDI_all_assets_stimulus_metadata_indexed"] is True
    assert boundaries["Gou_DANDI_Mi1_Tm3_asset_counts"] == {"Mi1": 144, "Tm3": 20}
    assert boundaries["Gou_DANDI_Mi1_Tm3_stimulus_metadata_available"] is False
    assert boundaries["T5_Figure4_relative_PD_ND_mapping_verified"] is True
    assert boundaries["T5_Figure4_native_coordinate_motion_mapping_verified"] is True
    assert boundaries["T5_Figure4_absolute_physical_direction_mapping_verified"] is False
    assert boundaries["T5_Kohn_Portes_record_level_direction_code_available"] is False
    assert (
        boundaries["T5_Kohn_Portes_record_level_physical_direction_available"]
        is False
    )
    assert boundaries["T5_cross_dataset_direction_mapping_authorized"] is False
    assert boundaries["Gou_Dryad_stable_biological_individual_IDs_verified"] is False
    assert boundaries["Gou_Dryad_experimental_membrane_voltage"] is False
    assert boundaries["v7_vertical_camera_ray_angles_declared"] is False
    assert boundaries["v7_vertical_FOV_declared"] is False
    assert boundaries["v7_vertical_pixel_to_angle_formula_declared"] is False
    assert boundaries["v7_vertical_motion_has_physical_angular_units"] is False
    assert boundaries["v7_looming_radius_has_physical_angular_units"] is False
    assert boundaries["v7_retinal_v_has_physical_angular_units"] is False
    assert boundaries["v7_offline_2D_engineering_angular_grid_complete"] is True
    assert boundaries["v7_offline_engineering_vertical_FOV_degrees"] == 70.09590046813251
    assert boundaries["v7_offline_engineering_angular_pixel_pitch_degrees"] == 3.047647846440544
    assert boundaries["v7_offline_engineering_grid_biologically_calibrated"] is False
    assert (
        boundaries["v7_controlled_stimulus_and_R1_R6_input_boundary_complete"]
        is True
    )
    assert boundaries["v7_controlled_base_stimulus_count"] == 20
    assert boundaries["v7_controlled_stimuli_per_independent_split"] == 172
    assert boundaries["v7_typed_LPLC_LC4_stimulus_count"] == 33
    assert boundaries["v7_external_drive_non_R1_R6_node_count"] == 0
    assert boundaries["v7_target_direct_external_drive_overlap"] == 0
    assert boundaries["Gou_DANDI_asset_level_stable_participant_IDs_verified"] is True
    assert boundaries["Gou_DANDI_unique_subject_ID_count"] == 282
    assert boundaries["Gou_Dryad_distinct_fliesUsed_label_count"] == 66
    assert boundaries["Gou_Dryad_DANDI_identity_crosswalk_match_count"] == 0
    assert boundaries["Gou_Dryad_rows_to_DANDI_subject_crosswalk_verified"] is False
    assert boundaries["v7_offline_time_coordinate_contract_complete"] is True
    assert boundaries["T5_transfer_synthesis_kernel_shape_source_count"] == 4
    assert boundaries["T5_transfer_synthesis_mapping_source_count"] == 4
    assert boundaries["T5_transfer_synthesis_positive_shape_evidence_count"] == 4
    assert boundaries["T5_transfer_synthesis_absolute_gain_available"] is False
    assert boundaries["T5_transfer_synthesis_official_target_model_available"] is True
    assert boundaries["T5_transfer_synthesis_target_to_source_mapping_available"] is False
    assert boundaries["T5_transfer_synthesis_Figure6_axolotl_source_available"] is False
    assert boundaries["T5_transfer_synthesis_related_axolotl_project_identified"] is True
    assert boundaries["T5_transfer_synthesis_axolotl_repository_readable"] is False
    assert boundaries["T5_measured_kernel_temporal_identifiability_passed"] is False
    assert boundaries["T5_measured_kernel_direction_scoring_performed"] is False
    assert boundaries["T5_measured_kernel_frame_alignment_verified"] is True
    assert boundaries["T5_measured_kernel_probe_solver_alignment_verified"] is False
    assert boundaries["T5_measured_kernel_physical_transfer_authorized"] is False
    assert boundaries["T5_measured_kernel_standard_substep_negative_reproduced"] is True
    assert boundaries["T5_measured_kernel_cross_substep_identifiability_passed"] is False
    assert (
        boundaries["T5_measured_kernel_cross_substep_direction_scoring_performed"]
        is False
    )
    assert boundaries["T5_transfer_synthesis_CT1_complete"] is False
    assert boundaries["T5_transfer_synthesis_ready"] is False
    assert boundaries["v7_offline_horizontal_coordinate_contract_complete"] is True
    assert boundaries["v7_offline_two_dimensional_angular_calibration_complete"] is False
    assert boundaries["v7_offline_frame_interval_milliseconds"] == 10.0
    assert boundaries["v7_offline_substep_interval_milliseconds"] == 2.5
    assert boundaries["T4_source_mapping_mode"] == "exact_type_average"
    assert boundaries["C2C3_version_of_record_DOI"] == "10.7554/eLife.108529.3"
    assert boundaries["C2C3_version_of_record_repository_revision"] == (
        "745c6114b72a61dabc9389a1f024dbe22f679edf"
    )
    assert boundaries["C2C3_version_of_record_new_payload_modality"] == (
        "calcium_fluorescence_STRF"
    )
    assert boundaries["C2C3_version_of_record_C3_fly_count"] == 8
    assert boundaries["C2C3_version_of_record_Mi1_control_fly_count"] == 7
    assert boundaries["C2C3_version_of_record_new_C3_Mi4_voltage_found"] is False
    assert boundaries["C2C3_version_of_record_new_Mi4_payload_found"] is False
    assert boundaries["C2C3_version_of_record_MaleCNS_crosswalk_found"] is False
    assert boundaries["C2C3_version_of_record_changed_T4_transfer_gate"] is False
    assert boundaries["T4_source_pool_camera_frame_bilateral_readouts"] == [
        "pairwise_distal_vector"
    ]
    assert boundaries["T4_source_pool_camera_frame_bilateral_subtypes"] == {
        "pairwise_distal_vector": ["d"]
    }
    assert boundaries["T4_source_pool_camera_frame_post_hoc_candidate_discovered"] is True
    assert boundaries["T4_source_pool_camera_frame_replication_preregistered"] is False
    assert boundaries["T4_source_pool_camera_frame_replication_evaluated"] is True
    assert boundaries["T4_source_pool_camera_frame_target_formula_authorized"] is False
    assert boundaries["T4_source_pool_camera_frame_ordered_replication_passed"] is False
    assert boundaries["T4_source_pool_camera_frame_controls_evaluated"] is False
    assert boundaries["T4_source_pool_camera_frame_replication_gate_passed"] is False
    assert boundaries["T4_source_pool_camera_frame_replication_results"][
        "LPLC-T02"
    ]["population_scores"]["T4d_L"]["median_signed_contrast"] == (
        -0.8703534267888534
    )
    assert boundaries["T4_synapse_RF_axis_joint_valid_target_count"] == 6749
    assert boundaries["T4_synapse_RF_axis_identity_median_angle_degrees"] == (
        68.18171979285492
    )
    assert boundaries["T4_synapse_RF_axis_identity_cardinal_match_fraction"] == (
        0.4951844717735961
    )
    assert boundaries["T4_synapse_RF_axis_descriptive_best_transform"] == (
        "reflect_horizontal"
    )
    assert boundaries["T4_synapse_RF_axis_best_median_angle_degrees"] == (
        21.112026999265463
    )
    assert boundaries["T4_source_RF_axis_interchangeability_verified"] is False
    assert boundaries["T4_RF_axis_replacement_authorized"] is False
    assert boundaries["T4_source_recording_level_body_assignment"] is False
    assert boundaries["T4_exact_type_average_mapping_complete"] is True
    assert boundaries["T4_author_minmax_formula_reproduced"] is True
    assert boundaries["T4_state_mapping_held_out_outside_fraction_by_source"] == {
        "Mi1": 0.1295625,
        "Tm3": 0.17478125,
        "Mi4": 0.2870446428571429,
        "C3": 0.18847916666666667,
    }
    assert boundaries["T4_author_minmax_semantics_match_v7_state"] is False
    assert boundaries["T4_millivolts_to_v7_state_mapping_available"] is False
    assert boundaries["T5_record_specific_stimulus_logs_available"] is False
    assert boundaries["Motyxia2_public_history_branch_count"] == 22
    assert boundaries["Motyxia2_public_history_commit_count"] == 447
    assert boundaries["T5_record_log_found_in_Motyxia2_public_history"] is False
    assert boundaries["T5_external_successful_indexes_linked_log_found"] is False
    assert boundaries["T5_PMC_supplement_content_inspected"] is False
    assert boundaries["T5_publisher_supplements_inspected"] is True
    assert boundaries["T5_publisher_supplements_contain_record_log"] is False
    assert boundaries["T5_Figshare_search_accessible"] is False
    assert boundaries["T5_stimulus_log_global_absence_claimed"] is False
    assert boundaries["T5_generator_defaults_used_as_record_fields"] is False
    assert boundaries["T5_stimulus_provenance_complete"] is False
    assert boundaries["CT1_audited_candidate_count"] == 11
    assert boundaries["CT1_incremental_2025_2026_candidate_count"] == 3
    assert boundaries["CT1_direct_experimental_voltage_candidate_found"] is False
    assert boundaries["Tm9_official_synapse_coordinate"] == [15, 2]
    assert boundaries["CT1_per_synapse_Lo1_columnar_retinotopy_available"] is True
    assert boundaries["CT1_complete_official_LO_column_coverage"] is False


def test_autonomy_status_alias_matches_hash_verified_v7_status() -> None:
    client = TestClient(app)
    v7 = client.get("/api/v7/status")
    autonomy = client.get("/api/autonomy/status")
    assert v7.status_code == 200
    assert autonomy.status_code == 200
    assert autonomy.json() == v7.json()


def test_v7_status_rejects_stale_evidence(tmp_path, monkeypatch) -> None:
    import hashlib
    import json

    monkeypatch.setattr(api_module, "ROOT", tmp_path)
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    report = {
        "protocol": {
            "dependencies_sha256": {
                "evidence.json": hashlib.sha256(evidence.read_bytes()).hexdigest()
            }
        },
        "checks": [],
        "summary": {},
    }
    target = tmp_path / "artifacts/v7-goal-audit.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(report))
    evidence.write_text("stale")
    response = TestClient(app).get("/api/v7/status")
    assert response.status_code == 503
    assert response.json()["detail"] == "Stale v7 audit dependency: evidence.json"
    alias = TestClient(app).get("/api/autonomy/status")
    assert alias.status_code == 503
    assert alias.json() == response.json()


def test_status_routes_report_invalid_audit_as_service_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "ROOT", tmp_path)
    target = tmp_path / "artifacts/v7-goal-audit.json"
    target.parent.mkdir(parents=True)
    target.write_text("{not-json", encoding="utf-8")
    client = TestClient(app)
    for route in ("/api/v7/status", "/api/autonomy/status"):
        response = client.get(route)
        assert response.status_code == 503
        assert response.json()["detail"] == "Invalid v7 goal audit"


def test_real_cached_skeleton_endpoint() -> None:
    response = TestClient(app).get("/api/skeleton/10001?max_edges=20000")
    assert response.status_code == 200
    assert response.json()["source_vertices"] == 2975


def test_real_connectome_assets() -> None:
    client = TestClient(app)
    overview = client.get("/api/connectome/overview").json()
    pathways = client.get("/api/connectome/pathways").json()
    assert overview["canonical_nodes"] == 166_700
    assert pathways["all_edges_accounted"] == 25_582_938


def test_driving_endpoints_are_stateful(monkeypatch) -> None:
    calls = []

    class FakeEngine:
        def state(self, **_):
            return {"environment": {"step": len(calls)}}

        def reset(self, seed, *, keep_learning, scenario, control_mode):
            calls.append(("reset", seed, keep_learning, scenario, control_mode))
            return self.state()

        def set_control_mode(self, control_mode):
            calls.append(("control_mode", control_mode))

        def step(self, *, learning, explore, safety_constraints=True, include_activity=True):
            calls.append(("step", learning, explore, safety_constraints))
            return {"environment": {"step": len(calls), "done": False}}

    monkeypatch.setattr(api_module, "engine", lambda: FakeEngine())
    client = TestClient(app)
    reset = client.post(
        "/api/driving/reset", json={"seed": 9, "keep_learning": False}
    )
    assert reset.status_code == 200
    response = client.post("/api/driving/step", json={"steps": 2, "learning": True})
    assert response.json()["environment"]["step"] == 3
    assert calls == [
        ("reset", 9, False, "highway", "assisted"),
        ("step", True, False, True),
        ("step", True, False, True),
    ]


def test_driving_step_defaults_to_frozen_execution(monkeypatch) -> None:
    calls = []

    class FakeEngine:
        def step(self, *, learning, explore, safety_constraints=True, include_activity=True):
            calls.append((learning, explore, safety_constraints))
            return {"environment": {"done": True}}

    monkeypatch.setattr(api_module, "engine", lambda: FakeEngine())
    response = TestClient(app).post("/api/driving/step", json={})
    assert response.status_code == 200
    assert calls == [(False, False, True)]


def test_engine_is_singleton_under_concurrent_first_load(monkeypatch) -> None:
    import threading
    import time
    created = []

    class FakeDrivingEngine:
        def __init__(self, *_args, **_kwargs):
            time.sleep(0.02)
            created.append(self)

    monkeypatch.setattr(api_module, "DrivingEngine", FakeDrivingEngine)
    monkeypatch.setattr(api_module, "_ENGINE", None)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(api_module.engine())) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(created) == 1 and len({id(value) for value in results}) == 1
    monkeypatch.setattr(api_module, "_ENGINE", None)
