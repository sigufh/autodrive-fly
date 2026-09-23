import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-dynamics-transfer-audit.json"


def test_T4_source_dynamics_transfer_audit_is_read_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["required_sources"] == ["Mi1", "Tm3", "Mi4", "C3"]
    assert report["protocol"]["read_only"] is True
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_T4_source_recordings_do_not_supply_label_blind_transfer_contract() -> None:
    report = json.loads(REPORT.read_text())
    assert report["available_source_recordings"]["axes"] == [
        "stimulus_on_off",
        "cell",
        "time_milliseconds",
    ]
    assert all(
        item["stable_MaleCNS_body_ids_available"] is False
        for item in report["available_source_recordings"]["sources"].values()
    )
    synthesis = report["paper_direction_synthesis"]
    assert synthesis["shift_samples"] == 160
    assert synthesis["target_PD_ND_conditioned"] is True
    assert synthesis["may_be_used_as_label_blind_v7_source_kernel"] is False
    assert report["required_transfer_fields_available"] == {
        "direction_independent_source_kernel": False,
        "source_to_MaleCNS_identity_mapping": True,
        "physical_v7_sample_interval": True,
        "millivolts_to_v7_normalized_state_mapping": False,
        "ordered_source_sequence_identifiability": False,
    }
    assert report["transfer_gates"]["crossfit_structure_axis_passed"] is True
    assert report["transfer_gates"]["physical_timebase_available"] is True
    mapping = report["source_to_MaleCNS_mapping"]
    assert mapping["mapping_mode"] == "exact_type_average"
    assert mapping["recording_level_body_assignment"] is False
    assert mapping["all_required_T4_sources_complete"] is True
    assert set(mapping["source_mapping_complete"].values()) == {True}
    assert not any(
        value
        for name, value in report["transfer_gates"].items()
        if name not in {"crossfit_structure_axis_passed", "physical_timebase_available"}
    )
    package = report["official_unified_model_package"]
    assert package["doi"] == "10.25378/janelia.16663486"
    assert package["article_id"] == 16663486
    assert package["size_bytes"] == 3_630_019
    assert package["description_names_optTables"] is True
    assert package["prior_live_audit_file_manifest_retrieved"] is False
    assert package["file_manifest_retrieved"] is True
    assert package["recovered_manifest_consistent"] is True
    assert package["recovered_file_manifest"] == {
        "recovery_source": "internet_archive_official_page_snapshot",
        "snapshot_timestamp": "20241127113612",
        "file_id": 31067761,
        "file_name": "modelFigure.zip",
        "size_bytes": 3_630_019,
        "md5": "14ba0fa761a513d55cacc41610881e80",
        "mime_type": "application/zip",
        "official_download_url": (
            "https://janelia.figshare.com/ndownloader/files/31067761"
        ),
    }
    assert package["payload_retrieved"] is True
    assert package["prior_live_audit_files_verified"] is False
    assert package["files_verified"] is True
    semantics = report["verified_unified_model_semantics"]
    assert semantics["T4_target_model_parameters_available"] is True
    assert semantics["target_components"] == ["E", "I", "E2", "I2"]
    assert semantics["source_type_mapping_available"] is False
    assert semantics["cardinal_diagonal_to_subtype_mapping_available"] is False
    assert semantics["independent_cell_holdout_available"] is False
    assert semantics["MaleCNS_source_kernel_transfer_authorized"] is False
    fig3 = report["verified_fig3_source_kernel_readiness"]
    assert fig3["source_workbook_verified"] is True
    assert fig3["ON_source_specific_kernels_ready"] is False
    assert fig3["prior_two_pool_kernel_authorized"] is False
    assert fig3["source_specific_kernel_candidate_authorized"] is False
    assert fig3["cross_cell_robustness_passed"] is False
    arenz = report["verified_Arenz_source_filter_readiness"]
    assert arenz["filter_parameters_verified"] is True
    assert arenz["current_source_coverage_fraction"] == 0.75
    assert arenz["missing_current_sources"] == ["C3"]
    assert arenz["physical_time_transfer_authorized"] is False
    assert arenz["source_filter_candidate_authorized"] is False
    c3 = report["verified_C3_STRF_readiness"]
    assert c3["numerical_data_verified"] is True
    assert c3["direct_temporal_measurement_available"] is True
    assert c3["all_required_sources_have_some_direct_temporal_evidence"] is True
    assert c3["all_required_sources_share_one_transferable_parameterization"] is False
    assert c3["membrane_voltage_or_validated_deconvolved_kernel_available"] is False
    assert c3["source_filter_candidate_authorized"] is False
    version = report["C2C3_version_of_record_update"]
    assert version["doi"] == "10.7554/eLife.108529.3"
    assert version["published_on"] == "2026-05-26"
    assert version["repository_revision"] == (
        "745c6114b72a61dabc9389a1f024dbe22f679edf"
    )
    assert version["repository_release_count"] == 0
    assert version["repository_tag_count"] == 0
    assert version["new_payload_source_types"] == ["C3", "Mi1"]
    assert version["measurement_modality"] == "calcium_fluorescence_STRF"
    assert version["C3_unique_Flyname_count"] == 8
    assert version["Mi1_control_unique_Flyname_count"] == 7
    assert version["new_C3_or_Mi4_membrane_voltage_payload_found"] is False
    assert version["new_Mi4_numerical_payload_found"] is False
    assert version["new_recording_to_MaleCNS_body_crosswalk_found"] is False
    assert version["source_dynamics_transfer_gate_changed"] is False
    flash = report["verified_C3_STRF_to_independent_flash_transfer"]
    assert flash["source_fly_axis_unit_count"] == 7
    assert flash["external_flash_fly_count"] == 22
    assert flash["mean_waveform_correlation"] > 0.97
    assert flash["minimum_external_fly_correlation"] > 0.83
    assert flash["bootstrap_correlation_p05"] < 0.12
    assert flash["transfer_passed"] is False
    assert flash["source_kernel_candidate_authorized"] is False
    directional = report["verified_C3_directional_edge_boundary"]
    assert directional["response_unit"] == "deltaF_over_F"
    assert directional["fly_count_in_public_payload"] == 6
    assert directional["fly_count_in_version_of_record_caption"] == 8
    assert directional["ROI_count"] == 77
    assert directional["direction_degrees"] == [90, 45, 0, 315, 270, 225, 180, 135]
    assert directional["edge_velocity_degrees_per_second"] == 20.0
    assert directional["sample_interval_seconds"] == 0.1
    assert directional["paper_reports_direction_preference"] is False
    assert directional["every_directional_fly_in_flash_cohort"] is True
    assert directional["every_directional_fly_in_STRF_cohort"] is True
    assert directional["independent_cohort"] is False
    assert directional["experimental_membrane_voltage"] is False
    assert directional["direction_specific_source_kernel_verified"] is False
    assert directional["source_dynamics_transfer_authorized"] is False
    fig1 = report["verified_Fig1_source_temporal_readiness"]
    assert fig1["workbooks_verified"] is True
    assert fig1["source_average_tables_have_time_axis"] is False
    assert fig1["individual_source_tables_have_time_axis"] is False
    assert fig1["object_payload_hash_verified"] is True
    assert fig1["object_payload_safely_inspected"] is True
    assert fig1["source_temporal_kernel_transfer_authorized"] is False
    c3_filter = report["verified_C3_analytic_filter_precheck"]
    assert c3_filter["band_pass_BIC_below_low_pass_BIC"] is True
    assert c3_filter["minimum_leave_one_fly_out_correlation"] < 0.67
    assert c3_filter["median_leave_one_fly_out_correlation"] > 0.84
    assert c3_filter["filter_family_precheck_passed"] is False
    assert c3_filter["source_filter_transfer_authorized"] is False
    timing = report["verified_TimingModels_source_filter_readiness"]
    assert timing["repository_commit"] == "100bb2f52cb9628477c3883e4b17774b0b244e67"
    assert timing["covered_source_filter_stability_verified"] is True
    assert timing["current_source_coverage_fraction"] == 0.75
    assert timing["missing_current_sources"] == ["C3"]
    assert timing["complete_source_filter_candidate_authorized"] is False
    independent_mi4 = report[
        "verified_Gonzalez_Suarez_independent_Mi4_readiness"
    ]
    assert independent_mi4["doi"] == "10.1016/j.cub.2022.06.075"
    assert independent_mi4["Mi4_GCaMP6f_fly_count"] == 15
    assert independent_mi4["Mi4_type_average_filter_available"] is True
    assert independent_mi4["filter_sample_interval_seconds"] == 1 / 30
    assert independent_mi4["individual_cell_axis_available"] is False
    assert independent_mi4["Mi4_experimental_membrane_voltage_available"] is False
    assert independent_mi4["C3_source_dynamics_available"] is False
    assert independent_mi4["Mi4_C3_voltage_transfer_authorized"] is False
    assert independent_mi4["bioRxiv_supplement_retrieved"] is True
    assert independent_mi4["individual_flies_are_statistical_units"] is True
    assert independent_mi4["public_individual_fly_numeric_payload_attached"] is False
    tanaka = report["verified_Tanaka_2023_independent_Mi4_calcium"]
    assert tanaka["Figure_6_named_Mi4_C3_member_count"] == 4
    assert tanaka["Figure_6_measurement_object"] == "walking_turning_angular_velocity"
    assert tanaka["Figure_6_neural_activity_recording"] is False
    assert tanaka["Figure_6_source_dynamics_payload"] is False
    assert tanaka["Figure_6_large_MAT_members_downloaded"] is False
    yuan = report["verified_Yuan_C3_intervention_boundary"]
    assert yuan["doi"] == "10.1111/jnc.15036"
    assert yuan["C3_intervention_candidate_verified"] is True
    assert yuan["directly_recorded_neural_activity_sources"] == ["L1", "L2"]
    assert yuan["C3_direct_recording_candidate_verified"] is False
    assert yuan["C3_public_numeric_source_dynamics_payload_verified"] is False
    assert yuan["C3_source_dynamics_transfer_authorized"] is False
    strother = report["verified_Strother_Mi4_public_index_boundary"]
    assert strother["doi"] == "10.1073/pnas.1703090115"
    assert strother["successful_index_numeric_Mi4_trace_found"] is False
    assert strother["PMC_numeric_data_attachment_count"] == 0
    assert strother["Figshare_search_interpretable"] is False
    assert strother["global_absence_claimed"] is False
    assert strother["author_public_repository_count"] == 2
    assert strother["author_repository_numeric_payload_verified"] is False
    assert strother["author_repository_search_is_bounded"] is True
    assert strother["neuron_image_analysis_complete_history_audited"] is True
    assert strother["supplementary_bundle_entry_count"] == 16
    assert strother["moving_grating_trace_figure_present"] is True
    assert strother["moving_grating_fly_count"] == 5
    assert strother["moving_grating_speed_degrees_per_second"] == 90.0
    assert strother["supplement_public_numeric_trace_attachment_verified"] is False
    assert strother["Mi4_source_dynamics_transfer_authorized"] is False
    graph = report["verified_C3_citation_graph_boundary"]
    assert graph["union_unique_work_count"] == 240
    assert graph["unresolved_reference_ID_count"] == 2
    assert graph["high_relevance_candidate"] == "Pang_2025"
    assert graph["directly_recorded_neuron_types"] == ["L1", "L2"]
    assert graph["Dryad_file_count"] == 75
    assert graph["Dryad_total_declared_bytes"] == 49_600_989_798
    assert graph["new_independent_C3_direct_recording_candidate_found"] is False
    legacy = report["verified_legacy_C3_candidate_boundary"]
    assert legacy["audited_candidate_count"] == 3
    assert legacy["intervention_only_candidates"] == ["Tuthill_2013"]
    assert legacy["direct_source_dynamics_candidates"] == []
    assert legacy["unresolved_fulltext_candidates"] == []
    assert legacy["Ramos_2020_fulltext_scope_resolved"] is True
    assert legacy["Ramos_2020_direct_recording_targets"] == ["Tm9"]
    assert legacy["Ramos_2020_C3_direct_recording_verified"] is False
    assert legacy["source_dynamics_transfer_authorized"] is False
    assert graph["C3_source_dynamics_transfer_authorized"] is False
    hao = report["unresolved_Hao_2026_ASAP7y_candidate"]
    assert hao["doi"] == "10.64898/2026.05.27.728040"
    assert hao["Drosophila_in_vivo_voltage_imaging"] is True
    assert hao["measurement_modality"] == "two_photon_ASAP7y_voltage_imaging"
    assert hao["paper_source_experimental_cell_types"] == []
    assert hao["public_author_figure_named_examples"] == ["Dm9", "MeLo13"]
    assert hao["complete_experimental_cell_type_set_resolved"] is False
    assert hao["candidate_classification"] == "unresolved_high_value_candidate"
    assert hao["public_numeric_payload_verified"] is False
    assert hao["Wayback_capture_view"] == "abstract_only"
    assert hao["Wayback_linked_route_contents_retrieved"] is False
    assert hao["Wayback_resolves_complete_cell_types"] is False
    assert hao["ClandininLab_public_repository_count"] == 31
    assert hao["ClandininLab_paper_repository_hits"] == []
    assert hao["bioRxiv_TDM_requester_pays_required"] is True
    assert hao["bioRxiv_TDM_payload_inventory_readable"] is False
    assert hao["Europe_PMC_annotation_count"] == 6
    assert hao["Europe_PMC_Mi4_annotation_hit"] is False
    assert hao["Europe_PMC_C3_annotation_hit"] is False
    assert hao["Europe_PMC_annotation_scope"] == (
        "abstract_only_because_in_PMC_is_false"
    )
    assert hao["author_dissertation_restricted_until"] == "2027-03-14"
    assert hao["author_dissertation_public_abstract_names_cell_types"] is False
    assert hao["successful_public_index_numeric_payload_found"] is False
    assert hao["global_payload_absence_claimed"] is False
    assert hao["source_dynamics_fit_authorized"] is False
    stable = report["verified_Gur_2024_stable_contrast_source_scope"]
    assert stable["doi"] == "10.1038/s41467-024-52724-5"
    assert {"Tm1", "Tm2", "Tm4", "Tm9", "Dm12"} <= set(
        stable["directly_recorded_neuron_types"]
    )
    assert stable["Mi4_direct_physiology_found"] is False
    assert stable["C3_direct_physiology_found"] is False
    assert stable["Mi4_proofreading_row_count"] == 723
    assert stable["C3_proofreading_row_count"] == 678
    assert stable["Mi4_C3_files_are_connectome_proofreading_only"] is True
    assert stable["external_recording_to_MaleCNS_crosswalk_found"] is False
    assert stable["source_dynamics_transfer_authorized"] is False
    tanaka = report["verified_Tanaka_2023_independent_Mi4_calcium"]
    assert tanaka["doi"] == "10.1016/j.cub.2023.10.011"
    assert tanaka["measurement_modality"] == "two_photon_jGCaMP7b_calcium"
    assert tanaka["response_unit"] == "deltaF_over_F"
    assert tanaka["fly_count"] == 10
    assert tanaka["selected_ROI_count"] == 201
    assert tanaka["individual_fly_axis_available"] is True
    assert tanaka["trial_and_ROI_axes_available"] is True
    assert tanaka["experimental_membrane_voltage"] is False
    assert tanaka["recording_to_MaleCNS_body_crosswalk_found"] is False
    assert tanaka["source_dynamics_transfer_authorized"] is False
    afterimages = report["verified_Wu_2026_afterimages_Mi4_phenotype"]
    assert afterimages["doi"] == "10.64898/2026.01.19.700413"
    assert afterimages["measurement_modality"] == "two_photon_GCaMP6f_calcium"
    assert afterimages["fly_count"] == 6
    assert afterimages["ROI_count"] == 113
    assert afterimages["public_numeric_payload_verified"] is False
    assert afterimages["experimental_membrane_voltage"] is False
    assert afterimages["source_dynamics_transfer_authorized"] is False
    flyvis = report["verified_FlyVis_C3_time_constant_readiness"]
    assert flyvis["repository_commit"] == "92b3845cc426dd309a1a0e1b3890156c42e14021"
    assert flyvis["pretrained_model_count"] == 50
    assert flyvis["solver_dt_seconds"] == 0.02
    assert 0.066 < flyvis["C3_median_time_constant_seconds"] < 0.068
    assert flyvis["C3_models_at_or_below_solver_dt"] == 21
    assert flyvis["C3_time_constant_transfer_authorized"] is False
    effective = report["verified_FlyVis_C3_effective_dynamics_readiness"]
    assert effective["pretrained_model_count"] == 50
    assert effective["fixed_denominator"] == 50
    assert effective["cross_dt_minimum_correlation"] < 0.67
    assert min(effective["leave_one_model_out_minimum_correlation_by_dt"].values()) < -0.81
    assert max(
        value
        for by_tau in effective[
            "external_ensemble_correlation_by_dt_and_calcium_assumption"
        ].values()
        for value in by_tau.values()
    ) < 0.72
    assert effective["effective_dynamics_transfer_authorized"] is False
    visual_time = report["verified_FlyVis_visual_source_time_constants"]
    assert visual_time["T4_missing_sources"] == []
    assert visual_time["T5_missing_sources"] == []
    assert visual_time["all_values_finite_and_positive"] is True
    assert visual_time["all_above_solver_dt"] is False
    assert visual_time["all_cross_model_IQR_stable"] is False
    assert visual_time["transferable"] is False
    measured = report["verified_C3_measured_filter_robustness"]
    assert measured["cross_deconvolution_stability_passed"] is True
    assert set(measured["minimum_held_out_correlation_by_assumption"]) == {
        "200ms",
        "250ms",
        "300ms",
        "350ms",
    }
    assert max(measured["minimum_held_out_correlation_by_assumption"].values()) < 0.58
    assert measured["every_deconvolution_assumption_passed"] is False
    assert measured["type_shared_kernel_authorized"] is False
    public_models = report["verified_public_T4_model_source_coverage"]
    assert public_models["Clark_required_source_coverage_fraction"] == 0.5
    assert public_models["Clark_missing_required_sources"] == ["C3", "Tm3"]
    assert public_models["ModelDB_required_source_coverage_fraction"] == 0.0
    assert public_models["C3_specific_parameters_available"] is False
    assert public_models["complete_source_dynamics_transfer_authorized"] is False
    crossfit = report["crossfit_dynamic_readiness"]
    assert crossfit["structure_axis_passed"] is True
    assert crossfit["functional_candidate_passed"] is False
    assert crossfit["maximum_ordered_source_sequence_direction_pass_count"] == 2
    assert all(not values for values in crossfit["ordered_bilateral_direction_subtypes"].values())
    assert crossfit["shuffle_signed_mean_bilateral_direction_subtypes"] == ["a"]
    assert crossfit["source_sequence_candidate_authorized"] is False
    assert crossfit["balanced_retina_ordered_direction_pass_count"] == 0
    assert crossfit["retinal_sampling_imbalance_explains_direction_failure"] is False


def test_T4_source_dynamics_transfer_stop_rule_preserves_boundaries() -> None:
    report = json.loads(REPORT.read_text())
    assert report["source_dynamics_transfer_authorized"] is False
    assert report["next_candidate_authorized"] is False
    assert report["stop_reason"] == (
        "published_source_dynamics_not_transferable_to_label_blind_v7"
    )
    assert report["boundary"]["infer_kernel_from_T4_target_labels"] is False
    assert report["boundary"]["use_160ms_shift_as_v7_lag"] is False
    assert report["boundary"]["archived_manifest_does_not_substitute_for_payload"] is True
    assert report["boundary"]["T5_and_LPLC_remain_frozen"] is True
    assert (
        report["boundary"][
            "prohibit_further_target_formula_scan_without_new_source_dynamics_evidence"
        ]
        is True
    )
