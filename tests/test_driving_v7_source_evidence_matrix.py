import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-source-evidence-matrix.json"


def test_source_evidence_matrix_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["source_order"] == [
        "Mi1",
        "Tm3",
        "Mi4",
        "C3",
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
        "CT1",
    ]
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_T4_has_numerical_voltage_but_no_complete_source() -> None:
    report = json.loads(REPORT.read_text())
    assert report["family_summary"]["T4"]["numerical_membrane_voltage_count"] == 4
    assert report["family_summary"]["T4"]["all_sources_contract_complete"] is False
    for source in report["family_summary"]["T4"]["sources"]:
        assert report["matrix"][source]["numerical_membrane_voltage"] is True
        assert (
            report["matrix"][source]["gates"]["stable_biological_individual_id_on_allowed_payload"]
            is True
        )
        assert (
            report["matrix"][source]["gates"][
                "complete_stimulus_and_baseline_fields_on_allowed_payload"
            ]
            is False
        )
        components = report["matrix"][source]["evidence_components"]
        if source == "Tm3":
            assert components[
                "MaleCNS_one_hop_same_type_blind_validation_available"
            ] is False
            assert components[
                "MaleCNS_one_hop_blind_replay_rounded_exact_fraction"
            ] is None
        else:
            assert components[
                "MaleCNS_one_hop_same_type_blind_validation_available"
            ] is True
        assert (
            report["matrix"][source]["evidence_components"][
                "fixed_T4_individual_split_passed_for_ON_and_OFF"
            ]
            is False
        )
        assert (
            report["matrix"][source]["evidence_components"][
                "fixed_T4_split_sample_interval_milliseconds"
            ]
            == 1.0
        )
        assert (
            report["matrix"][source]["evidence_components"][
                "Edmond_1khz_payload_currently_verified"
            ]
            is True
        )
        assert (
            report["matrix"][source]["evidence_components"][
                "Edmond_1khz_fixed_split_recompute_authorized"
            ]
            is True
        )
        components = report["matrix"][source]["evidence_components"]
        assert components["Edmond_complete_public_dataset_file_count"] == 74
        assert components["Edmond_Fig3_identity_or_metadata_sidecar_found"] is False
        assert components["Edmond_workbook_recording_metadata_recovered"] is False
        assert components["Edmond_Fig1_Fig3_cohort_relation_identifiable"] is False
        assert components["Edmond_Fig3_edge_baseline_window_declared"] is False
        row = report["matrix"][source]
        assert "T4_Fig3_verified_1khz_millivolt_array" in row["numerical_evidence_sources"]
    for source in ("Mi1", "C3"):
        components = report["matrix"][source]["evidence_components"]
        assert components["C2C3_version_of_record_repository_revision_verified"] is True
        assert components["C2C3_version_of_record_calcium_STRF_with_Flyname_verified"] is True
        assert components["C2C3_version_of_record_MaleCNS_crosswalk_found"] is False
    tm4_mapping = report["matrix"]["Tm4"]["evidence_components"]
    assert tm4_mapping["MaleCNS_one_hop_same_type_blind_validation_available"] is True
    assert tm4_mapping["MaleCNS_one_hop_blind_replay_rounded_exact_fraction"] < 0.32
    assert report["matrix"]["C3"]["evidence_components"][
        "C2C3_version_of_record_new_C3_or_Mi4_voltage_found"
    ] is False
    c3_components = report["matrix"]["C3"]["evidence_components"]
    assert c3_components["Henning_C3_directional_numeric_calcium_verified"] is True
    assert c3_components["Henning_C3_directional_fly_count_in_public_payload"] == 6
    assert c3_components["Henning_C3_directional_cohort_independent"] is False
    assert c3_components["Henning_C3_direction_specific_source_kernel_verified"] is False
    assert c3_components["Henning_C3_directional_experimental_membrane_voltage"] is False
    assert c3_components["Tuthill_2013_C3_intervention_only_verified"] is True
    assert c3_components["Maisak_2018_direct_C3_or_Mi4_recording_verified"] is False
    assert c3_components["Ramos_2020_complete_fulltext_scope_resolved"] is True
    assert "Henning_C3_directional_edge_GCaMP6f" in report["matrix"]["C3"][
        "numerical_evidence_sources"
    ]
    assert report["matrix"]["Mi4"]["evidence_components"][
        "C2C3_version_of_record_new_C3_or_Mi4_voltage_found"
    ] is False
    for source in ("Tm3", "Mi4"):
        assert report["matrix"][source]["evidence_components"][
            "C2C3_version_of_record_calcium_STRF_with_Flyname_verified"
        ] is False
    for source in ("Mi1", "Tm3", "Mi4", "C3"):
        components = report["matrix"][source]["evidence_components"]
        assert components["Ketkar_2019_official_attachment_count"] == 11
        assert components["Ketkar_2019_experimental_membrane_voltage_payload_found"] is False
    for source in ("Mi1", "Tm3"):
        assert report["matrix"][source]["evidence_components"][
            "Ketkar_2019_Mi1_Tm3_GCaMP_summary_evidence_found"
        ] is True
    for source in ("Mi4", "C3"):
        assert report["matrix"][source]["evidence_components"][
            "Ketkar_2019_Mi4_or_C3_attachment_payload_found"
        ] is False
    mi4_components = report["matrix"]["Mi4"]["evidence_components"]
    assert mi4_components[
        "Gonzalez_Suarez_2022_independent_Mi4_type_average_filter_available"
    ] is True
    assert mi4_components[
        "Gonzalez_Suarez_2022_individual_cell_axis_available"
    ] is False
    assert mi4_components[
        "Gonzalez_Suarez_2022_experimental_Mi4_voltage_available"
    ] is False
    assert report["matrix"]["Mi4"][
        "local_numerical_calcium_or_deconvolved_calcium"
    ] is True
    assert (
        "Gonzalez_Suarez_2022_type_average_deconvolved_GCaMP6f_filter"
        in report["matrix"]["Mi4"]["numerical_evidence_sources"]
    )
    assert (
        "Tanaka_2023_individual_jGCaMP7b_time_series"
        in report["matrix"]["Mi4"]["numerical_evidence_sources"]
    )
    assert mi4_components[
        "Tanaka_2023_independent_Mi4_numerical_calcium_dynamics_verified"
    ] is True
    assert mi4_components["Tanaka_2023_Mi4_fly_count"] == 10
    assert mi4_components["Tanaka_2023_Mi4_individual_fly_axis_available"] is True
    assert mi4_components["Tanaka_2023_Mi4_experimental_membrane_voltage"] is False
    assert mi4_components["Tanaka_2023_recording_to_MaleCNS_body_crosswalk_found"] is False
    assert mi4_components["Tanaka_2023_Figure_6_named_Mi4_C3_files_are_behavioral"] is True
    assert report["matrix"]["C3"]["evidence_components"][
        "Tanaka_2023_Figure_6_named_Mi4_C3_files_are_behavioral"
    ] is True
    assert mi4_components["Wu_2026_afterimages_Mi4_GCaMP6f_phenotype_verified"] is True
    assert mi4_components[
        "Wu_2026_afterimages_public_numeric_Mi4_payload_verified"
    ] is False
    assert mi4_components["Wu_2026_afterimages_experimental_membrane_voltage"] is False
    assert "Wu_2026_afterimages_Mi4_GCaMP6f_figure" in report["matrix"]["Mi4"][
        "phenotype_evidence_sources"
    ]
    assert report["matrix"]["C3"]["evidence_components"][
        "Gonzalez_Suarez_2022_C3_source_dynamics_available"
    ] is False
    c3_components = report["matrix"]["C3"]["evidence_components"]
    assert c3_components["Yuan_2020_C3_intervention_candidate_verified"] is True
    assert c3_components["Yuan_2020_C3_direct_recording_candidate_verified"] is False
    assert c3_components["Yuan_2020_C3_public_numeric_source_dynamics_verified"] is False
    assert mi4_components[
        "Strother_2018_successful_public_indexes_numeric_Mi4_trace_found"
    ] is False
    assert mi4_components["Strother_2018_Figshare_search_interpretable"] is False
    assert mi4_components["Strother_2018_global_absence_claimed"] is False
    assert mi4_components["Strother_2018_author_public_repository_count"] == 2
    assert (
        mi4_components[
            "Strother_2018_author_repository_numeric_payload_verified"
        ]
        is False
    )
    assert mi4_components["Strother_2018_author_repository_search_is_bounded"] is True
    assert (
        mi4_components[
            "Strother_2018_official_supplement_Mi4_moving_grating_figure_verified"
        ]
        is True
    )
    assert (
        mi4_components[
            "Strother_2018_supplement_public_numeric_trace_attachment_verified"
        ]
        is False
    )
    assert c3_components["C3_citation_graph_union_unique_work_count"] == 240
    assert c3_components["Pang_2025_L1_L2_voltage_not_C3"] is True
    assert c3_components["Pang_2025_Dryad_C3_filename_hit_count"] == 0
    for source in ("Mi4", "C3"):
        components = report["matrix"][source]["evidence_components"]
        assert components["Ramos_2020_complete_fulltext_scope_resolved"] is True
        assert components["Ramos_2020_required_source_direct_recording_verified"] is False
    assert c3_components["Ramos_2020_C3_perturbation_with_Tm9_calcium_readout"] is True
    for source in ("Mi4", "C3"):
        components = report["matrix"][source]["evidence_components"]
        assert components["Hao_2026_ASAP7y_Drosophila_voltage_candidate_unresolved"] is True
        assert components["Hao_2026_ASAP7y_required_source_direct_recording_verified"] is False
        assert components["Hao_2026_ASAP7y_public_numeric_payload_verified"] is False
        assert components["Hao_2026_Wayback_capture_is_abstract_only"] is True
        assert components["Hao_2026_Wayback_resolves_complete_cell_types"] is False
        assert components["Hao_2026_ClandininLab_public_repository_count"] == 31
        assert components["Hao_2026_ClandininLab_paper_repository_hit"] is False
        assert components["Hao_2026_bioRxiv_TDM_requester_pays_required"] is True
        assert components["Hao_2026_bioRxiv_TDM_payload_inventory_readable"] is False
        assert components["Hao_2026_Europe_PMC_annotation_hit"] is False
        assert (
            components[
                "Hao_2026_Europe_PMC_annotations_resolve_complete_cell_types"
            ]
            is False
        )
        assert components["Fendl_2021_target_Rdl_localization_verified"] is True
        assert components["Fendl_2021_pooled_GABAergic_T4_input_sign_supported"] is True
        assert components["Fendl_2021_source_specific_contact_resolved"] is False
        assert components["Fendl_2021_required_source_direct_recording_verified"] is False
        assert components["Sporar_2020_required_source_direct_recording_verified"] is False
    mi4_components = report["matrix"]["Mi4"]["evidence_components"]
    assert mi4_components["Drews_2020_individual_Mi4_numeric_GCaMP6f_verified"] is True
    assert mi4_components["Drews_2020_Mi4_pseudonymous_fly_count"] == 13
    assert mi4_components["Drews_2020_Mi4_experimental_membrane_voltage"] is False
    assert (
        mi4_components["Drews_2020_Mi4_tonic_weak_surround_phenotype_reproduced"]
        is True
    )
    assert mi4_components["Drews_2020_Mi4_full_temporal_kernel_identified"] is False
    assert (
        mi4_components["Drews_2020_preregistered_independent_validation_available"]
        is False
    )
    assert mi4_components["Drews_2020_recording_to_MaleCNS_body_crosswalk_found"] is False
    assert "Drews_2020_individual_GCaMP6f_time_series" in report["matrix"][
        "Mi4"
    ]["numerical_evidence_sources"]
    c3_components = report["matrix"]["C3"]["evidence_components"]
    assert c3_components["Shomar_2025_C3_behavioral_silencing_verified"] is True
    assert c3_components["Shomar_2025_direct_imaging_is_LC15_not_C3"] is True
    assert c3_components["Shomar_2025_C3_numeric_source_dynamics_verified"] is False
    for source in ("Mi1", "Tm3", "Mi4", "C3"):
        components = report["matrix"][source]["evidence_components"]
        assert components["T4_microstep_shuffle_R1_R6_energy_preserved"] is True
        assert components["T4_microstep_shuffle_source_energy_within_five_percent"] is True
        assert (
            components["T4_microstep_temporal_failure_retained_after_energy_audit"]
            is True
        )
        assert (
            components[
                "Fig3_baseline_or_gain_mismatch_explains_robustness_failure"
            ]
            is False
        )
        assert (
            components[
                "Fig3_bounded_latency_jitter_explains_every_negative_cell"
            ]
            is False
        )
        assert (
            components[
                "Fig3_source_kernel_robustness_failure_retained_after_alignment_audit"
            ]
            is True
        )
        assert components["Fig3_negative_cells_at_or_above_reference_SNR_count"] == 11
        assert components["Fig3_low_SNR_explains_every_negative_cell"] is False
        assert components["Fig3_ON_OFF_negative_cell_id_intersection_count"] == 0
        assert components["Fig3_post_hoc_cell_exclusion_authorized"] is False
    assert report["matrix"]["Mi4"]["evidence_components"][
        "Gur_2024_Mi4_C3_proofreading_table_row_count"
    ] == 723
    assert report["matrix"]["C3"]["evidence_components"][
        "Gur_2024_Mi4_C3_proofreading_table_row_count"
    ] == 678
    for source in ("Mi4", "C3"):
        components = report["matrix"][source]["evidence_components"]
        assert components["Gur_2024_Mi4_C3_named_files_are_proofreading_only"] is True
        assert components["Gur_2024_required_source_direct_physiology_found"] is False
    for source in ("Mi1", "Tm3"):
        assert report["matrix"][source]["published_optical_voltage_phenotype"] is True
        assert "Yang_2016_optical_voltage_figure" in report["matrix"][source][
            "phenotype_evidence_sources"
        ]


def test_gou_processed_calcium_is_local_but_not_allowed_voltage() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi1", "Tm3", "Tm1", "Tm2"):
        row = report["matrix"][source]
        assert row["publisher_described_unverified_numerical_payload"] is False
        assert "Gou_Dryad_processed_calcium" in row["numerical_evidence_sources"]
        assert row["publisher_described_sources"] == []
        assert (
            row["evidence_components"][
                "Gou_Dryad_local_processed_calcium_payload_verified"
            ]
            is True
        )
        assert (
            row["evidence_components"]["Gou_Dryad_processed_fly_axis_verified"]
            is True
        )
        assert (
            row["evidence_components"][
                "Gou_Dryad_stable_biological_individual_ID_verified"
            ]
            is False
        )
        assert (
            row["evidence_components"][
                "Gou_DANDI_asset_level_stable_participant_IDs_available"
            ]
            is True
        )
        assert (
            row["evidence_components"][
                "Gou_Dryad_rows_to_DANDI_subject_crosswalk_verified"
            ]
            is False
        )
        assert (
            row["evidence_components"]["Gou_Dryad_experimental_membrane_voltage"]
            is False
        )
        if source in {"Mi1", "Tm3"}:
            assert (
                row["evidence_components"][
                    "Gou_Dryad_record_level_direction_axis_identifiable"
                ]
                is False
            )
            assert row["evidence_components"]["Gou_Dryad_direction_invariance_evaluated"] is False
    assert (
        report["family_summary"]["T4"]["publisher_described_unverified_numerical_payload_count"]
        == 0
    )


def test_independent_T4_fast_phenotype_is_not_counted_as_numeric_payload() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi1", "Tm3"):
        row = report["matrix"][source]
        assert (
            row["evidence_components"][
                "independent_whole_cell_voltage_phenotype_without_numeric_payload"
            ]
            is True
        )
        assert "Behnia_2014_whole_cell_voltage_phenotype" in row["phenotype_evidence_sources"]
        assert (
            row["evidence_components"][
                "Behnia_2014_numeric_payload_found_in_audited_public_indexes"
            ]
            is False
        )
    for source in ("Mi4", "C3"):
        assert (
            report["matrix"][source]["evidence_components"][
                "independent_whole_cell_voltage_phenotype_without_numeric_payload"
            ]
            is False
        )
    mi1 = report["matrix"]["Mi1"]
    assert mi1["evidence_components"][
        "Matulis_2020_independent_Mi1_whole_cell_voltage_phenotype"
    ] is True
    assert mi1["evidence_components"][
        "Matulis_2020_public_numeric_voltage_payload_available"
    ] is False
    assert "Matulis_2020_Mi1_whole_cell_voltage_phenotype" in mi1[
        "phenotype_evidence_sources"
    ]


def test_borst_2025_parameterized_target_is_not_counted_as_measured_voltage() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi1", "Tm3", "Mi4", "Tm1", "Tm2", "Tm4", "Tm9"):
        components = report["matrix"][source]["evidence_components"]
        assert components["Borst_2025_parameterized_calcium_derived_target"] is True
        assert components["Borst_2025_target_is_experimental_membrane_voltage"] is False
        assert "Borst_2025_parameterized_calcium_derived_fit_target" in (
            report["matrix"][source]["phenotype_evidence_sources"]
        )
    for source in ("C3", "CT1"):
        assert (
            report["matrix"][source]["evidence_components"][
                "Borst_2025_parameterized_calcium_derived_target"
            ]
            is False
        )


def test_pirogova_numeric_calcium_does_not_cross_unit_or_identity_boundaries() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi1", "Tm3", "Tm1", "Tm2"):
        row = report["matrix"][source]
        components = row["evidence_components"]
        assert components["Pirogova_2023_local_numeric_calcium_time_series"] is True
        assert components["Pirogova_2023_allowed_membrane_voltage_unit"] is False
        assert components["Pirogova_2023_biological_individual_id_available"] is False
        assert "Pirogova_2023_historical_GCaMP6f_time_series" in (
            row["numerical_evidence_sources"]
        )


def test_braun_calcium_keeps_fly_identity_without_crossing_the_voltage_boundary() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Tm2", "Tm9", "CT1"):
        row = report["matrix"][source]
        components = row["evidence_components"]
        assert components["Braun_2023_local_GCaMP7f_time_series"] is True
        assert components["Braun_2023_stable_source_local_fly_IDs"] is True
        assert components["Braun_2023_complete_condition_grid"] is True
        assert components["Braun_2023_allowed_membrane_voltage_unit"] is False
        assert components["Braun_2023_baseline_window_declared"] is False
        assert components["stable_biological_individual_ids_in_any_numerical_payload"] is True
        assert "Braun_2023_GCaMP7f_edge_time_series" in row["numerical_evidence_sources"]
        assert row["gates"]["stable_biological_individual_id_on_allowed_payload"] is False
    for source in ("Mi4", "C3", "Tm4", "Tm9", "CT1"):
        assert (
            report["matrix"][source]["evidence_components"][
                "Pirogova_2023_local_numeric_calcium_time_series"
            ]
            is False
        )


def test_T5_relative_direction_labels_do_not_supply_physical_record_direction() -> None:
    report = json.loads(REPORT.read_text())
    for source in report["family_summary"]["T5"]["sources"]:
        components = report["matrix"][source]["evidence_components"]
        assert components["Figure4_relative_direction_code_to_PD_ND_verified"] is True
        assert (
            components["Figure4_direction_code_to_native_coordinate_motion_verified"]
            is True
        )
        assert (
            components["Figure4_direction_code_to_absolute_physical_motion_verified"]
            is False
        )
        assert components["Kohn_Portes_record_level_direction_code_available"] is False
        assert (
            components["Kohn_Portes_record_level_physical_direction_available"]
            is False
        )
        assert components["cross_dataset_direction_mapping_authorized"] is False


def test_T5_voltage_derived_kernels_do_not_claim_absolute_gain_or_invariance() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Tm1", "Tm2", "Tm4", "Tm9"):
        components = report["matrix"][source]["evidence_components"]
        assert components["Kohn_Portes_voltage_derived_temporal_kernel_verified"]
        assert components["Kohn_Portes_source_kernel_absolute_gain_transferable"] is False
        assert components["Kohn_Portes_source_kernel_stimulus_invariant"] is False
        assert components["Kohn_Portes_exact_volts_to_millivolts_scale_verified"] is True
        assert (
            components[
                "Kohn_Portes_voltage_or_filter_output_to_v7_state_mapping_available"
            ]
            is False
        )
        assert components["Kohn_Portes_saline_Tm9_relative_delay_supported"] is True
        assert components["Kohn_Portes_Tm9_delay_ordering_state_invariant"] is False
        assert components["Kohn_Portes_source_delay_transfer_to_v7_authorized"] is False
        assert components["Kohn_Portes_relative_low_frequency_Tm9_shape_evidence"] is True
        assert components["Kohn_Portes_preferred_frequency_summary_invariant"] is False
        assert components["Kohn_Portes_source_frequency_transfer_to_v7_authorized"] is False
        assert components["Kohn_Portes_Tm_to_T5_author_model_reproduced"] is True
        assert components["Kohn_Portes_Tm_to_T5_independently_validated"] is False
        assert components["Kohn_Portes_Tm_to_T5_model_transfer_to_v7_authorized"] is False
        assert components["FIB19_and_MaleCNS_population_source_rank_matches"] is True
        assert components["FIB19_to_MaleCNS_weight_transfer_authorized"] is False
        assert components[
            "Kohn_Portes_cross_stimulus_moving_bar_shape_candidate_available"
        ] is True
        assert components["Kohn_Portes_independent_moving_bar_validation_available"] is False
        assert components["Kohn_Portes_moving_bar_transfer_to_v7_authorized"] is False
    assert (
        report["matrix"]["CT1"]["evidence_components"][
            "Kohn_Portes_voltage_derived_temporal_kernel_verified"
        ]
        is False
    )


def test_T5_mapping_scope_is_complete_for_four_Tm_sources_but_not_CT1() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Tm1", "Tm2", "Tm4", "Tm9"):
        assert report["matrix"][source]["evidence_components"][
            "T5_source_exact_type_average_mapping_scope_complete"
        ]
    assert (
        report["matrix"]["CT1"]["evidence_components"][
            "T5_source_exact_type_average_mapping_scope_complete"
        ]
        is False
    )


def test_exact_type_average_mapping_is_explicit_without_body_assignment() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi1", "Tm3", "Mi4", "C3", "Tm1", "Tm2", "Tm4", "Tm9"):
        assert (
            report["matrix"][source]["gates"]["external_recording_to_body_or_explicit_type_average"]
            is True
        )
    assert (
        report["matrix"]["CT1"]["gates"]["external_recording_to_body_or_explicit_type_average"]
        is False
    )
    tm9 = report["matrix"]["Tm9"]["evidence_components"]
    assert tm9["Tm9_coordinate_identifiable_under_existing_rule"] is False
    assert tm9["Tm9_post_hoc_coordinate_repair_authorized"] is False
    assert tm9["Tm9_official_synapse_column_coordinate_identifiable"] is True
    assert tm9["Tm9_latest_public_release_is_v1_0"] is True
    assert tm9["Tm9_current_annotation_object_matches_frozen_release"] is True


def test_T5_modalities_are_not_combined_into_false_completeness() -> None:
    report = json.loads(REPORT.read_text())
    assert report["family_summary"]["T5"]["numerical_membrane_voltage_count"] == 4
    assert report["family_summary"]["T5"]["published_optical_voltage_phenotype_count"] == 2
    assert report["family_summary"]["T5"]["local_numerical_calcium_or_deconvolved_count"] == 5
    assert (
        report["matrix"]["Tm4"]["evidence_components"]["local_numerical_temporal_calcium_or_STRF"]
        is True
    )
    assert (
        report["matrix"]["Tm9"]["evidence_components"]["local_numerical_temporal_calcium_or_STRF"]
        is True
    )
    assert report["matrix"]["CT1"]["published_calcium_phenotype"] is True
    assert report["matrix"]["CT1"]["numerical_membrane_voltage"] is False
    for source in ("Tm1", "Tm2", "Tm4", "Tm9"):
        assert report["matrix"][source]["numerical_membrane_voltage"] is True
        assert report["matrix"][source]["numerical_evidence_sources"][0] == (
            "Kohn_Portes_whole_cell_voltage_flash_payload"
        )
    for source in ("Tm1", "Tm2"):
        components = report["matrix"][source]["evidence_components"]
        assert components["Yang_2016_PMC_attachment_count"] == 9
        assert components["Yang_2016_numeric_payload_found_in_successful_public_indexes"] is False
        assert components["Yang_2016_Figshare_search_accessible"] is False
    ct1 = report["matrix"]["CT1"]["evidence_components"]
    assert ct1["CT1_native_per_synapse_Lo1_columnar_retinotopy_available"] is True
    assert ct1["CT1_complete_official_LO_column_coverage"] is False
    assert ct1["local_numerical_spatial_calcium_only"] is True
    assert ct1["local_uncompartmented_type_average_deconvolved_calcium"] is True
    assert ct1["published_lobula_Lo1_dynamic_phenotype_only"] is True
    assert ct1["local_lobula_Lo1_numerical_time_series"] is False
    assert ct1["simulated_compartmental_model_voltage_not_experimental"] is True
    assert ct1["audited_CT1_candidate_set_has_direct_experimental_voltage"] is False
    assert ct1["audited_CT1_candidate_set_has_direct_Lo1_experimental_voltage"] is False
    assert ct1["CT1_incremental_2025_2026_candidate_count"] == 3
    assert ct1["CT1_incremental_2025_2026_direct_voltage_found"] is False
    assert ct1["CT1_PuRe_official_archive_candidate_count"] == 2
    assert ct1["CT1_PuRe_archive_contents_verified"] is True
    assert ct1["CT1_PuRe_new_numerical_payload_verified"] is False
    assert report["family_summary"]["T5"]["all_sources_contract_complete"] is False
    for source in report["family_summary"]["T5"]["sources"]:
        assert report["matrix"][source]["gates"][
            "complete_stimulus_and_baseline_fields_on_allowed_payload"
        ] is False
        assert report["matrix"][source]["evidence_components"][
            "Kohn_Portes_record_specific_stimulus_provenance_complete"
        ] is False


def test_Kohn_Portes_recording_capacity_does_not_replace_biological_identity() -> None:
    report = json.loads(REPORT.read_text())
    expected = {"Tm1": 8, "Tm2": 7, "Tm4": 7, "Tm9": 13}
    for source, count in expected.items():
        components = report["matrix"][source]["evidence_components"]
        assert components["Kohn_Portes_full_repository_recording_id_upper_bound"] == count
        assert components["Kohn_Portes_biological_individual_5_plus_3_authorized"] is False
    assert report["matrix"]["Tm1"]["evidence_components"][
        "Kohn_Portes_recording_id_upper_bound_meets_5_plus_3"
    ] is True
    assert report["matrix"]["Tm9"]["evidence_components"][
        "Kohn_Portes_recording_id_upper_bound_meets_5_plus_3"
    ] is True
    assert report["matrix"]["Tm2"]["evidence_components"][
        "Kohn_Portes_recording_id_upper_bound_meets_5_plus_3"
    ] is False
    assert report["matrix"]["Tm4"]["evidence_components"][
        "Kohn_Portes_recording_id_upper_bound_meets_5_plus_3"
    ] is False


def test_inhibitory_source_external_evidence_does_not_cross_modalities() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi4", "C3"):
        components = report["matrix"][source]["evidence_components"]
        assert components["independent_inhibitory_source_physiology_published"] is True
        assert components["independent_inhibitory_source_allowed_unit"] is False
        assert components[
            "bounded_candidate_set_direct_Mi4_C3_voltage_reference"
        ] is True
        assert components[
            "bounded_candidate_set_independent_Mi4_C3_voltage_found"
        ] is False


def test_C3_identity_and_external_cohort_are_preserved_without_false_authorization() -> None:
    report = json.loads(REPORT.read_text())
    row = report["matrix"]["C3"]
    components = row["evidence_components"]
    assert components["local_numerical_temporal_calcium_or_STRF"] is True
    assert components["stable_biological_individual_ids_in_any_numerical_payload"] is True
    assert components["independent_external_cohort_with_disjoint_individual_ids"] is True
    assert components["fixed_external_robustness_gate_passed"] is False
    assert row["gates"]["stable_biological_individual_id_on_allowed_payload"] is True
    assert row["gates"]["complete_stimulus_and_baseline_fields_on_allowed_payload"] is False
    assert row["all_contract_gates_passed"] is False


def test_no_source_or_downstream_gate_is_authorized() -> None:
    report = json.loads(REPORT.read_text())
    assert all(not row["all_contract_gates_passed"] for row in report["matrix"].values())
    assert report["all_nine_sources_contract_complete"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"]["evidence_modalities_are_not_interchangeable"] is True
    assert report["boundary"]["publisher_descriptions_are_not_local_numerical_payloads"] is True
