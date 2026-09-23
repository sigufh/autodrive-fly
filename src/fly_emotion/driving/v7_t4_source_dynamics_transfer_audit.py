"""Audit whether published T4-source recordings can define v7 source dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t4-source-dynamics-transfer-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_source_dynamics_transfer_audit.py"
)


def evaluate_v7_t4_source_dynamics_transfer_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    ephys_path = Path(config["electrophysiology_evidence"])
    interface_path = Path(config["electrophysiology_interface"])
    timebase_path = Path(config["timebase_evidence"])
    coordinate_path = Path(config["stimulus_coordinate_evidence"])
    coordinate_protocol_path = Path(config["stimulus_coordinate_protocol"])
    malecns_mapping_path = Path(config["malecns_mapping_evidence"])
    malecns_mapping_protocol_path = Path(config["malecns_mapping_protocol"])
    microstep_path = Path(config["microstep_evidence"])
    unified_path = Path(config["official_unified_model_evidence"])
    verified_unified_path = Path(config["verified_unified_model_evidence"])
    fig3_kernel_path = Path(config["fig3_source_kernel_evidence"])
    fig3_robustness_path = Path(config["fig3_source_kernel_robustness_evidence"])
    arenz_path = Path(config["arenz_source_dynamics_evidence"])
    c3_strf_path = Path(config["c3_strf_source_dynamics_evidence"])
    c2c3_vor_path = Path(config["c2c3_version_of_record_evidence"])
    c3_strf_flash_path = Path(config["c3_strf_flash_transfer_evidence"])
    c3_directional_path = Path(config["c3_directional_edge_evidence"])
    fig1_temporal_path = Path(config["fig1_source_temporal_readiness_evidence"])
    c3_filter_path = Path(config["c3_analytic_filter_evidence"])
    timing_models_path = Path(config["timing_models_source_filter_evidence"])
    gonzalez_suarez_path = Path(config["gonzalez_suarez_mi4_evidence"])
    yuan_c3_path = Path(config["yuan_c3_candidate_evidence"])
    strother_index_path = Path(config["strother_mi4_public_index_evidence"])
    c3_citation_graph_path = Path(config["c3_citation_graph_evidence"])
    legacy_c3_path = Path(config["legacy_C3_candidate_evidence"])
    hao_asap7y_path = Path(config["hao_ASAP7y_candidate_evidence"])
    fendl_path = Path(config["fendl_receptor_boundary_evidence"])
    drews_sporar_path = Path(config["drews_Mi4_sporar_C3_boundary_evidence"])
    drews_phenotype_path = Path(config["drews_Mi4_contrast_phenotype_evidence"])
    stable_contrast_path = Path(config["stable_contrast_source_scope_evidence"])
    tanaka_mi4_path = Path(config["tanaka_Mi4_calcium_evidence"])
    afterimages_mi4_path = Path(config["afterimages_Mi4_evidence"])
    flyvis_path = Path(config["flyvis_c3_time_constant_evidence"])
    flyvis_visual_path = Path(config["flyvis_visual_source_time_constants_evidence"])
    flyvis_effective_path = Path(config["flyvis_c3_effective_dynamics_evidence"])
    c3_measured_path = Path(config["c3_measured_filter_robustness_evidence"])
    public_models_path = Path(config["public_t4_model_source_coverage_evidence"])
    crossfit_axis_path = Path(config["crossfit_axis_evidence"])
    crossfit_functional_path = Path(config["crossfit_functional_evidence"])
    crossfit_sequence_path = Path(config["crossfit_sequence_evidence"])
    retinal_symmetry_path = Path(config["retinal_symmetry_evidence"])
    ephys = json.loads((root / ephys_path).read_text(encoding="utf-8"))
    interface = json.loads((root / interface_path).read_text(encoding="utf-8"))
    coordinates = json.loads((root / coordinate_path).read_text(encoding="utf-8"))
    malecns_mapping = json.loads(
        (root / malecns_mapping_path).read_text(encoding="utf-8")
    )
    microstep = json.loads((root / microstep_path).read_text(encoding="utf-8"))
    unified = json.loads((root / unified_path).read_text(encoding="utf-8"))
    verified_unified = json.loads(
        (root / verified_unified_path).read_text(encoding="utf-8")
    )
    fig3_kernel = json.loads((root / fig3_kernel_path).read_text(encoding="utf-8"))
    fig3_robustness = json.loads(
        (root / fig3_robustness_path).read_text(encoding="utf-8")
    )
    arenz = json.loads((root / arenz_path).read_text(encoding="utf-8"))
    c3_strf = json.loads((root / c3_strf_path).read_text(encoding="utf-8"))
    c2c3_vor = json.loads((root / c2c3_vor_path).read_text(encoding="utf-8"))
    c3_strf_flash = json.loads(
        (root / c3_strf_flash_path).read_text(encoding="utf-8")
    )
    c3_directional = json.loads(
        (root / c3_directional_path).read_text(encoding="utf-8")
    )
    fig1_temporal = json.loads(
        (root / fig1_temporal_path).read_text(encoding="utf-8")
    )
    c3_filter = json.loads((root / c3_filter_path).read_text(encoding="utf-8"))
    timing_models = json.loads(
        (root / timing_models_path).read_text(encoding="utf-8")
    )
    gonzalez_suarez = json.loads(
        (root / gonzalez_suarez_path).read_text(encoding="utf-8")
    )
    yuan_c3 = json.loads((root / yuan_c3_path).read_text(encoding="utf-8"))
    strother_index = json.loads(
        (root / strother_index_path).read_text(encoding="utf-8")
    )
    c3_citation_graph = json.loads(
        (root / c3_citation_graph_path).read_text(encoding="utf-8")
    )
    legacy_c3 = json.loads((root / legacy_c3_path).read_text(encoding="utf-8"))
    hao_asap7y = json.loads((root / hao_asap7y_path).read_text(encoding="utf-8"))
    fendl = json.loads((root / fendl_path).read_text(encoding="utf-8"))
    drews_sporar = json.loads((root / drews_sporar_path).read_text(encoding="utf-8"))
    drews_phenotype = json.loads(
        (root / drews_phenotype_path).read_text(encoding="utf-8")
    )
    stable_contrast = json.loads(
        (root / stable_contrast_path).read_text(encoding="utf-8")
    )
    tanaka_mi4 = json.loads((root / tanaka_mi4_path).read_text(encoding="utf-8"))
    afterimages_mi4 = json.loads(
        (root / afterimages_mi4_path).read_text(encoding="utf-8")
    )
    flyvis = json.loads((root / flyvis_path).read_text(encoding="utf-8"))
    flyvis_visual = json.loads((root / flyvis_visual_path).read_text(encoding="utf-8"))
    flyvis_effective = json.loads(
        (root / flyvis_effective_path).read_text(encoding="utf-8")
    )
    c3_measured = json.loads(
        (root / c3_measured_path).read_text(encoding="utf-8")
    )
    public_models = json.loads(
        (root / public_models_path).read_text(encoding="utf-8")
    )
    crossfit_axis = json.loads((root / crossfit_axis_path).read_text(encoding="utf-8"))
    crossfit_functional = json.loads(
        (root / crossfit_functional_path).read_text(encoding="utf-8")
    )
    crossfit_sequence = json.loads(
        (root / crossfit_sequence_path).read_text(encoding="utf-8")
    )
    retinal_symmetry = json.loads(
        (root / retinal_symmetry_path).read_text(encoding="utf-8")
    )
    replay = ephys["paper_model_replay"]
    source_axes = replay["array_axes"]["inputs"]
    synthesis = replay["direction_synthesis"]
    required_sources = config["required_sources"]
    verified_sources = {
        name: {
            "cell_count": replay["input_cells"][name],
            "sample_interval_milliseconds": interface["arrays"]["fig3_inputs"][name][
                "sample_interval_milliseconds"
            ],
            "stable_MaleCNS_body_ids_available": False,
        }
        for name in required_sources
    }
    source_direction_axis = any("direction" in axis for axis in source_axes)
    signed_shifts = {
        direction: {name: values.get(name, 0) for name in required_sources}
        for direction, values in (("pd", synthesis["pd"]), ("nd", synthesis["nd"]))
    }
    target_conditioned_shift = any(
        signed_shifts["pd"][name] == -signed_shifts["nd"][name]
        and signed_shifts["pd"][name] != 0
        for name in required_sources
    )
    if list(config["source_mapping_modes"]) != required_sources:
        raise ValueError("T4 source mapping declaration differs from required order")
    type_average_rows = {
        source: malecns_mapping["source_mapping"][source]
        for source in required_sources
    }
    type_average_mapping_complete = all(
        config["source_mapping_modes"][source] == "exact_type_average"
        and row["all_bodies_in_canonical_graph"]
        and row["soma_side_complete"]
        and row["columnar_retinotopy_available"]
        for source, row in type_average_rows.items()
    )
    fields = {
        "direction_independent_source_kernel": fig1_temporal[
            "source_temporal_kernel_transfer_authorized"
        ],
        "source_to_MaleCNS_identity_mapping": type_average_mapping_complete,
        "physical_v7_sample_interval": coordinates[
            "offline_time_coordinate_contract_complete"
        ],
        "millivolts_to_v7_normalized_state_mapping": interface["interface_boundary"][
            "convert_millivolts_to_normalized_drive"
        ],
        "ordered_source_sequence_identifiability": bool(
            crossfit_sequence["authorize_new_functional_candidate"]
        ),
    }
    gates = {
        "source_direction_axis_available": source_direction_axis,
        "source_shift_is_not_target_PD_ND_conditioned": not target_conditioned_shift,
        "physical_timebase_available": fields["physical_v7_sample_interval"],
        "state_unit_mapping_available": fields["millivolts_to_v7_normalized_state_mapping"],
        "independent_dynamic_validation_available": bool(
            interface["available_final_test"]
        ),
        "existing_microstep_temporal_control_passed": microstep[
            "temporal_identifiability"
        ]["passed"],
        "crossfit_structure_axis_passed": crossfit_axis[
            "T4_synapse_crossfit_axis_passed"
        ],
        "crossfit_functional_candidate_passed": crossfit_functional[
            "candidate_passed"
        ],
        "ordered_source_sequence_passed": crossfit_sequence[
            "authorize_new_functional_candidate"
        ],
    }
    transferable = all(gates.values()) and all(fields.values())
    recovered = config["official_unified_model_recovered_manifest"]
    package = unified["whole_cell_candidate"]["datasets"]["unified_model"]
    manifest_consistent = bool(
        int(recovered["article_id"]) == 16663486
        and recovered["file_name"] == "modelFigure.zip"
        and int(recovered["size_bytes"]) == int(package["size_bytes"])
        and recovered["mime_type"] == "application/zip"
        and len(recovered["md5"]) == 32
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(ephys_path): _sha256(root / ephys_path),
                str(interface_path): _sha256(root / interface_path),
                str(timebase_path): _sha256(root / timebase_path),
                str(coordinate_path): _sha256(root / coordinate_path),
                str(coordinate_protocol_path): _sha256(
                    root / coordinate_protocol_path
                ),
                str(malecns_mapping_path): _sha256(root / malecns_mapping_path),
                str(malecns_mapping_protocol_path): _sha256(
                    root / malecns_mapping_protocol_path
                ),
                str(microstep_path): _sha256(root / microstep_path),
                str(unified_path): _sha256(root / unified_path),
                str(verified_unified_path): _sha256(root / verified_unified_path),
                str(fig3_kernel_path): _sha256(root / fig3_kernel_path),
                str(fig3_robustness_path): _sha256(root / fig3_robustness_path),
                str(arenz_path): _sha256(root / arenz_path),
                str(c3_strf_path): _sha256(root / c3_strf_path),
                str(c2c3_vor_path): _sha256(root / c2c3_vor_path),
                str(c3_strf_flash_path): _sha256(root / c3_strf_flash_path),
                str(c3_directional_path): _sha256(root / c3_directional_path),
                str(fig1_temporal_path): _sha256(root / fig1_temporal_path),
                str(c3_filter_path): _sha256(root / c3_filter_path),
                str(timing_models_path): _sha256(root / timing_models_path),
                str(gonzalez_suarez_path): _sha256(root / gonzalez_suarez_path),
                str(yuan_c3_path): _sha256(root / yuan_c3_path),
                str(strother_index_path): _sha256(root / strother_index_path),
                str(c3_citation_graph_path): _sha256(root / c3_citation_graph_path),
                str(legacy_c3_path): _sha256(root / legacy_c3_path),
                str(hao_asap7y_path): _sha256(root / hao_asap7y_path),
                str(fendl_path): _sha256(root / fendl_path),
                str(drews_sporar_path): _sha256(root / drews_sporar_path),
                str(drews_phenotype_path): _sha256(root / drews_phenotype_path),
                str(stable_contrast_path): _sha256(root / stable_contrast_path),
                str(tanaka_mi4_path): _sha256(root / tanaka_mi4_path),
                str(afterimages_mi4_path): _sha256(root / afterimages_mi4_path),
                str(flyvis_path): _sha256(root / flyvis_path),
                str(flyvis_visual_path): _sha256(root / flyvis_visual_path),
                str(flyvis_effective_path): _sha256(root / flyvis_effective_path),
                str(c3_measured_path): _sha256(root / c3_measured_path),
                str(public_models_path): _sha256(root / public_models_path),
                str(crossfit_axis_path): _sha256(root / crossfit_axis_path),
                str(crossfit_functional_path): _sha256(root / crossfit_functional_path),
                str(crossfit_sequence_path): _sha256(root / crossfit_sequence_path),
                str(retinal_symmetry_path): _sha256(root / retinal_symmetry_path),
            },
            "required_sources": required_sources,
            "read_only": True,
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "available_source_recordings": {
            "axes": source_axes,
            "sample_interval_milliseconds": 1.0,
            "sources": verified_sources,
            "role": "training_reproduction",
        },
        "offline_v7_stimulus_coordinates": {
            "frame_interval_milliseconds": coordinates["time_coordinates"][
                "frame_interval_milliseconds"
            ],
            "substep_interval_milliseconds": coordinates["time_coordinates"][
                "substep_interval_milliseconds"
            ],
            "horizontal_fov_degrees": coordinates["camera_coordinates"][
                "horizontal_fov_degrees"
            ],
            "horizontal_coordinates_complete": coordinates[
                "offline_horizontal_stimulus_coordinate_contract_complete"
            ],
            "two_dimensional_angular_calibration_complete": coordinates[
                "offline_two_dimensional_stimulus_coordinate_contract_complete"
            ],
            "biologically_calibrated": coordinates["biological_timebase_calibrated"],
            "external_recording_alignment_verified": coordinates[
                "external_recording_alignment_verified"
            ],
        },
        "source_to_MaleCNS_mapping": {
            "accepted_contract_field": (
                "recording_unit_to_MaleCNS_body_id_or_explicit_type_average"
            ),
            "mapping_mode": "exact_type_average",
            "recording_level_body_assignment": False,
            "required_sources": required_sources,
            "source_mapping_complete": {
                source: bool(
                    row["all_bodies_in_canonical_graph"]
                    and row["soma_side_complete"]
                    and row["columnar_retinotopy_available"]
                )
                for source, row in type_average_rows.items()
            },
            "all_required_T4_sources_complete": type_average_mapping_complete,
            "broadcast_rule": "one_population_mean_kernel_to_all_exact_same_type_bodies",
        },
        "paper_direction_synthesis": {
            "shift_samples": synthesis["shift_samples"],
            "signed_shifts_by_target_direction": signed_shifts,
            "target_PD_ND_conditioned": target_conditioned_shift,
            "may_be_used_as_label_blind_v7_source_kernel": False,
        },
        "official_unified_model_package": {
            "doi": package["doi"],
            "article_id": recovered["article_id"],
            "size_bytes": package["size_bytes"],
            "description_names_optTables": "optTables.mat"
            in package["description"],
            "prior_live_audit_file_manifest_retrieved": unified["interface_status"][
                "unified_model_file_manifest_retrieved"
            ],
            "recovered_file_manifest": {
                key: recovered[key]
                for key in (
                    "recovery_source",
                    "snapshot_timestamp",
                    "file_id",
                    "file_name",
                    "size_bytes",
                    "md5",
                    "mime_type",
                    "official_download_url",
                )
            },
            "recovered_manifest_consistent": manifest_consistent,
            "file_manifest_retrieved": manifest_consistent,
            "prior_live_audit_files_verified": unified["interface_status"][
                "unified_model_files_verified"
            ],
            "files_verified": verified_unified["files_verified"],
            "payload_retrieved": recovered["payload_retrieved"],
            "retrieval_method": "Chrome DevTools after official WAF challenge",
            "model_package_sha256": verified_unified["packages"]["model"][
                "sha256"
            ],
            "supporting_package_sha256": verified_unified["packages"][
                "supporting"
            ]["sha256"],
        },
        "verified_unified_model_semantics": {
            "T4_target_model_parameters_available": verified_unified[
                "T4_target_model_parameters_available"
            ],
            "target_components": verified_unified["model_semantics"]["components"],
            "source_type_mapping_available": verified_unified["transfer_gates"][
                "E_I_E2_I2_to_MaleCNS_source_mapping_available"
            ],
            "cardinal_diagonal_to_subtype_mapping_available": verified_unified[
                "transfer_gates"
            ]["cardinal_diagonal_to_T4_subtype_mapping_available"],
            "independent_cell_holdout_available": verified_unified["transfer_gates"][
                "independent_cell_holdout_available"
            ],
            "MaleCNS_source_kernel_transfer_authorized": verified_unified[
                "MaleCNS_source_kernel_transfer_authorized"
            ],
        },
        "verified_fig3_source_kernel_readiness": {
            "source_workbook_verified": True,
            "ON_source_specific_kernels_ready": fig3_kernel[
                "ON_source_specific_kernels_ready"
            ],
            "prior_two_pool_kernel_authorized": fig3_kernel[
                "prior_two_pool_kernel_authorized"
            ],
            "source_specific_kernel_candidate_authorized": fig3_kernel[
                "source_specific_kernel_candidate_authorized"
            ],
            "cross_cell_robustness_passed": fig3_robustness[
                "all_source_kernel_robustness_gates_passed"
            ],
        },
        "verified_Arenz_source_filter_readiness": {
            "filter_parameters_verified": arenz["Arenz_filter_parameters_verified"],
            "current_source_coverage_fraction": arenz["current_v7_source_contract"][
                "coverage_fraction"
            ],
            "missing_current_sources": arenz["current_v7_source_contract"][
                "missing_from_Arenz"
            ],
            "physical_time_transfer_authorized": arenz[
                "physical_time_transfer_authorized"
            ],
            "source_filter_candidate_authorized": arenz[
                "source_filter_candidate_authorized"
            ],
        },
        "verified_C3_STRF_readiness": {
            "numerical_data_verified": c3_strf["C3_STRF_numerical_data_verified"],
            "direct_temporal_measurement_available": c3_strf[
                "combined_source_contract"
            ]["C3_direct_temporal_measurement_available"],
            "all_required_sources_have_some_direct_temporal_evidence": c3_strf[
                "combined_source_contract"
            ]["all_source_types_have_some_direct_temporal_evidence"],
            "all_required_sources_share_one_transferable_parameterization": c3_strf[
                "combined_source_contract"
            ]["all_source_types_share_one_transferable_parameterization"],
            "membrane_voltage_or_validated_deconvolved_kernel_available": c3_strf[
                "transfer_gates"
            ]["C3_membrane_voltage_or_validated_deconvolved_kernel_available"],
            "source_filter_candidate_authorized": c3_strf[
                "C3_source_filter_candidate_authorized"
            ],
        },
        "C2C3_version_of_record_update": {
            "doi": c2c3_vor["version_of_record"]["doi"],
            "published_on": c2c3_vor["version_of_record"]["published_on"],
            "repository_revision": c2c3_vor["protocol"][
                "remote_head_observed"
            ],
            "repository_release_count": c2c3_vor["repository_release_count"],
            "repository_tag_count": c2c3_vor["repository_tag_count"],
            "new_payload_source_types": c2c3_vor["new_payload_source_types"],
            "measurement_modality": c2c3_vor[
                "new_payload_measurement_modality"
            ],
            "C3_unique_Flyname_count": c2c3_vor["C3_unique_Flyname_count"],
            "Mi1_control_unique_Flyname_count": c2c3_vor[
                "Mi1_control_unique_Flyname_count"
            ],
            "new_C3_or_Mi4_membrane_voltage_payload_found": c2c3_vor[
                "new_C3_or_Mi4_membrane_voltage_payload_found"
            ],
            "new_Mi4_numerical_payload_found": c2c3_vor[
                "new_Mi4_numerical_payload_found"
            ],
            "new_recording_to_MaleCNS_body_crosswalk_found": c2c3_vor[
                "new_recording_to_MaleCNS_body_crosswalk_found"
            ],
            "source_dynamics_transfer_gate_changed": c2c3_vor[
                "source_dynamics_transfer_gate_changed"
            ],
        },
        "verified_C3_STRF_to_independent_flash_transfer": {
            "source_fly_axis_unit_count": c3_strf_flash["cohorts"][
                "source_fly_axis_unit_count"
            ],
            "external_flash_fly_count": c3_strf_flash["cohorts"][
                "external_flash_fly_count"
            ],
            "mean_waveform_correlation": c3_strf_flash["external_validation"][
                "mean_waveform_correlation"
            ],
            "minimum_external_fly_correlation": c3_strf_flash[
                "external_validation"
            ]["per_fly_correlation_summary"]["minimum"],
            "bootstrap_correlation_p05": c3_strf_flash["external_validation"][
                "bootstrap"
            ]["correlation_p05"],
            "transfer_passed": c3_strf_flash["C3_STRF_to_flash_transfer_passed"],
            "source_kernel_candidate_authorized": c3_strf_flash[
                "C3_source_kernel_candidate_authorized"
            ],
        },
        "verified_C3_directional_edge_boundary": {
            "response_unit": c3_directional["measurement"]["response_unit"],
            "fly_count_in_public_payload": c3_directional["measurement"][
                "fly_count_in_public_payload"
            ],
            "fly_count_in_version_of_record_caption": c3_directional[
                "measurement"
            ]["fly_count_in_version_of_record_caption"],
            "ROI_count": c3_directional["measurement"]["ROI_count"],
            "direction_degrees": c3_directional["measurement"][
                "direction_degrees"
            ],
            "edge_velocity_degrees_per_second": c3_directional["measurement"][
                "edge_velocity_degrees_per_second"
            ],
            "sample_interval_seconds": c3_directional["measurement"][
                "interpolated_sample_interval_seconds"
            ],
            "paper_reports_direction_preference": c3_directional["measurement"][
                "paper_reports_direction_preference"
            ],
            "every_directional_fly_in_flash_cohort": c3_directional[
                "cohort_relationship"
            ]["every_directional_fly_in_flash_cohort"],
            "every_directional_fly_in_STRF_cohort": c3_directional[
                "cohort_relationship"
            ]["every_directional_fly_in_STRF_cohort"],
            "independent_cohort": c3_directional["cohort_relationship"][
                "independent_cohort"
            ],
            "experimental_membrane_voltage": c3_directional[
                "experimental_membrane_voltage"
            ],
            "direction_specific_source_kernel_verified": c3_directional[
                "direction_specific_C3_source_kernel_verified"
            ],
            "source_dynamics_transfer_authorized": c3_directional[
                "authorize_C3_source_dynamics_transfer"
            ],
        },
        "verified_Fig1_source_temporal_readiness": {
            "workbooks_verified": fig1_temporal["observations"][
                "official_Fig1_workbooks_verified"
            ],
            "source_average_tables_have_time_axis": fig1_temporal["observations"][
                "source_type_average_tables_have_time_axis"
            ],
            "individual_source_tables_have_time_axis": fig1_temporal["observations"][
                "individual_source_tables_have_time_axis"
            ],
            "object_payload_hash_verified": fig1_temporal["edmond_object_payload"][
                "hash_verified"
            ],
            "object_payload_safely_inspected": fig1_temporal[
                "edmond_object_payload"
            ]["safely_inspected"],
            "source_temporal_kernel_transfer_authorized": fig1_temporal[
                "source_temporal_kernel_transfer_authorized"
            ],
        },
        "verified_C3_analytic_filter_precheck": {
            "band_pass_BIC_below_low_pass_BIC": c3_filter[
                "band_pass_BIC_below_low_pass_BIC"
            ],
            "minimum_leave_one_fly_out_correlation": c3_filter["model_families"][
                "band_pass"
            ]["cross_validation"]["minimum_held_out_correlation"],
            "median_leave_one_fly_out_correlation": c3_filter["model_families"][
                "band_pass"
            ]["cross_validation"]["median_held_out_correlation"],
            "filter_family_precheck_passed": c3_filter[
                "C3_analytic_filter_family_precheck_passed"
            ],
            "source_filter_transfer_authorized": c3_filter[
                "C3_analytic_filter_transfer_authorized"
            ],
        },
        "verified_TimingModels_source_filter_readiness": {
            "repository_commit": timing_models["protocol"]["repository_commit"],
            "covered_source_filter_stability_verified": timing_models[
                "covered_source_filter_stability_verified"
            ],
            "current_source_coverage_fraction": timing_models[
                "current_v7_source_contract"
            ]["coverage_fraction"],
            "missing_current_sources": timing_models["current_v7_source_contract"][
                "missing_sources"
            ],
            "complete_source_filter_candidate_authorized": timing_models[
                "complete_source_filter_candidate_authorized"
            ],
        },
        "verified_Gonzalez_Suarez_independent_Mi4_readiness": {
            "doi": gonzalez_suarez["paper"]["doi"],
            "Mi4_GCaMP6f_fly_count": gonzalez_suarez[
                "paper_measurement_evidence"
            ]["Mi4_GCaMP6f_fly_count"],
            "Mi4_type_average_filter_available": gonzalez_suarez[
                "repository_evidence"
            ]["Mi4_type_average_filter_available"],
            "filter_sample_interval_seconds": gonzalez_suarez[
                "repository_evidence"
            ]["filter_sample_interval_seconds"],
            "individual_cell_axis_available": gonzalez_suarez[
                "repository_evidence"
            ]["individual_cell_axis_available"],
            "Mi4_experimental_membrane_voltage_available": gonzalez_suarez[
                "independent_Mi4_experimental_membrane_voltage_available"
            ],
            "C3_source_dynamics_available": gonzalez_suarez[
                "independent_C3_source_dynamics_available"
            ],
            "Mi4_C3_voltage_transfer_authorized": gonzalez_suarez[
                "authorize_Mi4_C3_voltage_transfer"
            ],
            "bioRxiv_supplement_retrieved": gonzalez_suarez["public_indexes"][
                "bioRxiv_full_document_with_supplement_retrieved"
            ],
            "individual_flies_are_statistical_units": gonzalez_suarez[
                "preprint_evidence"
            ]["individual_flies_are_statistical_units"],
            "public_individual_fly_numeric_payload_attached": gonzalez_suarez[
                "preprint_evidence"
            ]["public_individual_fly_numeric_payload_attached"],
        },
        "verified_Yuan_C3_intervention_boundary": {
            "doi": yuan_c3["paper"]["doi"],
            "C3_intervention_candidate_verified": yuan_c3[
                "C3_intervention_candidate_verified"
            ],
            "directly_recorded_neural_activity_sources": yuan_c3[
                "abstract_evidence"
            ]["directly_recorded_neural_activity_sources"],
            "C3_direct_recording_candidate_verified": yuan_c3[
                "C3_direct_recording_candidate_verified"
            ],
            "C3_public_numeric_source_dynamics_payload_verified": yuan_c3[
                "C3_public_numeric_source_dynamics_payload_verified"
            ],
            "C3_source_dynamics_transfer_authorized": yuan_c3[
                "authorize_C3_source_dynamics_transfer"
            ],
        },
        "verified_Strother_Mi4_public_index_boundary": {
            "doi": strother_index["paper"]["doi"],
            "successful_index_numeric_Mi4_trace_found": strother_index[
                "local_numeric_Mi4_trace_payload_found_in_successful_indexes"
            ],
            "PMC_numeric_data_attachment_count": strother_index[
                "PMC_attachment_inventory"
            ]["numeric_data_attachment_count"],
            "Figshare_search_interpretable": strother_index["audited_indexes"][
                "Figshare_search_interpretable"
            ],
            "global_absence_claimed": strother_index["global_absence_claimed"],
            "author_public_repository_count": strother_index[
                "author_repository_index"
            ]["Bitbucket_public_repository_count"],
            "author_repository_numeric_payload_verified": strother_index[
                "author_repository_index"
            ]["paper_linked_numeric_repository_verified"],
            "author_repository_search_is_bounded": strother_index[
                "author_repository_index"
            ]["bounded_index_not_global_repository_absence"],
            "neuron_image_analysis_complete_history_audited": strother_index[
                "author_repository_index"
            ]["repositories"]["neuron_image_analysis"][
                "complete_history_audited"
            ],
            "supplementary_bundle_entry_count": strother_index[
                "PMC_attachment_inventory"
            ]["bundle_entry_count"],
            "moving_grating_trace_figure_present": strother_index[
                "supplementary_Mi4_evidence"
            ]["moving_grating_axonal_trace_figure_present"],
            "moving_grating_fly_count": strother_index[
                "supplementary_Mi4_evidence"
            ]["moving_grating_fly_count"],
            "moving_grating_speed_degrees_per_second": strother_index[
                "supplementary_Mi4_evidence"
            ]["moving_grating_speed_degrees_per_second"],
            "supplement_public_numeric_trace_attachment_verified": strother_index[
                "supplementary_Mi4_evidence"
            ]["public_numeric_trace_attachment_verified"],
            "Mi4_source_dynamics_transfer_authorized": strother_index[
                "authorize_Mi4_source_dynamics_transfer"
            ],
        },
        "verified_C3_citation_graph_boundary": {
            "union_unique_work_count": c3_citation_graph["citation_graph"][
                "union_unique_work_count"
            ],
            "unresolved_reference_ID_count": c3_citation_graph["citation_graph"][
                "unresolved_reference_ID_count"
            ],
            "high_relevance_candidate": c3_citation_graph[
                "high_relevance_candidate"
            ]["name"],
            "directly_recorded_neuron_types": c3_citation_graph[
                "high_relevance_candidate"
            ]["directly_recorded_neuron_types"],
            "Dryad_file_count": c3_citation_graph["Dryad"]["file_count"],
            "Dryad_total_declared_bytes": c3_citation_graph["Dryad"][
                "total_declared_bytes"
            ],
            "new_independent_C3_direct_recording_candidate_found": (
                c3_citation_graph[
                    "new_independent_C3_direct_recording_candidate_found"
                ]
            ),
            "C3_source_dynamics_transfer_authorized": c3_citation_graph[
                "authorize_C3_source_dynamics_transfer"
            ],
        },
        "verified_legacy_C3_candidate_boundary": {
            "audited_candidate_count": legacy_c3["audited_candidate_count"],
            "intervention_only_candidates": legacy_c3[
                "independent_C3_intervention_only_candidates"
            ],
            "direct_source_dynamics_candidates": legacy_c3[
                "direct_C3_or_Mi4_source_dynamics_candidates"
            ],
            "unresolved_fulltext_candidates": legacy_c3[
                "unresolved_fulltext_candidates"
            ],
            "Ramos_2020_fulltext_scope_resolved": legacy_c3["candidates"][
                "Ramos_Traslosheros_2020"
            ]["complete_fulltext_scope_resolved"],
            "Ramos_2020_direct_recording_targets": legacy_c3["candidates"][
                "Ramos_Traslosheros_2020"
            ]["direct_neural_recording_targets_in_relevant_C3_experiment"],
            "Ramos_2020_C3_direct_recording_verified": legacy_c3["candidates"][
                "Ramos_Traslosheros_2020"
            ]["C3_direct_neural_recording_verified"],
            "source_dynamics_transfer_authorized": legacy_c3[
                "authorize_Mi4_C3_source_dynamics_transfer"
            ],
        },
        "unresolved_Hao_2026_ASAP7y_candidate": {
            "doi": hao_asap7y["paper"]["doi"],
            "Drosophila_in_vivo_voltage_imaging": hao_asap7y["verified_scope"][
                "Drosophila_in_vivo_voltage_imaging"
            ],
            "measurement_modality": hao_asap7y["verified_scope"][
                "measurement_modality"
            ],
            "paper_source_experimental_cell_types": hao_asap7y[
                "cell_type_resolution"
            ]["experimental_Drosophila_cell_types_named_in_paper_sources"],
            "public_author_figure_named_examples": hao_asap7y[
                "cell_type_resolution"
            ]["public_author_figure_named_examples"],
            "complete_experimental_cell_type_set_resolved": hao_asap7y[
                "cell_type_resolution"
            ]["complete_experimental_cell_type_set_resolved"],
            "candidate_classification": hao_asap7y["cell_type_resolution"][
                "candidate_classification"
            ],
            "public_numeric_payload_verified": hao_asap7y[
                "public_numeric_payload_verified"
            ],
            "Wayback_capture_view": hao_asap7y["archived_biorxiv_page"][
                "captured_view"
            ],
            "Wayback_linked_route_contents_retrieved": hao_asap7y[
                "archived_biorxiv_page"
            ]["linked_route_contents_retrieved"],
            "Wayback_resolves_complete_cell_types": hao_asap7y[
                "archived_biorxiv_page"
            ]["resolves_complete_paper_cell_type_set"],
            "ClandininLab_public_repository_count": hao_asap7y[
                "public_repository_indexes"
            ]["ClandininLab_public_repository_count"],
            "ClandininLab_paper_repository_hits": hao_asap7y[
                "public_repository_indexes"
            ]["ClandininLab_paper_specific_name_or_description_hits"],
            "bioRxiv_TDM_requester_pays_required": hao_asap7y[
                "access_boundaries"
            ]["bioRxiv_TDM_requester_pays_authentication_required"],
            "bioRxiv_TDM_payload_inventory_readable": hao_asap7y[
                "access_boundaries"
            ]["bioRxiv_TDM_payload_inventory_readable"],
            "Europe_PMC_annotation_count": hao_asap7y[
                "Europe_PMC_text_mining"
            ]["annotation_count"],
            "Europe_PMC_Mi4_annotation_hit": hao_asap7y[
                "Europe_PMC_text_mining"
            ]["Mi4_annotation_hit"],
            "Europe_PMC_C3_annotation_hit": hao_asap7y[
                "Europe_PMC_text_mining"
            ]["C3_annotation_hit"],
            "Europe_PMC_annotation_scope": hao_asap7y[
                "Europe_PMC_text_mining"
            ]["annotation_scope"],
            "author_dissertation_restricted_until": hao_asap7y[
                "author_dissertation"
            ]["restricted_until"],
            "author_dissertation_public_abstract_names_cell_types": hao_asap7y[
                "author_dissertation"
            ]["public_abstract_names_experimental_cell_types"],
            "successful_public_index_numeric_payload_found": hao_asap7y[
                "public_repository_indexes"
            ]["successful_index_numeric_payload_found"],
            "global_payload_absence_claimed": hao_asap7y[
                "public_repository_indexes"
            ]["global_payload_absence_claimed"],
            "source_dynamics_fit_authorized": hao_asap7y[
                "authorize_source_dynamics_fit"
            ],
        },
        "verified_Fendl_2021_receptor_boundary": {
            "doi": fendl["thesis"]["doi"],
            "classification": fendl["classification"],
            "direct_functional_imaging_cell_types": fendl[
                "direct_functional_imaging_cell_types"
            ],
            "target_receptor_localization_cell_types": fendl[
                "target_receptor_localization_cell_types"
            ],
            "Rdl_localized_on_T4_T5_dendrites": fendl[
                "target_receptor_evidence"
            ]["Rdl_localized_on_T4_T5_dendrites"],
            "pooled_Mi4_C3_CT1_GABAergic_T4_input_sign_supported": fendl[
                "target_receptor_evidence"
            ]["Mi4_C3_CT1_assigned_as_GABAergic_T4_inputs"],
            "source_specific_Mi4_or_C3_contact_resolved": fendl[
                "target_receptor_evidence"
            ]["source_specific_Mi4_or_C3_receptor_contact_resolved"],
            "Mi4_direct_recording_verified": fendl[
                "Mi4_direct_neural_recording_verified"
            ],
            "C3_direct_recording_verified": fendl[
                "C3_direct_neural_recording_verified"
            ],
            "source_dynamics_transfer_authorized": fendl[
                "authorize_Mi4_C3_source_dynamics_transfer"
            ],
        },
        "verified_Drews_2020_independent_Mi4_calcium": {
            "doi": drews_sporar["Drews_2020"]["article_doi"],
            "measurement_modality": drews_sporar["Drews_2020"][
                "Mi4_payload"
            ]["response_modality"],
            "response_unit": drews_sporar["Drews_2020"]["Mi4_payload"][
                "response_unit"
            ],
            "ROI_count": drews_sporar["Drews_2020"]["Mi4_payload"][
                "ROI_count"
            ],
            "pseudonymous_fly_count": drews_sporar["Drews_2020"][
                "Mi4_payload"
            ]["pseudonymous_fly_count"],
            "individual_ROI_trial_numeric_time_series_verified": drews_sporar[
                "Drews_2020"
            ]["Mi4_payload"]["individual_ROI_trial_numeric_time_series_verified"],
            "experimental_membrane_voltage": drews_sporar["Drews_2020"][
                "experimental_membrane_voltage_verified"
            ],
            "C3_direct_recording_verified": drews_sporar["Drews_2020"][
                "C3_direct_neural_recording_verified"
            ],
            "recording_to_MaleCNS_body_crosswalk_found": drews_sporar[
                "Drews_2020"
            ]["Mi4_payload"]["recording_to_MaleCNS_body_crosswalk_found"],
            "source_dynamics_fit_authorized": drews_sporar[
                "authorize_T4_source_dynamics_fit"
            ],
            "tonic_weak_surround_phenotype_qualitatively_reproduced": (
                drews_phenotype[
                    "Mi4_tonic_weak_surround_suppression_qualitatively_reproduced"
                ]
            ),
            "preregistered_independent_dynamic_validation_available": (
                drews_phenotype[
                    "preregistered_independent_dynamic_validation_available"
                ]
            ),
            "full_temporal_kernel_identified": drews_phenotype[
                "full_Mi4_temporal_kernel_identified"
            ],
        },
        "verified_Sporar_2020_C3_context_boundary": {
            "doi": drews_sporar["Sporar_2020"]["doi"],
            "C3_exact_term_count": drews_sporar["Sporar_2020"][
                "term_evidence"
            ]["C3"]["exact_term_count"],
            "C3_direct_recording_verified": drews_sporar["Sporar_2020"][
                "C3_direct_neural_recording_verified"
            ],
            "classification": drews_sporar["Sporar_2020"]["classification"],
            "source_dynamics_transfer_authorized": drews_sporar["Sporar_2020"][
                "authorize_Mi4_C3_source_dynamics_transfer"
            ],
        },
        "verified_Gur_2024_stable_contrast_source_scope": {
            "doi": stable_contrast["paper"]["doi"],
            "directly_recorded_neuron_types": stable_contrast[
                "directly_recorded_neuron_types"
            ],
            "Mi4_direct_physiology_found": stable_contrast[
                "Mi4_direct_physiology_found"
            ],
            "C3_direct_physiology_found": stable_contrast[
                "C3_direct_physiology_found"
            ],
            "Mi4_proofreading_row_count": stable_contrast[
                "proofreading_workbooks"
            ]["Mi4"]["row_count"],
            "C3_proofreading_row_count": stable_contrast[
                "proofreading_workbooks"
            ]["C3"]["row_count"],
            "Mi4_C3_files_are_connectome_proofreading_only": stable_contrast[
                "Mi4_C3_files_are_connectome_proofreading_only"
            ],
            "external_recording_to_MaleCNS_crosswalk_found": stable_contrast[
                "external_recording_to_MaleCNS_crosswalk_found"
            ],
            "source_dynamics_transfer_authorized": stable_contrast[
                "authorize_Mi4_C3_source_dynamics_transfer"
            ],
        },
        "verified_Tanaka_2023_independent_Mi4_calcium": {
            "doi": tanaka_mi4["paper"]["doi"],
            "measurement_modality": tanaka_mi4["measurement"]["modality"],
            "response_unit": tanaka_mi4["measurement"]["response_unit"],
            "fly_count": tanaka_mi4["payload_inventory"]["fly_count"],
            "selected_ROI_count": tanaka_mi4["payload_inventory"][
                "selected_roi_count"
            ],
            "individual_fly_axis_available": tanaka_mi4["payload_inventory"][
                "individual_fly_axis_available"
            ],
            "trial_and_ROI_axes_available": tanaka_mi4["payload_inventory"][
                "trial_and_ROI_axes_available"
            ],
            "experimental_membrane_voltage": tanaka_mi4[
                "experimental_membrane_voltage"
            ],
            "recording_to_MaleCNS_body_crosswalk_found": tanaka_mi4[
                "recording_to_MaleCNS_body_crosswalk_found"
            ],
            "source_dynamics_transfer_authorized": tanaka_mi4[
                "authorize_Mi4_source_dynamics_transfer"
            ],
            "Figure_6_named_Mi4_C3_member_count": len(
                tanaka_mi4["Figure_6_named_Mi4_C3_members"]["members"]
            ),
            "Figure_6_measurement_object": tanaka_mi4[
                "Figure_6_named_Mi4_C3_members"
            ]["measurement_object"],
            "Figure_6_neural_activity_recording": tanaka_mi4[
                "Figure_6_named_Mi4_C3_members"
            ]["neural_activity_recording"],
            "Figure_6_source_dynamics_payload": tanaka_mi4[
                "Figure_6_named_Mi4_C3_members"
            ]["source_dynamics_payload"],
            "Figure_6_large_MAT_members_downloaded": tanaka_mi4[
                "Figure_6_named_Mi4_C3_members"
            ]["large_MAT_members_downloaded"],
        },
        "verified_Wu_2026_afterimages_Mi4_phenotype": {
            "doi": afterimages_mi4["paper"]["doi"],
            "measurement_modality": afterimages_mi4["measurement"][
                "Mi4_measurement_modality"
            ],
            "fly_count": afterimages_mi4["measurement"]["Mi4_fly_count"],
            "ROI_count": afterimages_mi4["measurement"]["Mi4_ROI_count"],
            "public_numeric_payload_verified": afterimages_mi4[
                "public_numeric_Mi4_payload_verified"
            ],
            "experimental_membrane_voltage": afterimages_mi4[
                "experimental_membrane_voltage"
            ],
            "source_dynamics_transfer_authorized": afterimages_mi4[
                "authorize_Mi4_source_dynamics_transfer"
            ],
        },
        "verified_FlyVis_C3_time_constant_readiness": {
            "repository_commit": flyvis["protocol"]["repository_commit"],
            "pretrained_model_count": flyvis["training_contract"]["model_count"],
            "solver_dt_seconds": flyvis["training_contract"]["dataset_dt_seconds"],
            "C3_median_time_constant_seconds": flyvis["source_time_constants"]["C3"][
                "median_seconds"
            ],
            "C3_models_at_or_below_solver_dt": flyvis["source_time_constants"]["C3"][
                "models_at_or_below_solver_dt"
            ],
            "C3_time_constant_transfer_authorized": flyvis[
                "C3_time_constant_transfer_authorized"
            ],
        },
        "verified_FlyVis_C3_effective_dynamics_readiness": {
            "pretrained_model_count": flyvis_effective["protocol"]["model_count"],
            "fixed_denominator": flyvis_effective["protocol"]["fixed_denominator"],
            "cross_dt_minimum_correlation": flyvis_effective["cross_dt_summary"][
                "minimum"
            ],
            "leave_one_model_out_minimum_correlation_by_dt": {
                name: item["unit_shape_leave_one_model_out_summary"]["minimum"]
                for name, item in flyvis_effective["ensemble_stability"].items()
            },
            "external_ensemble_correlation_by_dt_and_calcium_assumption": {
                dt: {
                    tau: item["ensemble_equal_shape_correlation"]
                    for tau, item in by_tau.items()
                }
                for dt, by_tau in flyvis_effective[
                    "external_C3_flash_consistency"
                ].items()
            },
            "effective_dynamics_transfer_authorized": flyvis_effective[
                "FlyVis_C3_effective_dynamics_transfer_authorized"
            ],
        },
        "verified_FlyVis_visual_source_time_constants": {
            "T4_missing_sources": flyvis_visual["source_coverage"]["T4"]["missing"],
            "T5_missing_sources": flyvis_visual["source_coverage"]["T5"]["missing"],
            "all_values_finite_and_positive": flyvis_visual["gates"][
                "all_models_finite_and_positive"
            ],
            "all_above_solver_dt": flyvis_visual["gates"][
                "every_source_above_solver_dt_in_every_model"
            ],
            "all_cross_model_IQR_stable": flyvis_visual["gates"][
                "every_source_cross_model_IQR_stable"
            ],
            "transferable": flyvis_visual[
                "FlyVis_visual_source_time_constants_transferable"
            ],
        },
        "verified_C3_measured_filter_robustness": {
            "cross_deconvolution_stability_passed": c3_measured[
                "cross_deconvolution_stability_passed"
            ],
            "minimum_held_out_correlation_by_assumption": {
                name: item["minimum_held_out_correlation"]
                for name, item in c3_measured[
                    "leave_one_fly_out_by_assumption"
                ].items()
            },
            "every_deconvolution_assumption_passed": c3_measured[
                "every_deconvolution_assumption_passed"
            ],
            "type_shared_kernel_authorized": c3_measured[
                "C3_type_shared_kernel_authorized"
            ],
        },
        "verified_public_T4_model_source_coverage": {
            "Clark_required_source_coverage_fraction": public_models["models"][
                "Clark_SynapticModel"
            ]["required_source_coverage_fraction"],
            "Clark_missing_required_sources": public_models["models"][
                "Clark_SynapticModel"
            ]["missing_required_sources"],
            "ModelDB_required_source_coverage_fraction": public_models["models"][
                "ModelDB_239435"
            ]["required_source_coverage_fraction"],
            "C3_specific_parameters_available": public_models["transfer_gates"][
                "C3_specific_parameters_available"
            ],
            "complete_source_dynamics_transfer_authorized": public_models[
                "complete_source_dynamics_transfer_authorized"
            ],
        },
        "crossfit_dynamic_readiness": {
            "structure_axis_passed": crossfit_axis[
                "T4_synapse_crossfit_axis_passed"
            ],
            "functional_candidate_passed": crossfit_functional["candidate_passed"],
            "maximum_ordered_source_sequence_direction_pass_count": (
                crossfit_sequence["maximum_ordered_direction_pass_count"]
            ),
            "ordered_bilateral_direction_subtypes": crossfit_sequence[
                "ordered_bilateral_direction_subtypes_by_reduction"
            ],
            "shuffle_signed_mean_bilateral_direction_subtypes": (
                crossfit_sequence["mode_reduction_results"]["temporal_shuffle"]
                ["signed_mean"]["bilateral_direction_subtypes"]
            ),
            "source_sequence_candidate_authorized": crossfit_sequence[
                "authorize_new_functional_candidate"
            ],
            "balanced_retina_ordered_direction_pass_count": retinal_symmetry[
                "maximum_balanced_ordered_direction_pass_count"
            ],
            "retinal_sampling_imbalance_explains_direction_failure": (
                retinal_symmetry[
                    "retinal_sampling_imbalance_explains_direction_failure"
                ]
            ),
        },
        "required_transfer_fields_available": fields,
        "transfer_gates": gates,
        "source_dynamics_transfer_authorized": transferable,
        "next_candidate_authorized": transferable,
        "blocking_data_requirements": [
            name for name, available in fields.items() if not available
        ],
        "stop_reason": (
            None
            if transferable
            else "published_source_dynamics_not_transferable_to_label_blind_v7"
        ),
        "boundary": config["boundary"],
    }
