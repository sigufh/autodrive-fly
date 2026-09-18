from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7 import V7_CONFIG, V7_IMPLEMENTATION, V7Contract, _sha256

CONFIG = Path("configs/driving-v7-goal-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_goal_audit.py")


def _load_evidence(root: Path, paths: dict[str, str]) -> tuple[dict[str, dict], dict[str, str]]:
    reports = {}
    hashes = {}
    for name, relative in paths.items():
        path = root / relative
        if not path.exists():
            raise ValueError(f"required goal-audit evidence is missing: {relative}")
        reports[name] = json.loads(path.read_text(encoding="utf-8"))
        hashes[relative] = _sha256(path)
    return reports, hashes


def evaluate_v7_goal_coverage(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract = V7Contract.load(root)
    reports, evidence_hashes = _load_evidence(root, config["evidence"])
    if contract.payload["stage_order"] != config["required_stage_order"]:
        raise ValueError("v7 stage order differs from objective audit")
    manifest = reports["manifest"]
    controlled = reports["controlled_vision"]
    t5 = reports["t5_phenotype"]
    t5_data = reports["t5_data"]
    t5_conductance = reports["t5_repository"]
    t5_labels = reports["t5_label_audit"]
    interface = reports["ephys_interface"]
    stage1_split = reports["stage1_split"]
    stage1_scoring = reports["stage1_scoring"]
    stage1_input = reports["stage1_input"]
    stage1_development = reports["stage1_development"]
    stage1_geometry_ab = reports["stage1_geometry_ab"]
    visual_target_inputs = reports["visual_target_inputs"]
    t5_supplement = reports["t5_supplement"]
    looming_mechanisms = reports["looming_mechanisms"]
    stage1_nested = reports["stage1_nested"]
    target_fit = reports["target_fit"]
    closed_loop_tuning = reports["closed_loop_tuning"]
    closed_loop_calibration = reports["closed_loop_calibration"]
    closed_loop_controls = reports["closed_loop_controls"]
    closed_loop_multi = reports["closed_loop_multi"]
    r1r6_multi_tuning = reports["r1r6_multi_tuning"]
    r1r6_multi_calibration = reports["r1r6_multi_calibration"]
    r1r6_local_tuning = reports["r1r6_local_tuning"]
    r1r6_local_calibration = reports["r1r6_local_calibration"]
    r1r6_local_controls = reports["r1r6_local_controls"]
    neural_channels_tuning = reports["neural_channels_tuning"]
    neural_channels_calibration = reports["neural_channels_calibration"]
    neural_channel_controls = reports["neural_channel_controls"]
    neural_topology_controls = reports["neural_topology_controls"]
    degree_preserving_control = reports["degree_preserving_control"]
    parameter_matched_baselines = reports["parameter_matched_baselines"]
    nested_neural_screen = reports["nested_neural_screen"]
    t4_source_resolved = reports["t4_source_resolved"]
    t4_continuous_pair_precheck = reports["t4_continuous_pair_precheck"]
    t4_pair_lag_audit = reports["t4_pair_lag_audit"]
    t4_normalized_correlator = reports["t4_normalized_correlator"]
    t4_local_correlator_precheck = reports["t4_local_correlator_precheck"]
    t4_source_pool_local = reports["t4_source_pool_local"]
    three_hop_moment = reports["three_hop_moment"]
    three_hop_temporal_consistency = reports["three_hop_temporal_consistency"]
    upstream_latency_audit = reports["upstream_latency_audit"]
    source_type_temporal_identifiability = reports[
        "source_type_temporal_identifiability"
    ]
    synapse_spatial_audit = reports["synapse_spatial_audit"]
    synapse_axis_calibration = reports["synapse_axis_calibration"]
    t4_synapse_crossfit_axis_audit = reports["t4_synapse_crossfit_axis_audit"]
    t4_synapse_correlator_precheck = reports["t4_synapse_correlator_precheck"]
    t4_synapse_antisymmetric_precheck = reports["t4_synapse_antisymmetric_precheck"]
    t4_synapse_crossfit_precheck = reports["t4_synapse_crossfit_precheck"]
    t4_crossfit_sequence_identifiability = reports[
        "t4_crossfit_sequence_identifiability"
    ]
    t4_crossfit_retinal_symmetry_audit = reports[
        "t4_crossfit_retinal_symmetry_audit"
    ]
    t4_synapse_centered_precheck = reports["t4_synapse_centered_precheck"]
    t4_synapse_microstep_precheck = reports["t4_synapse_microstep_precheck"]
    t4_source_dynamics_transfer_audit = reports["t4_source_dynamics_transfer_audit"]
    t4_source_identity_readiness_audit = reports[
        "t4_source_identity_readiness_audit"
    ]
    t4_individual_split_1khz_audit = reports["t4_individual_split_1khz_audit"]
    edmond_fig3_retrieval_audit = reports["edmond_fig3_retrieval_audit"]
    t4_recording_field_audit = reports["t4_recording_field_audit"]
    behnia_t4_fast_source_audit = reports["behnia_t4_fast_source_audit"]
    t4_inhibitory_source_external_audit = reports[
        "t4_inhibitory_source_external_audit"
    ]
    unified_model_package_audit = reports["unified_model_package_audit"]
    fig3_source_kernel_audit = reports["fig3_source_kernel_audit"]
    fig3_source_kernel_robustness = reports["fig3_source_kernel_robustness"]
    arenz_source_dynamics_audit = reports["arenz_source_dynamics_audit"]
    arenz_t5_source_dynamics_audit = reports["arenz_t5_source_dynamics_audit"]
    c3_strf_source_dynamics_audit = reports["c3_strf_source_dynamics_audit"]
    c3_strf_flash_transfer = reports["c3_strf_flash_transfer"]
    fig1_source_temporal_readiness_audit = reports[
        "fig1_source_temporal_readiness_audit"
    ]
    c3_analytic_filter_precheck = reports["c3_analytic_filter_precheck"]
    timing_models_source_filter_audit = reports["timing_models_source_filter_audit"]
    timing_models_ct1_compartment_audit = reports[
        "timing_models_ct1_compartment_audit"
    ]
    flyvis_c3_time_constant_audit = reports["flyvis_c3_time_constant_audit"]
    flyvis_visual_source_time_constants = reports[
        "flyvis_visual_source_time_constants"
    ]
    flyvis_c3_effective_dynamics_audit = reports[
        "flyvis_c3_effective_dynamics_audit"
    ]
    c3_measured_filter_robustness = reports["c3_measured_filter_robustness"]
    public_t4_model_source_coverage_audit = reports[
        "public_t4_model_source_coverage_audit"
    ]
    dryad_l1l2_source_dynamics_audit = reports[
        "dryad_l1l2_source_dynamics_audit"
    ]
    gou_sparsity_source_dynamics_audit = reports[
        "gou_sparsity_source_dynamics_audit"
    ]
    t5_contrast_opponency_source_data_audit = reports[
        "t5_contrast_opponency_source_data_audit"
    ]
    yang_t5_voltage_evidence_audit = reports["yang_t5_voltage_evidence_audit"]
    kohn_portes_t5_ephys_audit = reports["kohn_portes_t5_ephys_audit"]
    kohn_portes_identity_history_audit = reports[
        "kohn_portes_identity_history_audit"
    ]
    ct1_extreme_compartmentalization_audit = reports[
        "ct1_extreme_compartmentalization_audit"
    ]
    ct1_experimental_voltage_boundary_audit = reports[
        "ct1_experimental_voltage_boundary_audit"
    ]
    borst_2025_temporal_filtering_audit = reports[
        "borst_2025_temporal_filtering_audit"
    ]
    pirogova_source_calcium_audit = reports["pirogova_source_calcium_audit"]
    malecns_source_mapping_readiness_audit = reports[
        "malecns_source_mapping_readiness_audit"
    ]
    source_evidence_matrix = reports["source_evidence_matrix"]
    source_type_average_mapping_contract = reports[
        "source_type_average_mapping_contract"
    ]
    tm9_coordinate_identifiability_audit = reports[
        "tm9_coordinate_identifiability_audit"
    ]
    three_hop_source_coverage = reports["three_hop_source_coverage"]
    four_hop_scalar_precheck = reports["four_hop_scalar_precheck"]
    lc4_position_speed_precheck = reports["lc4_position_speed_precheck"]
    lc4_input_speed_precheck = reports["lc4_input_speed_precheck"]
    t5_lamina_split = reports["t5_lamina_split"]
    t5_lamina_scalar_precheck = reports["t5_lamina_scalar_precheck"]
    t5_source_axis_audit = reports["t5_source_axis_audit"]
    t5_source_pair_precheck = reports["t5_source_pair_precheck"]
    t5_typed_spatial_pair_precheck = reports["t5_typed_spatial_pair_precheck"]
    t5_native_direction_waveform_audit = reports[
        "t5_native_direction_waveform_audit"
    ]
    t5_ct1_terminal_axis_audit = reports["t5_ct1_terminal_axis_audit"]
    t5_ct1_axis_calibration = reports["t5_ct1_axis_calibration"]
    t5_ct1_crossfit_axis_audit = reports["t5_ct1_crossfit_axis_audit"]
    t5_ct1_source_dynamics_precheck = reports[
        "t5_ct1_source_dynamics_precheck"
    ]
    t5_ct1_multiplicative_precheck = reports[
        "t5_ct1_multiplicative_precheck"
    ]
    t5_ct1_axis_aware_precheck = reports["t5_ct1_axis_aware_precheck"]
    t5_ct1_axis_aware_antisymmetric_precheck = reports[
        "t5_ct1_axis_aware_antisymmetric_precheck"
    ]
    t5_ct1_axis_sequence_identifiability = reports[
        "t5_ct1_axis_sequence_identifiability"
    ]
    t5_physical_time_transfer_audit = reports["t5_physical_time_transfer_audit"]
    t4t5_source_dynamics_readiness = reports["t4t5_source_dynamics_readiness"]
    source_dynamics_external_evidence_contract = reports[
        "source_dynamics_external_evidence_contract"
    ]
    t5_continuous_moment_precheck = reports["t5_continuous_moment_precheck"]
    lplc1_near_collision_precheck = reports["lplc1_near_collision_precheck"]
    lplc1_input_structure = reports["lplc1_input_structure"]
    t5_spatial_order = reports["t5_spatial_order"]
    t4t5_local_edge_precheck = reports["t4t5_local_edge_precheck"]
    t4t5_local_edge_backends = reports["t4t5_local_edge_backends"]
    lplc_typed_screen = reports["lplc_typed_screen"]
    lplc2_radial_opponency = reports["lplc2_radial_opponency"]
    lplc2_position_coverage = reports["lplc2_position_coverage"]
    heading_ring = reports["heading_ring"]
    neural_episode_cv = reports["neural_episode_cv"]
    neural_episode_controls = reports["neural_episode_controls"]
    fc2_pfl_dna = reports["fc2_pfl_dna"]
    descending_paths = reports["descending_path_audit"]
    descending_propagation = reports["descending_propagation_precheck"]
    navigation_nested = reports["navigation_nested"]
    navigation_nested_eval = reports["navigation_nested_eval"]
    danger_throttle = reports["danger_throttle"]
    visual_corridor_goal = reports["visual_corridor_goal"]
    neural_corridor = reports["neural_corridor"]
    local_column_corridor = reports["local_column_corridor"]
    neural_dynamics_local = reports["neural_dynamics_local"]
    neural_corridor_dagger = reports["neural_corridor_dagger"]
    visual_layer_locality = reports["visual_layer_locality"]
    lamina_goal = reports["lamina_goal"]
    lamina_goal_symmetry = reports["lamina_goal_symmetry"]
    fc2_goal_memory = reports["fc2_goal_memory"]
    neural_goal_fusion = reports["neural_goal_fusion"]
    fusion_nested = reports["fusion_nested"]
    fusion_nested_eval = reports["fusion_nested_eval"]
    baseline_hashes_valid = all(
        _sha256(root / item["path"]) == item["sha256"]
        for item in contract.payload["baseline_contracts"].values()
    )
    disclosure_paths = {
        name: Path(path)
        for name, path in config["interface_disclosure"].items()
        if name != "contribution_layers"
    }
    if not all((root / path).exists() for path in disclosure_paths.values()):
        raise ValueError("v7 interface disclosure file is missing")
    disclosure_hashes = {str(path): _sha256(root / path) for path in disclosure_paths.values()}
    visual_inputs_only_at_receptors = (
        controlled["protocol"]["direct_input_type"] == "R1-R6"
        and controlled["protocol"]["target_direct_input_overlap"] == 0
    )
    visual_response_pass = bool(
        reports["typed_vision"]["controlled_response_gates_pass"]
        and reports["branched_t4"]["controlled_response_gates_pass"]
        and reports["conductance_t4"]["controlled_response_gates_pass"]
        and reports["fitted_t4"]["validation_passed"]
    )
    topology_complete = bool(
        controlled["stage1_topology_controls_complete"]
        and controlled["strict_degree_preserving_control_complete"]
        and controlled["real_topology_advantage"]
    )
    stage1_pass = visual_response_pass and topology_complete
    if manifest["advance_to_central_complex"] != stage1_pass:
        raise ValueError("manifest and independent coverage audit disagree on stage 1")
    downstream_authorized = bool(stage1_pass)
    checks = [
        {
            "item": 0,
            "deliverable": "independent v5/v6/v7 versions, checkpoints and claims",
            "status": "passed" if baseline_hashes_valid else "failed",
            "evidence": [str(V7_CONFIG), config["evidence"]["manifest"]],
            "observations": {
                "baseline_hashes_valid": baseline_hashes_valid,
                "v7_deployment_enabled": contract.payload["deployment_enabled"],
                "v7_city_expansion_enabled": contract.payload["city_expansion_enabled"],
                "default_runtime_changed": manifest["default_runtime_changed"],
            },
        },
        {
            "item": 1,
            "deliverable": "R1-R6 causal controlled vision through T4/T5 and LPLC/LC",
            "status": "failed",
            "evidence": [
                config["evidence"]["controlled_vision"],
                config["evidence"]["typed_vision"],
                config["evidence"]["branched_t4"],
                config["evidence"]["conductance_t4"],
                config["evidence"]["fitted_t4"],
                config["evidence"]["t5_phenotype"],
                config["evidence"]["t5_lamina_split"],
                config["evidence"]["t5_lamina_scalar_precheck"],
                config["evidence"]["t5_source_axis_audit"],
                config["evidence"]["t5_source_pair_precheck"],
                config["evidence"]["t5_typed_spatial_pair_precheck"],
                config["evidence"]["t5_native_direction_waveform_audit"],
                config["evidence"]["t5_ct1_terminal_axis_audit"],
                config["evidence"]["t5_ct1_axis_calibration"],
                config["evidence"]["t5_ct1_crossfit_axis_audit"],
                config["evidence"]["t5_ct1_source_dynamics_precheck"],
                config["evidence"]["t5_ct1_multiplicative_precheck"],
                config["evidence"]["t5_ct1_axis_aware_precheck"],
                config["evidence"]["t5_ct1_axis_aware_antisymmetric_precheck"],
                config["evidence"]["t5_ct1_axis_sequence_identifiability"],
                config["evidence"]["t5_physical_time_transfer_audit"],
                config["evidence"]["t4t5_source_dynamics_readiness"],
                config["evidence"]["source_dynamics_external_evidence_contract"],
                config["evidence"]["t5_continuous_moment_precheck"],
                config["evidence"]["t4_synapse_crossfit_axis_audit"],
                config["evidence"]["t4_synapse_correlator_precheck"],
                config["evidence"]["t4_synapse_antisymmetric_precheck"],
                config["evidence"]["t4_synapse_crossfit_precheck"],
                config["evidence"]["t4_crossfit_sequence_identifiability"],
                config["evidence"]["t4_crossfit_retinal_symmetry_audit"],
                config["evidence"]["t4_synapse_centered_precheck"],
                config["evidence"]["t4_synapse_microstep_precheck"],
                config["evidence"]["source_type_temporal_identifiability"],
                config["evidence"]["t4_source_dynamics_transfer_audit"],
                config["evidence"]["t4_source_identity_readiness_audit"],
                config["evidence"]["t4_individual_split_1khz_audit"],
                config["evidence"]["edmond_fig3_retrieval_audit"],
                config["evidence"]["t4_recording_field_audit"],
                config["evidence"]["behnia_t4_fast_source_audit"],
                config["evidence"]["t4_inhibitory_source_external_audit"],
                config["evidence"]["unified_model_package_audit"],
                config["evidence"]["fig3_source_kernel_audit"],
                config["evidence"]["fig3_source_kernel_robustness"],
                config["evidence"]["arenz_source_dynamics_audit"],
                config["evidence"]["arenz_t5_source_dynamics_audit"],
                config["evidence"]["c3_strf_source_dynamics_audit"],
                config["evidence"]["c3_strf_flash_transfer"],
                config["evidence"]["fig1_source_temporal_readiness_audit"],
                config["evidence"]["c3_analytic_filter_precheck"],
                config["evidence"]["timing_models_source_filter_audit"],
                config["evidence"]["timing_models_ct1_compartment_audit"],
                config["evidence"]["flyvis_c3_time_constant_audit"],
                config["evidence"]["flyvis_visual_source_time_constants"],
                config["evidence"]["flyvis_c3_effective_dynamics_audit"],
                config["evidence"]["c3_measured_filter_robustness"],
                config["evidence"]["public_t4_model_source_coverage_audit"],
                config["evidence"]["dryad_l1l2_source_dynamics_audit"],
                config["evidence"]["gou_sparsity_source_dynamics_audit"],
                config["evidence"]["t5_contrast_opponency_source_data_audit"],
                config["evidence"]["yang_t5_voltage_evidence_audit"],
                config["evidence"]["kohn_portes_t5_ephys_audit"],
                config["evidence"]["kohn_portes_identity_history_audit"],
                config["evidence"]["ct1_extreme_compartmentalization_audit"],
                config["evidence"]["ct1_experimental_voltage_boundary_audit"],
                config["evidence"]["borst_2025_temporal_filtering_audit"],
                config["evidence"]["pirogova_source_calcium_audit"],
                config["evidence"]["malecns_source_mapping_readiness_audit"],
                config["evidence"]["source_evidence_matrix"],
                config["evidence"]["source_type_average_mapping_contract"],
                config["evidence"]["tm9_coordinate_identifiability_audit"],
                config["evidence"]["lplc1_near_collision_precheck"],
                config["evidence"]["lplc1_input_structure"],
                config["evidence"]["lc4_input_speed_precheck"],
                config["evidence"]["lplc2_radial_opponency"],
                config["evidence"]["lplc2_position_coverage"],
            ],
            "observations": {
                "stimulus_count": controlled["protocol"]["stimulus_count"],
                "stimulus_families": sorted({item["family"] for item in controlled["stimuli"]}),
                "visual_inputs_only_at_receptors": visual_inputs_only_at_receptors,
                "controlled_response_gates_pass": visual_response_pass,
                "T5_measured_pair_count": t5["summary"]["all_pairs"]["pair_count"],
                "T5_biological_PD_code_assigned": t5["label_boundary"][
                    "biological_PD_code_assigned"
                ],
                "T5_external_direction_label_map_verified": t5_labels["label_status"][
                    "direction_code_to_PD_ND_mapping_verified"
                ],
                "T5_model_scoring_allowed": t5["label_boundary"]["model_scoring_allowed"],
                "T5_published_fitted_parameter_vectors_available": t5_supplement["replay_status"][
                    "published_17_cell_parameter_values_available"
                ],
                "T5_processed_cells_with_both_direction_codes": t5_conductance[
                    "direction_and_identity_readiness"
                ]["cells_with_both_moving_bar_direction_codes"],
                "T5_processed_direction_code_mapping_verified": t5_data[
                    "interface_status"
                ]["processed_T5_direction_code_to_PD_ND_mapping_verified"],
                "T5_processed_stable_biological_ids_available": t5_data[
                    "interface_status"
                ]["processed_T5_stable_biological_cell_ids_available"],
                "T5_processed_independent_cell_holdout_available": t5_data[
                    "interface_status"
                ]["processed_T5_independent_cell_holdout_available"],
                "T5_unified_model_files_verified": t5_data["interface_status"][
                    "unified_model_files_verified"
                ],
                "T5_typed_spatial_pair_joint_source_fraction": (
                    t5_typed_spatial_pair_precheck["source_coverage"][
                        "joint_valid_fraction"
                    ]
                ),
                "T5_typed_spatial_pair_direction_pass_counts": [
                    item["direction_pass_count"]
                    for item in t5_typed_spatial_pair_precheck["ordered_candidates"]
                ],
                "T5_typed_spatial_pair_polarity_pass_counts": [
                    item["polarity_pass_count"]
                    for item in t5_typed_spatial_pair_precheck["ordered_candidates"]
                ],
                "T5_typed_spatial_pair_control_eligible": bool(
                    t5_typed_spatial_pair_precheck["control_eligible_candidates"]
                ),
                "T5_native_waveform_condition_count": (
                    t5_native_direction_waveform_audit["condition_count"]
                ),
                "T5_native_waveform_passing_condition_count": (
                    t5_native_direction_waveform_audit["passing_condition_count"]
                ),
                "T5_native_waveform_template_authorized": (
                    t5_native_direction_waveform_audit[
                        "authorize_T5_direction_template"
                    ]
                ),
                "T5_CT1_terminal_mapping_R2_by_eye": {
                    eye: item["R2_by_coordinate"]
                    for eye, item in t5_ct1_terminal_axis_audit[
                        "mapping_holdout"
                    ].items()
                },
                "T5_CT1_terminal_axis_passing_populations": [
                    name
                    for name, item in t5_ct1_terminal_axis_audit[
                        "population_results"
                    ].items()
                    if item["passed"]
                ],
                "T5_CT1_terminal_axis_candidate_authorized": (
                    t5_ct1_terminal_axis_audit[
                        "authorize_CT1_spatial_dynamics_candidate"
                    ]
                ),
                "T5_CT1_axis_held_out_accuracy": t5_ct1_axis_calibration[
                    "held_out_accuracy_mean"
                ],
                "T5_CT1_axis_held_out_median_angle_degrees": (
                    t5_ct1_axis_calibration[
                        "held_out_median_angle_error_mean_degrees"
                    ]
                ),
                "T5_CT1_axis_calibration_passed": t5_ct1_axis_calibration[
                    "T5_CT1_axis_calibration_passed"
                ],
                "T5_CT1_crossfit_axis_accuracy": t5_ct1_crossfit_axis_audit[
                    "out_of_fold_overall"
                ]["accuracy"],
                "T5_CT1_crossfit_axis_median_angle_degrees": (
                    t5_ct1_crossfit_axis_audit["out_of_fold_overall"]
                    ["median_angle_error_degrees"]
                ),
                "T5_CT1_crossfit_axis_passed": t5_ct1_crossfit_axis_audit[
                    "T5_CT1_crossfit_axis_passed"
                ],
                "T5_CT1_dynamics_ordered_direction_pass_counts": [
                    item["direction_pass_count"]
                    for item in t5_ct1_source_dynamics_precheck[
                        "ordered_candidates"
                    ]
                ],
                "T5_CT1_dynamics_ordered_polarity_pass_counts": [
                    item["polarity_pass_count"]
                    for item in t5_ct1_source_dynamics_precheck[
                        "ordered_candidates"
                    ]
                ],
                "T5_CT1_dynamics_control_direction_pass_counts": {
                    mode: [
                        item["direction_pass_count"] for item in by_gain.values()
                    ]
                    for mode, by_gain in t5_ct1_source_dynamics_precheck[
                        "control_results"
                    ].items()
                },
                "T5_CT1_dynamics_strict_candidate_passed": (
                    t5_ct1_source_dynamics_precheck["ordered_precheck_passed"]
                ),
                "T5_CT1_multiplicative_ordered_direction_pass_counts": [
                    item["direction_pass_count"]
                    for item in t5_ct1_multiplicative_precheck[
                        "ordered_candidates"
                    ]
                ],
                "T5_CT1_multiplicative_ordered_polarity_pass_counts": [
                    item["polarity_pass_count"]
                    for item in t5_ct1_multiplicative_precheck[
                        "ordered_candidates"
                    ]
                ],
                "T5_CT1_multiplicative_control_eligible_gains": (
                    t5_ct1_multiplicative_precheck["control_eligible_gains"]
                ),
                "T5_CT1_multiplicative_controls_evaluated": (
                    t5_ct1_multiplicative_precheck["controls_evaluated"]
                ),
                "T5_CT1_multiplicative_strict_candidate_passed": (
                    t5_ct1_multiplicative_precheck["candidate_passed"]
                ),
                "T5_CT1_multiplicative_three_condition_evaluation_performed": (
                    t5_ct1_multiplicative_precheck[
                        "three_condition_evaluation_performed"
                    ]
                ),
                "T5_CT1_axis_aware_ordered_direction_pass_counts": [
                    item["direction_pass_count"]
                    for item in t5_ct1_axis_aware_precheck["ordered_candidates"]
                ],
                "T5_CT1_axis_aware_ordered_polarity_pass_counts": [
                    item["polarity_pass_count"]
                    for item in t5_ct1_axis_aware_precheck["ordered_candidates"]
                ],
                "T5_CT1_axis_aware_controls_evaluated": t5_ct1_axis_aware_precheck[
                    "controls_evaluated"
                ],
                "T5_CT1_axis_aware_strict_candidate_passed": (
                    t5_ct1_axis_aware_precheck["candidate_passed"]
                ),
                "T5_CT1_axis_aware_three_condition_evaluation_performed": (
                    t5_ct1_axis_aware_precheck["three_condition_evaluation_performed"]
                ),
                "T5_CT1_axis_aware_antisymmetric_direction_pass_counts": [
                    item["direction_pass_count"]
                    for item in t5_ct1_axis_aware_antisymmetric_precheck[
                        "ordered_candidates"
                    ]
                ],
                "T5_CT1_axis_aware_antisymmetric_polarity_pass_counts": [
                    item["polarity_pass_count"]
                    for item in t5_ct1_axis_aware_antisymmetric_precheck[
                        "ordered_candidates"
                    ]
                ],
                "T5_CT1_axis_aware_antisymmetric_controls_evaluated": (
                    t5_ct1_axis_aware_antisymmetric_precheck["controls_evaluated"]
                ),
                "T5_CT1_axis_aware_antisymmetric_candidate_passed": (
                    t5_ct1_axis_aware_antisymmetric_precheck["candidate_passed"]
                ),
                "T5_CT1_axis_sequence_maximum_direction_pass_count": (
                    t5_ct1_axis_sequence_identifiability[
                        "maximum_ordered_direction_pass_count"
                    ]
                ),
                "T5_CT1_axis_sequence_ordered_bilateral_subtypes": (
                    t5_ct1_axis_sequence_identifiability[
                        "ordered_bilateral_direction_subtypes_by_reduction"
                    ]
                ),
                "T5_CT1_axis_sequence_candidate_selected": (
                    t5_ct1_axis_sequence_identifiability["candidate_selected"]
                ),
                "T5_CT1_axis_sequence_authorizes_functional_candidate": (
                    t5_ct1_axis_sequence_identifiability[
                        "authorize_new_functional_candidate"
                    ]
                ),
                "T5_physical_source_transfer_required_fields": (
                    t5_physical_time_transfer_audit["required_fields"]
                ),
                "T5_physical_source_transfer_missing_fields": (
                    t5_physical_time_transfer_audit["missing_fields"]
                ),
                "T5_physical_source_transfer_ready": (
                    t5_physical_time_transfer_audit[
                        "T5_physical_source_transfer_ready"
                    ]
                ),
                "T4_T5_source_dynamics_ready": t4t5_source_dynamics_readiness[
                    "T4_T5_source_dynamics_ready"
                ],
                "T4_T5_source_dynamics_passing_gates": (
                    t4t5_source_dynamics_readiness["passing_gates"]
                ),
                "T4_T5_source_dynamics_failing_gates": (
                    t4t5_source_dynamics_readiness["failing_gates"]
                ),
                "T4_T5_source_dynamics_authorizes_new_candidate": (
                    t4t5_source_dynamics_readiness[
                        "authorize_new_T4_or_T5_functional_candidate"
                    ]
                ),
                "T4_millivolt_source_types_complete": (
                    t4_source_identity_readiness_audit["transfer_gates"][
                        "all_four_T4_source_types_have_millivolt_traces"
                    ]
                ),
                "T4_source_biological_individual_ID_available": (
                    t4_source_identity_readiness_audit["transfer_gates"][
                        "stable_pseudonymous_biological_individual_ID_available"
                    ]
                ),
                "T4_individual_level_source_validation_ready": (
                    t4_source_identity_readiness_audit[
                        "T4_individual_level_source_validation_ready"
                    ]
                ),
                "T4_individual_split_passing_source_conditions": [
                    f"{condition}:{source}"
                    for condition, details in t4_individual_split_1khz_audit[
                        "conditions"
                    ].items()
                    for source, result in details["sources"].items()
                    if result["passed"]
                ],
                "T4_all_individual_split_gates_passed": (
                    t4_individual_split_1khz_audit[
                        "all_source_condition_individual_split_gates_passed"
                    ]
                ),
                "T4_individual_split_sample_interval_milliseconds": (
                    t4_individual_split_1khz_audit["protocol"][
                        "sample_interval_milliseconds"
                    ]
                ),
                "T4_1khz_split_pass_fail_pattern_changed_from_10ms": (
                    t4_individual_split_1khz_audit["comparison_to_10ms"][
                        "pass_fail_pattern_changed"
                    ]
                ),
                "Edmond_Fig3_frozen_four_file_manifest_complete": (
                    edmond_fig3_retrieval_audit["gates"][
                        "frozen_four_file_manifest_complete"
                    ]
                ),
                "Edmond_Fig3_1khz_payload_currently_verified": (
                    edmond_fig3_retrieval_audit["gates"][
                        "all_four_local_payloads_hash_and_structure_verified"
                    ]
                ),
                "Edmond_Fig3_full_resolution_recompute_authorized": (
                    edmond_fig3_retrieval_audit["gates"][
                        "full_resolution_fixed_split_recompute_authorized"
                    ]
                ),
                "Edmond_network_failure_is_scientific_rejection": (
                    edmond_fig3_retrieval_audit["gates"]["scientific_source_rejected"]
                ),
                "T4_recording_field_count_available": (
                    t4_recording_field_audit["available_field_count"]
                ),
                "T4_recording_field_count_required": (
                    t4_recording_field_audit["required_field_count"]
                ),
                "T4_missing_recording_fields": t4_recording_field_audit[
                    "missing_fields"
                ],
                "T4_complete_stimulus_and_baseline_fields": (
                    t4_recording_field_audit[
                        "complete_stimulus_and_baseline_fields_on_allowed_payload"
                    ]
                ),
                "Behnia_T4_independent_fast_source_phenotypes": sorted(
                    behnia_t4_fast_source_audit["source_evidence"]
                ),
                "Behnia_T4_independent_numeric_payload_verified": (
                    behnia_t4_fast_source_audit["transfer_gates"][
                        "local_numeric_trace_payload_verified"
                    ]
                ),
                "Behnia_T4_independent_transfer_authorized": (
                    behnia_t4_fast_source_audit[
                        "independent_T4_source_dynamics_transfer_authorized"
                    ]
                ),
                "T4_inhibitory_external_Mi4_allowed_unit": (
                    t4_inhibitory_source_external_audit["source_evidence"]["Mi4"][
                        "allowed_response_unit"
                    ]
                ),
                "T4_inhibitory_external_C3_allowed_unit": (
                    t4_inhibitory_source_external_audit["source_evidence"]["C3"][
                        "allowed_response_unit"
                    ]
                ),
                "T4_inhibitory_external_C3_fixed_robustness_passed": (
                    t4_inhibitory_source_external_audit["source_evidence"]["C3"][
                        "independent_fixed_robustness_passed"
                    ]
                ),
                "T4_inhibitory_external_transfer_authorized": (
                    t4_inhibitory_source_external_audit[
                        "both_inhibitory_sources_have_transferable_external_validation"
                    ]
                ),
                "source_dynamics_external_contract_satisfied": (
                    source_dynamics_external_evidence_contract["contract_satisfied"]
                ),
                "source_dynamics_external_payload_present": (
                    source_dynamics_external_evidence_contract[
                        "external_payload_present"
                    ]
                ),
                "source_dynamics_external_required_split_roles": (
                    source_dynamics_external_evidence_contract[
                        "required_split_roles"
                    ]
                ),
                "Dryad_L1L2_required_source_coverage_count": (
                    dryad_l1l2_source_dynamics_audit[
                        "covered_required_source_count"
                    ]
                ),
                "Dryad_L1L2_source_dynamics_transfer_authorized": (
                    dryad_l1l2_source_dynamics_audit[
                        "suitable_external_source_dynamics_evidence"
                    ]
                ),
                "Dryad_L1L2_large_download_authorized": (
                    dryad_l1l2_source_dynamics_audit["download_disposition"][
                        "large_dataset_download_authorized"
                    ]
                ),
                "Gou_sparsity_required_source_coverage_count": (
                    gou_sparsity_source_dynamics_audit[
                        "covered_required_source_count"
                    ]
                ),
                "Gou_sparsity_partial_source_evidence_present": (
                    gou_sparsity_source_dynamics_audit[
                        "partial_source_dynamics_evidence_present"
                    ]
                ),
                "Gou_sparsity_source_dynamics_transfer_authorized": (
                    gou_sparsity_source_dynamics_audit[
                        "complete_external_source_dynamics_evidence"
                    ]
                ),
                "Gou_sparsity_bulk_download_authorized": any(
                    gou_sparsity_source_dynamics_audit["download_disposition"][
                        name
                    ]
                    for name in (
                        "Dryad_archive_download_authorized",
                        "DANDI_bulk_download_authorized",
                    )
                ),
                "T5_contrast_opponency_temporal_sources": (
                    t5_contrast_opponency_source_data_audit["T5_source_contract"][
                        "sources_with_verified_temporal_blocks"
                    ]
                ),
                "T5_contrast_opponency_spatial_only_sources": (
                    t5_contrast_opponency_source_data_audit["T5_source_contract"][
                        "sources_with_verified_spatial_but_not_temporal_blocks"
                    ]
                ),
                "T5_contrast_opponency_transfer_authorized": (
                    t5_contrast_opponency_source_data_audit[
                        "complete_T5_source_dynamics_transfer_authorized"
                    ]
                ),
                "Yang_T5_optical_voltage_phenotype_sources": (
                    yang_t5_voltage_evidence_audit["T5_source_contract"][
                        "sources_with_optical_voltage_phenotype"
                    ]
                ),
                "Yang_T5_numerical_voltage_payload_verified": (
                    yang_t5_voltage_evidence_audit[
                        "T5_numerical_voltage_payload_verified"
                    ]
                ),
                "Yang_T5_voltage_transfer_authorized": (
                    yang_t5_voltage_evidence_audit[
                        "T5_source_dynamics_transfer_authorized"
                    ]
                ),
                "Kohn_Portes_T5_numeric_voltage_sources": (
                    kohn_portes_t5_ephys_audit["T5_source_contract"][
                        "sources_with_local_numeric_membrane_voltage"
                    ]
                ),
                "Kohn_Portes_T5_missing_voltage_sources": (
                    kohn_portes_t5_ephys_audit["T5_source_contract"][
                        "missing_numeric_membrane_voltage_sources"
                    ]
                ),
                "Kohn_Portes_T5_identity_retained": (
                    kohn_portes_t5_ephys_audit["transfer_gates"][
                        "biological_individual_ids_retained"
                    ]
                ),
                "Kohn_Portes_T5_recording_ID_field_retained": (
                    kohn_portes_t5_ephys_audit["transfer_gates"][
                        "stable_recording_id_field_retained"
                    ]
                ),
                "Kohn_Portes_T5_minimum_train_validation_capacity_ready": (
                    kohn_portes_t5_ephys_audit["transfer_gates"][
                        "even_recording_id_upper_bound_enough_for_training_and_validation"
                    ]
                ),
                "Kohn_Portes_T5_transfer_authorized": (
                    kohn_portes_t5_ephys_audit[
                        "T5_source_dynamics_transfer_authorized"
                    ]
                ),
                "Kohn_Portes_full_repository_recording_ID_upper_bounds": {
                    source: details[
                        "full_repository_unique_recording_id_upper_bound"
                    ]
                    for source, details in kohn_portes_identity_history_audit[
                        "source_capacity"
                    ].items()
                },
                "Kohn_Portes_sources_meeting_numeric_5_plus_3_recording_ID_upper_bound": [
                    source
                    for source, details in kohn_portes_identity_history_audit[
                        "source_capacity"
                    ].items()
                    if details["recording_id_upper_bound_meets_numeric_5_plus_3"]
                ],
                "Kohn_Portes_biological_individual_semantics_verified": (
                    kohn_portes_identity_history_audit["transfer_gates"][
                        "biological_individual_semantics_verified"
                    ]
                ),
                "Kohn_Portes_biological_individual_split_authorized": (
                    kohn_portes_identity_history_audit[
                        "T5_biological_individual_split_authorized"
                    ]
                ),
                "CT1_extreme_experimental_calcium_verified": (
                    ct1_extreme_compartmentalization_audit["transfer_gates"][
                        "experimental_CT1_calcium_phenotype_verified"
                    ]
                ),
                "CT1_extreme_model_voltage_is_simulated": (
                    ct1_extreme_compartmentalization_audit["model_evidence"][
                        "output_is_simulated"
                    ]
                ),
                "CT1_extreme_experimental_voltage_available": (
                    ct1_extreme_compartmentalization_audit["transfer_gates"][
                        "experimental_CT1_allowed_response_unit_available"
                    ]
                ),
                "CT1_extreme_transfer_authorized": (
                    ct1_extreme_compartmentalization_audit[
                        "CT1_experimental_source_dynamics_transfer_authorized"
                    ]
                ),
                "CT1_voltage_audited_candidate_count": (
                    ct1_experimental_voltage_boundary_audit["candidate_summary"][
                        "audited_candidate_count"
                    ]
                ),
                "CT1_direct_experimental_voltage_candidate_found": (
                    ct1_experimental_voltage_boundary_audit["transfer_gates"][
                        "direct_CT1_experimental_voltage_phenotype_found"
                    ]
                ),
                "CT1_direct_Lo1_experimental_voltage_candidate_found": (
                    ct1_experimental_voltage_boundary_audit["transfer_gates"][
                        "direct_CT1_Lo1_experimental_voltage_found"
                    ]
                ),
                "CT1_experimental_voltage_transfer_authorized": (
                    ct1_experimental_voltage_boundary_audit[
                        "CT1_experimental_voltage_transfer_authorized"
                    ]
                ),
                "Borst_2025_parameterized_target_verified": (
                    borst_2025_temporal_filtering_audit["transfer_gates"][
                        "local_numeric_parameterized_target_verified"
                    ]
                ),
                "Borst_2025_target_is_experimental_membrane_voltage": (
                    borst_2025_temporal_filtering_audit["transfer_gates"][
                        "experimental_membrane_voltage_payload"
                    ]
                ),
                "Borst_2025_source_transfer_authorized": (
                    borst_2025_temporal_filtering_audit[
                        "source_dynamics_transfer_authorized"
                    ]
                ),
                "Pirogova_numeric_calcium_sources": (
                    pirogova_source_calcium_audit["source_coverage"][
                        "covered_required_sources"
                    ]
                ),
                "Pirogova_biological_individual_ids_available": (
                    pirogova_source_calcium_audit["transfer_gates"][
                        "biological_individual_ids_available"
                    ]
                ),
                "Pirogova_source_transfer_authorized": (
                    pirogova_source_calcium_audit["source_dynamics_transfer_authorized"]
                ),
                "MaleCNS_nine_source_type_body_sets_available": (
                    malecns_source_mapping_readiness_audit[
                        "MaleCNS_type_average_body_sets_available"
                    ]
                ),
                "MaleCNS_external_source_mapping_contract_satisfied": (
                    malecns_source_mapping_readiness_audit[
                        "external_source_mapping_contract_satisfied"
                    ]
                ),
                "MaleCNS_Tm9_unlocated_source_body_count": len(
                    malecns_source_mapping_readiness_audit["source_mapping"]["Tm9"][
                        "unlocated_body_ids"
                    ]
                ),
                "MaleCNS_Tm9_532266_existing_rule_identifiable": (
                    tm9_coordinate_identifiability_audit[
                        "Tm9_532266_coordinate_identifiable_under_existing_rule"
                    ]
                ),
                "MaleCNS_Tm9_532266_post_hoc_repair_authorized": (
                    tm9_coordinate_identifiability_audit[
                        "Tm9_532266_coordinate_repair_authorized"
                    ]
                ),
                "MaleCNS_Tm9_complete_columnar_retinotopy_after_audit": (
                    tm9_coordinate_identifiability_audit[
                        "complete_Tm9_columnar_retinotopy_available"
                    ]
                ),
                "MaleCNS_CT1_columnar_Lo1_retinotopy_available": (
                    malecns_source_mapping_readiness_audit["CT1"][
                        "columnar_Lo1_retinotopy_available"
                    ]
                ),
                "source_evidence_matrix_all_nine_complete": (
                    source_evidence_matrix["all_nine_sources_contract_complete"]
                ),
                "source_evidence_matrix_authorizes_fit": (
                    source_evidence_matrix["authorize_source_dynamics_fit"]
                ),
                "source_evidence_matrix_T4_local_millivolt_count": (
                    source_evidence_matrix["family_summary"]["T4"][
                        "numerical_membrane_voltage_count"
                    ]
                ),
                "source_evidence_matrix_T5_local_calcium_count": (
                    source_evidence_matrix["family_summary"]["T5"][
                        "local_numerical_calcium_or_deconvolved_count"
                    ]
                ),
                "source_evidence_matrix_T5_local_millivolt_count": (
                    source_evidence_matrix["family_summary"]["T5"][
                        "numerical_membrane_voltage_count"
                    ]
                ),
                "source_evidence_matrix_C3_has_any_numerical_fly_IDs": (
                    source_evidence_matrix["matrix"]["C3"][
                        "evidence_components"
                    ]["stable_biological_individual_ids_in_any_numerical_payload"]
                ),
                "source_evidence_matrix_C3_fixed_external_robustness_passed": (
                    source_evidence_matrix["matrix"]["C3"][
                        "evidence_components"
                    ]["fixed_external_robustness_gate_passed"]
                ),
                "source_type_average_mapped_sources": (
                    source_type_average_mapping_contract[
                        "sources_with_explicit_type_average_mapping"
                    ]
                ),
                "source_type_average_all_nine_mapping_complete": (
                    source_type_average_mapping_contract[
                        "all_nine_source_mapping_contracts_complete"
                    ]
                ),
                "TimingModels_CT1_type_average_dynamics_verified": (
                    timing_models_ct1_compartment_audit[
                        "CT1_type_average_dynamics_verified"
                    ]
                ),
                "TimingModels_T5_lobula_CT1_dynamics_verified": (
                    timing_models_ct1_compartment_audit[
                        "T5_lobula_CT1_dynamics_verified"
                    ]
                ),
                "TimingModels_T5_lobula_CT1_phenotype_published": (
                    timing_models_ct1_compartment_audit[
                        "T5_lobula_CT1_dynamic_phenotype_published"
                    ]
                ),
                "TimingModels_T5_lobula_CT1_numerical_payload_verified": (
                    timing_models_ct1_compartment_audit[
                        "T5_lobula_CT1_numerical_payload_verified"
                    ]
                ),
                "TimingModels_T5_CT1_transfer_authorized": (
                    timing_models_ct1_compartment_audit[
                        "T5_CT1_source_dynamics_transfer_authorized"
                    ]
                ),
                "physical_timebase_identified": reports["timebase"]["identifiability"][
                    "physical_timebase_identified"
                ],
                "strict_scoring_contract_frozen": all(
                    stage1_scoring["synthetic_controls"].values()
                ),
                "development_R1_R6_input_gates_pass": stage1_input["input_gates_pass"],
                "development_neural_evaluation_allowed": stage1_input[
                    "development_neural_evaluation_allowed"
                ],
                "development_neural_screen_performed": True,
                "development_response_gates_pass": stage1_development[
                    "development_response_gates_pass"
                ],
                "development_passing_retinal_backends": stage1_development[
                    "passing_retinal_backends"
                ],
                "development_T4_conductance_target_fraction": {
                    backend: result["T4_conductance_target_fraction"]
                    for backend, result in stage1_development["retinal_results"].items()
                },
                "development_geometry_ab_performed": True,
                "reversed_geometry_development_gates_pass": stage1_geometry_ab[
                    "development_response_gates_pass"
                ],
                "reversed_geometry_passing_retinal_backends": stage1_geometry_ab[
                    "passing_retinal_backends"
                ],
                "T5_targets_with_every_fast_and_any_delayed_source": visual_target_inputs[
                    "structural_findings"
                ]["T5_targets_with_every_fast_and_any_delayed_source"],
                "T5_target_count_in_structure_audit": visual_target_inputs["structural_findings"][
                    "T5_target_count"
                ],
                "looming_targets_with_any_T4_and_any_T5": visual_target_inputs[
                    "structural_findings"
                ]["looming_targets_with_any_T4_and_any_T5"],
                "looming_target_count_in_structure_audit": visual_target_inputs[
                    "structural_findings"
                ]["looming_target_count"],
                "existing_artifacts_rescored_under_new_contract": stage1_scoring["protocol"][
                    "existing_artifacts_rescored"
                ],
                "legacy_172_role": stage1_nested["legacy_regression"]["role"],
                "legacy_172_scientific_gate_eligible": stage1_nested["legacy_regression"][
                    "scientific_gate_eligible"
                ],
                "new_tuning_condition_count": len(
                    [
                        item
                        for item in stage1_nested["local_condition_manifests"]
                        if item["role"] == "tuning"
                    ]
                ),
                "new_calibration_condition_count": len(
                    [
                        item
                        for item in stage1_nested["local_condition_manifests"]
                        if item["role"] == "calibration"
                    ]
                ),
                "external_final_committed": stage1_nested["external_final"]["committed"],
                "target_fit_contract_frozen": True,
                "target_fit_real_fit_performed": target_fit["protocol"]["real_fit_performed"],
                "looming_mechanism_axis_coverage": {
                    name: item["coverage_fraction"]
                    for name, item in looming_mechanisms["target_contracts"].items()
                },
                "closed_loop_tuning_successes": max(
                    item["success_count"] for item in closed_loop_tuning["candidate_summaries"]
                ),
                "closed_loop_tuning_episode_count": sum(
                    len(pair) for pair in closed_loop_tuning["condition_pairs"].values()
                ),
                "closed_loop_calibration_passed": closed_loop_calibration["calibration_passed"],
                "closed_loop_calibration_successes": sum(
                    item["success"] for item in closed_loop_calibration["episodes"]
                ),
                "closed_loop_causal_controls_passed": closed_loop_controls[
                    "causal_controls_passed"
                ],
                "closed_loop_zero_action_successes": closed_loop_controls["arms"]["zero_action"][
                    "success_count"
                ],
                "closed_loop_wrong_sign_successes": closed_loop_controls["arms"][
                    "wrong_action_sign"
                ]["success_count"],
                "closed_loop_point_sample_successes": closed_loop_controls["arms"][
                    "point_sampled_R1_R6"
                ]["success_count"],
                "multi_obstacle_diagnostic_successes": closed_loop_multi["summary"][
                    "success_count"
                ],
                "multi_obstacle_diagnostic_episode_count": closed_loop_multi["summary"][
                    "episode_count"
                ],
                "multi_obstacle_mean_obstacles_passed": closed_loop_multi["summary"][
                    "mean_obstacles_passed"
                ],
                "mass_balanced_R1_R6_global_tuning_passed": r1r6_multi_tuning["tuning_passed"],
                "mass_balanced_R1_R6_global_calibration_passed": r1r6_multi_calibration[
                    "calibration_passed"
                ],
                "mass_balanced_R1_R6_local_tuning_passed": r1r6_local_tuning["tuning_passed"],
                "mass_balanced_R1_R6_local_calibration_passed": r1r6_local_calibration[
                    "calibration_passed"
                ],
                "mass_balanced_R1_R6_local_calibration_obstacles": sum(
                    item["obstacles_passed"] for item in r1r6_local_calibration["episodes"]
                ),
                "mass_balanced_R1_R6_local_controls_passed": r1r6_local_controls[
                    "causal_controls_passed"
                ],
                "structured_neural_channels_tuning_passed": neural_channels_tuning["tuning_passed"],
                "structured_neural_channels_calibration_passed": neural_channels_calibration[
                    "calibration_passed"
                ],
                "structured_neural_channels_calibration_obstacles": sum(
                    item["obstacles_passed"] for item in neural_channels_calibration["episodes"]
                ),
                "neural_channel_source_controls": {
                    name: {
                        "success_count": item["success_count"],
                        "total_obstacles_passed": item["total_obstacles_passed"],
                    }
                    for name, item in neural_channel_controls["arms"].items()
                },
                "structured_neural_real_topology_advantage": neural_topology_controls[
                    "real_topology_advantage_passed"
                ],
                "nested_neural_strict_screen": nested_neural_screen["summaries"],
                "T4_source_resolved_passing_variants": t4_source_resolved["passing_variants"],
                "T4_continuous_pair_structural_axis_passed": (
                    t4_continuous_pair_precheck["structural_axis"]
                    ["all_population_and_mirror_gates_passed"]
                ),
                "T4_continuous_pair_direction_pass_counts": {
                    name: result["direction_pass_count"]
                    for name, result in t4_continuous_pair_precheck[
                        "ordered_activity_pair"
                    ].items()
                },
                "T4_continuous_pair_polarity_pass_counts": {
                    name: result["polarity_pass_count"]
                    for name, result in t4_continuous_pair_precheck[
                        "ordered_activity_pair"
                    ].items()
                },
                "T4_continuous_pair_controls_evaluated": t4_continuous_pair_precheck[
                    "controls_evaluated"
                ],
                "T4_continuous_pair_three_condition_evaluation_performed": (
                    t4_continuous_pair_precheck["three_condition_evaluation_performed"]
                ),
                "T4_pair_lag_candidate_count": t4_pair_lag_audit["protocol"][
                    "candidate_count"
                ],
                "T4_pair_lag_direction_pass_counts": {
                    name: result["direction_pass_count"]
                    for name, result in t4_pair_lag_audit["candidates"].items()
                },
                "T4_pair_lag_polarity_pass_counts": {
                    name: result["polarity_pass_count"]
                    for name, result in t4_pair_lag_audit["candidates"].items()
                },
                "T4_pair_lag_audit_passed": t4_pair_lag_audit["lag_audit_passed"],
                "T4_normalized_correlator_best_gate_count": t4_normalized_correlator[
                    "selected_candidate"
                ]["passed_gate_count"],
                "T4_local_correlator_direction_pass_counts": {
                    name: result["T4_direction_pass_count"]
                    for name, result in t4_local_correlator_precheck["candidates"].items()
                },
                "T4_local_correlator_gate_passed": t4_local_correlator_precheck[
                    "local_correlator_gate_passed"
                ],
                "T4_source_pool_direction_pass_counts": t4_source_pool_local[
                    "direction_pass_counts"
                ],
                "T4_source_pool_maximum_bilateral_pair_count": t4_source_pool_local[
                    "maximum_bilateral_direction_pair_count"
                ],
                "T4_source_pool_authorizes_target_formula": t4_source_pool_local[
                    "authorize_new_target_formula"
                ],
                "three_hop_main_direction_bilateral_populations": three_hop_moment[
                    "bilateral_direction_populations"
                ],
                "three_hop_temporal_shuffle_control_passed": three_hop_moment[
                    "temporal_shuffle_control_passed"
                ],
                "three_hop_static_sham_control_passed": three_hop_moment[
                    "static_sham_control_passed"
                ],
                "three_hop_temporal_shuffle_attenuated_population_count": three_hop_moment[
                    "temporal_shuffle_energy_attenuation"
                ]["attenuated_population_count"],
                "three_hop_temporal_shuffle_minimum_energy_ratio": min(
                    three_hop_moment["temporal_shuffle_energy_attenuation"][
                        "shuffle_to_ordered_mean_absolute_energy_ratio_by_population"
                    ].values()
                ),
                "three_hop_strict_gates_passed": three_hop_moment["strict_three_hop_gates_passed"],
                "three_hop_temporal_consistency_passing_family_metrics": (
                    three_hop_temporal_consistency["passing_family_metrics"]
                ),
                "three_hop_temporal_consistency_minimum_ratios": {
                    family: {
                        metric: result["minimum_shuffle_to_ordered_ratio"]
                        for metric, result in family_result["metrics"].items()
                    }
                    for family, family_result in three_hop_temporal_consistency[
                        "families"
                    ].items()
                },
                "three_hop_temporal_consistency_gate_passed": (
                    three_hop_temporal_consistency["temporal_consistency_gate_passed"]
                ),
                "upstream_ordered_latency_passing_populations": {
                    family: result["passing_population_count"]
                    for family, result in upstream_latency_audit["modes"]["ordered"].items()
                },
                "upstream_shuffle_latency_passing_populations": {
                    family: result["passing_population_count"]
                    for family, result in upstream_latency_audit["modes"][
                        "temporal_shuffle"
                    ].items()
                },
                "upstream_static_latency_passing_populations": {
                    family: result["passing_population_count"]
                    for family, result in upstream_latency_audit["modes"]["static_sham"].items()
                },
                "upstream_source_latency_temporal_identifiability_passed": (
                    upstream_latency_audit["source_latency_temporal_identifiability_passed"]
                ),
                "upstream_paired_residual_passing_populations": {
                    family: result["passing_population_count"]
                    for family, result in upstream_latency_audit["paired_static_residual"].items()
                },
                "upstream_paired_residual_minimum_shuffle_ratios": {
                    family: min(
                        channel["shuffle_to_ordered_residual_energy_ratio"]
                        for population in result["populations"].values()
                        for channel in population["channel_residual_energy"].values()
                    )
                    for family, result in upstream_latency_audit["paired_static_residual"].items()
                },
                "upstream_paired_residual_temporal_identifiability_passed": (
                    upstream_latency_audit[
                        "paired_static_residual_temporal_identifiability_passed"
                    ]
                ),
                "synapse_spatial_selected_row_count": synapse_spatial_audit["protocol"][
                    "source_file"
                ]["selected_row_count"],
                "synapse_spatial_complete_target_counts": {
                    family: result["complete_target_count"]
                    for family, result in synapse_spatial_audit["families"].items()
                },
                "synapse_spatial_aggregate_weight_matches": {
                    family: result["synapse_rows_match_aggregate_graph_weight"]
                    for family, result in synapse_spatial_audit["families"].items()
                },
                "synapse_spatial_structure_gate_passed": synapse_spatial_audit[
                    "strict_synapse_spatial_structure_gate_passed"
                ],
                "synapse_axis_held_out_T4_accuracy": synapse_axis_calibration[
                    "held_out_T4"
                ]["accuracy"],
                "synapse_axis_held_out_T4_median_angle_degrees": synapse_axis_calibration[
                    "held_out_T4"
                ]["median_angle_error_degrees"],
                "synapse_axis_T4_calibration_passed": synapse_axis_calibration[
                    "T4_synapse_axis_calibration_passed"
                ],
                "T4_synapse_crossfit_axis_accuracy": t4_synapse_crossfit_axis_audit[
                    "out_of_fold_overall"
                ]["accuracy"],
                "T4_synapse_crossfit_axis_median_angle_degrees": (
                    t4_synapse_crossfit_axis_audit["out_of_fold_overall"]
                    ["median_angle_error_degrees"]
                ),
                "T4_synapse_crossfit_axis_passed": t4_synapse_crossfit_axis_audit[
                    "T4_synapse_crossfit_axis_passed"
                ],
                "synapse_axis_zero_shot_T5_accuracy": synapse_axis_calibration[
                    "zero_shot_T5"
                ]["accuracy"],
                "synapse_axis_zero_shot_T5_mapping_passed": synapse_axis_calibration[
                    "zero_shot_T5_mapping_passed"
                ],
                "synapse_axis_T5_mapping_application_authorized": synapse_axis_calibration[
                    "authorize_T5_mapping_application"
                ],
                "T4_synapse_correlator_fixed_population_denominator": (
                    t4_synapse_correlator_precheck["fixed_T4_population_denominator"]
                ),
                "T4_synapse_correlator_valid_target_count": (
                    t4_synapse_correlator_precheck["diagnostic_valid_target_count"]
                ),
                "T4_synapse_correlator_direction_pass_counts": [
                    item["direction_pass_count"]
                    for item in t4_synapse_correlator_precheck["candidates"]
                ],
                "T4_synapse_correlator_polarity_pass_counts": [
                    item["polarity_pass_count"]
                    for item in t4_synapse_correlator_precheck["candidates"]
                ],
                "T4_synapse_correlator_bilateral_direction_subtypes": [
                    item["bilateral_direction_subtypes"]
                    for item in t4_synapse_correlator_precheck["candidates"]
                ],
                "T4_synapse_correlator_main_gate_passed": (
                    t4_synapse_correlator_precheck["main_gate_passed"]
                ),
                "T4_synapse_correlator_controls_evaluated": (
                    t4_synapse_correlator_precheck["controls_evaluated"]
                ),
                "T4_synapse_correlator_three_condition_evaluation_performed": (
                    t4_synapse_correlator_precheck[
                        "three_condition_evaluation_performed"
                    ]
                ),
                "T4_synapse_antisymmetric_direction_pass_counts": [
                    item["direction_pass_count"]
                    for item in t4_synapse_antisymmetric_precheck["ordered_candidates"]
                ],
                "T4_synapse_antisymmetric_polarity_pass_counts": [
                    item["polarity_pass_count"]
                    for item in t4_synapse_antisymmetric_precheck["ordered_candidates"]
                ],
                "T4_synapse_antisymmetric_bilateral_direction_subtypes": [
                    item["bilateral_direction_subtypes"]
                    for item in t4_synapse_antisymmetric_precheck["ordered_candidates"]
                ],
                "T4_synapse_antisymmetric_controls_evaluated": (
                    t4_synapse_antisymmetric_precheck["controls_evaluated"]
                ),
                "T4_synapse_antisymmetric_three_condition_evaluation_performed": (
                    t4_synapse_antisymmetric_precheck[
                        "three_condition_evaluation_performed"
                    ]
                ),
                "T4_synapse_crossfit_direction_pass_counts": [
                    item["direction_pass_count"]
                    for item in t4_synapse_crossfit_precheck["ordered_candidates"]
                ],
                "T4_synapse_crossfit_polarity_pass_counts": [
                    item["polarity_pass_count"]
                    for item in t4_synapse_crossfit_precheck["ordered_candidates"]
                ],
                "T4_synapse_crossfit_controls_evaluated": t4_synapse_crossfit_precheck[
                    "controls_evaluated"
                ],
                "T4_synapse_crossfit_candidate_passed": t4_synapse_crossfit_precheck[
                    "candidate_passed"
                ],
                "T4_crossfit_sequence_maximum_direction_pass_count": (
                    t4_crossfit_sequence_identifiability[
                        "maximum_ordered_direction_pass_count"
                    ]
                ),
                "T4_crossfit_sequence_ordered_bilateral_subtypes": (
                    t4_crossfit_sequence_identifiability[
                        "ordered_bilateral_direction_subtypes_by_reduction"
                    ]
                ),
                "T4_crossfit_sequence_shuffle_signed_mean_bilateral_subtypes": (
                    t4_crossfit_sequence_identifiability["mode_reduction_results"]
                    ["temporal_shuffle"]["signed_mean"][
                        "bilateral_direction_subtypes"
                    ]
                ),
                "T4_crossfit_sequence_authorizes_functional_candidate": (
                    t4_crossfit_sequence_identifiability[
                        "authorize_new_functional_candidate"
                    ]
                ),
                "T4_balanced_retina_ordered_direction_pass_count": (
                    t4_crossfit_retinal_symmetry_audit[
                        "maximum_balanced_ordered_direction_pass_count"
                    ]
                ),
                "T4_retinal_sampling_imbalance_explains_direction_failure": (
                    t4_crossfit_retinal_symmetry_audit[
                        "retinal_sampling_imbalance_explains_direction_failure"
                    ]
                ),
                "T4_synapse_centered_direction_pass_counts": [
                    item["direction_pass_count"]
                    for item in t4_synapse_centered_precheck["ordered_candidates"]
                ],
                "T4_synapse_centered_polarity_pass_counts": [
                    item["polarity_pass_count"]
                    for item in t4_synapse_centered_precheck["ordered_candidates"]
                ],
                "T4_synapse_centered_bilateral_direction_subtypes": [
                    item["bilateral_direction_subtypes"]
                    for item in t4_synapse_centered_precheck["ordered_candidates"]
                ],
                "T4_synapse_centered_controls_evaluated": t4_synapse_centered_precheck[
                    "controls_evaluated"
                ],
                "T4_synapse_centered_three_condition_evaluation_performed": (
                    t4_synapse_centered_precheck["three_condition_evaluation_performed"]
                ),
                "T4_synapse_microstep_shuffle_residual_ratio": (
                    t4_synapse_microstep_precheck["temporal_identifiability"][
                        "shuffle_to_ordered_residual_energy_ratio"
                    ]
                ),
                "T4_synapse_microstep_static_energy_ratio": (
                    t4_synapse_microstep_precheck["temporal_identifiability"][
                        "static_to_ordered_energy_ratio"
                    ]
                ),
                "T4_synapse_microstep_temporal_identifiability_passed": (
                    t4_synapse_microstep_precheck["temporal_identifiability"]["passed"]
                ),
                "T4_synapse_microstep_direction_scoring_performed": (
                    t4_synapse_microstep_precheck["direction_scoring_performed"]
                ),
                "source_type_temporal_passing_counts": {
                    family: {
                        "passed": item["passing_source_population_count"],
                        "denominator": item["source_population_denominator"],
                    }
                    for family, item in source_type_temporal_identifiability[
                        "families"
                    ].items()
                },
                "source_type_temporal_identifiability_passed": (
                    source_type_temporal_identifiability[
                        "source_type_temporal_identifiability_passed"
                    ]
                ),
                "T4_source_dynamics_transfer_fields_available": (
                    t4_source_dynamics_transfer_audit[
                        "required_transfer_fields_available"
                    ]
                ),
                "T4_source_dynamics_transfer_authorized": (
                    t4_source_dynamics_transfer_audit["source_dynamics_transfer_authorized"]
                ),
                "T4_source_dynamics_next_candidate_authorized": (
                    t4_source_dynamics_transfer_audit["next_candidate_authorized"]
                ),
                "unified_model_package_files_verified": unified_model_package_audit[
                    "files_verified"
                ],
                "unified_model_T4_target_parameters_available": (
                    unified_model_package_audit["T4_target_model_parameters_available"]
                ),
                "unified_model_MaleCNS_source_transfer_authorized": (
                    unified_model_package_audit[
                        "MaleCNS_source_kernel_transfer_authorized"
                    ]
                ),
                "fig3_ON_source_specific_kernels_ready": fig3_source_kernel_audit[
                    "ON_source_specific_kernels_ready"
                ],
                "fig3_prior_two_pool_kernel_authorized": fig3_source_kernel_audit[
                    "prior_two_pool_kernel_authorized"
                ],
                "fig3_source_specific_kernel_candidate_authorized": (
                    fig3_source_kernel_audit["source_specific_kernel_candidate_authorized"]
                ),
                "fig3_source_kernel_cross_cell_robustness_passed": (
                    fig3_source_kernel_robustness[
                        "all_source_kernel_robustness_gates_passed"
                    ]
                ),
                "Arenz_source_filter_parameters_verified": arenz_source_dynamics_audit[
                    "Arenz_filter_parameters_verified"
                ],
                "Arenz_current_source_coverage_fraction": arenz_source_dynamics_audit[
                    "current_v7_source_contract"
                ]["coverage_fraction"],
                "Arenz_missing_current_sources": arenz_source_dynamics_audit[
                    "current_v7_source_contract"
                ]["missing_from_Arenz"],
                "Arenz_source_filter_candidate_authorized": arenz_source_dynamics_audit[
                    "source_filter_candidate_authorized"
                ],
                "Arenz_T5_source_coverage_fraction": arenz_t5_source_dynamics_audit[
                    "T5_source_contract"
                ]["coverage_fraction"],
                "Arenz_T5_raw_filter_contract_complete": arenz_t5_source_dynamics_audit[
                    "raw_T5_calcium_filter_contract_complete"
                ],
                "Arenz_T5_deconvolved_filter_contract_complete": (
                    arenz_t5_source_dynamics_audit[
                        "deconvolved_T5_filter_contract_complete"
                    ]
                ),
                "Arenz_T5_Tm9_deconvolved_R2": arenz_t5_source_dynamics_audit[
                    "paper_model"
                ]["parameters"]["Tm9"]["deconvolved"]["R2"],
                "Arenz_T5_source_filter_transfer_authorized": (
                    arenz_t5_source_dynamics_audit[
                        "T5_source_filter_transfer_authorized"
                    ]
                ),
                "C3_STRF_numerical_data_verified": c3_strf_source_dynamics_audit[
                    "C3_STRF_numerical_data_verified"
                ],
                "C3_direct_temporal_measurement_available": (
                    c3_strf_source_dynamics_audit["combined_source_contract"][
                        "C3_direct_temporal_measurement_available"
                    ]
                ),
                "combined_source_temporal_evidence_complete": (
                    c3_strf_source_dynamics_audit["combined_source_contract"][
                        "all_source_types_have_some_direct_temporal_evidence"
                    ]
                ),
                "combined_source_parameterization_transferable": (
                    c3_strf_source_dynamics_audit["combined_source_contract"][
                        "all_source_types_share_one_transferable_parameterization"
                    ]
                ),
                "C3_source_filter_candidate_authorized": (
                    c3_strf_source_dynamics_audit[
                        "C3_source_filter_candidate_authorized"
                    ]
                ),
                "Fig1_source_spatial_tables_verified": (
                    fig1_source_temporal_readiness_audit["observations"][
                        "official_Fig1_workbooks_verified"
                    ]
                ),
                "Fig1_source_average_tables_have_time_axis": (
                    fig1_source_temporal_readiness_audit["observations"][
                        "source_type_average_tables_have_time_axis"
                    ]
                ),
                "Fig1_source_object_payload_safely_inspected": (
                    fig1_source_temporal_readiness_audit["edmond_object_payload"][
                        "safely_inspected"
                    ]
                ),
                "Fig1_source_temporal_kernel_transfer_authorized": (
                    fig1_source_temporal_readiness_audit[
                        "source_temporal_kernel_transfer_authorized"
                    ]
                ),
                "C3_analytic_filter_band_pass_preferred": (
                    c3_analytic_filter_precheck[
                        "band_pass_BIC_below_low_pass_BIC"
                    ]
                ),
                "C3_analytic_filter_minimum_LOFO_correlation": (
                    c3_analytic_filter_precheck["model_families"]["band_pass"][
                        "cross_validation"
                    ]["minimum_held_out_correlation"]
                ),
                "C3_analytic_filter_family_precheck_passed": (
                    c3_analytic_filter_precheck[
                        "C3_analytic_filter_family_precheck_passed"
                    ]
                ),
                "C3_analytic_filter_transfer_authorized": (
                    c3_analytic_filter_precheck[
                        "C3_analytic_filter_transfer_authorized"
                    ]
                ),
                "TimingModels_covered_source_filters_stable": (
                    timing_models_source_filter_audit[
                        "covered_source_filter_stability_verified"
                    ]
                ),
                "TimingModels_current_source_coverage_fraction": (
                    timing_models_source_filter_audit["current_v7_source_contract"][
                        "coverage_fraction"
                    ]
                ),
                "TimingModels_missing_current_sources": (
                    timing_models_source_filter_audit["current_v7_source_contract"][
                        "missing_sources"
                    ]
                ),
                "TimingModels_complete_source_candidate_authorized": (
                    timing_models_source_filter_audit[
                        "complete_source_filter_candidate_authorized"
                    ]
                ),
                "FlyVis_C3_pretrained_model_count": flyvis_c3_time_constant_audit[
                    "training_contract"
                ]["model_count"],
                "FlyVis_C3_median_time_constant_seconds": (
                    flyvis_c3_time_constant_audit["source_time_constants"]["C3"][
                        "median_seconds"
                    ]
                ),
                "FlyVis_C3_models_at_or_below_solver_dt": (
                    flyvis_c3_time_constant_audit["source_time_constants"]["C3"][
                        "models_at_or_below_solver_dt"
                    ]
                ),
                "FlyVis_C3_time_constant_transfer_authorized": (
                    flyvis_c3_time_constant_audit[
                        "C3_time_constant_transfer_authorized"
                    ]
                ),
                "FlyVis_visual_source_coverage_complete": all(
                    not item["missing"]
                    for item in flyvis_visual_source_time_constants[
                        "source_coverage"
                    ].values()
                ),
                "FlyVis_visual_source_time_constants_transferable": (
                    flyvis_visual_source_time_constants[
                        "FlyVis_visual_source_time_constants_transferable"
                    ]
                ),
                "FlyVis_C3_effective_cross_dt_minimum_correlation": (
                    flyvis_c3_effective_dynamics_audit["cross_dt_summary"]["minimum"]
                ),
                "FlyVis_C3_effective_LOMO_minimum_by_dt": {
                    name: item["unit_shape_leave_one_model_out_summary"]["minimum"]
                    for name, item in flyvis_c3_effective_dynamics_audit[
                        "ensemble_stability"
                    ].items()
                },
                "FlyVis_C3_effective_external_maximum_correlation": max(
                    item["ensemble_equal_shape_correlation"]
                    for by_tau in flyvis_c3_effective_dynamics_audit[
                        "external_C3_flash_consistency"
                    ].values()
                    for item in by_tau.values()
                ),
                "FlyVis_C3_effective_dynamics_transfer_authorized": (
                    flyvis_c3_effective_dynamics_audit[
                        "FlyVis_C3_effective_dynamics_transfer_authorized"
                    ]
                ),
                "C3_measured_filter_cross_deconvolution_stable": (
                    c3_measured_filter_robustness[
                        "cross_deconvolution_stability_passed"
                    ]
                ),
                "C3_measured_filter_minimum_LOFO_by_assumption": {
                    name: item["minimum_held_out_correlation"]
                    for name, item in c3_measured_filter_robustness[
                        "leave_one_fly_out_by_assumption"
                    ].items()
                },
                "C3_measured_filter_type_shared_kernel_authorized": (
                    c3_measured_filter_robustness[
                        "C3_type_shared_kernel_authorized"
                    ]
                ),
                "C3_STRF_flash_external_fly_count": c3_strf_flash_transfer[
                    "cohorts"
                ]["external_flash_fly_count"],
                "C3_STRF_flash_mean_correlation": c3_strf_flash_transfer[
                    "external_validation"
                ]["mean_waveform_correlation"],
                "C3_STRF_flash_bootstrap_p05": c3_strf_flash_transfer[
                    "external_validation"
                ]["bootstrap"]["correlation_p05"],
                "C3_STRF_flash_transfer_passed": c3_strf_flash_transfer[
                    "C3_STRF_to_flash_transfer_passed"
                ],
                "public_T4_models_C3_specific_parameters_available": (
                    public_t4_model_source_coverage_audit["transfer_gates"][
                        "C3_specific_parameters_available"
                    ]
                ),
                "public_T4_models_complete_source_transfer_authorized": (
                    public_t4_model_source_coverage_audit[
                        "complete_source_dynamics_transfer_authorized"
                    ]
                ),
                "three_hop_independent_channel_coverage_passed": (
                    three_hop_source_coverage["independent_fast_delayed_coverage_gate_passed"]
                ),
                "three_hop_scalar_reichardt_authorized": three_hop_source_coverage[
                    "authorize_scalar_reichardt"
                ],
                "four_hop_scalar_ordered_direction_pass_counts": {
                    name: result["modes"]["ordered"]["direction_pass_count"]
                    for name, result in four_hop_scalar_precheck["candidates"].items()
                },
                "four_hop_scalar_shuffle_bilateral_subtypes": {
                    name: result["modes"]["temporal_shuffle"]["bilateral_direction_subtypes"]
                    for name, result in four_hop_scalar_precheck["candidates"].items()
                },
                "four_hop_scalar_gate_passed": four_hop_scalar_precheck[
                    "four_hop_scalar_gate_passed"
                ],
                "LC4_position_speed_precheck_passed": lc4_position_speed_precheck[
                    "LC4_position_speed_precheck_passed"
                ],
                "LC4_position_speed_valid_fraction": {
                    side: result["valid_cell_fraction"]
                    for side, result in lc4_position_speed_precheck["per_side"].items()
                },
                "LC4_position_speed_positive_slope_fraction": {
                    side: result["positive_slope_fraction"]
                    for side, result in lc4_position_speed_precheck["per_side"].items()
                },
                "LC4_position_speed_primary_readout": lc4_position_speed_precheck[
                    "primary_precheck_readout"
                ],
                "LC4_input_speed_valid_fraction": {
                    side: result["valid_cell_fraction"]
                    for side, result in lc4_input_speed_precheck["per_side"].items()
                },
                "LC4_input_speed_positive_slope_fraction": {
                    side: result["positive_slope_fraction_all_cells"]
                    for side, result in lc4_input_speed_precheck["per_side"].items()
                },
                "LC4_input_speed_mirror_passed": lc4_input_speed_precheck["mirror"][
                    lc4_input_speed_precheck["primary_precheck_readout"]
                ]["passed"],
                "LC4_input_speed_precheck_passed": lc4_input_speed_precheck[
                    "LC4_input_speed_precheck_passed"
                ],
                "LC4_excitatory_input_derivative_valid_counts": {
                    side: result["valid_cell_count"]
                    for side, result in lc4_input_speed_precheck["readouts"]
                    ["peak_positive_excitatory_input_derivative"].items()
                },
                "LC4_inhibitory_input_derivative_monotonic_fractions": {
                    side: result["monotonic_cell_fraction"]
                    for side, result in lc4_input_speed_precheck["readouts"]
                    ["peak_positive_inhibitory_input_derivative"].items()
                },
                "T5_lamina_split_polarity_passing_condition_counts": {
                    name: result["polarity"]["passing_condition_count"]
                    for name, result in t5_lamina_split["population_consistency"].items()
                },
                "T5_lamina_split_direction_passing_condition_counts": {
                    name: result["direction"]["passing_condition_count"]
                    for name, result in t5_lamina_split["population_consistency"].items()
                },
                "T5_lamina_split_mirror_gate_passed": t5_lamina_split[
                    "mirror_gate_passed"
                ],
                "T5_lamina_split_strict_gates_passed": t5_lamina_split[
                    "strict_T5_gates_passed"
                ],
                "T5_lamina_scalar_joint_source_fraction": t5_lamina_scalar_precheck[
                    "source_coverage"
                ]["joint_present_fraction"],
                "T5_lamina_scalar_ordered_direction_pass_count": (
                    t5_lamina_scalar_precheck["modes"]["ordered"]
                    ["fast_delayed_correlation"]["direction_pass_count"]
                ),
                "T5_lamina_scalar_ordered_polarity_pass_count": (
                    t5_lamina_scalar_precheck["modes"]["ordered"]
                    ["fast_delayed_correlation"]["polarity_pass_count"]
                ),
                "T5_lamina_scalar_shuffle_bilateral_subtypes": (
                    t5_lamina_scalar_precheck["modes"]["temporal_shuffle"]
                    ["fast_delayed_correlation"]["bilateral_direction_subtypes"]
                ),
                "T5_lamina_scalar_precheck_passed": t5_lamina_scalar_precheck[
                    "candidate_passed"
                ],
                "T5_source_axis_passing_populations": [
                    name
                    for name, result in t5_source_axis_audit["population_summaries"].items()
                    if result["passed"]
                ],
                "T5_source_axis_mirror_passing_subtypes": [
                    name
                    for name, result in t5_source_axis_audit["population_mirror"].items()
                    if result["passed"]
                ],
                "T5_source_axis_strict_gate_passed": t5_source_axis_audit[
                    "strict_source_axis_gate_passed"
                ],
                "T5_source_axis_dynamic_candidate_authorized": t5_source_axis_audit[
                    "authorize_dynamic_axis_candidate"
                ],
                "T5_zero_shot_T4_axis_calibration_accuracy": t5_source_axis_audit[
                    "independent_T4_axis_calibration"
                ]["zero_shot_T5_accuracy"],
                "T5_source_axis_transform_application_authorized": t5_source_axis_audit[
                    "independent_T4_axis_calibration"
                ]["transform_application_authorized"],
                "T5_source_pair_joint_source_fraction": t5_source_pair_precheck[
                    "source_coverage"
                ]["joint_present_fraction"],
                "T5_source_pair_direction_pass_counts": [
                    item["direction_pass_count"]
                    for item in t5_source_pair_precheck["candidates"]
                ],
                "T5_source_pair_polarity_pass_counts": [
                    item["polarity_pass_count"]
                    for item in t5_source_pair_precheck["candidates"]
                ],
                "T5_source_pair_bilateral_direction_subtypes": [
                    item["bilateral_direction_subtypes"]
                    for item in t5_source_pair_precheck["candidates"]
                ],
                "T5_source_pair_precheck_passed": t5_source_pair_precheck[
                    "ordered_precheck_passed"
                ],
                "T5_source_pair_controls_evaluated": t5_source_pair_precheck[
                    "controls_evaluated"
                ],
                "T5_source_pair_three_condition_evaluation_performed": (
                    t5_source_pair_precheck["three_condition_evaluation_performed"]
                ),
                "LPLC_mechanism_repair_authorized_by_T4_T5_gate": (
                    t5_source_pair_precheck["advance_to_LPLC_mechanism_repair"]
                ),
                "T5_continuous_moment_structural_axis_passed": (
                    t5_continuous_moment_precheck["structural_axis"]
                    ["all_population_and_mirror_gates_passed"]
                ),
                "T5_continuous_moment_direction_pass_counts": {
                    name: result["direction_pass_count"]
                    for name, result in t5_continuous_moment_precheck[
                        "ordered_activity_moment"
                    ].items()
                },
                "T5_continuous_moment_controls_evaluated": (
                    t5_continuous_moment_precheck["controls_evaluated"]
                ),
                "T5_continuous_moment_three_condition_evaluation_performed": (
                    t5_continuous_moment_precheck["three_condition_evaluation_performed"]
                ),
                "LPLC1_near_collision_direct_input_fraction": {
                    side: result["targets_with_direct_T4_T5_fraction"]
                    for side, result in lplc1_near_collision_precheck[
                        "direct_source_coverage"
                    ].items()
                },
                "LPLC1_near_collision_response_pass_counts": {
                    name: sum(
                        lplc1_near_collision_precheck["per_side"][side][name]["passed"]
                        for side in "LR"
                    )
                    for name in (
                        "near_vs_miss",
                        "approach_vs_recede",
                        "stationary_vs_rotating_background",
                    )
                },
                "LPLC1_near_collision_stimulus_geometry_passed": (
                    lplc1_near_collision_precheck["stimulus_geometry_gate_passed"]
                ),
                "LPLC1_near_collision_mirror_passed": lplc1_near_collision_precheck[
                    "mirror_gate_passed"
                ],
                "LPLC1_near_collision_precheck_passed": lplc1_near_collision_precheck[
                    "LPLC1_near_collision_precheck_passed"
                ],
                "LPLC1_input_group_target_fractions": {
                    side: {
                        name: result["targets_with_input_fraction"]
                        for name, result in side_result["groups"].items()
                    }
                    for side, side_result in lplc1_input_structure["per_side"].items()
                },
                "LPLC1_input_group_located_weight_fractions": {
                    side: {
                        name: result["located_source_weight_fraction"]
                        for name, result in side_result["groups"].items()
                    }
                    for side, side_result in lplc1_input_structure["per_side"].items()
                },
                "LPLC1_input_structure_gate_passed": lplc1_input_structure[
                    "strict_input_structure_gate_passed"
                ],
                "LPLC1_spatial_inhibition_mechanism_authorized": lplc1_input_structure[
                    "authorize_spatial_inhibition_mechanism"
                ],
                "T5_spatial_order_b_d_reachable_fraction": t5_spatial_order["b_d_reachability"][
                    "reachable_fraction"
                ],
                "local_edge_precheck_direction_pass_count": t4t5_local_edge_precheck["summary"][
                    "direction_pass_count"
                ],
                "local_edge_precheck_polarity_pass_count": t4t5_local_edge_precheck["summary"][
                    "polarity_pass_count"
                ],
                "local_edge_precheck_expand_to_three_conditions": t4t5_local_edge_precheck[
                    "summary"
                ]["expand_to_three_conditions"],
                "local_edge_backend_direction_pass_counts": {
                    name: result["direction_pass_count"]
                    for name, result in t4t5_local_edge_backends["candidates"].items()
                },
                "local_edge_backend_polarity_pass_counts": {
                    name: result["polarity_pass_count"]
                    for name, result in t4t5_local_edge_backends["candidates"].items()
                },
                "existing_backend_reuse_gate_passed": t4t5_local_edge_backends[
                    "existing_backend_reuse_gate_passed"
                ],
                "typed_LPLC_LC4_gates_passed": lplc_typed_screen["typed_lplc_lc4_gates_passed"],
                "LPLC2_radial_opponency_gates_passed": lplc2_radial_opponency[
                    "radial_opponency_mechanism_gates_passed"
                ],
                "LPLC2_radial_fixed_target_denominators": lplc2_radial_opponency["protocol"][
                    "fixed_target_denominators"
                ],
                "LPLC2_radial_direction_labels_used": lplc2_radial_opponency["protocol"][
                    "direction_labels_used_by_mechanism"
                ],
                "LPLC2_radial_may_authorize_strict_visual_gate": lplc2_radial_opponency["protocol"][
                    "may_authorize_strict_visual_gate"
                ],
                "LPLC2_radial_temporal_identifiability_passed": lplc2_radial_opponency[
                    "temporal_identifiability"
                ]["passed"],
                "LPLC2_radial_all_condition_separated_fraction": {
                    side: result["coverage"]["all_condition_separated_fraction"]
                    for side, result in lplc2_radial_opponency["temporal_identifiability"][
                        "population_consistency"
                    ].items()
                },
                "radial_layer_localization_passing_populations": lplc2_radial_opponency[
                    "layer_localization"
                ]["passing_populations"],
                "mapped_R1_R6_all_condition_separated_fraction": lplc2_radial_opponency[
                    "layer_localization"
                ]["population_consistency"]["mapped_R1-R6"]["all_condition_separated_fraction"],
                "visual_graph_unselected_R1_R6_count": lplc2_radial_opponency["layer_localization"][
                    "selected_receptor_edge_audit"
                ]["unselected_graph_R1_R6_count"],
                "selected_R1_R6_renormalization_AB_gate_restored": lplc2_radial_opponency[
                    "layer_localization"
                ]["selected_receptor_edge_audit"]["renormalization_AB_gate_restored"],
                "full_retina_projection_AB_passing_populations": lplc2_radial_opponency[
                    "input_projection_ab"
                ]["passing_populations"],
                "full_retina_mapping_is_exact_mirror": lplc2_radial_opponency[
                    "input_projection_ab"
                ]["full_mapping_is_exact_mirror"],
                "full_retina_lamina_all_condition_separated_fraction": {
                    name: lplc2_radial_opponency["input_projection_ab"]["population_consistency"][
                        name
                    ]["all_condition_separated_fraction"]
                    for name in ("L1", "L2", "L3")
                },
                "unmatched_noise_layer_localization_confounded": lplc2_radial_opponency[
                    "paired_noise_control"
                ]["unmatched_noise_layer_localization_confounded"],
                "matched_noise_passing_populations": lplc2_radial_opponency["paired_noise_control"][
                    "controls"
                ]["matched_noise"]["passing_populations"],
                "matched_noise_mapped_R1_R6_all_condition_separated_fraction": (
                    lplc2_radial_opponency["paired_noise_control"]["controls"]["matched_noise"][
                        "population_consistency"
                    ]["mapped_R1-R6"]["all_condition_separated_fraction"]
                ),
                "LPLC2_position_observability_gate_passed": lplc2_position_coverage[
                    "observability_gate_passed"
                ],
                "LPLC2_position_observability_all_condition_fraction": lplc2_position_coverage[
                    "observability_consistency"
                ]["all_condition_separated_fraction"],
                "LPLC2_position_coverage_gates_passed": lplc2_position_coverage[
                    "LPLC2_position_coverage_gates_passed"
                ],
            },
            "missing": [
                "complete T4/T5 direction and ON/OFF response gates",
                "LPLC1/LPLC2/LC4 looming and collision response gates",
                "an independent T5 label map suitable for model scoring",
                "a calibrated physical visual/neural timebase",
                "an externally committed and independently custodied final condition",
                "multi-obstacle and dynamic OOD closed-loop validation",
            ],
        },
        {
            "item": 2,
            "deliverable": "EPG/PEN/PEG heading, occlusion memory and FC2/PFL comparison",
            "status": "partially_validated_not_stage_authorized",
            "evidence": [
                config["evidence"]["heading_ring"],
                config["evidence"]["fc2_pfl_dna"],
            ],
            "observations": {
                "stage1_pass": stage1_pass,
                "single_obstacle_closed_loop_calibration_passed": closed_loop_calibration[
                    "calibration_passed"
                ],
                "closed_loop_navigation_release_authorized": closed_loop_calibration[
                    "advance_to_navigation_release"
                ],
                "multi_obstacle_diagnostic_passed": closed_loop_multi["diagnostic_passed"],
                "R1_R6_local_multi_obstacle_upper_bound_passed": r1r6_local_calibration[
                    "calibration_passed"
                ],
                "R1_R6_upper_bound_authorizes_neural_release": r1r6_local_calibration[
                    "advance_to_navigation_release"
                ],
                "structured_neural_channels_calibration_passed": neural_channels_calibration[
                    "calibration_passed"
                ],
                "structured_neural_channels_release_authorized": neural_channels_calibration[
                    "advance_to_navigation_release"
                ],
                "EPG_PEN_PEG_heading_assays_passed": heading_ring["heading_assays"]["passed"],
                "heading_tuning_obstacles": heading_ring["navigation_arms"]["neural_heading"][
                    "tuning_obstacles_passed"
                ],
                "heading_fresh_calibration_obstacles": heading_ring["navigation_arms"][
                    "neural_heading"
                ]["calibration_obstacles_passed"],
                "heading_failure_attributed_to_visual_readout": heading_ring[
                    "calibration_failure_attribution"
                ]["diagnosis"]
                == "frozen_neural_visual_readout_generalization_gap",
            },
        },
        {
            "item": 3,
            "deliverable": "DNp20, DNa01/DNa02 and PFL3-to-DNa readout comparison",
            "status": "partially_validated_not_stage_authorized",
            "evidence": [
                config["evidence"]["fc2_pfl_dna"],
                config["evidence"]["descending_path_audit"],
                config["evidence"]["descending_propagation_precheck"],
            ],
            "observations": {
                "stage1_pass": stage1_pass,
                "turn_termination_passed": fc2_pfl_dna["turn_termination_assay"]["passed"],
                "readout_action_equivalence": fc2_pfl_dna["readout_action_equivalence"],
                "PFL3_to_DNa02_edges": fc2_pfl_dna["malecns_structure"]["edges"]["PFL3->DNa02"][
                    "edge_count"
                ],
                "PFL3_to_DNa01_edges": fc2_pfl_dna["malecns_structure"]["edges"]["PFL3->DNa01"][
                    "edge_count"
                ],
                "bilateral_shortest_paths_supported": descending_paths[
                    "structural_bilateral_path_gate_passed"
                ],
                "shortest_hops_by_target": {
                    target_type: {
                        item["target"]["soma_side"]: item["shortest_hops"]
                        for item in records
                    }
                    for target_type, records in descending_paths[
                        "target_shortest_paths"
                    ].items()
                },
                "action_equivalence_is_formula_level_only": descending_paths[
                    "algebraic_action_readout_boundary"
                ]["equivalence_is_formula_level_not_neural_state_level"],
                "real_neural_state_readout_performed": descending_paths[
                    "algebraic_action_readout_boundary"
                ]["real_neural_state_readout_performed"],
                "fixed_input_propagation_precheck_evaluated": True,
                "functional_neural_readout_validated": descending_propagation[
                    "functional_neural_readout_validated"
                ],
                "real_state_precheck_passing_populations": descending_propagation[
                    "passing_target_populations"
                ],
                "real_state_precheck_failed_populations": descending_propagation[
                    "failed_target_populations"
                ],
                "all_descending_readouts_precheck_passed": descending_propagation[
                    "all_descending_readouts_precheck_passed"
                ],
                "fresh_calibration_passed": fc2_pfl_dna["calibration_passed"],
            },
        },
        {
            "item": 4,
            "deliverable": (
                "separate self-motion, looming, near-collision, heading and target signals"
            ),
            "status": "partially_validated_not_stage_authorized",
            "evidence": [
                config["evidence"]["neural_episode_controls"],
                config["evidence"]["fc2_pfl_dna"],
            ],
            "observations": {
                "stage1_pass": stage1_pass,
                "T4_T5_spatial_is_causal": neural_episode_controls["causal_gates"][
                    "T4_T5_spatial_is_causal"
                ],
                "LPLC_LC_joint_is_causal": neural_episode_controls["causal_gates"][
                    "all_LPLC_LC_is_causal"
                ],
                "heading_is_causal": neural_episode_controls["causal_gates"]["heading_is_causal"],
                "FC2_PFL_DNa_controls_passed": fc2_pfl_dna["causal_controls_passed"],
                "danger_throttle_tuning_passed": danger_throttle["tuning_passed"],
                "R1_R6_visual_corridor_tuning_passed": visual_corridor_goal["tuning_passed"],
                "six_bin_neural_corridor_CV_passed": neural_corridor["cross_validation_passed"],
                "local_column_neural_corridor_CV_passed": local_column_corridor[
                    "cross_validation_passed"
                ],
                "best_local_dynamics_backend": neural_dynamics_local["selected_backend"],
                "best_local_dynamics_closed_loop_passed": neural_dynamics_local[
                    "cross_validation_passed"
                ],
                "closed_loop_data_aggregation_CV_passed": neural_corridor_dagger[
                    "cross_validation_passed"
                ],
                "best_local_visual_family": visual_layer_locality["selected_family"],
                "lamina_direct_goal_obstacles": lamina_goal["total_obstacles_passed"],
                "lamina_equivariant_goal_obstacles": lamina_goal_symmetry["selected_summary"][
                    "held_out_obstacles_passed"
                ],
                "FC2_goal_memory_obstacles": fc2_goal_memory["selected_summary"][
                    "held_out_obstacles_passed"
                ],
                "complementary_goal_fusion_development_CV_passed": neural_goal_fusion[
                    "cross_validation_passed"
                ],
                "fusion_nested_fresh_CV_passed": fusion_nested_eval["cross_validation"]["passed"],
            },
        },
        {
            "item": 5,
            "deliverable": "KC-MBON value learning gated by PAM/PPL1",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
            "observations": {"causal_visual_navigation_pass": False},
        },
        {
            "item": 6,
            "deliverable": (
                "real topology, strict graph shuffles, parametric baselines and ablations"
            ),
            "status": "incomplete",
            "evidence": [config["evidence"]["controlled_vision"]],
            "observations": {
                "preliminary_stage1_controls_complete": controlled[
                    "stage1_topology_controls_complete"
                ],
                "strict_degree_preserving_control_complete": controlled[
                    "strict_degree_preserving_control_complete"
                ],
                "strict_visual_target_degree_preserving_control_constructed": (
                    degree_preserving_control["strict_degree_preserving_control_constructed"]
                ),
                "strict_visual_target_degree_preserving_rewired_fraction": (
                    degree_preserving_control["swap_statistics"]["rewired_edge_fraction"]
                ),
                "strict_visual_target_control_response_evaluated": (
                    degree_preserving_control["visual_response_evaluation_performed"]
                ),
                "real_topology_advantage": controlled["real_topology_advantage"],
                "structured_neural_topology_control_successes": {
                    name: item["success_count"]
                    for name, item in neural_topology_controls["controls"].items()
                },
                "compact_neural_CV_passed": neural_episode_cv["cross_validation_passed"],
                "compact_neural_calibration_passed": neural_episode_cv["calibration_passed"],
                "compact_neural_real_topology_advantage": neural_episode_controls[
                    "real_topology_advantage_passed"
                ],
                "parameter_matched_linear_mlp_gru_complete": False,
                "parameter_matched_baseline_protocol_ready": parameter_matched_baselines[
                    "protocol_ready"
                ],
                "parameter_matched_baseline_budget": parameter_matched_baselines[
                    "protocol"
                ]["reference_trainable_parameter_budget"],
                "parameter_matched_baseline_counts": {
                    name: result["trainable_parameter_count"]
                    for name, result in parameter_matched_baselines["architectures"].items()
                },
                "parameter_matched_baseline_evaluation_performed": (
                    parameter_matched_baselines["evaluation_performed"]
                ),
                "downstream_ablations_authorized": downstream_authorized,
            },
        },
        {
            "item": 7,
            "deliverable": "independent train/validation/final and dynamic OOD release gates",
            "status": "not_authorized",
            "evidence": [
                config["evidence"]["fitted_t4"],
                config["evidence"]["ephys_interface"],
            ],
            "observations": {
                "fitted_T4_final_test_evaluated": reports["fitted_t4"]["test"]["evaluated"],
                "ephys_final_test_available": interface["available_final_test"],
                "task_and_OOD_release_evaluated": False,
                "stage1_stimulus_splits_disjoint": all(
                    value["identity_overlap"] == value["frame_hash_overlap"] == 0
                    for value in stage1_split["cross_split_overlap"].values()
                ),
                "reserved_final_evaluated": stage1_split["split_manifests"]["final"]["evaluated"],
                "blinded_one_time_final_available": False,
                "legacy_172_regression_only": not stage1_nested["legacy_regression"][
                    "decision_eligible"
                ],
                "nested_role_counts": stage1_nested["condition_contract"]["required_role_counts"],
                "external_final_committed": stage1_nested["external_final"]["committed"],
                "external_final_evaluated": stage1_nested["external_final"]["evaluated"],
                "closed_loop_calibration_passed": closed_loop_calibration["calibration_passed"],
                "closed_loop_final_evaluated": closed_loop_calibration["final_evaluated"],
                "navigation_nested_role_counts": navigation_nested["role_counts"],
                "navigation_nested_CV_passed": navigation_nested_eval["cross_validation"]["passed"],
                "navigation_nested_tuning_passed": navigation_nested_eval["tuning_passed"],
                "navigation_nested_calibration_attempt_count": navigation_nested_eval[
                    "calibration_receipt"
                ]["attempt_count"],
                "navigation_external_final_committed": navigation_nested["external_final"][
                    "committed"
                ],
                "fusion_nested_role_counts": fusion_nested["role_counts"],
                "fusion_nested_CV_passed": fusion_nested_eval["cross_validation"]["passed"],
                "fusion_nested_calibration_attempt_count": fusion_nested_eval[
                    "calibration_receipt"
                ]["attempt_count"],
                "fusion_external_final_committed": fusion_nested["external_final"]["committed"],
            },
        },
        {
            "item": 8,
            "deliverable": "pause city/LLM expansion and separate planner, fly core and executor",
            "status": "passed_boundary_only",
            "evidence": [str(V7_CONFIG), config["evidence"]["manifest"]],
            "observations": {
                "city_expansion_enabled": contract.payload["city_expansion_enabled"],
                "v7_deployment_enabled": contract.payload["deployment_enabled"],
                "three_way_contribution_report_available": True,
                "contribution_layers": config["interface_disclosure"][
                    "contribution_layers"
                ],
                "v7_status_is_offline_only": True,
            },
        },
    ]
    requirement_checklist = [
        {
            "requirement": "0.version_config_evidence_claim_boundaries",
            "status": "passed",
            "evidence": [str(V7_CONFIG), config["evidence"]["manifest"]],
        },
        {
            "requirement": "0.preserve_assisted_v5_checkpoint",
            "status": "passed" if baseline_hashes_valid else "failed",
            "evidence": [contract.payload["baseline_contracts"]["assisted_v5"]["path"]],
        },
        {
            "requirement": "0.preserve_neural_v6_front_checkpoint",
            "status": "passed" if baseline_hashes_valid else "failed",
            "evidence": [contract.payload["baseline_contracts"]["neural_v6_front"]["path"]],
        },
        {
            "requirement": "0.keep_default_service_unchanged",
            "status": "passed" if not manifest["default_runtime_changed"] else "failed",
            "evidence": [config["evidence"]["manifest"]],
        },
        {
            "requirement": "1.controlled_brightness_on_off_motion_looming_flow_mirror_stimuli",
            "status": "passed",
            "evidence": [
                config["evidence"]["controlled_vision"],
                config["evidence"]["degree_preserving_control"],
            ],
        },
        {
            "requirement": "1.external_visual_input_only_via_R1_R6",
            "status": "passed" if visual_inputs_only_at_receptors else "failed",
            "evidence": [config["evidence"]["controlled_vision"]],
        },
        {
            "requirement": "1.T4_T5_direction_and_ON_OFF_validation",
            "status": "failed",
            "evidence": [
                config["evidence"]["nested_neural_screen"],
                config["evidence"]["t5_data"],
                config["evidence"]["t5_repository"],
                config["evidence"]["ephys_interface"],
                config["evidence"]["t4_source_resolved"],
                config["evidence"]["t4_continuous_pair_precheck"],
                config["evidence"]["t4_pair_lag_audit"],
                config["evidence"]["t4_normalized_correlator"],
                config["evidence"]["t4_local_correlator_precheck"],
                config["evidence"]["t4_source_pool_local"],
                config["evidence"]["three_hop_moment"],
                config["evidence"]["three_hop_temporal_consistency"],
                config["evidence"]["upstream_latency_audit"],
                config["evidence"]["synapse_spatial_audit"],
                config["evidence"]["synapse_axis_calibration"],
                config["evidence"]["three_hop_source_coverage"],
                config["evidence"]["four_hop_scalar_precheck"],
                config["evidence"]["lc4_position_speed_precheck"],
                config["evidence"]["t5_spatial_order"],
                config["evidence"]["t4t5_local_edge_precheck"],
                config["evidence"]["t4t5_local_edge_backends"],
            ],
        },
        {
            "requirement": "1.LPLC1_LPLC2_LC4_approach_collision_validation",
            "status": "failed",
            "evidence": [
                config["evidence"]["lplc_typed_screen"],
                config["evidence"]["lplc2_radial_opponency"],
                config["evidence"]["lplc2_position_coverage"],
            ],
        },
        {
            "requirement": "2.EPG_PEN_PEG_heading_and_occlusion",
            "status": "passed_component_not_stage_gate",
            "evidence": [config["evidence"]["heading_ring"]],
        },
        {
            "requirement": "2.FC2_PFL3_PFL2_heading_goal_and_turn_termination",
            "status": "passed_component_not_stage_gate",
            "evidence": [config["evidence"]["fc2_pfl_dna"]],
        },
        {
            "requirement": "3.DNp20_DNa01_DNa02_PFL3_DNa_readout_comparison",
            "status": "partially_validated_not_stage_authorized",
            "evidence": [
                config["evidence"]["fc2_pfl_dna"],
                config["evidence"]["descending_path_audit"],
                config["evidence"]["descending_propagation_precheck"],
            ],
        },
        {
            "requirement": "3.transparent_environment_blind_vehicle_mapping",
            "status": "passed_component_not_stage_gate",
            "evidence": [config["evidence"]["fc2_pfl_dna"]],
        },
        {
            "requirement": "4.separate_self_motion_looming_slowing_heading_target_channels",
            "status": "partially_validated_not_stage_authorized",
            "evidence": [config["evidence"]["neural_episode_controls"]],
        },
        {
            "requirement": "5.KC_MBON_PAM_PPL1_value_learning_after_navigation_gate",
            "status": "partially_validated_not_stage_authorized",
            "evidence": [config["evidence"]["neural_episode_controls"]],
        },
        {
            "requirement": "6.real_MaleCNS_and_strict_randomized_graph_controls",
            "status": "incomplete",
            "evidence": [config["evidence"]["controlled_vision"]],
        },
        {
            "requirement": "6.parameter_matched_linear_MLP_GRU_baselines",
            "status": "protocol_frozen_not_evaluated",
            "evidence": [config["evidence"]["parameter_matched_baselines"]],
        },
        {
            "requirement": "6.T4_T5_LPLC_EPG_PFL_DNa_MBON_DAN_ablations",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
        },
        {
            "requirement": "7.independent_train_validation_one_time_final_split",
            "status": "protocol_frozen_not_evaluated",
            "evidence": [
                config["evidence"]["fitted_t4"],
                config["evidence"]["ephys_interface"],
                config["evidence"]["stage1_split"],
                config["evidence"]["stage1_scoring"],
                config["evidence"]["stage1_nested"],
                config["evidence"]["target_fit"],
                config["evidence"]["closed_loop_tuning"],
                config["evidence"]["closed_loop_calibration"],
                config["evidence"]["closed_loop_controls"],
                config["evidence"]["closed_loop_multi"],
                config["evidence"]["r1r6_multi_tuning"],
                config["evidence"]["r1r6_multi_calibration"],
                config["evidence"]["r1r6_local_tuning"],
                config["evidence"]["r1r6_local_calibration"],
                config["evidence"]["r1r6_local_controls"],
                config["evidence"]["neural_channels_tuning"],
                config["evidence"]["neural_channels_calibration"],
                config["evidence"]["neural_channel_controls"],
                config["evidence"]["neural_topology_controls"],
                config["evidence"]["nested_neural_screen"],
                config["evidence"]["navigation_nested"],
                config["evidence"]["navigation_nested_eval"],
            ],
        },
        {
            "requirement": "7.dynamic_obstacle_density_curvature_speed_noise_OOD",
            "status": "not_authorized",
            "evidence": [
                config["evidence"]["navigation_nested_eval"],
                config["evidence"]["visual_corridor_goal"],
            ],
        },
        {
            "requirement": (
                "7.task_collision_exit_window_early_failure_steering_drift_"
                "mirror_zero_action_topology_gates"
            ),
            "status": "not_authorized",
            "evidence": [str(V7_CONFIG), config["evidence"]["manifest"]],
        },
        {
            "requirement": "8.pause_city_and_large_model_expansion",
            "status": "passed",
            "evidence": [str(V7_CONFIG), config["evidence"]["manifest"]],
        },
        {
            "requirement": "8.separate_planner_fly_core_executor_contributions",
            "status": "passed",
            "evidence": [str(path) for path in disclosure_paths.values()],
        },
        {
            "requirement": "8.continuous_tests_evidence_docs_frontend_and_remote_updates",
            "status": "in_progress",
            "evidence": [
                "tests",
                "docs/driving-v7.zh-CN.md",
                "apps/web/src/components/DrivingPanel.tsx",
            ],
        },
        {
            "requirement": "8.preserve_negative_results_and_stop_conditions",
            "status": "passed",
            "evidence": [config["evidence"]["manifest"], config["evidence"]["t5_phenotype"]],
        },
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(V7_CONFIG): contract.sha256,
                str(V7_IMPLEMENTATION): _sha256(root / V7_IMPLEMENTATION),
                **evidence_hashes,
                **disclosure_hashes,
            },
        },
        "objective_items": config["objective_items"],
        "checks": checks,
        "requirement_checklist": requirement_checklist,
        "summary": {
            "complete_items": [item["item"] for item in checks if item["status"] == "passed"],
            "boundary_only_items": [
                item["item"] for item in checks if item["status"] == "passed_boundary_only"
            ],
            "component_pass_items": sorted(
                {
                    int(item["requirement"].split(".", 1)[0])
                    for item in requirement_checklist
                    if item["status"] == "passed_component_not_stage_gate"
                }
            ),
            "failed_items": [item["item"] for item in checks if item["status"] == "failed"],
            "incomplete_items": [item["item"] for item in checks if item["status"] == "incomplete"],
            "partially_validated_items": [
                item["item"]
                for item in checks
                if item["status"] == "partially_validated_not_stage_authorized"
            ],
            "not_authorized_items": [
                item["item"] for item in checks if item["status"] == "not_authorized"
            ],
            "objective_complete": False,
            "current_stage": "controlled_vision",
            "next_allowed_work": [
                "repair T4/T5 direction and polarity dynamics without subtype label injection",
                "repair typed LPLC1/LPLC2/LC4 responses under separate mechanism scores",
                (
                    "preserve EPG/PEN/PEG heading and FC2/PFL3/PFL2/DNa02 component "
                    "evidence without treating it as downstream stage authorization"
                ),
                (
                    "improve local neural column information before retrying the frozen "
                    "navigation nested protocol"
                ),
                "obtain externally custodied independent-cell and one-time final manifests",
            ],
        },
    }
