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
    nested_neural_screen = reports["nested_neural_screen"]
    t4_source_resolved = reports["t4_source_resolved"]
    t4_normalized_correlator = reports["t4_normalized_correlator"]
    t4_local_correlator_precheck = reports["t4_local_correlator_precheck"]
    t4_source_pool_local = reports["t4_source_pool_local"]
    three_hop_moment = reports["three_hop_moment"]
    three_hop_source_coverage = reports["three_hop_source_coverage"]
    four_hop_scalar_precheck = reports["four_hop_scalar_precheck"]
    lc4_position_speed_precheck = reports["lc4_position_speed_precheck"]
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
            "evidence": [config["evidence"]["fc2_pfl_dna"]],
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
                "three_way_contribution_report_available": False,
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
            "evidence": [config["evidence"]["controlled_vision"]],
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
                config["evidence"]["t4_source_resolved"],
                config["evidence"]["t4_normalized_correlator"],
                config["evidence"]["t4_local_correlator_precheck"],
                config["evidence"]["t4_source_pool_local"],
                config["evidence"]["three_hop_moment"],
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
            "evidence": [config["evidence"]["fc2_pfl_dna"]],
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
            "status": "not_started",
            "evidence": [],
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
            "status": "missing",
            "evidence": [],
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
