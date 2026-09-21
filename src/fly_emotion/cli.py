from __future__ import annotations

import argparse
import json
from pathlib import Path

from .connectome.graph import build_canonical_graph, load_graph
from .connectome.overview import build_overview
from .connectome.pathways import build_pathways
from .data.audit import audit_annotations, audit_connections, audit_neurotransmitters, write_json
from .data.download import download_file
from .data.manifest import iter_files, load_manifest
from .driving.evaluate import (
    calibrate_stable_policy,
    evaluate_body_motor_interaction,
    evaluate_city_alpha,
    evaluate_constraints,
    evaluate_neural_decision_baseline,
    evaluate_neural_motor_adaptation,
    evaluate_neural_transfer,
    evaluate_panorama_release,
    evaluate_sensory_ablation,
    evaluate_sensory_gain_audit,
    train_neural_curriculum,
    train_sensory_pathway_curriculum,
    write_evaluation,
)
from .driving.v7 import (
    evaluate_v7_controlled_vision,
    evaluate_v7_optic_hex_axis_calibration,
    evaluate_v7_typed_visual_candidate,
    write_v7_manifest,
)
from .driving.v7_arenz_source_dynamics_audit import (
    evaluate_v7_arenz_source_dynamics_audit,
)
from .driving.v7_arenz_t5_source_dynamics_audit import (
    evaluate_v7_arenz_t5_source_dynamics_audit,
)
from .driving.v7_behnia_t4_fast_source_audit import (
    evaluate_v7_behnia_t4_fast_source_audit,
)
from .driving.v7_behnia_t4_fast_source_availability_audit import (
    evaluate_v7_behnia_t4_fast_source_availability_audit,
)
from .driving.v7_borst_2025_temporal_filtering_audit import (
    evaluate_v7_borst_2025_temporal_filtering_audit,
)
from .driving.v7_branched import evaluate_v7_branched_t4_candidate
from .driving.v7_c3_analytic_filter_precheck import (
    evaluate_v7_c3_analytic_filter_precheck,
)
from .driving.v7_c3_flash_preregistration import (
    evaluate_v7_c3_flash_preregistration,
)
from .driving.v7_c3_measured_filter_robustness import (
    evaluate_v7_c3_measured_filter_robustness,
)
from .driving.v7_c3_strf_flash_transfer import evaluate_v7_c3_strf_flash_transfer
from .driving.v7_c3_strf_source_dynamics_audit import (
    evaluate_v7_c3_strf_source_dynamics_audit,
)
from .driving.v7_closed_loop import (
    evaluate_v7_closed_loop_calibration,
    evaluate_v7_closed_loop_tuning,
)
from .driving.v7_closed_loop_controls import evaluate_v7_closed_loop_controls
from .driving.v7_closed_loop_multi import evaluate_v7_closed_loop_multi
from .driving.v7_conductance import evaluate_v7_published_conductance
from .driving.v7_controlled_stimulus_angular_grid import (
    evaluate_v7_controlled_stimulus_angular_grid,
)
from .driving.v7_controlled_stimulus_input_boundary_audit import (
    evaluate_v7_controlled_stimulus_input_boundary_audit,
)
from .driving.v7_coverage_response import evaluate_v7_coverage_response
from .driving.v7_ct1_experimental_voltage_boundary_audit import (
    evaluate_v7_ct1_experimental_voltage_boundary_audit,
)
from .driving.v7_ct1_extreme_compartmentalization_audit import (
    evaluate_v7_ct1_extreme_compartmentalization_audit,
)
from .driving.v7_ct1_pure_data_index_audit import (
    evaluate_v7_ct1_pure_data_index_audit,
)
from .driving.v7_danger_throttle import evaluate_v7_danger_throttle
from .driving.v7_degree_preserving_control import evaluate_v7_degree_preserving_control
from .driving.v7_descending_path_audit import evaluate_v7_descending_path_audit
from .driving.v7_descending_propagation_precheck import (
    evaluate_v7_descending_propagation_precheck,
)
from .driving.v7_disinhibition import evaluate_v7_conductance_order, evaluate_v7_disinhibition
from .driving.v7_dryad_l1l2_source_dynamics_audit import (
    evaluate_v7_dryad_l1l2_source_dynamics_audit,
)
from .driving.v7_edmond_fig3_retrieval_audit import (
    evaluate_v7_edmond_fig3_retrieval_audit,
)
from .driving.v7_ephys_audit import evaluate_v7_electrophysiology_audit
from .driving.v7_ephys_interface import evaluate_v7_ephys_interface
from .driving.v7_fc2_goal_memory import evaluate_v7_fc2_goal_memory
from .driving.v7_fc2_pfl_dna import evaluate_v7_fc2_pfl_dna
from .driving.v7_fib19_malecns_t5_weight_transfer_audit import (
    evaluate_v7_fib19_malecns_t5_weight_transfer_audit,
)
from .driving.v7_fig1_source_temporal_readiness_audit import (
    evaluate_v7_fig1_source_temporal_readiness_audit,
)
from .driving.v7_fig3_source_kernel_audit import evaluate_v7_fig3_source_kernel_audit
from .driving.v7_fig3_source_kernel_robustness import (
    evaluate_v7_fig3_source_kernel_robustness,
)
from .driving.v7_fig5_validation import evaluate_v7_fig5_validation
from .driving.v7_fit import fit_v7_t4_conductance
from .driving.v7_flyvis_c3_effective_dynamics_audit import (
    evaluate_v7_flyvis_c3_effective_dynamics_audit,
)
from .driving.v7_flyvis_c3_time_constant_audit import (
    evaluate_v7_flyvis_c3_time_constant_audit,
)
from .driving.v7_flyvis_visual_source_time_constants import (
    evaluate_v7_flyvis_visual_source_time_constants,
)
from .driving.v7_four_hop_scalar_precheck import evaluate_v7_four_hop_scalar_precheck
from .driving.v7_fusion_nested import evaluate_v7_fusion_nested
from .driving.v7_fusion_nested_eval import evaluate_v7_fusion_nested_candidate
from .driving.v7_gain_audit import evaluate_v7_normalization_gain
from .driving.v7_geometry_sign import evaluate_v7_geometry_sign
from .driving.v7_goal_audit import evaluate_v7_goal_coverage
from .driving.v7_gou_dandi_identity_audit import (
    evaluate_v7_gou_dandi_identity_audit,
)
from .driving.v7_gou_dandi_stimulus_metadata_audit import (
    evaluate_v7_gou_dandi_stimulus_metadata_audit,
)
from .driving.v7_gou_moving_bar_direction_audit import (
    evaluate_v7_gou_moving_bar_direction_audit,
)
from .driving.v7_gou_sparsity_source_dynamics_audit import (
    evaluate_v7_gou_sparsity_source_dynamics_audit,
)
from .driving.v7_heading_ring import evaluate_v7_heading_ring
from .driving.v7_kohn_portes_axolotl_availability_audit import (
    evaluate_v7_kohn_portes_axolotl_availability_audit,
)
from .driving.v7_kohn_portes_external_index_audit import (
    evaluate_v7_kohn_portes_external_index_audit,
)
from .driving.v7_kohn_portes_figure6_model_identity_audit import (
    evaluate_v7_kohn_portes_figure6_model_identity_audit,
)
from .driving.v7_kohn_portes_identity_history_audit import (
    evaluate_v7_kohn_portes_identity_history_audit,
)
from .driving.v7_kohn_portes_stimulus_provenance_audit import (
    evaluate_v7_kohn_portes_stimulus_provenance_audit,
)
from .driving.v7_kohn_portes_t5_ephys_audit import (
    evaluate_v7_kohn_portes_t5_ephys_audit,
)
from .driving.v7_kohn_portes_t5_frequency_tuning_audit import (
    evaluate_v7_kohn_portes_t5_frequency_tuning_audit,
)
from .driving.v7_kohn_portes_t5_moving_bar_generalization_audit import (
    evaluate_v7_kohn_portes_t5_moving_bar_generalization_audit,
)
from .driving.v7_kohn_portes_t5_peak_latency_audit import (
    evaluate_v7_kohn_portes_t5_peak_latency_audit,
)
from .driving.v7_kohn_portes_t5_source_kernel_audit import (
    evaluate_v7_kohn_portes_t5_source_kernel_audit,
)
from .driving.v7_kohn_portes_t5_state_unit_mapping_audit import (
    evaluate_v7_kohn_portes_t5_state_unit_mapping_audit,
)
from .driving.v7_kohn_portes_tm_to_t5_model_audit import (
    evaluate_v7_kohn_portes_tm_to_t5_model_audit,
)
from .driving.v7_lamina_goal import evaluate_v7_lamina_goal
from .driving.v7_lamina_goal_symmetry import evaluate_v7_lamina_goal_symmetry
from .driving.v7_lc4_input_speed_precheck import evaluate_v7_lc4_input_speed_precheck
from .driving.v7_lc4_position_speed_precheck import evaluate_v7_lc4_position_speed_precheck
from .driving.v7_local_input_audit import (
    evaluate_v7_local_input_audit,
    evaluate_v7_receptor_mask_audit,
    evaluate_v7_t4_input_coverage,
)
from .driving.v7_looming_mechanism_audit import evaluate_v7_looming_mechanism_audit
from .driving.v7_lplc1_input_structure import evaluate_v7_lplc1_input_structure
from .driving.v7_lplc1_near_collision_precheck import (
    evaluate_v7_lplc1_near_collision_precheck,
)
from .driving.v7_lplc2_phenotype import evaluate_v7_lplc2_phenotype
from .driving.v7_lplc2_position_coverage import evaluate_v7_lplc2_position_coverage
from .driving.v7_lplc2_radial_opponency import evaluate_v7_lplc2_radial_opponency
from .driving.v7_lplc_typed_screen import evaluate_v7_lplc_typed_screen
from .driving.v7_malecns_ct1_columnar_audit import (
    evaluate_v7_malecns_ct1_columnar_audit,
)
from .driving.v7_malecns_source_mapping_readiness_audit import (
    evaluate_v7_malecns_source_mapping_readiness_audit,
)
from .driving.v7_malecns_synapse_column_audit import (
    evaluate_v7_malecns_synapse_column_audit,
)
from .driving.v7_matulis_mi1_voltage_availability_audit import (
    evaluate_v7_matulis_mi1_voltage_availability_audit,
)
from .driving.v7_mi4_c3_whole_cell_candidate_audit import (
    evaluate_v7_mi4_c3_whole_cell_candidate_audit,
)
from .driving.v7_mirror_audit import evaluate_v7_layerwise_mirror_audit
from .driving.v7_motyxia2_public_history_audit import (
    evaluate_v7_motyxia2_public_history_audit,
)
from .driving.v7_navigation_nested import evaluate_v7_navigation_nested
from .driving.v7_navigation_nested_eval import evaluate_v7_navigation_nested_candidate
from .driving.v7_nested_neural_screen import evaluate_v7_nested_neural_screen
from .driving.v7_neural_channel_controls import evaluate_v7_neural_channel_controls
from .driving.v7_neural_channels import (
    evaluate_v7_neural_channels_calibration,
    evaluate_v7_neural_channels_tuning,
)
from .driving.v7_neural_corridor import (
    evaluate_v7_local_column_corridor,
    evaluate_v7_neural_corridor,
)
from .driving.v7_neural_corridor_dagger import evaluate_v7_neural_corridor_dagger
from .driving.v7_neural_dynamics_local import evaluate_v7_neural_dynamics_local
from .driving.v7_neural_episode_controls import evaluate_v7_neural_episode_controls
from .driving.v7_neural_episode_cv import evaluate_v7_neural_episode_cv
from .driving.v7_neural_goal_fusion import evaluate_v7_neural_goal_fusion
from .driving.v7_neural_local_columns import evaluate_v7_neural_local_columns
from .driving.v7_neural_spectra import evaluate_v7_neural_spectra
from .driving.v7_neural_topology_controls import evaluate_v7_neural_topology_controls
from .driving.v7_parameter_matched_baselines import evaluate_v7_parameter_matched_baselines
from .driving.v7_perturbation import evaluate_v7_perturbation
from .driving.v7_phase_motion import evaluate_v7_phase_motion
from .driving.v7_pirogova_source_calcium_audit import (
    evaluate_v7_pirogova_source_calcium_audit,
)
from .driving.v7_pixel_sampling import evaluate_v7_pixel_sampling
from .driving.v7_public_t4_model_source_coverage_audit import (
    evaluate_v7_public_t4_model_source_coverage_audit,
)
from .driving.v7_r1r6_local import (
    evaluate_v7_r1r6_local_calibration,
    evaluate_v7_r1r6_local_tuning,
)
from .driving.v7_r1r6_local_controls import evaluate_v7_r1r6_local_controls
from .driving.v7_r1r6_multi import (
    evaluate_v7_r1r6_multi_calibration,
    evaluate_v7_r1r6_multi_tuning,
)
from .driving.v7_retina_audit import evaluate_v7_retina_column_audit
from .driving.v7_source_audit import evaluate_v7_t4_source_audit
from .driving.v7_source_dynamics_external_evidence_contract import (
    evaluate_v7_source_dynamics_external_evidence_contract,
)
from .driving.v7_source_evidence_matrix import evaluate_v7_source_evidence_matrix
from .driving.v7_source_type_average_mapping_contract import (
    evaluate_v7_source_type_average_mapping_contract,
)
from .driving.v7_source_type_temporal_identifiability import (
    evaluate_v7_source_type_temporal_identifiability,
)
from .driving.v7_spectral_controls import evaluate_v7_spectral_controls
from .driving.v7_stability import evaluate_v7_background_stability, evaluate_v7_feedback_cut
from .driving.v7_stage1_development import evaluate_v7_stage1_development
from .driving.v7_stage1_geometry_ab import evaluate_v7_stage1_geometry_ab
from .driving.v7_stage1_input_audit import evaluate_v7_stage1_input_audit
from .driving.v7_stage1_nested import evaluate_v7_stage1_nested
from .driving.v7_stage1_scoring import evaluate_v7_stage1_scoring
from .driving.v7_stage1_split import evaluate_v7_stage1_split
from .driving.v7_stimulus_coordinate_contract import (
    evaluate_v7_stimulus_coordinate_contract,
)
from .driving.v7_synapse_axis_calibration import evaluate_v7_synapse_axis_calibration
from .driving.v7_synapse_spatial_audit import evaluate_v7_synapse_spatial_audit
from .driving.v7_synchronous import evaluate_v7_synchronous_update
from .driving.v7_t4_continuous_pair_precheck import (
    evaluate_v7_t4_continuous_pair_precheck,
)
from .driving.v7_t4_crossfit_retinal_symmetry_audit import (
    evaluate_v7_t4_crossfit_retinal_symmetry_audit,
)
from .driving.v7_t4_crossfit_sequence_identifiability import (
    evaluate_v7_t4_crossfit_sequence_identifiability,
)
from .driving.v7_t4_individual_split_1khz_audit import (
    evaluate_v7_t4_individual_split_1khz_audit,
)
from .driving.v7_t4_individual_split_audit import evaluate_v7_t4_individual_split_audit
from .driving.v7_t4_inhibitory_source_external_audit import (
    evaluate_v7_t4_inhibitory_source_external_audit,
)
from .driving.v7_t4_local_correlator_precheck import evaluate_v7_t4_local_correlator_precheck
from .driving.v7_t4_normalized_correlator import evaluate_v7_t4_normalized_correlator
from .driving.v7_t4_pair_lag_audit import evaluate_v7_t4_pair_lag_audit
from .driving.v7_t4_recording_field_audit import evaluate_v7_t4_recording_field_audit
from .driving.v7_t4_source_dynamics_transfer_audit import (
    evaluate_v7_t4_source_dynamics_transfer_audit,
)
from .driving.v7_t4_source_identity_readiness_audit import (
    evaluate_v7_t4_source_identity_readiness_audit,
)
from .driving.v7_t4_source_pool_local import evaluate_v7_t4_source_pool_local
from .driving.v7_t4_source_resolved import evaluate_v7_t4_source_resolved
from .driving.v7_t4_state_unit_mapping_audit import (
    evaluate_v7_t4_state_unit_mapping_audit,
)
from .driving.v7_t4_synapse_antisymmetric_precheck import (
    evaluate_v7_t4_synapse_antisymmetric_precheck,
)
from .driving.v7_t4_synapse_centered_precheck import (
    evaluate_v7_t4_synapse_centered_precheck,
)
from .driving.v7_t4_synapse_correlator_precheck import (
    evaluate_v7_t4_synapse_correlator_precheck,
)
from .driving.v7_t4_synapse_crossfit_axis_audit import (
    evaluate_v7_t4_synapse_crossfit_axis_audit,
)
from .driving.v7_t4_synapse_crossfit_precheck import (
    evaluate_v7_t4_synapse_crossfit_precheck,
)
from .driving.v7_t4_synapse_microstep_precheck import (
    evaluate_v7_t4_synapse_microstep_precheck,
)
from .driving.v7_t4t5_local_edge_backends import evaluate_v7_t4t5_local_edge_backends
from .driving.v7_t4t5_local_edge_precheck import evaluate_v7_t4t5_local_edge_precheck
from .driving.v7_t4t5_source_dynamics_readiness import (
    evaluate_v7_t4t5_source_dynamics_readiness,
)
from .driving.v7_t5_author_row_weighted_full_support_sensitivity import (
    evaluate_v7_t5_author_row_weighted_full_support_sensitivity,
)
from .driving.v7_t5_conductance_audit import evaluate_v7_t5_conductance_audit
from .driving.v7_t5_continuous_moment_precheck import (
    evaluate_v7_t5_continuous_moment_precheck,
)
from .driving.v7_t5_contrast_opponency_source_data_audit import (
    evaluate_v7_t5_contrast_opponency_source_data_audit,
)
from .driving.v7_t5_ct1_axis_aware_antisymmetric_precheck import (
    evaluate_v7_t5_ct1_axis_aware_antisymmetric_precheck,
)
from .driving.v7_t5_ct1_axis_aware_precheck import (
    evaluate_v7_t5_ct1_axis_aware_precheck,
)
from .driving.v7_t5_ct1_axis_calibration import (
    evaluate_v7_t5_ct1_axis_calibration,
)
from .driving.v7_t5_ct1_axis_sequence_identifiability import (
    evaluate_v7_t5_ct1_axis_sequence_identifiability,
)
from .driving.v7_t5_ct1_crossfit_axis_audit import (
    evaluate_v7_t5_ct1_crossfit_axis_audit,
)
from .driving.v7_t5_ct1_multiplicative_precheck import (
    evaluate_v7_t5_ct1_multiplicative_precheck,
)
from .driving.v7_t5_ct1_source_dynamics_precheck import (
    evaluate_v7_t5_ct1_source_dynamics_precheck,
)
from .driving.v7_t5_ct1_terminal_axis_audit import (
    evaluate_v7_t5_ct1_terminal_axis_audit,
)
from .driving.v7_t5_data_audit import evaluate_v7_t5_data_audit
from .driving.v7_t5_direction_code_provenance_audit import (
    evaluate_v7_t5_direction_code_provenance_audit,
)
from .driving.v7_t5_label_audit import evaluate_v7_t5_label_audit
from .driving.v7_t5_lamina_scalar_precheck import evaluate_v7_t5_lamina_scalar_precheck
from .driving.v7_t5_lamina_split import evaluate_v7_t5_lamina_split
from .driving.v7_t5_measured_kernel_cascade_semantics_audit import (
    evaluate_v7_t5_measured_kernel_cascade_semantics_audit,
)
from .driving.v7_t5_measured_kernel_full_support_identifiability import (
    evaluate_v7_t5_measured_kernel_full_support_identifiability,
)
from .driving.v7_t5_measured_kernel_identifiability import (
    evaluate_v7_t5_measured_kernel_identifiability,
)
from .driving.v7_t5_measured_kernel_substep_sensitivity import (
    evaluate_v7_t5_measured_kernel_substep_sensitivity,
)
from .driving.v7_t5_measured_kernel_support_coverage_audit import (
    evaluate_v7_t5_measured_kernel_support_coverage_audit,
)
from .driving.v7_t5_native_direction_waveform_audit import (
    evaluate_v7_t5_native_direction_waveform_audit,
)
from .driving.v7_t5_phenotype import evaluate_v7_t5_phenotype
from .driving.v7_t5_physical_time_transfer_audit import (
    evaluate_v7_t5_physical_time_transfer_audit,
)
from .driving.v7_t5_population_kernel_aggregation_semantics_audit import (
    evaluate_v7_t5_population_kernel_aggregation_semantics_audit,
)
from .driving.v7_t5_population_kernel_robustness_audit import (
    evaluate_v7_t5_population_kernel_robustness_audit,
)
from .driving.v7_t5_recording_field_audit import evaluate_v7_t5_recording_field_audit
from .driving.v7_t5_source_axis_audit import evaluate_v7_t5_source_axis_audit
from .driving.v7_t5_source_mapping_scope_audit import (
    evaluate_v7_t5_source_mapping_scope_audit,
)
from .driving.v7_t5_source_pair_precheck import evaluate_v7_t5_source_pair_precheck
from .driving.v7_t5_source_transfer_synthesis_audit import (
    evaluate_v7_t5_source_transfer_synthesis_audit,
)
from .driving.v7_t5_spatial_order import evaluate_v7_t5_spatial_order
from .driving.v7_t5_supplement_audit import evaluate_v7_t5_supplement_audit
from .driving.v7_t5_tm2_loo_full_support_sensitivity import (
    evaluate_v7_t5_tm2_loo_full_support_sensitivity,
)
from .driving.v7_t5_typed_spatial_pair_precheck import (
    evaluate_v7_t5_typed_spatial_pair_precheck,
)
from .driving.v7_target_fit import evaluate_v7_target_fit_contract
from .driving.v7_temporal_audit import evaluate_v7_temporal_input_audit
from .driving.v7_three_hop_moment import evaluate_v7_three_hop_moment
from .driving.v7_three_hop_source_coverage import evaluate_v7_three_hop_source_coverage
from .driving.v7_three_hop_temporal_consistency import (
    evaluate_v7_three_hop_temporal_consistency,
)
from .driving.v7_timebase_audit import evaluate_v7_timebase_audit
from .driving.v7_timing_models_ct1_compartment_audit import (
    evaluate_v7_timing_models_ct1_compartment_audit,
)
from .driving.v7_timing_models_source_filter_audit import (
    evaluate_v7_timing_models_source_filter_audit,
)
from .driving.v7_tm9_coordinate_identifiability_audit import (
    evaluate_v7_tm9_coordinate_identifiability_audit,
)
from .driving.v7_unified_model_package_audit import evaluate_v7_unified_model_package_audit
from .driving.v7_upstream_latency_audit import evaluate_v7_upstream_latency_audit
from .driving.v7_vertical_angular_coordinate_audit import (
    evaluate_v7_vertical_angular_coordinate_audit,
)
from .driving.v7_visual_corridor_goal import evaluate_v7_visual_corridor_goal
from .driving.v7_visual_layer_locality import evaluate_v7_visual_layer_locality
from .driving.v7_visual_target_input_audit import evaluate_v7_visual_target_input_audit
from .driving.v7_yang_t5_voltage_evidence_audit import (
    evaluate_v7_yang_t5_voltage_evidence_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autodrive-fly")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("configs/data-manifest.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--dataset", action="append")
    download = subparsers.add_parser("download")
    download.add_argument("--dataset", action="append")
    download.add_argument("--include-optional", action="store_true")
    subparsers.add_parser("audit")
    subparsers.add_parser("build-graph")
    subparsers.add_parser("build-overview")
    subparsers.add_parser("build-pathways")
    evaluate_driving = subparsers.add_parser("evaluate-driving")
    evaluate_driving.add_argument("--train-episodes", type=int, default=48)
    evaluate_driving.add_argument("--evaluation-seeds", type=int, default=32)
    evaluate_driving.add_argument("--evaluation-start", type=int, default=400)
    calibrate = subparsers.add_parser("calibrate-policy")
    calibrate.add_argument("--episodes", type=int, default=48)
    calibrate.add_argument("--evaluation-seeds", type=int, default=32)
    calibrate.add_argument("--evaluation-start", type=int, default=400)
    constraints = subparsers.add_parser("evaluate-constraints")
    constraints.add_argument("--evaluation-seeds", type=int, default=32)
    constraints.add_argument("--evaluation-start", type=int, default=400)
    constraints.add_argument("--checkpoint", type=Path)
    constraints.add_argument("--output", type=Path)
    constraints.add_argument("--reference-report", type=Path)
    city = subparsers.add_parser("evaluate-city-alpha")
    city.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 7])
    neural = subparsers.add_parser("evaluate-neural-decision")
    neural.add_argument("--start", type=int, default=400)
    neural.add_argument("--count", type=int, default=8)
    train_neural = subparsers.add_parser("train-neural-v6")
    train_neural.add_argument("--train-episodes", type=int, default=24)
    train_neural.add_argument("--evaluation-start", type=int, default=600)
    train_neural.add_argument("--evaluation-seeds", type=int, default=8)
    train_neural.add_argument(
        "--stage",
        choices=["single", "triple", "nine"],
        default="single",
    )
    train_neural.add_argument("--resume", action="store_true")
    train_neural.add_argument(
        "--sensory-profile",
        choices=["front", "panorama", "panorama_flow", "panorama_flow_body"],
        default="front",
    )
    subparsers.add_parser("evaluate-neural-transfer")
    subparsers.add_parser("evaluate-neural-motor-adaptation")
    sensory = subparsers.add_parser("evaluate-sensory-ablation")
    sensory.add_argument("--train-episodes", type=int, default=4)
    sensory.add_argument("--evaluation-start", type=int, default=960)
    sensory.add_argument("--evaluation-seeds", type=int, default=4)
    gain_audit = subparsers.add_parser("evaluate-sensory-gains")
    gain_audit.add_argument("--seed", type=int, default=960)
    gain_audit.add_argument("--steps", type=int, default=120)
    body_interaction = subparsers.add_parser("evaluate-body-motor-interaction")
    body_interaction.add_argument("--start", type=int, default=960)
    body_interaction.add_argument("--count", type=int, default=4)
    body_interaction.add_argument("--body-gain", type=float, default=0.0003)
    sensory_train = subparsers.add_parser("train-sensory-pathway")
    sensory_train.add_argument("--episodes-per-stage", type=int, default=2)
    sensory_train.add_argument("--evaluation-start", type=int, default=980)
    sensory_train.add_argument("--evaluation-seeds", type=int, default=4)
    panorama_release = subparsers.add_parser("evaluate-panorama-release")
    panorama_release.add_argument("--start", type=int, default=1000)
    panorama_release.add_argument("--count", type=int, default=16)
    subparsers.add_parser("v7-init")
    subparsers.add_parser("v7-evaluate-vision")
    subparsers.add_parser("v7-evaluate-typed-vision")
    subparsers.add_parser("v7-evaluate-optic-axis")
    subparsers.add_parser("v7-audit-t4-sources")
    subparsers.add_parser("v7-evaluate-branched-t4")
    subparsers.add_parser("v7-evaluate-t4-conductance")
    subparsers.add_parser("v7-fit-t4-conductance")
    subparsers.add_parser("v7-audit-retina-columns")
    subparsers.add_parser("v7-audit-layer-mirror")
    subparsers.add_parser("v7-audit-temporal-input")
    subparsers.add_parser("v7-audit-local-input")
    subparsers.add_parser("v7-audit-receptor-mask")
    subparsers.add_parser("v7-audit-t4-input-coverage")
    subparsers.add_parser("v7-audit-coverage-response")
    subparsers.add_parser("v7-evaluate-mi9-sign")
    subparsers.add_parser("v7-evaluate-conductance-order")
    subparsers.add_parser("v7-audit-background-stability")
    subparsers.add_parser("v7-audit-feedback-cut")
    subparsers.add_parser("v7-evaluate-synchronous-update")
    subparsers.add_parser("v7-audit-normalization-gain")
    subparsers.add_parser("v7-audit-full-update-perturbation")
    subparsers.add_parser("v7-evaluate-phase-motion")
    subparsers.add_parser("v7-evaluate-geometry-sign")
    subparsers.add_parser("v7-evaluate-pixel-sampling")
    subparsers.add_parser("v7-audit-spectral-controls")
    subparsers.add_parser("v7-evaluate-neural-spectra")
    subparsers.add_parser("v7-audit-electrophysiology")
    subparsers.add_parser("v7-audit-malecns-source-mapping-readiness")
    subparsers.add_parser("v7-audit-timebase")
    subparsers.add_parser("v7-audit-stimulus-coordinates")
    subparsers.add_parser("v7-audit-controlled-stimulus-angular-grid")
    subparsers.add_parser("v7-audit-controlled-stimulus-input-boundary")
    subparsers.add_parser("v7-audit-vertical-angular-coordinates")
    subparsers.add_parser("v7-validate-published-fig5")
    subparsers.add_parser("v7-build-ephys-interface")
    subparsers.add_parser("v7-audit-t5-data")
    subparsers.add_parser("v7-audit-t5-conductance")
    subparsers.add_parser("v7-extract-t5-phenotype")
    subparsers.add_parser("v7-audit-goal-coverage")
    subparsers.add_parser("v7-audit-t5-labels")
    subparsers.add_parser("v7-freeze-stage1-split")
    subparsers.add_parser("v7-audit-t5-supplement")
    subparsers.add_parser("v7-freeze-stage1-scoring")
    subparsers.add_parser("v7-audit-stage1-input")
    subparsers.add_parser("v7-evaluate-stage1-development")
    subparsers.add_parser("v7-evaluate-stage1-geometry-ab")
    subparsers.add_parser("v7-audit-visual-target-inputs")
    subparsers.add_parser("v7-audit-looming-mechanisms")
    subparsers.add_parser("v7-freeze-stage1-nested")
    subparsers.add_parser("v7-freeze-target-fit")
    subparsers.add_parser("v7-evaluate-closed-loop-tuning")
    subparsers.add_parser("v7-evaluate-closed-loop-calibration")
    subparsers.add_parser("v7-evaluate-closed-loop-controls")
    subparsers.add_parser("v7-evaluate-closed-loop-multi")
    subparsers.add_parser("v7-evaluate-r1r6-multi-tuning")
    subparsers.add_parser("v7-evaluate-r1r6-multi-calibration")
    subparsers.add_parser("v7-evaluate-r1r6-local-tuning")
    subparsers.add_parser("v7-evaluate-r1r6-local-calibration")
    subparsers.add_parser("v7-evaluate-r1r6-local-controls")
    subparsers.add_parser("v7-evaluate-neural-channels-tuning")
    subparsers.add_parser("v7-evaluate-neural-channels-calibration")
    subparsers.add_parser("v7-evaluate-neural-channel-controls")
    subparsers.add_parser("v7-evaluate-neural-topology-controls")
    subparsers.add_parser("v7-evaluate-nested-neural-screen")
    subparsers.add_parser("v7-evaluate-t5-spatial-order")
    subparsers.add_parser("v7-evaluate-lplc2-phenotype")
    subparsers.add_parser("v7-evaluate-t4-source-resolved")
    subparsers.add_parser("v7-evaluate-heading-ring")
    subparsers.add_parser("v7-evaluate-neural-local-columns")
    subparsers.add_parser("v7-evaluate-neural-episode-cv")
    subparsers.add_parser("v7-evaluate-neural-episode-controls")
    subparsers.add_parser("v7-evaluate-fc2-pfl-dna")
    subparsers.add_parser("v7-audit-descending-paths")
    subparsers.add_parser("v7-evaluate-descending-propagation-precheck")
    subparsers.add_parser("v7-freeze-navigation-nested")
    subparsers.add_parser("v7-evaluate-navigation-nested")
    subparsers.add_parser("v7-evaluate-danger-throttle")
    subparsers.add_parser("v7-evaluate-visual-corridor")
    subparsers.add_parser("v7-evaluate-neural-corridor")
    subparsers.add_parser("v7-evaluate-local-column-corridor")
    subparsers.add_parser("v7-evaluate-neural-dynamics-local")
    subparsers.add_parser("v7-evaluate-neural-corridor-dagger")
    subparsers.add_parser("v7-evaluate-visual-layer-locality")
    subparsers.add_parser("v7-evaluate-lamina-goal")
    subparsers.add_parser("v7-evaluate-lamina-goal-symmetry")
    subparsers.add_parser("v7-evaluate-t4-normalized-correlator")
    subparsers.add_parser("v7-evaluate-fc2-goal-memory")
    subparsers.add_parser("v7-evaluate-lplc-typed-screen")
    subparsers.add_parser("v7-evaluate-lplc2-radial-opponency")
    subparsers.add_parser("v7-evaluate-lplc2-position-coverage")
    subparsers.add_parser("v7-evaluate-t4t5-local-edge-precheck")
    subparsers.add_parser("v7-evaluate-t4t5-local-edge-backends")
    subparsers.add_parser("v7-audit-t4t5-source-dynamics-readiness")
    subparsers.add_parser("v7-evaluate-t4-local-correlator-precheck")
    subparsers.add_parser("v7-evaluate-t4-continuous-pair-precheck")
    subparsers.add_parser("v7-audit-t4-pair-lags")
    subparsers.add_parser("v7-audit-upstream-latency")
    subparsers.add_parser("v7-audit-source-type-temporal-identifiability")
    subparsers.add_parser("v7-freeze-source-dynamics-external-evidence-contract")
    subparsers.add_parser("v7-audit-source-evidence-matrix")
    subparsers.add_parser("v7-audit-source-type-average-mapping")
    subparsers.add_parser("v7-audit-synapse-spatial")
    subparsers.add_parser("v7-calibrate-synapse-axis")
    subparsers.add_parser("v7-evaluate-t4-synapse-correlator-precheck")
    subparsers.add_parser("v7-audit-t4-synapse-crossfit-axis")
    subparsers.add_parser("v7-evaluate-t4-synapse-crossfit-precheck")
    subparsers.add_parser("v7-audit-t4-crossfit-sequence-identifiability")
    subparsers.add_parser("v7-audit-t4-crossfit-retinal-symmetry")
    subparsers.add_parser("v7-evaluate-t4-synapse-antisymmetric-precheck")
    subparsers.add_parser("v7-evaluate-t4-synapse-centered-precheck")
    subparsers.add_parser("v7-evaluate-t4-synapse-microstep-precheck")
    subparsers.add_parser("v7-audit-t4-source-dynamics-transfer")
    subparsers.add_parser("v7-audit-t4-state-unit-mapping")
    subparsers.add_parser("v7-audit-edmond-fig3-retrieval")
    subparsers.add_parser("v7-audit-t4-source-identity-readiness")
    subparsers.add_parser("v7-audit-t4-individual-split")
    subparsers.add_parser("v7-audit-t4-individual-split-1khz")
    subparsers.add_parser("v7-audit-t4-recording-fields")
    subparsers.add_parser("v7-audit-t5-recording-fields")
    subparsers.add_parser("v7-audit-behnia-t4-fast-sources")
    subparsers.add_parser("v7-audit-behnia-t4-fast-source-availability")
    subparsers.add_parser("v7-audit-matulis-mi1-voltage-availability")
    subparsers.add_parser("v7-audit-borst-2025-temporal-filtering")
    subparsers.add_parser("v7-audit-t4-inhibitory-source-external")
    subparsers.add_parser("v7-audit-mi4-c3-whole-cell-candidates")
    subparsers.add_parser("v7-audit-unified-model-package")
    subparsers.add_parser("v7-audit-fig3-source-kernels")
    subparsers.add_parser("v7-audit-fig3-source-kernel-robustness")
    subparsers.add_parser("v7-audit-arenz-source-dynamics")
    subparsers.add_parser("v7-audit-arenz-t5-source-dynamics")
    subparsers.add_parser("v7-audit-c3-strf-source-dynamics")
    subparsers.add_parser("v7-audit-c3-flash-preregistration")
    subparsers.add_parser("v7-audit-c3-strf-flash-transfer")
    subparsers.add_parser("v7-precheck-c3-analytic-filter")
    subparsers.add_parser("v7-audit-c3-measured-filter-robustness")
    subparsers.add_parser("v7-audit-timing-models-source-filters")
    subparsers.add_parser("v7-audit-timing-models-ct1-compartment")
    subparsers.add_parser("v7-audit-flyvis-c3-time-constant")
    subparsers.add_parser("v7-audit-flyvis-visual-source-time-constants")
    subparsers.add_parser("v7-audit-flyvis-c3-effective-dynamics")
    subparsers.add_parser("v7-audit-public-t4-model-source-coverage")
    subparsers.add_parser("v7-audit-dryad-l1l2-source-dynamics")
    subparsers.add_parser("v7-audit-gou-sparsity-source-dynamics")
    subparsers.add_parser("v7-audit-gou-moving-bar-directions")
    subparsers.add_parser("v7-audit-gou-dandi-identity")
    subparsers.add_parser("v7-audit-gou-dandi-stimulus-metadata")
    subparsers.add_parser("v7-audit-pirogova-source-calcium")
    subparsers.add_parser("v7-audit-t5-contrast-opponency-source-data")
    subparsers.add_parser("v7-audit-yang-t5-voltage-evidence")
    subparsers.add_parser("v7-audit-kohn-portes-t5-ephys")
    subparsers.add_parser("v7-audit-kohn-portes-axolotl-availability")
    subparsers.add_parser("v7-audit-kohn-portes-figure6-model-identity")
    subparsers.add_parser("v7-audit-kohn-portes-t5-frequency-tuning")
    subparsers.add_parser("v7-audit-kohn-portes-t5-moving-bar-generalization")
    subparsers.add_parser("v7-audit-kohn-portes-t5-peak-latency")
    subparsers.add_parser("v7-audit-kohn-portes-t5-source-kernels")
    subparsers.add_parser("v7-audit-kohn-portes-t5-state-unit-mapping")
    subparsers.add_parser("v7-audit-kohn-portes-tm-to-t5-model")
    subparsers.add_parser("v7-audit-fib19-malecns-t5-weight-transfer")
    subparsers.add_parser("v7-audit-kohn-portes-identity-history")
    subparsers.add_parser("v7-audit-kohn-portes-external-indexes")
    subparsers.add_parser("v7-audit-kohn-portes-stimulus-provenance")
    subparsers.add_parser("v7-audit-t5-direction-code-provenance")
    subparsers.add_parser("v7-audit-motyxia2-public-history")
    subparsers.add_parser("v7-audit-ct1-extreme-compartmentalization")
    subparsers.add_parser("v7-audit-ct1-pure-data-index")
    subparsers.add_parser("v7-audit-ct1-experimental-voltage-boundary")
    subparsers.add_parser("v7-audit-tm9-coordinate-identifiability")
    subparsers.add_parser("v7-audit-malecns-synapse-columns")
    subparsers.add_parser("v7-audit-malecns-ct1-columns")
    subparsers.add_parser("v7-audit-fig1-source-temporal-readiness")
    subparsers.add_parser("v7-evaluate-t4-source-pool-local")
    subparsers.add_parser("v7-evaluate-three-hop-moment")
    subparsers.add_parser("v7-audit-three-hop-source-coverage")
    subparsers.add_parser("v7-audit-three-hop-temporal-consistency")
    subparsers.add_parser("v7-evaluate-four-hop-scalar-precheck")
    subparsers.add_parser("v7-evaluate-lc4-position-speed-precheck")
    subparsers.add_parser("v7-evaluate-lc4-input-speed-precheck")
    subparsers.add_parser("v7-audit-degree-preserving-control")
    subparsers.add_parser("v7-freeze-parameter-matched-baselines")
    subparsers.add_parser("v7-evaluate-t5-lamina-split")
    subparsers.add_parser("v7-evaluate-t5-lamina-scalar-precheck")
    subparsers.add_parser("v7-audit-t5-source-axis")
    subparsers.add_parser("v7-audit-t5-source-mapping-scope")
    subparsers.add_parser("v7-audit-t5-author-row-weighted-full-support")
    subparsers.add_parser("v7-audit-t5-measured-kernel-identifiability")
    subparsers.add_parser("v7-audit-t5-measured-kernel-full-support")
    subparsers.add_parser("v7-audit-t5-measured-kernel-cascade-semantics")
    subparsers.add_parser("v7-audit-t5-measured-kernel-substep-sensitivity")
    subparsers.add_parser("v7-audit-t5-measured-kernel-support-coverage")
    subparsers.add_parser("v7-audit-t5-source-transfer-synthesis")
    subparsers.add_parser("v7-evaluate-t5-source-pair-precheck")
    subparsers.add_parser("v7-evaluate-t5-typed-spatial-pair-precheck")
    subparsers.add_parser("v7-audit-t5-native-direction-waveforms")
    subparsers.add_parser("v7-audit-t5-physical-time-transfer")
    subparsers.add_parser("v7-audit-t5-population-kernel-robustness")
    subparsers.add_parser("v7-audit-t5-population-kernel-aggregation-semantics")
    subparsers.add_parser("v7-audit-t5-tm2-loo-full-support")
    subparsers.add_parser("v7-audit-t5-ct1-terminal-axis")
    subparsers.add_parser("v7-calibrate-t5-ct1-axis")
    subparsers.add_parser("v7-audit-t5-ct1-crossfit-axis")
    subparsers.add_parser("v7-evaluate-t5-ct1-axis-aware-precheck")
    subparsers.add_parser("v7-evaluate-t5-ct1-axis-aware-antisymmetric-precheck")
    subparsers.add_parser("v7-audit-t5-ct1-axis-sequence-identifiability")
    subparsers.add_parser("v7-evaluate-t5-ct1-source-dynamics")
    subparsers.add_parser("v7-evaluate-t5-ct1-multiplicative-precheck")
    subparsers.add_parser("v7-evaluate-t5-continuous-moment-precheck")
    subparsers.add_parser("v7-evaluate-lplc1-near-collision-precheck")
    subparsers.add_parser("v7-audit-lplc1-input-structure")
    subparsers.add_parser("v7-evaluate-neural-goal-fusion")
    subparsers.add_parser("v7-freeze-fusion-nested")
    subparsers.add_parser("v7-evaluate-fusion-nested")
    train_neural.add_argument("--publish", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    if args.command == "evaluate-driving":
        report = write_evaluation(
            root,
            root / "artifacts/driving-evaluation.json",
            train_episodes=args.train_episodes,
            evaluation_seeds=args.evaluation_seeds,
            evaluation_start=args.evaluation_start,
        )
        print(
            f"frozen {report['frozen']['mean_distance']:.2f} m -> "
            f"learned {report['learned']['mean_distance']:.2f} m "
            f"(delta {report['delta_mean_distance']:+.2f} m)"
        )
        return
    if args.command == "calibrate-policy":
        report = calibrate_stable_policy(
            root,
            episodes=args.episodes,
            evaluation_seeds=args.evaluation_seeds,
            evaluation_start=args.evaluation_start,
        )
        target = root / "artifacts/stable-policy-calibration.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-constraints":
        report = evaluate_constraints(
            root,
            evaluation_seeds=args.evaluation_seeds,
            evaluation_start=args.evaluation_start,
            checkpoint=root / args.checkpoint if args.checkpoint else None,
            reference_report=root / args.reference_report if args.reference_report else None,
        )
        target = args.output or root / "artifacts/behavior-constraint-ablation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        delta = report["paired"]["distance"]["delta_mean"]
        print(f"lane-constraint distance delta {delta:+.2f} m -> {target}")
        return
    if args.command == "evaluate-city-alpha":
        report = evaluate_city_alpha(root, seeds=tuple(args.seeds))
        target = root / "artifacts/city-alpha-evaluation.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-neural-decision":
        report = evaluate_neural_decision_baseline(root, start=args.start, count=args.count)
        target = root / "artifacts/neural-decision-current.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "train-neural-v6":
        report = train_neural_curriculum(
            root,
            train_episodes=args.train_episodes,
            evaluation_start=args.evaluation_start,
            evaluation_seeds=args.evaluation_seeds,
            publish=args.publish,
            stage=args.stage,
            resume=args.resume,
            sensory_profile=args.sensory_profile,
        )
        profile_suffix = "" if args.sensory_profile == "front" else f"-{args.sensory_profile}"
        target = root / f"artifacts/neural-v6-{args.stage}{profile_suffix}-curriculum.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-neural-transfer":
        report = evaluate_neural_transfer(root)
        target = root / "artifacts/neural-v6-transfer.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-neural-motor-adaptation":
        report = evaluate_neural_motor_adaptation(root)
        target = root / "artifacts/neural-v6-motor-adaptation.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-sensory-ablation":
        report = evaluate_sensory_ablation(
            root,
            train_episodes=args.train_episodes,
            evaluation_start=args.evaluation_start,
            evaluation_seeds=args.evaluation_seeds,
        )
        target = root / "artifacts/neural-v6-sensory-ablation.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-sensory-gains":
        report = evaluate_sensory_gain_audit(root, seed=args.seed, steps=args.steps)
        target = root / "artifacts/neural-v6-sensory-gain-audit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-body-motor-interaction":
        report = evaluate_body_motor_interaction(
            root, start=args.start, count=args.count, body_gain=args.body_gain
        )
        target = root / "artifacts/neural-v6-body-motor-interaction.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "train-sensory-pathway":
        report = train_sensory_pathway_curriculum(
            root,
            episodes_per_stage=args.episodes_per_stage,
            evaluation_start=args.evaluation_start,
            evaluation_seeds=args.evaluation_seeds,
        )
        target = root / "artifacts/neural-v6-sensory-pathway-curriculum.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-panorama-release":
        report = evaluate_panorama_release(root, start=args.start, count=args.count)
        target = root / "artifacts/neural-v6-panorama-release.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-init":
        report = write_v7_manifest(root)
        target = root / "artifacts/v7-manifest.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-vision":
        report = evaluate_v7_controlled_vision(root)
        target = root / "artifacts/v7-controlled-vision.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-typed-vision":
        report = evaluate_v7_typed_visual_candidate(root)
        target = root / "artifacts/v7-typed-visual-candidate.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-optic-axis":
        report = evaluate_v7_optic_hex_axis_calibration(root)
        target = root / "artifacts/v7-optic-hex-axis-calibration.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t4-sources":
        report = evaluate_v7_t4_source_audit(root)
        target = root / "artifacts/v7-t4-source-audit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-branched-t4":
        report = evaluate_v7_branched_t4_candidate(root)
        target = root / "artifacts/v7-branched-t4-candidate.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4-conductance":
        report = evaluate_v7_published_conductance(root)
        target = root / "artifacts/v7-t4-conductance-candidate.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-fit-t4-conductance":
        report = fit_v7_t4_conductance(root)
        target = root / "artifacts/v7-t4-conductance-fit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-retina-columns":
        report = evaluate_v7_retina_column_audit(root)
        target = root / "artifacts/v7-retina-column-audit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-layer-mirror":
        report = evaluate_v7_layerwise_mirror_audit(root)
        target = root / "artifacts/v7-layerwise-mirror-audit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-temporal-input":
        report = evaluate_v7_temporal_input_audit(root)
        target = root / "artifacts/v7-temporal-input-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-local-input":
        report = evaluate_v7_local_input_audit(root)
        target = root / "artifacts/v7-local-input-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-receptor-mask":
        report = evaluate_v7_receptor_mask_audit(root)
        target = root / "artifacts/v7-receptor-mask-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t4-input-coverage":
        report = evaluate_v7_t4_input_coverage(root)
        target = root / "artifacts/v7-t4-input-coverage.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-coverage-response":
        report = evaluate_v7_coverage_response(root)
        target = root / "artifacts/v7-t4-coverage-response.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-mi9-sign":
        report = evaluate_v7_disinhibition(root)
        target = root / "artifacts/v7-mi9-sign-comparison.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-conductance-order":
        report = evaluate_v7_conductance_order(root)
        target = root / "artifacts/v7-conductance-order-comparison.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-background-stability":
        report = evaluate_v7_background_stability(root)
        target = root / "artifacts/v7-background-stability.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-feedback-cut":
        report = evaluate_v7_feedback_cut(root)
        target = root / "artifacts/v7-feedback-cut.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-synchronous-update":
        report = evaluate_v7_synchronous_update(root)
        target = root / "artifacts/v7-synchronous-update.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-normalization-gain":
        report = evaluate_v7_normalization_gain(root)
        target = root / "artifacts/v7-normalization-gain.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-full-update-perturbation":
        report = evaluate_v7_perturbation(root)
        target = root / "artifacts/v7-full-update-perturbation.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t5-data":
        report = evaluate_v7_t5_data_audit(root)
        target = root / "artifacts/v7-t5-data-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-malecns-source-mapping-readiness":
        report = evaluate_v7_malecns_source_mapping_readiness_audit(root)
        target = root / "artifacts/v7-malecns-source-mapping-readiness-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-conductance":
        report = evaluate_v7_t5_conductance_audit(root)
        target = root / "artifacts/v7-t5-conductance-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-extract-t5-phenotype":
        report = evaluate_v7_t5_phenotype(root)
        target = root / "artifacts/v7-t5-phenotype.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-goal-coverage":
        report = evaluate_v7_goal_coverage(root)
        target = root / "artifacts/v7-goal-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t5-labels":
        report = evaluate_v7_t5_label_audit(root)
        target = root / "artifacts/v7-t5-label-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-stage1-split":
        report = evaluate_v7_stage1_split(root)
        target = root / "artifacts/v7-stage1-split.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t5-supplement":
        report = evaluate_v7_t5_supplement_audit(root)
        target = root / "artifacts/v7-t5-supplement-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-stage1-scoring":
        report = evaluate_v7_stage1_scoring(root)
        target = root / "artifacts/v7-stage1-scoring.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-stage1-input":
        report = evaluate_v7_stage1_input_audit(root)
        target = root / "artifacts/v7-stage1-input-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-stage1-development":
        report = evaluate_v7_stage1_development(root)
        target = root / "artifacts/v7-stage1-development.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-stage1-geometry-ab":
        report = evaluate_v7_stage1_geometry_ab(root)
        target = root / "artifacts/v7-stage1-geometry-ab.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-visual-target-inputs":
        report = evaluate_v7_visual_target_input_audit(root)
        target = root / "artifacts/v7-visual-target-input-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-looming-mechanisms":
        report = evaluate_v7_looming_mechanism_audit(root)
        target = root / "artifacts/v7-looming-mechanism-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-stage1-nested":
        report = evaluate_v7_stage1_nested(root)
        target = root / "artifacts/v7-stage1-nested.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-target-fit":
        report = evaluate_v7_target_fit_contract(root)
        target = root / "artifacts/v7-target-fit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-closed-loop-tuning":
        report = evaluate_v7_closed_loop_tuning(root)
        target = root / "artifacts/v7-closed-loop-tuning.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-closed-loop-calibration":
        report = evaluate_v7_closed_loop_calibration(root)
        target = root / "artifacts/v7-closed-loop-calibration.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-closed-loop-controls":
        report = evaluate_v7_closed_loop_controls(root)
        target = root / "artifacts/v7-closed-loop-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-closed-loop-multi":
        report = evaluate_v7_closed_loop_multi(root)
        target = root / "artifacts/v7-closed-loop-multi.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-r1r6-multi-tuning":
        report = evaluate_v7_r1r6_multi_tuning(root)
        target = root / "artifacts/v7-r1r6-multi-tuning.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-r1r6-multi-calibration":
        report = evaluate_v7_r1r6_multi_calibration(root)
        target = root / "artifacts/v7-r1r6-multi-calibration.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-r1r6-local-tuning":
        report = evaluate_v7_r1r6_local_tuning(root)
        target = root / "artifacts/v7-r1r6-local-tuning.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-r1r6-local-calibration":
        report = evaluate_v7_r1r6_local_calibration(root)
        target = root / "artifacts/v7-r1r6-local-calibration.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-r1r6-local-controls":
        report = evaluate_v7_r1r6_local_controls(root)
        target = root / "artifacts/v7-r1r6-local-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-channels-tuning":
        report = evaluate_v7_neural_channels_tuning(root)
        target = root / "artifacts/v7-neural-channels-tuning.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-channels-calibration":
        report = evaluate_v7_neural_channels_calibration(root)
        target = root / "artifacts/v7-neural-channels-calibration.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-channel-controls":
        report = evaluate_v7_neural_channel_controls(root)
        target = root / "artifacts/v7-neural-channel-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-topology-controls":
        report = evaluate_v7_neural_topology_controls(root)
        target = root / "artifacts/v7-neural-topology-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-nested-neural-screen":
        report = evaluate_v7_nested_neural_screen(root)
        target = root / "artifacts/v7-nested-neural-screen.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t5-spatial-order":
        report = evaluate_v7_t5_spatial_order(root)
        target = root / "artifacts/v7-t5-spatial-order.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lplc2-phenotype":
        report = evaluate_v7_lplc2_phenotype(root)
        target = root / "artifacts/v7-lplc2-phenotype.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4-source-resolved":
        report = evaluate_v7_t4_source_resolved(root)
        target = root / "artifacts/v7-t4-source-resolved.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-heading-ring":
        report = evaluate_v7_heading_ring(root)
        target = root / "artifacts/v7-heading-ring.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-local-columns":
        report = evaluate_v7_neural_local_columns(root)
        target = root / "artifacts/v7-neural-local-columns.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-episode-cv":
        report = evaluate_v7_neural_episode_cv(root)
        target = root / "artifacts/v7-neural-episode-cv.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-episode-controls":
        report = evaluate_v7_neural_episode_controls(root)
        target = root / "artifacts/v7-neural-episode-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-fc2-pfl-dna":
        report = evaluate_v7_fc2_pfl_dna(root)
        target = root / "artifacts/v7-fc2-pfl-dna.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-descending-paths":
        report = evaluate_v7_descending_path_audit(root)
        target = root / "artifacts/v7-descending-path-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-descending-propagation-precheck":
        report = evaluate_v7_descending_propagation_precheck(root)
        target = root / "artifacts/v7-descending-propagation-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-navigation-nested":
        report = evaluate_v7_navigation_nested(root)
        target = root / "artifacts/v7-navigation-nested.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-navigation-nested":
        report = evaluate_v7_navigation_nested_candidate(root)
        target = root / "artifacts/v7-navigation-nested-eval.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-danger-throttle":
        report = evaluate_v7_danger_throttle(root)
        target = root / "artifacts/v7-danger-throttle.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-visual-corridor":
        report = evaluate_v7_visual_corridor_goal(root)
        target = root / "artifacts/v7-visual-corridor-goal.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-corridor":
        report = evaluate_v7_neural_corridor(root)
        target = root / "artifacts/v7-neural-corridor.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-local-column-corridor":
        report = evaluate_v7_local_column_corridor(root)
        target = root / "artifacts/v7-local-column-corridor.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-dynamics-local":
        report = evaluate_v7_neural_dynamics_local(root)
        target = root / "artifacts/v7-neural-dynamics-local.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-corridor-dagger":
        report = evaluate_v7_neural_corridor_dagger(root)
        target = root / "artifacts/v7-neural-corridor-dagger.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-visual-layer-locality":
        report = evaluate_v7_visual_layer_locality(root)
        target = root / "artifacts/v7-visual-layer-locality.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lamina-goal":
        report = evaluate_v7_lamina_goal(root)
        target = root / "artifacts/v7-lamina-goal.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lamina-goal-symmetry":
        report = evaluate_v7_lamina_goal_symmetry(root)
        target = root / "artifacts/v7-lamina-goal-symmetry.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4-normalized-correlator":
        report = evaluate_v7_t4_normalized_correlator(root)
        target = root / "artifacts/v7-t4-normalized-correlator.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-fc2-goal-memory":
        report = evaluate_v7_fc2_goal_memory(root)
        target = root / "artifacts/v7-fc2-goal-memory.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lplc-typed-screen":
        report = evaluate_v7_lplc_typed_screen(root)
        target = root / "artifacts/v7-lplc-typed-screen.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lplc2-radial-opponency":
        report = evaluate_v7_lplc2_radial_opponency(root)
        target = root / "artifacts/v7-lplc2-radial-opponency.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lplc2-position-coverage":
        report = evaluate_v7_lplc2_position_coverage(root)
        target = root / "artifacts/v7-lplc2-position-coverage.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4t5-local-edge-precheck":
        report = evaluate_v7_t4t5_local_edge_precheck(root)
        target = root / "artifacts/v7-t4t5-local-edge-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4t5-local-edge-backends":
        report = evaluate_v7_t4t5_local_edge_backends(root)
        target = root / "artifacts/v7-t4t5-local-edge-backends.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t4t5-source-dynamics-readiness":
        report = evaluate_v7_t4t5_source_dynamics_readiness(root)
        target = root / "artifacts/v7-t4t5-source-dynamics-readiness.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-evaluate-t4-local-correlator-precheck":
        report = evaluate_v7_t4_local_correlator_precheck(root)
        target = root / "artifacts/v7-t4-local-correlator-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4-source-pool-local":
        report = evaluate_v7_t4_source_pool_local(root)
        target = root / "artifacts/v7-t4-source-pool-local.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-three-hop-moment":
        report = evaluate_v7_three_hop_moment(root)
        target = root / "artifacts/v7-three-hop-moment.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-three-hop-source-coverage":
        report = evaluate_v7_three_hop_source_coverage(root)
        target = root / "artifacts/v7-three-hop-source-coverage.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-four-hop-scalar-precheck":
        report = evaluate_v7_four_hop_scalar_precheck(root)
        target = root / "artifacts/v7-four-hop-scalar-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lc4-position-speed-precheck":
        report = evaluate_v7_lc4_position_speed_precheck(root)
        target = root / "artifacts/v7-lc4-position-speed-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lc4-input-speed-precheck":
        report = evaluate_v7_lc4_input_speed_precheck(root)
        target = root / "artifacts/v7-lc4-input-speed-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-degree-preserving-control":
        report = evaluate_v7_degree_preserving_control(root)
        target = root / "artifacts/v7-degree-preserving-control.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-parameter-matched-baselines":
        report = evaluate_v7_parameter_matched_baselines(root)
        target = root / "artifacts/v7-parameter-matched-baselines.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t5-lamina-split":
        report = evaluate_v7_t5_lamina_split(root)
        target = root / "artifacts/v7-t5-lamina-split.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t5-lamina-scalar-precheck":
        report = evaluate_v7_t5_lamina_scalar_precheck(root)
        target = root / "artifacts/v7-t5-lamina-scalar-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t5-source-axis":
        report = evaluate_v7_t5_source_axis_audit(root)
        target = root / "artifacts/v7-t5-source-axis-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t5-source-mapping-scope":
        report = evaluate_v7_t5_source_mapping_scope_audit(root)
        target = root / "artifacts/v7-t5-source-mapping-scope-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-source-transfer-synthesis":
        report = evaluate_v7_t5_source_transfer_synthesis_audit(root)
        target = root / "artifacts/v7-t5-source-transfer-synthesis-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-author-row-weighted-full-support":
        report = evaluate_v7_t5_author_row_weighted_full_support_sensitivity(root)
        target = root / "artifacts/v7-t5-author-row-weighted-full-support-sensitivity.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-measured-kernel-identifiability":
        report = evaluate_v7_t5_measured_kernel_identifiability(root)
        target = root / "artifacts/v7-t5-measured-kernel-identifiability.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-measured-kernel-full-support":
        report = evaluate_v7_t5_measured_kernel_full_support_identifiability(root)
        target = root / "artifacts/v7-t5-measured-kernel-full-support-identifiability.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-measured-kernel-cascade-semantics":
        report = evaluate_v7_t5_measured_kernel_cascade_semantics_audit(root)
        target = root / "artifacts/v7-t5-measured-kernel-cascade-semantics-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-population-kernel-robustness":
        report = evaluate_v7_t5_population_kernel_robustness_audit(root)
        target = root / "artifacts/v7-t5-population-kernel-robustness-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-population-kernel-aggregation-semantics":
        report = evaluate_v7_t5_population_kernel_aggregation_semantics_audit(root)
        target = root / "artifacts/v7-t5-population-kernel-aggregation-semantics-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-tm2-loo-full-support":
        report = evaluate_v7_t5_tm2_loo_full_support_sensitivity(root)
        target = root / "artifacts/v7-t5-tm2-loo-full-support-sensitivity.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-measured-kernel-substep-sensitivity":
        report = evaluate_v7_t5_measured_kernel_substep_sensitivity(root)
        target = root / "artifacts/v7-t5-measured-kernel-substep-sensitivity.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-measured-kernel-support-coverage":
        report = evaluate_v7_t5_measured_kernel_support_coverage_audit(root)
        target = root / "artifacts/v7-t5-measured-kernel-support-coverage-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-evaluate-t5-source-pair-precheck":
        report = evaluate_v7_t5_source_pair_precheck(root)
        target = root / "artifacts/v7-t5-source-pair-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t5-typed-spatial-pair-precheck":
        report = evaluate_v7_t5_typed_spatial_pair_precheck(root)
        target = root / "artifacts/v7-t5-typed-spatial-pair-precheck.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-native-direction-waveforms":
        report = evaluate_v7_t5_native_direction_waveform_audit(root)
        target = root / "artifacts/v7-t5-native-direction-waveform-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-physical-time-transfer":
        report = evaluate_v7_t5_physical_time_transfer_audit(root)
        target = root / "artifacts/v7-t5-physical-time-transfer-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-ct1-terminal-axis":
        report = evaluate_v7_t5_ct1_terminal_axis_audit(root)
        target = root / "artifacts/v7-t5-ct1-terminal-axis-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-calibrate-t5-ct1-axis":
        report = evaluate_v7_t5_ct1_axis_calibration(root)
        target = root / "artifacts/v7-t5-ct1-axis-calibration.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-ct1-crossfit-axis":
        report = evaluate_v7_t5_ct1_crossfit_axis_audit(root)
        target = root / "artifacts/v7-t5-ct1-crossfit-axis-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-evaluate-t5-ct1-axis-aware-precheck":
        report = evaluate_v7_t5_ct1_axis_aware_precheck(root)
        target = root / "artifacts/v7-t5-ct1-axis-aware-precheck.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-evaluate-t5-ct1-axis-aware-antisymmetric-precheck":
        report = evaluate_v7_t5_ct1_axis_aware_antisymmetric_precheck(root)
        target = root / "artifacts/v7-t5-ct1-axis-aware-antisymmetric-precheck.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-ct1-axis-sequence-identifiability":
        report = evaluate_v7_t5_ct1_axis_sequence_identifiability(root)
        target = root / "artifacts/v7-t5-ct1-axis-sequence-identifiability.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-evaluate-t5-ct1-source-dynamics":
        report = evaluate_v7_t5_ct1_source_dynamics_precheck(root)
        target = root / "artifacts/v7-t5-ct1-source-dynamics-precheck.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-evaluate-t5-ct1-multiplicative-precheck":
        report = evaluate_v7_t5_ct1_multiplicative_precheck(root)
        target = root / "artifacts/v7-t5-ct1-multiplicative-precheck.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-evaluate-t5-continuous-moment-precheck":
        report = evaluate_v7_t5_continuous_moment_precheck(root)
        target = root / "artifacts/v7-t5-continuous-moment-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4-continuous-pair-precheck":
        report = evaluate_v7_t4_continuous_pair_precheck(root)
        target = root / "artifacts/v7-t4-continuous-pair-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-three-hop-temporal-consistency":
        report = evaluate_v7_three_hop_temporal_consistency(root)
        target = root / "artifacts/v7-three-hop-temporal-consistency.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-source-type-temporal-identifiability":
        report = evaluate_v7_source_type_temporal_identifiability(root)
        target = root / "artifacts/v7-source-type-temporal-identifiability.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-freeze-source-dynamics-external-evidence-contract":
        report = evaluate_v7_source_dynamics_external_evidence_contract(root)
        target = root / "artifacts/v7-source-dynamics-external-evidence-contract.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t4-pair-lags":
        report = evaluate_v7_t4_pair_lag_audit(root)
        target = root / "artifacts/v7-t4-pair-lag-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-upstream-latency":
        report = evaluate_v7_upstream_latency_audit(root)
        target = root / "artifacts/v7-upstream-latency-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-synapse-spatial":
        report = evaluate_v7_synapse_spatial_audit(root)
        target = root / "artifacts/v7-synapse-spatial-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-calibrate-synapse-axis":
        report = evaluate_v7_synapse_axis_calibration(root)
        target = root / "artifacts/v7-synapse-axis-calibration.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4-synapse-correlator-precheck":
        report = evaluate_v7_t4_synapse_correlator_precheck(root)
        target = root / "artifacts/v7-t4-synapse-correlator-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t4-synapse-crossfit-axis":
        report = evaluate_v7_t4_synapse_crossfit_axis_audit(root)
        target = root / "artifacts/v7-t4-synapse-crossfit-axis-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-evaluate-t4-synapse-crossfit-precheck":
        report = evaluate_v7_t4_synapse_crossfit_precheck(root)
        target = root / "artifacts/v7-t4-synapse-crossfit-precheck.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t4-crossfit-sequence-identifiability":
        report = evaluate_v7_t4_crossfit_sequence_identifiability(root)
        target = root / "artifacts/v7-t4-crossfit-sequence-identifiability.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t4-crossfit-retinal-symmetry":
        report = evaluate_v7_t4_crossfit_retinal_symmetry_audit(root)
        target = root / "artifacts/v7-t4-crossfit-retinal-symmetry-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-evaluate-t4-synapse-antisymmetric-precheck":
        report = evaluate_v7_t4_synapse_antisymmetric_precheck(root)
        target = root / "artifacts/v7-t4-synapse-antisymmetric-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4-synapse-centered-precheck":
        report = evaluate_v7_t4_synapse_centered_precheck(root)
        target = root / "artifacts/v7-t4-synapse-centered-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4-synapse-microstep-precheck":
        report = evaluate_v7_t4_synapse_microstep_precheck(root)
        target = root / "artifacts/v7-t4-synapse-microstep-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t4-source-dynamics-transfer":
        report = evaluate_v7_t4_source_dynamics_transfer_audit(root)
        target = root / "artifacts/v7-t4-source-dynamics-transfer-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t4-state-unit-mapping":
        report = evaluate_v7_t4_state_unit_mapping_audit(root)
        target = root / "artifacts/v7-t4-state-unit-mapping-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-edmond-fig3-retrieval":
        report = evaluate_v7_edmond_fig3_retrieval_audit(root)
        target = root / "artifacts/v7-edmond-fig3-retrieval-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t4-source-identity-readiness":
        report = evaluate_v7_t4_source_identity_readiness_audit(root)
        target = root / "artifacts/v7-t4-source-identity-readiness-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t4-individual-split":
        report = evaluate_v7_t4_individual_split_audit(root)
        target = root / "artifacts/v7-t4-individual-split-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t4-individual-split-1khz":
        report = evaluate_v7_t4_individual_split_1khz_audit(root)
        target = root / "artifacts/v7-t4-individual-split-1khz-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t4-recording-fields":
        report = evaluate_v7_t4_recording_field_audit(root)
        target = root / "artifacts/v7-t4-recording-field-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-recording-fields":
        report = evaluate_v7_t5_recording_field_audit(root)
        target = root / "artifacts/v7-t5-recording-field-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-behnia-t4-fast-sources":
        report = evaluate_v7_behnia_t4_fast_source_audit(root)
        target = root / "artifacts/v7-behnia-t4-fast-source-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-behnia-t4-fast-source-availability":
        report = evaluate_v7_behnia_t4_fast_source_availability_audit(root)
        target = root / "artifacts/v7-behnia-t4-fast-source-availability-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-matulis-mi1-voltage-availability":
        report = evaluate_v7_matulis_mi1_voltage_availability_audit(root)
        target = root / "artifacts/v7-matulis-mi1-voltage-availability-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-borst-2025-temporal-filtering":
        report = evaluate_v7_borst_2025_temporal_filtering_audit(root)
        target = root / "artifacts/v7-borst-2025-temporal-filtering-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t4-inhibitory-source-external":
        report = evaluate_v7_t4_inhibitory_source_external_audit(root)
        target = root / "artifacts/v7-t4-inhibitory-source-external-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-mi4-c3-whole-cell-candidates":
        report = evaluate_v7_mi4_c3_whole_cell_candidate_audit(root)
        target = root / "artifacts/v7-mi4-c3-whole-cell-candidate-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-unified-model-package":
        report = evaluate_v7_unified_model_package_audit(root)
        target = root / "artifacts/v7-unified-model-package-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-fig3-source-kernels":
        report = evaluate_v7_fig3_source_kernel_audit(root)
        target = root / "artifacts/v7-fig3-source-kernel-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-fig3-source-kernel-robustness":
        report = evaluate_v7_fig3_source_kernel_robustness(root)
        target = root / "artifacts/v7-fig3-source-kernel-robustness.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-arenz-source-dynamics":
        report = evaluate_v7_arenz_source_dynamics_audit(root)
        target = root / "artifacts/v7-arenz-source-dynamics-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-arenz-t5-source-dynamics":
        report = evaluate_v7_arenz_t5_source_dynamics_audit(root)
        target = root / "artifacts/v7-arenz-t5-source-dynamics-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-c3-strf-source-dynamics":
        report = evaluate_v7_c3_strf_source_dynamics_audit(root)
        target = root / "artifacts/v7-c3-strf-source-dynamics-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-c3-flash-preregistration":
        report = evaluate_v7_c3_flash_preregistration(root)
        target = root / "artifacts/v7-c3-flash-preregistration.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-c3-strf-flash-transfer":
        report = evaluate_v7_c3_strf_flash_transfer(root)
        target = root / "artifacts/v7-c3-strf-flash-transfer.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-precheck-c3-analytic-filter":
        report = evaluate_v7_c3_analytic_filter_precheck(root)
        target = root / "artifacts/v7-c3-analytic-filter-precheck.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-c3-measured-filter-robustness":
        report = evaluate_v7_c3_measured_filter_robustness(root)
        target = root / "artifacts/v7-c3-measured-filter-robustness.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-timing-models-source-filters":
        report = evaluate_v7_timing_models_source_filter_audit(root)
        target = root / "artifacts/v7-timing-models-source-filter-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-timing-models-ct1-compartment":
        report = evaluate_v7_timing_models_ct1_compartment_audit(root)
        target = root / "artifacts/v7-timing-models-ct1-compartment-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-flyvis-c3-time-constant":
        report = evaluate_v7_flyvis_c3_time_constant_audit(root)
        target = root / "artifacts/v7-flyvis-c3-time-constant-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-flyvis-visual-source-time-constants":
        report = evaluate_v7_flyvis_visual_source_time_constants(root)
        target = root / "artifacts/v7-flyvis-visual-source-time-constants.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-flyvis-c3-effective-dynamics":
        report = evaluate_v7_flyvis_c3_effective_dynamics_audit(root)
        target = root / "artifacts/v7-flyvis-c3-effective-dynamics-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-public-t4-model-source-coverage":
        report = evaluate_v7_public_t4_model_source_coverage_audit(root)
        target = root / "artifacts/v7-public-t4-model-source-coverage-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-dryad-l1l2-source-dynamics":
        report = evaluate_v7_dryad_l1l2_source_dynamics_audit(root)
        target = root / "artifacts/v7-dryad-l1l2-source-dynamics-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-gou-sparsity-source-dynamics":
        report = evaluate_v7_gou_sparsity_source_dynamics_audit(root)
        target = root / "artifacts/v7-gou-sparsity-source-dynamics-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-gou-moving-bar-directions":
        report = evaluate_v7_gou_moving_bar_direction_audit(root)
        target = root / "artifacts/v7-gou-moving-bar-direction-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-gou-dandi-identity":
        report = evaluate_v7_gou_dandi_identity_audit(root)
        target = root / "artifacts/v7-gou-dandi-identity-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-gou-dandi-stimulus-metadata":
        report = evaluate_v7_gou_dandi_stimulus_metadata_audit(root)
        target = root / "artifacts/v7-gou-dandi-stimulus-metadata-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-pirogova-source-calcium":
        report = evaluate_v7_pirogova_source_calcium_audit(root)
        target = root / "artifacts/v7-pirogova-source-calcium-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-contrast-opponency-source-data":
        report = evaluate_v7_t5_contrast_opponency_source_data_audit(root)
        target = root / "artifacts/v7-t5-contrast-opponency-source-data-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-yang-t5-voltage-evidence":
        report = evaluate_v7_yang_t5_voltage_evidence_audit(root)
        target = root / "artifacts/v7-yang-t5-voltage-evidence-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-source-evidence-matrix":
        report = evaluate_v7_source_evidence_matrix(root)
        target = root / "artifacts/v7-source-evidence-matrix.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-source-type-average-mapping":
        report = evaluate_v7_source_type_average_mapping_contract(root)
        target = root / "artifacts/v7-source-type-average-mapping-contract.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-t5-ephys":
        report = evaluate_v7_kohn_portes_t5_ephys_audit(root)
        target = root / "artifacts/v7-kohn-portes-t5-ephys-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-t5-frequency-tuning":
        report = evaluate_v7_kohn_portes_t5_frequency_tuning_audit(root)
        target = root / "artifacts/v7-kohn-portes-t5-frequency-tuning-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-t5-moving-bar-generalization":
        report = evaluate_v7_kohn_portes_t5_moving_bar_generalization_audit(root)
        target = root / "artifacts/v7-kohn-portes-t5-moving-bar-generalization-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-t5-peak-latency":
        report = evaluate_v7_kohn_portes_t5_peak_latency_audit(root)
        target = root / "artifacts/v7-kohn-portes-t5-peak-latency-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-t5-source-kernels":
        report = evaluate_v7_kohn_portes_t5_source_kernel_audit(root)
        target = root / "artifacts/v7-kohn-portes-t5-source-kernel-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-t5-state-unit-mapping":
        report = evaluate_v7_kohn_portes_t5_state_unit_mapping_audit(root)
        target = root / "artifacts/v7-kohn-portes-t5-state-unit-mapping-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-tm-to-t5-model":
        report = evaluate_v7_kohn_portes_tm_to_t5_model_audit(root)
        target = root / "artifacts/v7-kohn-portes-tm-to-t5-model-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-figure6-model-identity":
        report = evaluate_v7_kohn_portes_figure6_model_identity_audit(root)
        target = root / "artifacts/v7-kohn-portes-figure6-model-identity-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-axolotl-availability":
        report = evaluate_v7_kohn_portes_axolotl_availability_audit(root)
        target = root / "artifacts/v7-kohn-portes-axolotl-availability-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-fib19-malecns-t5-weight-transfer":
        report = evaluate_v7_fib19_malecns_t5_weight_transfer_audit(root)
        target = root / "artifacts/v7-fib19-malecns-t5-weight-transfer-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-identity-history":
        report = evaluate_v7_kohn_portes_identity_history_audit(root)
        target = root / "artifacts/v7-kohn-portes-identity-history-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-external-indexes":
        report = evaluate_v7_kohn_portes_external_index_audit(root)
        target = root / "artifacts/v7-kohn-portes-external-index-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-kohn-portes-stimulus-provenance":
        report = evaluate_v7_kohn_portes_stimulus_provenance_audit(root)
        target = root / "artifacts/v7-kohn-portes-stimulus-provenance-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-t5-direction-code-provenance":
        report = evaluate_v7_t5_direction_code_provenance_audit(root)
        target = root / "artifacts/v7-t5-direction-code-provenance-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-motyxia2-public-history":
        report = evaluate_v7_motyxia2_public_history_audit(root)
        target = root / "artifacts/v7-motyxia2-public-history-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-ct1-extreme-compartmentalization":
        report = evaluate_v7_ct1_extreme_compartmentalization_audit(root)
        target = root / "artifacts/v7-ct1-extreme-compartmentalization-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-ct1-pure-data-index":
        report = evaluate_v7_ct1_pure_data_index_audit(root)
        target = root / "artifacts/v7-ct1-pure-data-index-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-ct1-experimental-voltage-boundary":
        report = evaluate_v7_ct1_experimental_voltage_boundary_audit(root)
        target = root / "artifacts/v7-ct1-experimental-voltage-boundary-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-tm9-coordinate-identifiability":
        report = evaluate_v7_tm9_coordinate_identifiability_audit(root)
        target = root / "artifacts/v7-tm9-coordinate-identifiability-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-malecns-synapse-columns":
        report = evaluate_v7_malecns_synapse_column_audit(root)
        target = root / "artifacts/v7-malecns-synapse-column-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-malecns-ct1-columns":
        report = evaluate_v7_malecns_ct1_columnar_audit(root)
        target = root / "artifacts/v7-malecns-ct1-columnar-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-fig1-source-temporal-readiness":
        report = evaluate_v7_fig1_source_temporal_readiness_audit(root)
        target = root / "artifacts/v7-fig1-source-temporal-readiness-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-evaluate-lplc1-near-collision-precheck":
        report = evaluate_v7_lplc1_near_collision_precheck(root)
        target = root / "artifacts/v7-lplc1-near-collision-precheck.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-lplc1-input-structure":
        report = evaluate_v7_lplc1_input_structure(root)
        target = root / "artifacts/v7-lplc1-input-structure.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-goal-fusion":
        report = evaluate_v7_neural_goal_fusion(root)
        target = root / "artifacts/v7-neural-goal-fusion.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-fusion-nested":
        report = evaluate_v7_fusion_nested(root)
        target = root / "artifacts/v7-fusion-nested.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-fusion-nested":
        report = evaluate_v7_fusion_nested_candidate(root)
        target = root / "artifacts/v7-fusion-nested-eval.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-build-ephys-interface":
        report = evaluate_v7_ephys_interface(root)
        target = root / "artifacts/v7-ephys-interface.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-validate-published-fig5":
        report = evaluate_v7_fig5_validation(root)
        target = root / "artifacts/v7-fig5-validation.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-timebase":
        report = evaluate_v7_timebase_audit(root)
        target = root / "artifacts/v7-timebase-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-stimulus-coordinates":
        report = evaluate_v7_stimulus_coordinate_contract(root)
        target = root / "artifacts/v7-stimulus-coordinate-contract.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-controlled-stimulus-angular-grid":
        report = evaluate_v7_controlled_stimulus_angular_grid(root)
        target = root / "artifacts/v7-controlled-stimulus-angular-grid.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-controlled-stimulus-input-boundary":
        report = evaluate_v7_controlled_stimulus_input_boundary_audit(root)
        target = root / "artifacts/v7-controlled-stimulus-input-boundary-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-vertical-angular-coordinates":
        report = evaluate_v7_vertical_angular_coordinate_audit(root)
        target = root / "artifacts/v7-vertical-angular-coordinate-audit.json"
        target.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(target)
        return
    if args.command == "v7-audit-electrophysiology":
        report = evaluate_v7_electrophysiology_audit(root)
        target = root / "artifacts/v7-electrophysiology-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-spectra":
        report = evaluate_v7_neural_spectra(root)
        target = root / "artifacts/v7-neural-spectra.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-spectral-controls":
        report = evaluate_v7_spectral_controls(root)
        target = root / "artifacts/v7-spectral-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-pixel-sampling":
        report = evaluate_v7_pixel_sampling(root)
        target = root / "artifacts/v7-pixel-sampling.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-geometry-sign":
        report = evaluate_v7_geometry_sign(root)
        target = root / "artifacts/v7-geometry-sign.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-phase-motion":
        report = evaluate_v7_phase_motion(root)
        target = root / "artifacts/v7-phase-motion.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    manifest = load_manifest(root / args.manifest)
    if args.command in {"verify", "download"}:
        selected = set(args.dataset) if args.dataset else None
        failures: list[str] = []
        for spec in iter_files(manifest, selected):
            if spec.optional and not getattr(args, "include_optional", False):
                continue
            if args.command == "download":
                print(f"downloading {spec.dataset}/{spec.name} -> {spec.path}")
                download_file(spec, root)
            errors = spec.verify(root)
            print(f"{'OK' if not errors else 'FAIL'} {spec.dataset}/{spec.name}")
            failures.extend(errors)
        if failures:
            raise SystemExit("\n".join(failures))
        return

    annotations = root / "data/raw/malecns-v1.0/body-annotations.feather"
    if args.command == "build-graph":
        graph = build_canonical_graph(
            annotations,
            root / "data/raw/malecns-v1.0/connectome-weights.feather",
            root / "data/processed/malecns-v1.0",
        )
        print(f"built {graph.node_count} nodes and {graph.edge_count} edges")
        return
    if args.command == "build-overview":
        overview = build_overview(annotations, root / "data/processed/malecns-v1.0/overview.json")
        print(f"built overview for {overview['positioned_nodes']} positioned neurons")
        return
    if args.command == "build-pathways":
        graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
        pathways = build_pathways(
            graph, annotations, root / "data/processed/malecns-v1.0/pathways.json"
        )
        print(
            f"built {len(pathways['bundled_paths'])} bundles and "
            f"{len(pathways['strong_paths'])} strong paths from "
            f"{pathways['all_edges_accounted']} edges"
        )
        return
    transmitters = root / "data/raw/malecns-v1.0/body-neurotransmitters.feather"
    annotation_report, canonical = audit_annotations(annotations)
    report = {
        "malecns": {
            "annotations": annotation_report,
            "neurotransmitters": audit_neurotransmitters(transmitters, canonical),
            "connections": audit_connections(
                root / "data/raw/malecns-v1.0/connectome-weights.feather", canonical
            ),
        }
    }
    expected = manifest["datasets"]["malecns"]
    observed = report["malecns"]
    checks = {
        "expected_canonical_nodes": observed["annotations"]["canonical_nodes"],
        "expected_source_connection_rows": observed["connections"]["source_rows"],
        "expected_canonical_edges": observed["connections"]["canonical_edges"],
        "expected_canonical_synapse_weight_sum": observed["connections"][
            "canonical_synapse_weight_sum"
        ],
    }
    mismatches = {
        key: (expected[key], value) for key, value in checks.items() if expected[key] != value
    }
    if mismatches:
        raise ValueError(f"MaleCNS manifest count mismatch: {mismatches}")
    target = root / "artifacts/data-audit.json"
    write_json(report, target)
    print(target)


if __name__ == "__main__":
    main()
