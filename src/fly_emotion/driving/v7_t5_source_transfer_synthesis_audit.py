"""Synthesize T5 source evidence without changing the frozen transfer gate."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t5-source-transfer-synthesis-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_source_transfer_synthesis_audit.py"
)


def evaluate_v7_t5_source_transfer_synthesis_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_paths = {name: Path(path) for name, path in config["evidence"].items()}
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }
    physical = evidence["physical_transfer"]
    if list(physical["required_fields"]) != config["expected"]["original_required_fields"]:
        raise ValueError("original T5 physical-transfer field contract changed")
    if physical["missing_fields"] != config["expected"]["original_missing_fields"]:
        raise ValueError("original T5 physical-transfer missing fields changed")
    if physical["T5_physical_source_transfer_ready"]:
        raise ValueError("original T5 physical-transfer gate unexpectedly passed")

    source_order = config["expected"]["source_order"]
    mapping = evidence["source_mapping"]
    model_identity = evidence["figure6_model_identity"]
    axolotl_availability = evidence["axolotl_availability"]
    measured_kernel = evidence["measured_kernel_identifiability"]
    measured_kernel_substeps = evidence["measured_kernel_substep_sensitivity"]
    measured_kernel_support = evidence["measured_kernel_support_coverage"]
    measured_kernel_full_support = evidence["measured_kernel_full_support"]
    population_kernel_robustness = evidence["population_kernel_robustness"]
    population_kernel_aggregation = evidence[
        "population_kernel_aggregation_semantics"
    ]
    author_row_weighted = evidence["author_row_weighted_full_support"]
    tm2_loo_full_support = evidence["tm2_loo_full_support"]
    measured_kernel_cascade = evidence["measured_kernel_cascade_semantics"]
    measured_kernel_inputs = evidence[
        "measured_kernel_source_input_decomposition"
    ]
    lamina_only_replacement = evidence["lamina_only_measured_kernel_replacement"]
    shuffle_input_energy = evidence["temporal_shuffle_input_energy"]
    rows = {}
    for source in source_order:
        has_kernel = source in evidence["source_kernels"]["source_results"]
        rows[source] = {
            "voltage_derived_temporal_kernel": has_kernel,
            "exact_type_average_mapping": mapping["source_mappings"][source][
                "mapping_scope_complete"
            ],
            "state_mapping_available": bool(
                has_kernel
                and evidence["state_units"][
                    "millivolts_or_filter_output_to_v7_state_mapping_available"
                ]
            ),
            "delay_transfer_authorized": bool(
                has_kernel
                and evidence["peak_latency"]["authorize_source_delay_transfer_to_v7"]
            ),
            "frequency_transfer_authorized": bool(
                has_kernel
                and evidence["frequency_tuning"][
                    "authorize_source_frequency_transfer_to_v7"
                ]
            ),
            "Tm_to_T5_model_transfer_authorized": bool(
                has_kernel
                and evidence["author_model"][
                    "authorize_Tm_to_T5_model_transfer_to_v7"
                ]
            ),
            "connectome_weight_transfer_authorized": bool(
                has_kernel
                and evidence["connectome_weights"][
                    "authorize_FIB19_weight_transfer_to_MaleCNS"
                ]
            ),
            "moving_bar_generalization_transfer_authorized": bool(
                has_kernel
                and evidence["moving_bar_generalization"][
                    "authorize_moving_bar_generalization_transfer_to_v7"
                ]
            ),
        }
    gates = {
        "four_Tm_voltage_derived_kernel_shapes_available": all(
            rows[source]["voltage_derived_temporal_kernel"] for source in source_order[:4]
        ),
        "four_Tm_exact_type_average_mapping_complete": all(
            rows[source]["exact_type_average_mapping"] for source in source_order[:4]
        ),
        "saline_relative_Tm9_delay_candidate_available": evidence["peak_latency"][
            "relative_Tm9_delay_candidate_supported_in_saline"
        ],
        "relative_low_frequency_Tm9_shape_evidence_available": evidence[
            "frequency_tuning"
        ]["relative_low_frequency_Tm9_shape_evidence_available"],
        "cross_stimulus_moving_bar_shape_candidate_available": evidence[
            "moving_bar_generalization"
        ]["cross_stimulus_relative_shape_candidate_available"],
        "official_T5_target_model_parameters_available": model_identity[
            "official_T5_target_model_parameters_available"
        ],
        "official_target_model_to_T5_source_mapping_available": model_identity[
            "T5_source_mapping_available"
        ],
        "external_Figure6_axolotl_tmodel_source_available": axolotl_availability[
            "gates"
        ]["Figure6_axolotl_source_recovered"],
        "related_axolotl_project_identified": axolotl_availability["gates"][
            "related_GitLab_project_metadata_found"
        ],
        "related_axolotl_repository_anonymously_readable": axolotl_availability[
            "gates"
        ]["related_GitLab_repository_anonymously_readable"],
        "measured_kernel_temporal_identifiability_passed": measured_kernel[
            "temporal_identifiability_passed"
        ],
        "measured_kernel_direction_scoring_performed": measured_kernel[
            "direction_scoring_performed"
        ],
        "measured_kernel_to_stimulus_frame_alignment_verified": measured_kernel[
            "timebase_contract"
        ]["kernel_to_stimulus_frame_alignment_verified"],
        "measured_kernel_to_probe_solver_alignment_verified": measured_kernel[
            "timebase_contract"
        ]["external_recording_to_probe_solver_alignment_verified"],
        "measured_kernel_physical_source_dynamics_transfer_authorized": measured_kernel[
            "authorize_physical_source_dynamics_transfer"
        ],
        "measured_kernel_standard_substep_negative_result_reproduced": (
            measured_kernel_substeps["standard_substep_negative_result_reproduced"]
        ),
        "measured_kernel_cross_substep_temporal_identifiability_passed": (
            measured_kernel_substeps[
                "cross_substep_temporal_identifiability_passed"
            ]
        ),
        "measured_kernel_cross_substep_direction_scoring_performed": (
            measured_kernel_substeps["direction_scoring_performed"]
        ),
        "measured_kernel_cross_substep_direction_scoring_authorized": (
            measured_kernel_substeps["direction_scoring_authorized"]
        ),
        "measured_kernel_causal_index_zero_supported": measured_kernel_support[
            "author_convolution_semantics"
        ]["causal_index_zero_interpretation_supported"],
        "measured_kernel_all_population_peaks_covered": measured_kernel_support[
            "coverage_gate"
        ]["all_population_absolute_peaks_within_scored_trace"],
        "measured_kernel_full_L1_support_covered": measured_kernel_support[
            "coverage_gate"
        ]["all_population_kernel_L1_mass_coverage_passed"],
        "measured_kernel_prefix_alone_full_support_negative_authorized": (
            measured_kernel_support["full_support_negative_conclusion_authorized"]
        ),
        "measured_kernel_zero_tail_full_support_evaluated": (
            measured_kernel_full_support["full_kernel_support_evaluated"]
        ),
        "measured_kernel_zero_tail_cross_substep_identifiability_passed": (
            measured_kernel_full_support[
                "cross_substep_temporal_identifiability_passed"
            ]
        ),
        "measured_kernel_zero_tail_all_candidates_failed_every_update_count": (
            measured_kernel_full_support[
                "all_candidates_failed_every_update_count"
            ]
        ),
        "measured_kernel_zero_tail_direction_scoring_authorized": (
            measured_kernel_full_support["direction_scoring_authorized"]
        ),
        "measured_kernel_zero_tail_direction_scoring_performed": (
            measured_kernel_full_support["direction_scoring_performed"]
        ),
        "population_kernel_recording_id_robustness_passed": (
            population_kernel_robustness[
                "all_population_kernel_robustness_gates_passed"
            ]
        ),
        "Tm2_population_kernel_recording_id_robustness_passed": (
            population_kernel_robustness["source_results"]["Tm2"]["passed"]
        ),
        "population_kernel_independent_biological_validation_available": (
            population_kernel_robustness[
                "independent_biological_validation_available"
            ]
        ),
        "population_kernel_transfer_authorized": population_kernel_robustness[
            "authorize_population_kernel_transfer"
        ],
        "population_kernel_author_default_baseline_verified": (
            population_kernel_aggregation[
                "author_default_baseline_semantics_verified"
            ]
        ),
        "population_kernel_current_no_baseline_matches_author_default": (
            population_kernel_aggregation[
                "current_no_baseline_matches_author_default"
            ]
        ),
        "population_kernel_author_row_weighting_exactly_reproduced": (
            population_kernel_aggregation[
                "author_row_weighted_aggregation_exactly_reproduced"
            ]
        ),
        "population_kernel_alternative_tail_baseline_authorized": (
            population_kernel_aggregation[
                "alternative_tail_baseline_variant_authorized"
            ]
        ),
        "author_row_weighted_full_support_all_candidates_failed": (
            author_row_weighted["all_candidates_failed_every_update_count"]
        ),
        "author_row_weighted_cross_substep_identifiability_passed": (
            author_row_weighted["cross_substep_temporal_identifiability_passed"]
        ),
        "author_row_weighted_direction_scoring_authorized": author_row_weighted[
            "direction_scoring_authorized"
        ],
        "author_row_weighted_direction_scoring_performed": author_row_weighted[
            "direction_scoring_performed"
        ],
        "Tm2_LOO_full_support_all_evaluations_failed": tm2_loo_full_support[
            "all_candidates_failed_every_fold_and_update_count"
        ],
        "Tm2_LOO_robust_temporal_identifiability_passed": tm2_loo_full_support[
            "robust_temporal_identifiability_passed"
        ],
        "Tm2_LOO_direction_scoring_authorized": tm2_loo_full_support[
            "direction_scoring_authorized"
        ],
        "Tm2_LOO_direction_scoring_performed": tm2_loo_full_support[
            "direction_scoring_performed"
        ],
        "Tm2_LOO_independent_biological_validation_performed": (
            tm2_loo_full_support["independent_biological_validation_performed"]
        ),
        "measured_kernel_typed_recurrent_cascade_verified": (
            measured_kernel_cascade[
                "typed_recurrent_source_state_then_measured_kernel_cascade_verified"
            ]
        ),
        "measured_kernel_replaces_existing_source_dynamics": (
            measured_kernel_cascade[
                "measured_kernel_replaces_existing_source_dynamics"
            ]
        ),
        "measured_kernel_single_stage_biological_interpretation_authorized": (
            measured_kernel_cascade[
                "single_stage_biological_source_model_interpretation_authorized"
            ]
        ),
        "measured_kernel_external_state_mapping_available": measured_kernel_cascade[
            "external_recording_to_v7_source_state_mapping_available"
        ],
        "measured_kernel_source_inputs_partitioned_exactly_once": (
            measured_kernel_inputs["all_direct_input_edges_partitioned_exactly_once"]
        ),
        "measured_kernel_every_source_has_recurrent_or_feedback_input": (
            measured_kernel_inputs[
                "every_source_has_recurrent_or_target_feedback_input"
            ]
        ),
        "measured_kernel_feedforward_only_source_drive_available": (
            measured_kernel_inputs[
                "feedforward_only_source_drive_available_from_current_trace"
            ]
        ),
        "measured_kernel_source_dynamics_replacement_evaluated": (
            measured_kernel_inputs[
                "measured_kernel_replacement_of_source_dynamics_evaluated"
            ]
        ),
        "measured_kernel_source_dynamics_replacement_authorized": (
            measured_kernel_inputs["authorize_source_dynamics_replacement"]
        ),
        "lamina_only_measured_kernel_replacement_evaluated": True,
        "lamina_only_measured_kernel_all_candidates_failed": (
            lamina_only_replacement["all_candidates_failed_every_update_count"]
        ),
        "lamina_only_measured_kernel_temporal_identifiability_passed": (
            lamina_only_replacement["temporal_identifiability_passed"]
        ),
        "lamina_only_measured_kernel_direction_scoring_authorized": (
            lamina_only_replacement["direction_scoring_authorized"]
        ),
        "lamina_only_measured_kernel_direction_scoring_performed": (
            lamina_only_replacement["direction_scoring_performed"]
        ),
        "lamina_only_measured_kernel_physical_transfer_authorized": (
            lamina_only_replacement["authorize_physical_source_dynamics_transfer"]
        ),
        "temporal_shuffle_frame_multiset_preserved": shuffle_input_energy[
            "frame_multiset_preserved_for_every_shuffle"
        ],
        "temporal_shuffle_retinal_energy_preserved": shuffle_input_energy[
            "retinal_temporal_energy_exactly_preserved"
        ],
        "temporal_shuffle_lamina_source_energy_preserved": shuffle_input_energy[
            "lamina_only_source_energy_exactly_preserved"
        ],
        "temporal_shuffle_energy_matched_control_verified": shuffle_input_energy[
            "energy_matched_temporal_shuffle_control_verified"
        ],
        "equal_energy_temporal_selectivity_interpretation_authorized": (
            shuffle_input_energy[
                "equal_energy_temporal_selectivity_interpretation_authorized"
            ]
        ),
        "new_energy_normalized_gate_authorized": shuffle_input_energy[
            "authorize_new_energy_normalized_gate"
        ],
        "absolute_source_gain_available": evidence["source_kernels"]["gates"][
            "raw_temporal_filter_absolute_gain_transferable"
        ],
        "source_to_v7_state_mapping_available": evidence["state_units"][
            "millivolts_or_filter_output_to_v7_state_mapping_available"
        ],
        "state_invariant_source_timing_available": evidence["peak_latency"][
            "relative_Tm9_delay_candidate_supported_across_states"
        ],
        "independent_moving_bar_validation_available": evidence[
            "moving_bar_generalization"
        ]["independent_moving_bar_validation_available"],
        "MaleCNS_connectome_weight_transfer_available": evidence[
            "connectome_weights"
        ]["authorize_FIB19_weight_transfer_to_MaleCNS"],
        "CT1_allowed_dynamics_and_mapping_complete": rows["CT1"][
            "voltage_derived_temporal_kernel"
        ]
        and rows["CT1"]["exact_type_average_mapping"],
        "original_physical_transfer_gate_passed": physical[
            "T5_physical_source_transfer_ready"
        ],
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in evidence_paths.values()},
            },
            "parameter_fit": False,
            "functional_stimulus_evaluated": False,
            "runtime_modified": False,
        },
        "original_physical_transfer_contract": {
            "required_fields": physical["required_fields"],
            "missing_fields": physical["missing_fields"],
            "ready": physical["T5_physical_source_transfer_ready"],
        },
        "source_rows": rows,
        "incremental_evidence": {
            "voltage_derived_kernel_source_count": sum(
                row["voltage_derived_temporal_kernel"] for row in rows.values()
            ),
            "exact_type_average_mapping_source_count": sum(
                row["exact_type_average_mapping"] for row in rows.values()
            ),
            "Figure5_training_fit_count": sum(
                values["count"]
                for values in evidence["author_model"][
                    "Figure5_Tm1_Tm9_static_flash_regression"
                ]["aggregate_training_R2"].values()
            ),
            "moving_bar_condition_count": len(
                evidence["moving_bar_generalization"]["condition_results"]
            ),
            "measured_kernel_trace_samples": measured_kernel_support["window"][
                "post_baseline_trace_samples"
            ],
            "measured_kernel_minimum_prefix_L1_mass_fraction": min(
                item["scored_prefix_L1_mass_fraction"]
                for item in measured_kernel_support["source_results"].values()
            ),
            "measured_kernel_maximum_prefix_L1_mass_fraction": max(
                item["scored_prefix_L1_mass_fraction"]
                for item in measured_kernel_support["source_results"].values()
            ),
            "measured_kernel_zero_tail_samples": measured_kernel_full_support[
                "full_support_contract"
            ]["zero_tail_samples"],
            "measured_kernel_full_support_output_samples": measured_kernel_full_support[
                "full_support_contract"
            ]["output_samples"],
            "population_kernel_robustness_passing_source_count": len(
                population_kernel_robustness["passing_sources"]
            ),
            "population_kernel_robustness_failing_sources": (
                population_kernel_robustness["failing_sources"]
            ),
            "Tm2_recording_id_vs_rest_median_correlation": (
                population_kernel_robustness["source_results"]["Tm2"][
                    "held_out_recording_id_vs_rest"
                ]["summary"]["median"]
            ),
            "Tm2_partition_correlation_p05": population_kernel_robustness[
                "source_results"
            ]["Tm2"]["exhaustive_near_equal_recording_id_partitions"][
                "summary"
            ]["p05"],
            "population_kernel_author_call_count": population_kernel_aggregation[
                "call_summary"
            ]["call_count"],
            "population_kernel_explicit_baseline_call_count": (
                population_kernel_aggregation["call_summary"][
                    "calls_with_explicit_baseline"
                ]
            ),
            "Tm1_row_vs_recording_id_weighted_correlation": (
                population_kernel_aggregation["source_results"]["Tm1"][
                    "row_weighted_vs_equal_recording_id_correlation"
                ]
            ),
            "author_row_weighted_changed_sources": author_row_weighted[
                "aggregation_contract"
            ]["changed_sources"],
            "Tm2_LOO_fold_count": tm2_loo_full_support["sensitivity_contract"][
                "fold_count"
            ],
            "Tm2_LOO_evaluation_count": tm2_loo_full_support["evaluation_count"],
            "Tm2_LOO_passed_evaluation_count": tm2_loo_full_support[
                "passed_evaluation_count"
            ],
            "Tm2_LOO_candidate_ratio_ranges": {
                name: {
                    "shuffle": item["shuffle_ratio_range"],
                    "static": item["static_ratio_range"],
                }
                for name, item in tm2_loo_full_support["candidate_results"].items()
            },
            "measured_kernel_source_leaks": measured_kernel_cascade[
                "configured_path"
            ]["source_leaks"],
            "measured_kernel_source_node_counts": measured_kernel_cascade[
                "runtime_source_node_counts"
            ],
            "measured_kernel_recurrent_or_feedback_fraction_by_source": {
                source: item["recurrent_or_target_feedback_input_fraction"]
                for source, item in measured_kernel_inputs["source_results"].items()
            },
            "measured_kernel_Tm9_CT1_input_fraction": measured_kernel_inputs[
                "source_results"
            ]["Tm9"]["category_results"]["CT1"][
                "normalized_absolute_input_fraction"
            ],
            "lamina_only_source_coverage": lamina_only_replacement[
                "replacement_contract"
            ]["source_input_coverage"],
            "lamina_only_candidate_ratios_by_update": {
                name: {
                    update: {
                        "shuffle": result[
                            "shuffle_to_ordered_residual_energy_ratio"
                        ],
                        "static": result["static_to_ordered_energy_ratio"],
                    }
                    for update, result in item["by_brain_updates_per_frame"].items()
                }
                for name, item in lamina_only_replacement["candidate_results"].items()
            },
            "temporal_shuffle_input_energy_ratios_by_update": {
                update: result["temporal_shuffle_to_ordered_energy_ratio"]
                for update, result in shuffle_input_energy[
                    "by_brain_updates_per_frame"
                ].items()
            },
            "static_sham_input_energy_ratios_by_update": {
                update: result["static_sham_to_ordered_energy_ratio"]
                for update, result in shuffle_input_energy[
                    "by_brain_updates_per_frame"
                ].items()
            },
        },
        "gates": gates,
        "T5_source_transfer_ready": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "incremental_shape_evidence_does_not_supply_gain_state_CT1_mapping_or_independent_validation"
        ),
        "boundary": config["boundary"],
    }
