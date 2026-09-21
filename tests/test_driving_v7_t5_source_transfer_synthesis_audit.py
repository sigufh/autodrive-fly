import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-source-transfer-synthesis-audit.json"


def test_synthesis_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["functional_stimulus_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_incremental_shape_evidence_is_kept_separate_from_transfer_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["incremental_evidence"] == {
        "voltage_derived_kernel_source_count": 4,
        "exact_type_average_mapping_source_count": 4,
        "Figure5_training_fit_count": 168,
        "moving_bar_condition_count": 24,
        "measured_kernel_trace_samples": 27,
        "measured_kernel_minimum_prefix_L1_mass_fraction": 0.4082240394891599,
        "measured_kernel_maximum_prefix_L1_mass_fraction": 0.47168529082482163,
        "measured_kernel_zero_tail_samples": 498,
        "measured_kernel_full_support_output_samples": 525,
        "population_kernel_robustness_passing_source_count": 3,
        "population_kernel_robustness_failing_sources": ["Tm2"],
        "Tm2_recording_id_vs_rest_median_correlation": 0.785694273562359,
        "Tm2_partition_correlation_p05": 0.76950266256901,
        "population_kernel_author_call_count": 9,
        "population_kernel_explicit_baseline_call_count": 0,
        "Tm1_row_vs_recording_id_weighted_correlation": 0.9995598341098436,
        "author_row_weighted_changed_sources": ["Tm1"],
        "Tm2_LOO_fold_count": 5,
        "Tm2_LOO_evaluation_count": 30,
        "Tm2_LOO_passed_evaluation_count": 0,
        "Tm2_LOO_candidate_ratio_ranges": {
            "summed_filtered_source_centroid_projection": {
                "shuffle": [2.101569895482767, 2.353461648721484],
                "static": [0.9994170965302039, 0.9998918524707392],
            },
            "fast_pool_vs_Tm9_centroid_difference": {
                "shuffle": [1.3648448324511815, 1.5063573039683826],
                "static": [0.7466918261547935, 0.7940622246002749],
            },
            "temporal_difference_filtered_Tm_pair_reichardt": {
                "shuffle": [0.9860891425025265, 1.2287928486318291],
                "static": [0.5921415935025431, 0.6741595238667628],
            },
        },
        "measured_kernel_source_leaks": {
            "Tm1": 0.28,
            "Tm2": 0.55,
            "Tm4": 0.34,
            "Tm9": 0.16,
        },
        "measured_kernel_source_node_counts": {
            "Tm1": 1777,
            "Tm2": 1766,
            "Tm4": 1670,
            "Tm9": 1771,
        },
        "measured_kernel_recurrent_or_feedback_fraction_by_source": {
            "Tm1": 0.06242381162002561,
            "Tm2": 0.04116725419043349,
            "Tm4": 0.19606226563946827,
            "Tm9": 0.23058754793814995,
        },
        "measured_kernel_Tm9_CT1_input_fraction": 0.09073247310668342,
        "lamina_only_source_coverage": {
            "Tm1": {
                "source_node_count": 1777,
                "source_node_with_lamina_input_count": 1774,
                "source_node_without_lamina_input_count": 3,
                "source_node_with_lamina_input_fraction": 0.9983117613956106,
            },
            "Tm2": {
                "source_node_count": 1766,
                "source_node_with_lamina_input_count": 1765,
                "source_node_without_lamina_input_count": 1,
                "source_node_with_lamina_input_fraction": 0.9994337485843715,
            },
            "Tm4": {
                "source_node_count": 1670,
                "source_node_with_lamina_input_count": 1670,
                "source_node_without_lamina_input_count": 0,
                "source_node_with_lamina_input_fraction": 1.0,
            },
            "Tm9": {
                "source_node_count": 1771,
                "source_node_with_lamina_input_count": 1770,
                "source_node_without_lamina_input_count": 1,
                "source_node_with_lamina_input_fraction": 0.9994353472614342,
            },
        },
        "lamina_only_candidate_ratios_by_update": {
            "summed_filtered_source_centroid_projection": {
                "1": {"shuffle": 1.5698382891422213, "static": 0.5056834500354587},
                "4": {"shuffle": 1.6164007523973414, "static": 0.4988382782934325},
            },
            "fast_pool_vs_Tm9_centroid_difference": {
                "1": {"shuffle": 2.3605189049473343, "static": 0.44503033278039983},
                "4": {"shuffle": 2.2398989473633173, "static": 0.42111038663450145},
            },
            "temporal_difference_filtered_Tm_pair_reichardt": {
                "1": {"shuffle": 1.6853864631000266, "static": 0.3191557479999038},
                "4": {"shuffle": 1.6277322953083266, "static": 0.3327291401605626},
            },
        },
        "temporal_shuffle_input_energy_ratios_by_update": {
            "1": {
                "pixel_temporal_difference": 6.93426194129865,
                "R1_R6_signed_frame_difference": 5.837713478478663,
                "lamina_only_Tm_preactivation": {
                    "Tm1": 4.908033023142721,
                    "Tm2": 4.905535265160078,
                    "Tm4": 4.902718804860586,
                    "Tm9": 4.710199918312225,
                },
            },
            "4": {
                "pixel_temporal_difference": 6.93426194129865,
                "R1_R6_signed_frame_difference": 5.837713478478663,
                "lamina_only_Tm_preactivation": {
                    "Tm1": 5.158457721706764,
                    "Tm2": 5.155179138220397,
                    "Tm4": 5.148909292181183,
                    "Tm9": 5.127327256460109,
                },
            },
        },
        "static_sham_input_energy_ratios_by_update": {
            "1": {
                "pixel_temporal_difference": 0.3333333416568422,
                "R1_R6_signed_frame_difference": 0.45652174898942627,
                "lamina_only_Tm_preactivation": {
                    "Tm1": 0.4380513339747843,
                    "Tm2": 0.4379028051340378,
                    "Tm4": 0.43657969359191173,
                    "Tm9": 0.45422753830080753,
                },
            },
            "4": {
                "pixel_temporal_difference": 0.3333333416568422,
                "R1_R6_signed_frame_difference": 0.45652174898942627,
                "lamina_only_Tm_preactivation": {
                    "Tm1": 0.43223462778410937,
                    "Tm2": 0.43199250364279845,
                    "Tm4": 0.43063522424125444,
                    "Tm9": 0.4363820143500369,
                },
            },
        },
    }
    gates = report["gates"]
    assert gates["four_Tm_voltage_derived_kernel_shapes_available"] is True
    assert gates["four_Tm_exact_type_average_mapping_complete"] is True
    assert gates["saline_relative_Tm9_delay_candidate_available"] is True
    assert gates["relative_low_frequency_Tm9_shape_evidence_available"] is True
    assert gates["cross_stimulus_moving_bar_shape_candidate_available"] is True
    assert gates["official_T5_target_model_parameters_available"] is True
    assert gates["official_target_model_to_T5_source_mapping_available"] is False
    assert gates["external_Figure6_axolotl_tmodel_source_available"] is False
    assert gates["related_axolotl_project_identified"] is True
    assert gates["related_axolotl_repository_anonymously_readable"] is False
    assert gates["measured_kernel_temporal_identifiability_passed"] is False
    assert gates["measured_kernel_direction_scoring_performed"] is False
    assert gates["measured_kernel_to_stimulus_frame_alignment_verified"] is True
    assert gates["measured_kernel_to_probe_solver_alignment_verified"] is False
    assert gates["measured_kernel_physical_source_dynamics_transfer_authorized"] is False
    assert gates["measured_kernel_standard_substep_negative_result_reproduced"] is True
    assert gates["measured_kernel_cross_substep_temporal_identifiability_passed"] is False
    assert gates["measured_kernel_cross_substep_direction_scoring_performed"] is False
    assert gates["measured_kernel_cross_substep_direction_scoring_authorized"] is False
    assert gates["measured_kernel_causal_index_zero_supported"] is True
    assert gates["measured_kernel_all_population_peaks_covered"] is True
    assert gates["measured_kernel_full_L1_support_covered"] is False
    assert gates["measured_kernel_prefix_alone_full_support_negative_authorized"] is False
    assert gates["measured_kernel_zero_tail_full_support_evaluated"] is True
    assert gates["measured_kernel_zero_tail_cross_substep_identifiability_passed"] is False
    assert gates["measured_kernel_zero_tail_all_candidates_failed_every_update_count"] is True
    assert gates["measured_kernel_zero_tail_direction_scoring_authorized"] is False
    assert gates["measured_kernel_zero_tail_direction_scoring_performed"] is False
    assert gates["population_kernel_recording_id_robustness_passed"] is False
    assert gates["Tm2_population_kernel_recording_id_robustness_passed"] is False
    assert gates["population_kernel_independent_biological_validation_available"] is False
    assert gates["population_kernel_transfer_authorized"] is False
    assert gates["population_kernel_author_default_baseline_verified"] is True
    assert gates["population_kernel_current_no_baseline_matches_author_default"] is True
    assert gates["population_kernel_author_row_weighting_exactly_reproduced"] is False
    assert gates["population_kernel_alternative_tail_baseline_authorized"] is False
    assert gates["author_row_weighted_full_support_all_candidates_failed"] is True
    assert gates["author_row_weighted_cross_substep_identifiability_passed"] is False
    assert gates["author_row_weighted_direction_scoring_authorized"] is False
    assert gates["author_row_weighted_direction_scoring_performed"] is False
    assert gates["Tm2_LOO_full_support_all_evaluations_failed"] is True
    assert gates["Tm2_LOO_robust_temporal_identifiability_passed"] is False
    assert gates["Tm2_LOO_direction_scoring_authorized"] is False
    assert gates["Tm2_LOO_direction_scoring_performed"] is False
    assert gates["Tm2_LOO_independent_biological_validation_performed"] is False
    assert gates["measured_kernel_typed_recurrent_cascade_verified"] is True
    assert gates["measured_kernel_replaces_existing_source_dynamics"] is False
    assert gates["measured_kernel_single_stage_biological_interpretation_authorized"] is False
    assert gates["measured_kernel_external_state_mapping_available"] is False
    assert gates["measured_kernel_source_inputs_partitioned_exactly_once"] is True
    assert gates["measured_kernel_every_source_has_recurrent_or_feedback_input"] is True
    assert gates["measured_kernel_feedforward_only_source_drive_available"] is False
    assert gates["measured_kernel_source_dynamics_replacement_evaluated"] is False
    assert gates["measured_kernel_source_dynamics_replacement_authorized"] is False
    assert gates["lamina_only_measured_kernel_replacement_evaluated"] is True
    assert gates["lamina_only_measured_kernel_all_candidates_failed"] is True
    assert gates["lamina_only_measured_kernel_temporal_identifiability_passed"] is False
    assert gates["lamina_only_measured_kernel_direction_scoring_authorized"] is False
    assert gates["lamina_only_measured_kernel_direction_scoring_performed"] is False
    assert gates["lamina_only_measured_kernel_physical_transfer_authorized"] is False
    assert gates["temporal_shuffle_frame_multiset_preserved"] is True
    assert gates["temporal_shuffle_retinal_energy_preserved"] is False
    assert gates["temporal_shuffle_lamina_source_energy_preserved"] is False
    assert gates["temporal_shuffle_energy_matched_control_verified"] is False
    assert gates["equal_energy_temporal_selectivity_interpretation_authorized"] is False
    assert gates["new_energy_normalized_gate_authorized"] is False
    assert gates["absolute_source_gain_available"] is False
    assert gates["source_to_v7_state_mapping_available"] is False
    assert gates["state_invariant_source_timing_available"] is False
    assert gates["independent_moving_bar_validation_available"] is False
    assert gates["MaleCNS_connectome_weight_transfer_available"] is False
    assert gates["CT1_allowed_dynamics_and_mapping_complete"] is False
    assert gates["original_physical_transfer_gate_passed"] is False


def test_synthesis_preserves_original_transfer_stop() -> None:
    report = json.loads(REPORT.read_text())
    original = report["original_physical_transfer_contract"]
    assert original["ready"] is False
    assert set(original["missing_fields"]) == {
        "v7_camera_angular_calibration",
        "Tm1_Tm2_Tm4_Tm9_membrane_like_kernels",
        "CT1_membrane_like_kernel",
        "stable_source_or_target_cell_to_MaleCNS_mapping",
        "independent_dynamic_validation_cohort",
    }
    assert report["T5_source_transfer_ready"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
