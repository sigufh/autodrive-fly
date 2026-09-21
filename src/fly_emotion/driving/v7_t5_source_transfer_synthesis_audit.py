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
