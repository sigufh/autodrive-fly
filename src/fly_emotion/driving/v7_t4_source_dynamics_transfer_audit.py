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
    microstep_path = Path(config["microstep_evidence"])
    unified_path = Path(config["official_unified_model_evidence"])
    verified_unified_path = Path(config["verified_unified_model_evidence"])
    fig3_kernel_path = Path(config["fig3_source_kernel_evidence"])
    ephys = json.loads((root / ephys_path).read_text(encoding="utf-8"))
    interface = json.loads((root / interface_path).read_text(encoding="utf-8"))
    timebase = json.loads((root / timebase_path).read_text(encoding="utf-8"))
    microstep = json.loads((root / microstep_path).read_text(encoding="utf-8"))
    unified = json.loads((root / unified_path).read_text(encoding="utf-8"))
    verified_unified = json.loads(
        (root / verified_unified_path).read_text(encoding="utf-8")
    )
    fig3_kernel = json.loads((root / fig3_kernel_path).read_text(encoding="utf-8"))
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
    fields = {
        "direction_independent_source_kernel": False,
        "source_to_MaleCNS_identity_mapping": all(
            item["stable_MaleCNS_body_ids_available"] for item in verified_sources.values()
        ),
        "physical_v7_sample_interval": timebase["identifiability"][
            "physical_timebase_identified"
        ],
        "millivolts_to_v7_normalized_state_mapping": interface["interface_boundary"][
            "convert_millivolts_to_normalized_drive"
        ],
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
                str(microstep_path): _sha256(root / microstep_path),
                str(unified_path): _sha256(root / unified_path),
                str(verified_unified_path): _sha256(root / verified_unified_path),
                str(fig3_kernel_path): _sha256(root / fig3_kernel_path),
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
        },
        "required_transfer_fields_available": fields,
        "transfer_gates": gates,
        "source_dynamics_transfer_authorized": transferable,
        "next_candidate_authorized": False,
        "blocking_data_requirements": [
            name for name, available in fields.items() if not available
        ],
        "stop_reason": "published_source_dynamics_not_transferable_to_label_blind_v7",
        "boundary": config["boundary"],
    }
