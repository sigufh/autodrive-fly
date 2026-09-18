"""Audit T5 voltage fields without combining incompatible recording modalities."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t5-recording-field-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_recording_field_audit.py")


def evaluate_v7_t5_recording_field_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_paths = {
        name: Path(config[name])
        for name in ("contract", "kohn_portes_ephys", "kohn_portes_identity")
    }
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }
    contract = evidence["contract"]
    ephys = evidence["kohn_portes_ephys"]
    identity = evidence["kohn_portes_identity"]
    required_fields = contract["required_recording_fields"]
    required_sources = contract["required_families"]["T5"]["source_types"]
    if required_sources != [*config["source_types"], config["missing_source_type"]]:
        raise ValueError("T5 source order differs from the external contract")
    for name, fields in config["modalities"].items():
        if list(fields) != required_fields:
            raise ValueError(f"{name} does not preserve recording-field order")

    expected_voltage_sources = ephys["T5_source_contract"][
        "sources_with_local_numeric_membrane_voltage"
    ]
    if expected_voltage_sources != config["source_types"]:
        raise ValueError("Kohn-Portes voltage source coverage changed")
    if ephys["T5_source_contract"]["missing_numeric_membrane_voltage_sources"] != [
        config["missing_source_type"]
    ]:
        raise ValueError("Kohn-Portes missing source changed")

    modality_summary = {}
    for name, fields in config["modalities"].items():
        available = [field for field, item in fields.items() if item["available"]]
        missing = [field for field, item in fields.items() if not item["available"]]
        modality_summary[name] = {
            "field_status": fields,
            "available_fields": available,
            "missing_fields": missing,
            "available_field_count": len(available),
            "required_field_count": len(required_fields),
            "all_required_fields_coexist": not missing,
        }

    white_noise = ephys["white_noise_payloads"]
    source_checks = {}
    for source in config["source_types"]:
        flash = ephys["source_payloads"][source]
        noise = white_noise[source]
        history = identity["source_capacity"][source]
        source_checks[source] = {
            "full_field_flash_condition_count": flash["condition_count"],
            "full_field_flash_all_conditions_have_voltage": flash[
                "all_conditions_have_numerical_membrane_voltage"
            ],
            "white_noise_record_count": noise["record_count"],
            "white_noise_stable_recording_id_field_retained": noise[
                "stable_recording_id_field_retained"
            ],
            "full_repository_recording_id_upper_bound": history[
                "full_repository_unique_recording_id_upper_bound"
            ],
            "biological_individual_semantics_verified": history[
                "biological_individual_count_verified"
            ],
            "all_required_fields_coexist_in_any_allowed_voltage_modality": False,
        }
    missing_source = config["missing_source_type"]
    source_checks[missing_source] = {
        "full_field_flash_condition_count": 0,
        "full_field_flash_all_conditions_have_voltage": False,
        "white_noise_record_count": 0,
        "white_noise_stable_recording_id_field_retained": False,
        "full_repository_recording_id_upper_bound": 0,
        "biological_individual_semantics_verified": False,
        "all_required_fields_coexist_in_any_allowed_voltage_modality": False,
    }

    union_available = [
        field
        for field in required_fields
        if any(
            modality["field_status"][field]["available"]
            for modality in modality_summary.values()
        )
    ]
    complete = all(
        item["all_required_fields_coexist_in_any_allowed_voltage_modality"]
        for item in source_checks.values()
    )
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
            "runtime_modified": False,
        },
        "required_fields": required_fields,
        "modalities": modality_summary,
        "cross_modality_union": {
            "available_fields": union_available,
            "available_field_count": len(union_available),
            "accepted_as_single_recording_payload": False,
        },
        "source_checks": source_checks,
        "all_five_T5_sources_have_complete_coexisting_recording_fields": complete,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "no_voltage_modality_contains_all_required_fields_and_CT1_voltage_is_absent"
        ),
        "boundary": config["boundary"],
    }
