"""Freeze the external evidence schema required to unblock T4/T5 dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path(
    "configs/driving-v7-source-dynamics-external-evidence-contract.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_source_dynamics_external_evidence_contract.py"
)


def evaluate_v7_source_dynamics_external_evidence_contract(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    readiness_path = Path(config["readiness_evidence"])
    protocol_path = Path(config["readiness_protocol"])
    readiness = json.loads((root / readiness_path).read_text())
    if readiness["T4_T5_source_dynamics_ready"]:
        raise ValueError("external evidence contract is only valid while readiness fails")
    expected_sources = {
        "T4": ["Mi1", "Tm3", "Mi4", "C3"],
        "T5": ["Tm1", "Tm2", "Tm4", "Tm9", "CT1"],
    }
    actual_sources = {
        family: item["source_types"]
        for family, item in config["required_families"].items()
    }
    if actual_sources != expected_sources:
        raise ValueError("external source contract changed required T4/T5 types")
    template = {
        "schema_version": 1,
        "dataset_id": None,
        "dataset_version": None,
        "license": None,
        "source_url": None,
        "payload_sha256": None,
        "recording_fields": {name: None for name in config["required_recording_fields"]},
        "mapping_fields": {name: None for name in config["required_mapping_fields"]},
        "external_final_commitment_sha256": None,
        "external_final_custodian": None,
        "external_final_consumed": False,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(readiness_path): _sha256(root / readiness_path),
                str(protocol_path): _sha256(root / protocol_path),
            },
            "parameter_fit": False,
            "external_payload_evaluated": False,
            "runtime_modified": False,
        },
        "required_families": config["required_families"],
        "required_recording_fields": config["required_recording_fields"],
        "required_mapping_fields": config["required_mapping_fields"],
        "required_split_roles": config["required_split_roles"],
        "allowed_response_units": config["allowed_response_units"],
        "gates": config["gates"],
        "manifest_template": template,
        "external_payload_present": False,
        "contract_satisfied": False,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "external_source_dynamics_payload_not_present",
        "boundary": config["boundary"],
    }
