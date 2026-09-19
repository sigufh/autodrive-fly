"""Declare and verify exact-type-average source-to-MaleCNS mappings."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-source-type-average-mapping-contract.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_source_type_average_mapping_contract.py")


def evaluate_v7_source_type_average_mapping_contract(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_paths = {
        name: Path(config[name])
        for name in (
            "external_contract",
            "malecns_mapping",
            "T4_identity",
            "T5_ephys",
        )
    }
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }
    required = [
        *evidence["external_contract"]["required_families"]["T4"]["source_types"],
        *evidence["external_contract"]["required_families"]["T5"]["source_types"],
    ]
    if list(config["mapping_modes"]) != required:
        raise ValueError("type-average mapping order differs from source contract")
    t4_sources = evidence["external_contract"]["required_families"]["T4"]["source_types"]
    local_voltage = set(t4_sources) | set(
        evidence["T5_ephys"]["T5_source_contract"]["sources_with_local_numeric_membrane_voltage"]
    )
    t4_identity = evidence["T4_identity"]["transfer_gates"]
    t5_ephys = evidence["T5_ephys"]["transfer_gates"]
    rows = {}
    for source in required:
        mapping = evidence["malecns_mapping"]["source_mapping"][source]
        mode = config["mapping_modes"][source]
        explicit_type_average = mode == "exact_type_average" and source in local_voltage
        if source in t4_sources:
            source_identity_supported = bool(
                t4_identity["stable_pseudonymous_biological_individual_ID_available"]
            )
        else:
            source_identity_supported = bool(t5_ephys["stable_recording_id_field_retained"])
        gates = {
            "local_allowed_unit_payload_available": source in local_voltage,
            "explicit_exact_type_average_declared": explicit_type_average,
            "exact_MaleCNS_type_body_set_available": bool(mapping["all_bodies_in_canonical_graph"]),
            "soma_side_available": bool(mapping["soma_side_complete"]),
            "complete_columnar_retinotopy_available": bool(
                mapping["columnar_retinotopy_available"]
            ),
        }
        rows[source] = {
            "mapping_mode": mode,
            "recording_level_body_assignment": False,
            "source_recording_identity_support_available": source_identity_supported,
            "MaleCNS_body_count": mapping["body_count"],
            "MaleCNS_body_ids_sha256": mapping["body_ids_sha256"],
            "broadcast_rule": (
                "one_population_mean_kernel_to_all_exact_same_type_bodies"
                if explicit_type_average
                else None
            ),
            "gates": gates,
            "mapping_contract_complete": all(gates.values()),
        }
    complete = all(row["mapping_contract_complete"] for row in rows.values())
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
        "source_mappings": rows,
        "sources_with_explicit_type_average_mapping": [
            source
            for source, row in rows.items()
            if row["gates"]["explicit_exact_type_average_declared"]
        ],
        "all_nine_source_mapping_contracts_complete": complete,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            None
            if complete
            else "CT1_allowed_payload_and_single_body_coordinate_mapping_gates_incomplete"
        ),
        "boundary": config["boundary"],
    }
