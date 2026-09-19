"""Audit Gou et al. sparsity data against the v7 source-dynamics contract."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-gou-sparsity-source-dynamics-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_gou_sparsity_source_dynamics_audit.py")
INSPECTOR = Path("scripts/inspect_gou_sparsity_payload.py")


def evaluate_v7_gou_sparsity_source_dynamics_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))

    def verified_file(spec: dict, label: str) -> Path:
        path = root / spec["path"]
        if path.stat().st_size != int(spec["bytes"]):
            raise ValueError(f"Gou {label} size changed")
        if _sha256(path) != spec["sha256"]:
            raise ValueError(f"Gou {label} SHA-256 changed")
        return path

    dryad = config["dryad"]
    aggregate_path = verified_file(dryad["aggregate_download"], "aggregate")
    publisher_path = verified_file(dryad["publisher_archive"], "publisher archive")
    readme_path = verified_file(dryad["readme"], "README")
    inventory_spec = config["local_payload_inventory"]
    inventory_path = verified_file(inventory_spec, "numerical inventory")
    script_paths = {
        name: verified_file(spec, f"analysis script {name}")
        for name, spec in config["analysis_scripts"].items()
    }
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    identity_path = Path(config["dandi_identity_evidence"])
    identity_protocol_path = Path(config["dandi_identity_protocol"])
    identity = json.loads((root / identity_path).read_text(encoding="utf-8"))
    identity_conclusions = identity["identity_conclusions"]
    if (
        bool(config["dandi"]["asset_level_stable_participant_ids_verified"])
        != bool(
            identity_conclusions[
                "DANDI_asset_level_stable_participant_IDs_available"
            ]
        )
        or bool(
            config["dandi"][
                "Dryad_fliesUsed_to_DANDI_subject_crosswalk_verified"
            ]
        )
        != bool(
            identity_conclusions[
                "Dryad_fliesUsed_to_DANDI_subject_crosswalk_verified"
            ]
        )
    ):
        raise ValueError("Gou DANDI identity boundary changed")
    publisher = inventory["archive"]["publisher_archive"]
    if (
        publisher["sha256"]
        != dryad["files"]["sparsity_Dryad_upload.zip"]["publisher_reported_sha256"]
        or not publisher["zip_integrity_passed"]
        or publisher["real_file_count"] != int(inventory_spec["real_file_count"])
        or publisher["real_mat_file_count"] != int(inventory_spec["real_mat_file_count"])
    ):
        raise ValueError("Gou publisher archive inventory changed")
    if inventory["source_coverage"]["covered_required_sources"] != [
        "Mi1",
        "Tm3",
        "Tm1",
        "Tm2",
    ]:
        raise ValueError("Gou local source coverage changed")
    if any(
        inventory["source_coverage"][
            "missing_source_name_occurrences_in_real_member_paths"
        ].values()
    ):
        raise ValueError("Gou missing-source filename audit changed")
    for protocol_name in ("flash", "moving_bar"):
        for source, item in inventory[protocol_name].items():
            if item["source_type"] != source:
                raise ValueError(f"Gou {protocol_name} source identity changed: {source}")
            for array in item["arrays"].values():
                if (
                    array["finite_count"] != array["element_count"]
                    or array["nan_count"]
                    or array["infinite_count"]
                ):
                    raise ValueError(f"Gou {protocol_name} nonfinite array: {source}")
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    required_by_family = {
        family: item["source_types"] for family, item in contract["required_families"].items()
    }
    evidence = config["source_evidence"]
    covered_by_family = {
        family: [source for source in sources if source in evidence]
        for family, sources in required_by_family.items()
    }
    missing_by_family = {
        family: [source for source in sources if source not in evidence]
        for family, sources in required_by_family.items()
    }
    required_count = sum(len(sources) for sources in required_by_family.values())
    covered_count = sum(len(sources) for sources in covered_by_family.values())
    source_unit_gates = {
        source: bool(item["response_unit_contract_satisfied"]) for source, item in evidence.items()
    }
    fields = config["stimulus_and_recording"]
    transfer_gates = {
        "every_T4_source_type_measured": not missing_by_family["T4"],
        "every_T5_source_type_measured": not missing_by_family["T5"],
        "every_covered_source_has_allowed_response_unit": all(source_unit_gates.values()),
        "processed_rows_linked_to_stable_subject_ids": bool(
            identity_conclusions[
                "Dryad_fliesUsed_to_DANDI_subject_crosswalk_verified"
            ]
            and fields["processed_fly_rows_to_DANDI_subject_ids_verified"]
        ),
        "exact_sample_timestamps_and_baselines_verified": bool(
            fields["exact_sample_timestamps_verified_from_payload"]
            and fields["baseline_windows_verified_from_payload"]
        ),
        "source_type_to_MaleCNS_mapping_available": bool(fields["MaleCNS_mapping_present"]),
        "required_split_roles_present": bool(
            fields["preregistered_training_validation_external_final_roles_present"]
        ),
        "payload_hash_locally_verified": bool(config["dryad"]["payload_sha256_locally_verified"]),
    }
    transferable = all(transfer_gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(INSPECTOR): _sha256(root / INSPECTOR),
                str(contract_path): _sha256(root / contract_path),
                str(aggregate_path.relative_to(root)): _sha256(aggregate_path),
                str(publisher_path.relative_to(root)): _sha256(publisher_path),
                str(readme_path.relative_to(root)): _sha256(readme_path),
                str(inventory_path.relative_to(root)): _sha256(inventory_path),
                str(identity_path): _sha256(root / identity_path),
                str(identity_protocol_path): _sha256(root / identity_protocol_path),
                **{str(path.relative_to(root)): _sha256(path) for path in script_paths.values()},
            },
            "paper": config["paper"],
            "parameter_fit": False,
            "external_payload_evaluated": True,
            "runtime_modified": False,
        },
        "repositories": {
            "Dryad": config["dryad"],
            "DANDI": config["dandi"],
        },
        "source_evidence": evidence,
        "local_payload_inventory": inventory,
        "DANDI_identity_audit": identity,
        "stimulus_and_recording": fields,
        "required_sources_by_family": required_by_family,
        "covered_required_sources_by_family": covered_by_family,
        "missing_required_sources_by_family": missing_by_family,
        "covered_required_source_count": covered_count,
        "required_source_count": required_count,
        "coverage_fraction": covered_count / required_count,
        "source_response_unit_gates": source_unit_gates,
        "transfer_gates": transfer_gates,
        "partial_source_dynamics_evidence_present": covered_count > 0,
        "complete_external_source_dynamics_evidence": transferable,
        "authorize_source_dynamics_fit": transferable,
        "advance_to_T4_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in transfer_gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else "partial_Mi1_Tm3_Tm1_Tm2_fluorescence_evidence_does_not_satisfy_unified_contract"
        ),
        "download_disposition": {
            "Dryad_archive_download_completed": True,
            "Dryad_archive_download_authorized": False,
            "DANDI_bulk_download_authorized": False,
            "reason": (
                "the bounded Dryad archive is now locally validated; incomplete source, "
                "identity, timestamp, mapping, split-role, and response-unit coverage "
                "still forbids fitting and makes further bulk download unnecessary"
            ),
        },
        "boundary": config["boundary"],
    }
