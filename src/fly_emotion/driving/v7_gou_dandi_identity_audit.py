"""Audit whether Gou Dryad rows can be linked to DANDI participant identities."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-gou-dandi-identity-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_gou_dandi_identity_audit.py")


def _verified(root: Path, spec: dict, label: str) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"Gou DANDI {label} size changed")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"Gou DANDI {label} SHA-256 changed")
    return path


def _dryad_labels(inventory: dict) -> list[str]:
    labels: set[str] = set()
    for protocol in ("flash", "moving_bar"):
        for item in inventory[protocol].values():
            labels.update(str(value) for value in item["identity"]["fliesUsed_values"])
    return sorted(labels, key=int)


def evaluate_v7_gou_dandi_identity_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {name: _verified(root, spec, name) for name, spec in config["snapshots"].items()}
    inventory_path = _verified(root, config["dryad_inventory"], "Dryad inventory")
    nwb_snapshot_path = _verified(
        root,
        config["representative_NWB_range_inspection"],
        "representative NWB range metadata",
    )
    version = json.loads(paths["version"].read_text(encoding="utf-8"))
    index = json.loads(paths["assets_index"].read_text(encoding="utf-8"))
    manifest = yaml.safe_load(paths["assets_manifest"].read_text(encoding="utf-8"))
    representative = json.loads(paths["representative_asset"].read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if version["id"] != f"{config['dataset']['identifier']}/{config['dataset']['version']}":
        raise ValueError("Gou DANDI version identity changed")
    if index["count"] != len(index["results"]) or len(manifest) != index["count"]:
        raise ValueError("Gou DANDI asset count changed")

    manifest_by_id = {item["identifier"]: item for item in manifest}
    index_ids = {item["asset_id"] for item in index["results"]}
    if set(manifest_by_id) != index_ids:
        raise ValueError("Gou DANDI JSON index and YAML manifest differ")
    if any(
        manifest_by_id[item["asset_id"]]["path"] != item["path"]
        or int(manifest_by_id[item["asset_id"]]["contentSize"]) != int(item["size"])
        for item in index["results"]
    ):
        raise ValueError("Gou DANDI asset path or size differs between indexes")

    participant_fields: set[str] = set()
    session_fields: set[str] = set()
    subjects: list[str] = []
    sessions: list[str] = []
    for asset in manifest:
        participants = asset.get("wasAttributedTo", [])
        acquisition_sessions = [
            item for item in asset.get("wasGeneratedBy", []) if item.get("schemaKey") == "Session"
        ]
        if len(participants) != 1 or len(acquisition_sessions) != 1:
            raise ValueError("Gou DANDI asset lacks one participant or acquisition session")
        participant_fields.update(participants[0])
        session_fields.update(acquisition_sessions[0])
        subjects.append(str(participants[0]["identifier"]))
        sessions.append(str(acquisition_sessions[0]["startDate"]))

    labels = _dryad_labels(inventory)
    exact_matches = {
        label: [
            asset["path"]
            for asset in manifest
            if re.search(rf"(?<![0-9]){re.escape(label)}(?![0-9])", asset["path"])
            or any(
                str(value) == label
                for participant in asset.get("wasAttributedTo", [])
                for value in participant.values()
            )
        ]
        for label in labels
    }
    exact_matches = {key: value for key, value in exact_matches.items() if value}
    representative_spec = config["snapshots"]["representative_asset"]
    if (
        representative["identifier"] != representative_spec["asset_id"]
        or representative["path"] != representative_spec["asset_path"]
        or representative["digest"]["dandi:sha2-256"] != representative_spec["content_sha256"]
    ):
        raise ValueError("Gou representative DANDI asset identity changed")

    representative_nwb = json.loads(nwb_snapshot_path.read_text(encoding="utf-8"))
    participant = representative["wasAttributedTo"][0]
    session = next(
        item for item in representative["wasGeneratedBy"] if item["schemaKey"] == "Session"
    )
    if participant["identifier"] != representative_nwb["subject"][
        "subject_id"
    ] or datetime.fromisoformat(session["startDate"]) != datetime.fromisoformat(
        representative_nwb["session_start_time"]
    ):
        raise ValueError("Gou representative NWB and DANDI metadata differ")

    stable_dandi_ids = len(subjects) == len(set(subjects)) == index["count"]
    unique_sessions = len(sessions) == len(set(sessions)) == index["count"]
    crosswalk_verified = bool(exact_matches)
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path.relative_to(root)): _sha256(path) for path in paths.values()},
                str(inventory_path.relative_to(root)): _sha256(inventory_path),
                str(nwb_snapshot_path.relative_to(root)): _sha256(nwb_snapshot_path),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "dataset": config["dataset"],
        "asset_index": {
            "asset_count": index["count"],
            "manifest_asset_count": len(manifest),
            "JSON_and_YAML_asset_ids_match": True,
            "JSON_and_YAML_paths_and_sizes_match": True,
            "unique_asset_path_count": len({item["path"] for item in manifest}),
            "unique_subject_id_count": len(set(subjects)),
            "unique_session_start_count": len(set(sessions)),
            "one_unique_subject_per_asset": stable_dandi_ids,
            "one_unique_session_per_asset": unique_sessions,
            "participant_fields": sorted(participant_fields),
            "session_fields": sorted(session_fields),
            "genotype_field_present": "genotype" in participant_fields,
            "cell_type_field_present": "cellType" in participant_fields,
            "strain_field_present": "strain" in participant_fields,
            "internal_experiment_id_field_present": False,
        },
        "Dryad_identity_labels": {
            "distinct_fliesUsed_label_count": len(labels),
            "values": [int(value) for value in labels],
            "exact_token_matches_in_DANDI_paths_or_participant_fields": exact_matches,
            "exact_match_count": len(exact_matches),
        },
        "representative_NWB_range_inspection": representative_nwb,
        "identity_conclusions": {
            "DANDI_asset_level_stable_participant_IDs_available": stable_dandi_ids,
            "DANDI_asset_level_session_timestamps_available": unique_sessions,
            "representative_NWB_genotype_like_description_available": True,
            "representative_NWB_cell_location_available": True,
            "representative_NWB_original_Dryad_fly_ID_available": False,
            "Dryad_fliesUsed_to_DANDI_subject_crosswalk_verified": crosswalk_verified,
            "Dryad_processed_rows_have_stable_biological_IDs": False,
            "fit_held_out_biological_ID_disjointness_authorized": False,
        },
        "authorize_source_dynamics_transfer": False,
        "advance_to_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "DANDI_has_stable_asset_subjects_but_no_verified_Dryad_row_crosswalk",
        "boundary": config["boundary"],
    }
