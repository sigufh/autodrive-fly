"""Audit Kohn-Portes identity and capacity across the fixed repository history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
)

CONFIG = Path("configs/driving-v7-kohn-portes-identity-history-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_kohn_portes_identity_history_audit.py"
)


def _git_blob(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(prefix + payload, usedforsecurity=False).hexdigest()


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"Kohn-Portes history file size changed: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"Kohn-Portes history file SHA-256 changed: {path}")
    if "git_blob" in spec and _git_blob(path.read_bytes()) != spec["git_blob"]:
        raise ValueError(f"Kohn-Portes history Git blob changed: {path}")
    return path


def _record_summary(records: list[dict], source: str, bath: str) -> dict:
    if any(record.get("cell_type") != source for record in records):
        raise ValueError(f"Kohn-Portes history source label changed: {source}:{bath}")
    if any(record.get("bath_solution") != bath for record in records):
        raise ValueError(f"Kohn-Portes history bath label changed: {source}:{bath}")
    required = {
        "recording_id",
        "recording_date",
        "cell_type",
        "cell_number",
        "subrecording_number",
        "bath_solution",
        "ephys_trace",
        "timestamps",
        "sampling_rate",
    }
    if any(not required.issubset(record) for record in records):
        raise ValueError(f"Kohn-Portes history record fields changed: {source}:{bath}")
    ids = [str(record["recording_id"]) for record in records]
    dates = [str(record["recording_date"]) for record in records]
    if any(not np.asarray(record["ephys_trace"]).size for record in records):
        raise ValueError(f"Kohn-Portes history raw voltage trace is empty: {source}:{bath}")
    return {
        "record_count": len(records),
        "recording_ids": ids,
        "unique_recording_ids": sorted(set(ids)),
        "unique_recording_id_count": len(set(ids)),
        "duplicate_recording_ids": sorted(
            recording_id for recording_id in set(ids) if ids.count(recording_id) > 1
        ),
        "recording_dates": dates,
        "all_cell_number_fields_missing": all(
            record.get("cell_number") is None for record in records
        ),
        "raw_voltage_trace_present": True,
        "physical_timestamps_present": True,
    }


def evaluate_v7_kohn_portes_identity_history_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    prior_path = Path(config["prior_evidence"])
    prior = json.loads((root / prior_path).read_text(encoding="utf-8"))

    repository = config["repository"]
    tree_path = _verify_file(root, repository["current_tree"])
    tree_entries = [json.loads(line) for line in tree_path.read_text().splitlines() if line]
    if len(tree_entries) != int(repository["current_tree"]["entry_count"]):
        raise ValueError("Kohn-Portes fixed tree entry count changed")
    if sum(item["type"] == "blob" for item in tree_entries) != int(
        repository["current_tree"]["blob_count"]
    ):
        raise ValueError("Kohn-Portes fixed tree blob count changed")
    if sum(item["type"] == "tree" for item in tree_entries) != int(
        repository["current_tree"]["tree_count"]
    ):
        raise ValueError("Kohn-Portes fixed tree directory count changed")

    commits_path = _verify_file(root, repository["commits"])
    commits = json.loads(commits_path.read_text(encoding="utf-8"))
    if len(commits) != int(repository["commits"]["count"]):
        raise ValueError("Kohn-Portes commit count changed")
    if commits[0]["id"] != repository["commit"]:
        raise ValueError("Kohn-Portes fixed head changed")

    history_path = _verify_file(root, repository["historical_paths"])
    history_rows = [line.split("\t", 3) for line in history_path.read_text().splitlines()]
    if len(history_rows) != int(repository["historical_paths"]["commit_path_row_count"]):
        raise ValueError("Kohn-Portes historical commit/path row count changed")
    historical_paths = sorted({row[3] for row in history_rows})
    if len(historical_paths) != int(repository["historical_paths"]["unique_path_count"]):
        raise ValueError("Kohn-Portes historical unique path count changed")
    forbidden_tokens = [
        token.lower() for token in repository["forbidden_identity_sidecar_tokens"]
    ]
    identity_sidecar_paths = [
        path
        for path in historical_paths
        if any(token in Path(path).name.lower() for token in forbidden_tokens)
    ]
    if identity_sidecar_paths:
        raise ValueError("Kohn-Portes history gained an unreviewed identity sidecar")

    paper_path = _verify_file(root, config["paper"])
    paper_root = ElementTree.parse(paper_path).getroot()
    paper_text = " ".join(
        (element.text or "") for element in paper_root.findall(".//text")
    )
    required_paper_phrases = (
        "whole-cell patch clamp electrophysiology",
        "Drivers were expressed homozygously",
        "All experimental animals were collected approximately 24 hours post-eclosion",
    )
    if any(phrase not in paper_text for phrase in required_paper_phrases):
        raise ValueError("Kohn-Portes paper identity text changed")
    forbidden_identity_phrases = (
        "biological_individual_id",
        "animal_id",
        "fly_id",
        "subject_id",
    )
    if any(phrase in paper_text for phrase in forbidden_identity_phrases):
        raise ValueError("Kohn-Portes paper gained an unreviewed explicit identity field")

    code_paths = {name: _verify_file(root, spec) for name, spec in config["author_code"].items()}
    returnmeans = code_paths["returnmeans"].read_text(encoding="utf-8")
    notebook = json.loads(code_paths["white_noise_notebook"].read_text(encoding="utf-8"))
    notebook_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    historical_notebook = json.loads(
        code_paths["historical_Tm9_notebook"].read_text(encoding="utf-8")
    )
    historical_notebook_text = "\n".join(
        "".join(cell.get("source", [])) for cell in historical_notebook["cells"]
    )
    required_code_phrases = (
        "a single 'recording_id' might have multiple subrecording_numbers",
        "Average within cell",
        "r_id = [r['recording_id'] for r in fff_list]",
    )
    if any(phrase not in returnmeans for phrase in required_code_phrases):
        raise ValueError("Kohn-Portes recording-ID aggregation semantics changed")
    recording_description = (
        "White noise data for each cell is stored in a list of dictionaries for each "
        "recording"
    )
    if recording_description not in notebook_text:
        raise ValueError("Kohn-Portes notebook recording description changed")
    if "Tm9_temporal_all['n'] = 15" not in historical_notebook_text:
        raise ValueError("Kohn-Portes historical Tm9 selection changed")

    contrast = {}
    for source, spec in config["contrast_flash_files"].items():
        path = _verify_file(root, spec)
        payload = _load_restricted(path)
        if not isinstance(payload, dict) or len(payload) != 4:
            raise ValueError(f"Kohn-Portes contrast-flash payload changed: {source}")
        expected_fields = {"mean", "std", "n", "sampling_rate", "all"}
        if any(set(item) != expected_fields for item in payload.values()):
            raise ValueError(f"Kohn-Portes contrast-flash fields changed: {source}")
        contrast[source] = {
            "file": spec,
            "condition_count": len(payload),
            "condition_trace_counts": {
                name: int(np.asarray(item["all"]).shape[0]) for name, item in payload.items()
            },
            "recording_id_retained": False,
            "biological_individual_id_retained": False,
        }

    drifting = {}
    drifting_ids = {source: set() for source in ("Tm1", "Tm2", "Tm4", "Tm9")}
    for name, spec in config["drifting_grating_files"].items():
        path = _verify_file(root, spec)
        payload = _load_restricted(path)
        source = spec["source"]
        summary = _record_summary(payload, source, spec["bath"])
        drifting[name] = {"file": spec, **summary}
        drifting_ids[source].update(summary["unique_recording_ids"])

    all_tm9 = {}
    all_tm9_ids = set()
    for bath_name, spec in config["Tm9_all_white_noise_files"].items():
        path = _verify_file(root, spec)
        payload = _load_restricted(path)
        summary = _record_summary(payload, "Tm9", spec["bath"])
        if summary["record_count"] != int(spec["expected_record_count"]):
            raise ValueError(f"Kohn-Portes all-Tm9 record count changed: {bath_name}")
        if summary["unique_recording_id_count"] != int(
            spec["expected_unique_recording_id_count"]
        ):
            raise ValueError(f"Kohn-Portes all-Tm9 ID count changed: {bath_name}")
        all_tm9[bath_name] = {"file": spec, **summary}
        all_tm9_ids.update(summary["unique_recording_ids"])

    main_ids = {
        source: {
            str(record["recording_id"])
            for record in prior["white_noise_payloads"][source]["records"]
        }
        | set(prior["white_noise_OA_payloads"][source]["recording_ids"])
        for source in ("Tm1", "Tm2", "Tm4", "Tm9")
    }
    full_ids = {
        source: main_ids[source] | drifting_ids[source] for source in main_ids
    }
    full_ids["Tm9"] |= all_tm9_ids
    expected_main = config["expected_main_selected_unique_recording_ids"]
    expected_full = config["expected_full_repository_unique_recording_id_upper_bounds"]
    expected_new = config["expected_new_ids_beyond_main_selection"]
    source_capacity = {}
    minimum = int(
        contract["gates"]["minimum_recording_units_per_source_per_training_cohort"]
    ) + int(
        contract["gates"]["minimum_recording_units_per_source_per_validation_cohort"]
    )
    for source in main_ids:
        if len(main_ids[source]) != int(expected_main[source]):
            raise ValueError(f"Kohn-Portes main ID count changed: {source}")
        if len(full_ids[source]) != int(expected_full[source]):
            raise ValueError(f"Kohn-Portes full-history ID count changed: {source}")
        new_ids = sorted(full_ids[source] - main_ids[source])
        if new_ids != expected_new[source]:
            raise ValueError(f"Kohn-Portes new historical IDs changed: {source}")
        source_capacity[source] = {
            "main_selected_unique_recording_id_count": len(main_ids[source]),
            "full_repository_unique_recording_id_upper_bound": len(full_ids[source]),
            "main_selected_recording_ids": sorted(main_ids[source]),
            "full_repository_recording_ids": sorted(full_ids[source]),
            "new_ids_beyond_main_selection": new_ids,
            "minimum_training_plus_validation_count": minimum,
            "recording_id_upper_bound_meets_numeric_5_plus_3": len(full_ids[source])
            >= minimum,
            "biological_individual_count_verified": False,
            "biological_individual_5_plus_3_split_authorized": False,
        }

    gates = {
        "fixed_repository_tree_verified": True,
        "complete_public_commit_history_paths_verified": True,
        "no_identity_sidecar_in_public_history": not identity_sidecar_paths,
        "author_uses_recording_id_as_cell_aggregation_key": True,
        "every_source_recording_id_upper_bound_meets_numeric_5_plus_3": all(
            item["recording_id_upper_bound_meets_numeric_5_plus_3"]
            for item in source_capacity.values()
        ),
        "biological_individual_semantics_verified": False,
        "every_source_biological_individual_5_plus_3_split_authorized": False,
        "external_final_commitment_available": False,
    }
    transferable = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
                str(prior_path): _sha256(root / prior_path),
            },
            "repository": config["repository"],
            "paper": config["paper"],
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "repository_history": {
            "commit_count": len(commits),
            "fixed_tree_entry_count": len(tree_entries),
            "historical_commit_path_row_count": len(history_rows),
            "historical_unique_path_count": len(historical_paths),
            "identity_sidecar_paths": identity_sidecar_paths,
            "complete_public_history_inspected": True,
        },
        "author_identity_semantics": {
            "recording_id_is_used_as_cell_aggregation_key": True,
            "multiple_subrecording_numbers_per_recording_id_documented": True,
            "recording_id_is_date_formatted_in_payloads": True,
            "explicit_biological_fly_id_field_in_payloads": False,
            "paper_states_one_recording_cell_per_fly": False,
            "recording_id_promoted_to_biological_individual_id": False,
        },
        "contrast_flash_payloads": contrast,
        "drifting_grating_payloads": drifting,
        "Tm9_all_white_noise_payloads": all_tm9,
        "source_capacity": source_capacity,
        "transfer_gates": gates,
        "T5_biological_individual_split_authorized": transferable,
        "authorize_source_dynamics_fit": transferable,
        "authorize_T4_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else (
                "expanded_recording_key_capacity_still_lacks_biological_individual_"
                "semantics_and_all_source_5_plus_3"
            )
        ),
        "boundary": config["boundary"],
    }
