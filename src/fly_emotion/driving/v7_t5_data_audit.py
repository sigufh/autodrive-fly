from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t5-data-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_data_audit.py")
INTERFACE_CONFIG = Path("configs/driving-v7-ephys-interface.yaml")
T5_CONDUCTANCE_EVIDENCE = Path("artifacts/v7-t5-conductance-audit.json")


def _fetch(url: str) -> tuple[bytes, int]:
    request = urllib.request.Request(url, headers={"User-Agent": "AutoDrive-Fly-v7-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(), int(response.status)


def _observe(url: str, *, method: str = "GET") -> dict:
    request = urllib.request.Request(
        url, method=method, headers={"User-Agent": "AutoDrive-Fly-v7-audit/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return {"status_code": int(response.status), "accessible": True}
    except urllib.error.HTTPError as error:
        return {"status_code": int(error.code), "accessible": False}
    except urllib.error.URLError as error:
        return {"status_code": None, "accessible": False, "error_type": type(error).__name__}


def _plain(value: str) -> str:
    output, inside = [], False
    for char in unescape(value):
        if char == "<":
            inside = True
        elif char == ">":
            inside = False
            output.append(" ")
        elif not inside:
            output.append(char)
    return " ".join("".join(output).split())


def _datacite_metadata(raw: bytes, expected: dict) -> dict:
    attributes = json.loads(raw)["data"]["attributes"]
    description = _plain(" ".join(item["description"] for item in attributes["descriptions"]))
    title = attributes["titles"][0]["title"]
    sizes = [int(value.split()[0]) for value in attributes.get("sizes", [])]
    rights = [
        item.get("rightsIdentifier", item.get("rights", "")) for item in attributes["rightsList"]
    ]
    for phrase in expected["required_description_phrases"]:
        if str(phrase).lower() not in description.lower():
            raise ValueError(f"DataCite description is missing expected phrase: {phrase}")
    if int(expected["expected_size_bytes"]) not in sizes:
        raise ValueError(f"DataCite size mismatch for {expected['doi']}")
    return {
        "doi": attributes["doi"],
        "title": title,
        "size_bytes": int(expected["expected_size_bytes"]),
        "license_identifiers": rights,
        "description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        "description": description,
        "stimulus_scope": expected["stimulus_scope"],
    }


def evidence_classification(modality: str, allowed_roles: dict) -> dict:
    roles = allowed_roles[modality]
    return {
        "modality": modality,
        "allowed_roles": roles,
        "supports_absolute_voltage_calibration": "absolute_voltage_calibration" in roles,
        "supports_relative_voltage_validation": "relative_voltage_polarity" in roles,
        "supports_calcium_activity_validation": "relative_calcium_activity" in roles,
        "supports_topology_only": roles
        == ["topology", "spatial_input_order", "synapse_distribution"],
    }


def evaluate_v7_t5_data_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    interface = yaml.safe_load((root / INTERFACE_CONFIG).read_text(encoding="utf-8"))
    t5_conductance = json.loads((root / T5_CONDUCTANCE_EVIDENCE).read_text(encoding="utf-8"))
    if not config["exploratory"] or config["advance_allowed"]:
        raise ValueError("T5 data audit must remain exploratory and non-advancing")
    source = config["sources"]["gruntman_2021_whole_cell"]
    pubmed_raw, pubmed_status = _fetch(source["pubmed_url"])
    pubmed_text = pubmed_raw.decode("utf-8")
    for phrase in source["required_abstract_phrases"]:
        if phrase.lower() not in pubmed_text.lower():
            raise ValueError(f"PubMed abstract is missing expected phrase: {phrase}")
    collection_raw, collection_status = _fetch(source["collection_datacite_url"])
    collection = json.loads(collection_raw)["data"]["attributes"]
    datasets = {}
    access = {}
    total_large_bytes = 0
    for name, expected in source["datasets"].items():
        raw, status = _fetch(expected["datacite_url"])
        metadata = _datacite_metadata(raw, expected)
        metadata["metadata_status_code"] = status
        metadata["metadata_sha256"] = hashlib.sha256(raw).hexdigest()
        metadata["raw_files_downloaded"] = False
        datasets[name] = metadata
        if expected["expected_size_bytes"] > config["audit_boundary"]["maximum_download_bytes"]:
            total_large_bytes += int(expected["expected_size_bytes"])
        access[name] = {
            "article_api": _observe(
                f"https://api.figshare.com/v2/articles/{expected['article_id']}"
            ),
            "article_page": _observe(
                f"https://janelia.figshare.com/articles/_/{expected['article_id']}", method="HEAD"
            ),
        }
    inspected = {}
    for name in ("wienecke_2018_voltage_imaging", "ramos_2021_calcium"):
        item = config["sources"][name]
        raw, status = _fetch(item["pmc_url"])
        text = raw.decode("utf-8")
        for phrase in item["required_phrases"]:
            if phrase.lower() not in text.lower():
                raise ValueError(f"{name} is missing expected phrase: {phrase}")
        inspected[name] = {
            "paper_doi": item["paper_doi"],
            "http_status": status,
            "page_sha256": hashlib.sha256(raw).hexdigest(),
            "signal_unit": item["signal_unit"],
            "absolute_millivolts_possible": item["absolute_millivolts_possible"],
            **evidence_classification(item["modality"], config["allowed_roles"]),
        }
    connectome = config["sources"]["shinomiya_2025_connectome"]
    raw, status = _fetch(connectome["datacite_url"])
    connectome_attributes = json.loads(raw)["data"]["attributes"]
    interface_has_t5 = bool(interface["split_contract"]["t5_data_available"])
    return {
        "protocol": {
            "name": config["name"],
            "exploratory": True,
            "advance_allowed": False,
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(INTERFACE_CONFIG): _sha256(root / INTERFACE_CONFIG),
                str(T5_CONDUCTANCE_EVIDENCE): _sha256(root / T5_CONDUCTANCE_EVIDENCE),
            },
            "large_raw_files_downloaded": False,
            "parameter_fitting": False,
        },
        "whole_cell_candidate": {
            "paper_doi": source["paper_doi"],
            "pubmed_id": source["pubmed_id"],
            "pubmed_status_code": pubmed_status,
            "pubmed_sha256": hashlib.sha256(pubmed_raw).hexdigest(),
            "collection_doi": collection["doi"],
            "collection_status_code": collection_status,
            "collection_title": collection["titles"][0]["title"],
            "collection_license_identifiers": [
                item.get("rightsIdentifier", item.get("rights", ""))
                for item in collection["rightsList"]
            ],
            "classification": evidence_classification(source["modality"], config["allowed_roles"]),
            "datasets": datasets,
            "access_observation": access,
            "large_dataset_bytes_not_downloaded": total_large_bytes,
            "candidate_discovered": True,
            "file_manifest_retrieved": False,
            "raw_data_verified": False,
            "usable_in_current_interface": False,
        },
        "other_modalities": {
            **inspected,
            "shinomiya_2025_connectome": {
                "dataset_doi": connectome_attributes["doi"],
                "metadata_status_code": status,
                "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                "signal_unit": connectome["signal_unit"],
                **evidence_classification(connectome["modality"], config["allowed_roles"]),
            },
        },
        "interface_status": {
            "current_interface_contains_T5_data": interface_has_t5,
            "absolute_T5_voltage_candidate_discovered": True,
            "absolute_T5_voltage_files_verified": False,
            "processed_baseline_subtracted_T5_voltage_files_verified": True,
            "processed_T5_repository_commit": t5_conductance["repository"]["commit"],
            "processed_T5_cell_count": len(t5_conductance["cells"]),
            "T5_fit_allowed": False,
            "reason": (
                "The 2021 raw whole-cell datasets still lack a verified file manifest here. "
                "A separate fixed-commit audit verified processed baseline-subtracted 2019 T5 "
                "voltage traces and added a read-only, native-time interface; it is same-cell "
                "condition generalization and remains unavailable for fitting."
            ),
        },
        "forbidden_cross_modal_claims": config["forbidden_cross_modal_claims"],
        "audit_boundary": config["audit_boundary"],
        "limitations": [
            "Figshare access codes are dated environment observations, not permanence claims.",
            "DataCite descriptions establish scope, not file integrity or array axes.",
            "The 6.32-GB and 4.31-GB datasets were not downloaded under the bounded audit.",
            "Verified 2019 processed T5 traces do not verify the separate 2021 raw datasets.",
            "ASAP2f is relative fluorescence at about 15 Hz, not patch-clamp millivolts.",
            "Calcium imaging and connectome structure cannot calibrate membrane voltage.",
        ],
        "advance_to_T5_fit": False,
        "advance_to_visual_gate": False,
        "advance_to_central_complex": False,
    }
