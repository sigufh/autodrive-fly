from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t5_data_audit import _plain

CONFIG = Path("configs/driving-v7-t5-label-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_label_audit.py")


def _request(url: str, method: str = "GET") -> tuple[bytes, int]:
    request = urllib.request.Request(
        url, method=method, headers={"User-Agent": "AutoDrive-Fly-v7-audit/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read(), int(response.status)
    except urllib.error.HTTPError as error:
        return error.read(), int(error.code)
    except urllib.error.URLError:
        return b"", 0


def evaluate_v7_t5_label_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    dataset = config["figure4_dataset"]
    metadata_raw, metadata_status = _request(dataset["datacite_url"])
    if metadata_status != 200:
        raise ValueError("Figure 4 DataCite metadata was not accessible")
    attributes = json.loads(metadata_raw)["data"]["attributes"]
    description = _plain(
        " ".join(item["description"] for item in attributes.get("descriptions", []))
    )
    title = attributes["titles"][0]["title"]
    sizes = [int(value.split()[0]) for value in attributes.get("sizes", [])]
    licenses = [
        item.get("rightsIdentifier", item.get("rights", ""))
        for item in attributes.get("rightsList", [])
    ]
    if attributes["doi"].lower() != dataset["doi"].lower():
        raise ValueError("Figure 4 DOI mismatch")
    if title != dataset["expected_title"]:
        raise ValueError("Figure 4 title mismatch")
    if dataset["expected_size_bytes"] not in sizes:
        raise ValueError("Figure 4 byte size mismatch")
    if dataset["expected_license"] not in licenses:
        raise ValueError("Figure 4 license mismatch")
    if any(
        phrase.lower() not in description.lower()
        for phrase in dataset["required_description_phrases"]
    ):
        raise ValueError("Figure 4 description does not declare data and readme")

    access = {}
    for name, url in (
        ("article_api", dataset["article_api_url"]),
        ("article_page", dataset["article_page_url"]),
        ("archive", dataset["archive_url"]),
    ):
        raw, status = _request(url, method="HEAD" if name != "article_api" else "GET")
        access[name] = {
            "status_code": status or None,
            "accessible": status == 200,
            "response_bytes": len(raw),
        }
    manifest_retrieved = access["article_api"]["accessible"]
    archive_exceeds_budget = int(dataset["expected_size_bytes"]) > int(
        config["audit_boundary"]["maximum_download_bytes"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "parameter_fitting": False,
            "raw_files_downloaded": False,
        },
        "figure4_dataset": {
            "doi": attributes["doi"],
            "title": title,
            "description": description,
            "size_bytes": dataset["expected_size_bytes"],
            "license_identifiers": licenses,
            "metadata_status_code": metadata_status,
            "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
        },
        "access_observation": access,
        "bounded_download_decision": {
            "maximum_download_bytes": config["audit_boundary"]["maximum_download_bytes"],
            "archive_exceeds_budget": archive_exceeds_budget,
            "archive_downloaded": False,
            "file_manifest_retrieved": manifest_retrieved,
            "individual_readme_or_plotting_file_addressable": False,
        },
        "label_status": {
            "direction_code_to_PD_ND_mapping_verified": False,
            "biological_PD_code_assigned": None,
            "reason": (
                "DataCite verifies the Figure 4 data-and-code package and declares a readme, "
                "but the environment could not retrieve its file manifest or bounded individual "
                "files. The 1.392-GB archive exceeds this audit's download budget."
            ),
        },
        "next_protocol_if_manifest_available": config["next_protocol_if_manifest_available"],
        "audit_boundary": config["audit_boundary"],
        "advance_to_model_scoring": False,
        "advance_to_visual_gate": False,
        "advance_to_central_complex": False,
    }
