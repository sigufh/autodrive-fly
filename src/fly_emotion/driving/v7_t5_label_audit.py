from __future__ import annotations

import binascii
import hashlib
import json
import struct
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
    repository_evidence_path = Path(config["processed_repository_evidence"])
    repository_config_path = Path(config["processed_repository_config"])
    repository = json.loads((root / repository_evidence_path).read_text())
    repository_config = yaml.safe_load((root / repository_config_path).read_text())
    if repository["repository"]["commit"] != repository_config["repository"]["commit"]:
        raise ValueError("T5 label audit repository commit mismatch")
    readiness = repository["direction_and_identity_readiness"]
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
        raw, status = _request(url, method="GET")
        access[name] = {
            "status_code": status or None,
            "accessible": status == 200,
            "response_bytes": len(raw),
        }
    paper_raw, paper_status = _request(config["paper_full_text_url"])
    paper_text = _plain(paper_raw.decode("utf-8", errors="replace"))
    paper_phrase_checks = {
        phrase: phrase.lower() in paper_text.lower()
        for phrase in config["required_paper_alignment_phrases"]
    }
    archive_exceeds_budget = int(dataset["expected_size_bytes"]) > int(
        config["audit_boundary"]["maximum_download_bytes"]
    )
    article_spec = dataset["local_article_metadata"]
    article_path = root / article_spec["path"]
    if article_path.stat().st_size != int(article_spec["bytes"]) or _sha256(
        article_path
    ) != article_spec["sha256"]:
        raise ValueError("Figure 4 local article metadata mismatch")
    article = json.loads(article_path.read_text(encoding="utf-8"))
    official = dataset["official_file"]
    if article.get("files") != [
        {
            "id": official["id"],
            "name": official["name"],
            "size": official["size_bytes"],
            "is_link_only": False,
            "download_url": f"https://ndownloader.figshare.com/files/{official['id']}",
            "supplied_md5": official["md5"],
            "computed_md5": official["md5"],
            "mimetype": "application/zip",
        }
    ]:
        raise ValueError("Figure 4 official file manifest mismatch")
    directory_spec = dataset["central_directory"]
    directory_path = root / directory_spec["path"]
    if directory_path.stat().st_size != int(directory_spec["bytes"]) or _sha256(
        directory_path
    ) != directory_spec["sha256"]:
        raise ValueError("Figure 4 central directory mismatch")
    directory = directory_path.read_bytes()
    cursor = 0
    entries = {}
    while cursor < len(directory):
        if directory[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError("invalid Figure 4 ZIP central directory")
        fields = struct.unpack_from("<IHHHHHHIIIHHHHHII", directory, cursor)
        _, _, _, _, _, _, _, crc, compressed, size, name_len, extra_len, comment_len, *_ = fields
        name = directory[cursor + 46 : cursor + 46 + name_len].decode()
        entries[name] = {"crc32": f"{crc:08x}", "compressed": compressed, "bytes": size}
        cursor += 46 + name_len + extra_len + comment_len
    if len(entries) != int(directory_spec["archive_entry_count"]):
        raise ValueError("Figure 4 ZIP entry count mismatch")
    extracted = []
    for spec in dataset["extracted_files"]:
        path = root / spec["path"]
        payload = path.read_bytes()
        if len(payload) != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
            raise ValueError(f"Figure 4 extracted file mismatch: {spec['path']}")
        name = path.name
        if name not in entries or entries[name]["bytes"] != len(payload):
            raise ValueError(f"Figure 4 extracted file absent from archive: {name}")
        crc = f"{binascii.crc32(payload) & 0xFFFFFFFF:08x}"
        if crc != spec["zip_crc32"] or crc != entries[name]["crc32"]:
            raise ValueError(f"Figure 4 extracted file CRC mismatch: {name}")
        extracted.append({**spec, "archive_entry_verified": True})
    organizing = (root / dataset["extracted_files"][2]["path"]).read_text()
    plotting = (root / dataset["extracted_files"][3]["path"]).read_text()
    required_organizing = (
        "p.direction_mb'",
        "'direction'",
    )
    required_plotting = (
        "assert(relDirs(1) == 0 & relDirs(2) == 1, 'directions are flipped')",
        "dataND = tempDat.MB(relInds(1)).data;",
        "dataPD = tempDat.MB(relInds(2)).data;",
    )
    if not all(value in organizing for value in required_organizing) or not all(
        value in plotting for value in required_plotting
    ):
        raise ValueError("Figure 4 plotting code no longer establishes numeric mapping")
    mapping = {"0": "ND", "1": "PD"}
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(repository_evidence_path): _sha256(root / repository_evidence_path),
                str(repository_config_path): _sha256(root / repository_config_path),
                article_spec["path"]: _sha256(article_path),
                directory_spec["path"]: _sha256(directory_path),
                **{
                    spec["path"]: _sha256(root / spec["path"])
                    for spec in dataset["extracted_files"]
                },
            },
            "parameter_fitting": False,
            "raw_files_downloaded": False,
            "full_archive_downloaded": False,
            "bounded_range_extraction_only": True,
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
        "processed_repository_evidence": {
            "commit": repository["repository"]["commit"],
            "manifest_sha256": repository["repository"]["manifest_sha256"],
            "recorded_cell_count": readiness["recorded_cell_count"],
            "cells_with_both_direction_codes": readiness[
                "cells_with_both_moving_bar_direction_codes"
            ],
            "direction_codes_present": readiness["direction_codes_present"],
            "direction_code_to_PD_ND_mapping_verified": readiness[
                "direction_code_to_PD_ND_mapping_verified"
            ],
            "stable_biological_cell_ids_available": readiness[
                "stable_biological_cell_ids_available"
            ],
            "code_effect": config["repository_direction_semantics"]["code_effect"],
        },
        "paper_alignment_evidence": {
            "url": config["paper_full_text_url"],
            "http_status": paper_status or None,
            "response_sha256": hashlib.sha256(paper_raw).hexdigest(),
            "required_phrase_checks": paper_phrase_checks,
            "required_phrases_verified": all(paper_phrase_checks.values()),
            "scope": (
                "paper confirms per-cell PD-ND alignment, not the repository's numeric "
                "direction-code mapping"
            ),
        },
        "bounded_download_decision": {
            "maximum_download_bytes": config["audit_boundary"]["maximum_download_bytes"],
            "archive_exceeds_budget": archive_exceeds_budget,
            "archive_downloaded": False,
            "file_manifest_retrieved": True,
            "manifest_source": "official_article_API_via_read_only_translation_proxy",
            "official_file": official,
            "central_directory_bytes_retrieved": len(directory),
            "archive_entry_count": len(entries),
            "individual_readme_or_plotting_file_addressable": True,
            "extracted_files": extracted,
        },
        "label_status": {
            "direction_code_to_PD_ND_mapping_verified": True,
            "direction_code_to_PD_ND": mapping,
            "biological_PD_code_assigned": 1,
            "biological_ND_code_assigned": 0,
            "mapping_evidence": (
                "organizingClusterData.m copies p.direction_mb into the direction column; "
                "sourceDataPlottingFig4Script.m asserts ordered codes [0,1], then assigns "
                "the first trace to dataND/modelND and the second to dataPD/modelPD"
            ),
            "reason": (
                "Official Figure 4 plotting code independently maps numeric direction code "
                "0 to ND and 1 to PD; response magnitude was not used to infer the mapping."
            ),
        },
        "next_protocol_if_manifest_available": config["next_protocol_if_manifest_available"],
        "audit_boundary": config["audit_boundary"],
        "advance_to_model_scoring": True,
        "advance_to_visual_gate": False,
        "advance_to_central_complex": False,
    }
