"""Audit the official eLife 49373 source-data DOCX attachments."""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path(
    "configs/driving-v7-ketkar-2019-source-data-attachment-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_ketkar_2019_source_data_attachment_audit.py"
)


def _docx_inventory(path: Path, terms: list[str]) -> dict:
    try:
        with ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"corrupt DOCX member: {path.name}:{bad_member}")
            members = archive.namelist()
            text_fragments: list[str] = []
            for member in members:
                if member.startswith("word/") and member.endswith(".xml"):
                    root = ElementTree.fromstring(archive.read(member))
                    text_fragments.extend(
                        node.text or ""
                        for node in root.iter()
                        if node.tag.endswith("}t")
                    )
    except BadZipFile as error:
        raise ValueError(f"invalid DOCX ZIP: {path.name}") from error
    text = " ".join(text_fragments)
    normalized = " ".join(text.split())
    hits = {
        term: len(re.findall(re.escape(term), normalized, flags=re.IGNORECASE))
        for term in terms
    }
    return {
        "valid_docx_zip": True,
        "member_count": len(members),
        "embedded_object_count": sum(
            member.startswith("word/embeddings/") for member in members
        ),
        "contains_mean_plus_minus_sem_description": bool(
            re.search(r"mean\s*±\s*s\.?\s*e\.?\s*m", normalized, re.I)
        ),
        "term_hits": hits,
    }


def evaluate_v7_ketkar_2019_source_data_attachment_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    directory = root / config["attachment_directory"]
    expected_names = [item["name"] for item in config["attachments"]]
    actual_names = sorted(path.name for path in directory.glob("*.docx"))
    if actual_names != sorted(expected_names):
        raise ValueError("Ketkar source-data attachment set changed")

    attachments = {}
    for spec in config["attachments"]:
        path = directory / spec["name"]
        if path.stat().st_size != int(spec["bytes"]):
            raise ValueError(f"attachment size mismatch: {spec['name']}")
        digest = _sha256(path)
        if digest != spec["sha256"]:
            raise ValueError(f"attachment SHA-256 mismatch: {spec['name']}")
        attachments[spec["name"]] = {
            "official_url": (
                f"{config['official_attachment_base_url']}/{spec['name']}"
            ),
            "bytes": path.stat().st_size,
            "sha256": digest,
            **_docx_inventory(path, config["content_terms"]),
        }

    aggregate_hits = {
        term: sum(item["term_hits"][term] for item in attachments.values())
        for term in config["content_terms"]
    }
    all_valid = all(item["valid_docx_zip"] for item in attachments.values())
    all_summary = all(
        item["contains_mean_plus_minus_sem_description"]
        for item in attachments.values()
    )
    embedded_count = sum(
        item["embedded_object_count"] for item in attachments.values()
    )
    required_source_hits = {name: aggregate_hits[name] for name in ("Mi4", "C3")}
    individual_hits = {
        name: aggregate_hits[name]
        for name in ("individual", "Flyname", "fly id", "per-fly")
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "raw_attachment_identity": {
                str(Path(config["attachment_directory"]) / name): item["sha256"]
                for name, item in attachments.items()
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "paper": config["paper"],
        "attachment_count": len(attachments),
        "attachments": attachments,
        "all_attachments_valid_docx_zip": all_valid,
        "all_attachments_describe_mean_plus_minus_sem_tables": all_summary,
        "embedded_object_count": embedded_count,
        "aggregate_term_hits": aggregate_hits,
        "required_source_term_hits": required_source_hits,
        "individual_identifier_term_hits": individual_hits,
        "Mi1_Tm3_GCaMP_summary_evidence_found": all(
            aggregate_hits[name] > 0 for name in ("Mi1", "Tm3", "GCaMP")
        ),
        "Mi4_or_C3_attachment_payload_found": any(required_source_hits.values()),
        "individual_source_dynamics_payload_found": (
            any(individual_hits.values()) or embedded_count > 0
        ),
        "experimental_membrane_voltage_payload_found": aggregate_hits["voltage"] > 0,
        "source_dynamics_transfer_gate_changed": False,
        "authorize_T4_source_dynamics_fit": False,
        "authorize_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "attachments_are_mean_SEM_tables_without_Mi4_C3_individual_voltage",
        "boundary": config["boundary"],
    }
