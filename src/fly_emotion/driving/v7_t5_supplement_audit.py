from __future__ import annotations

import hashlib
import io
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t5-supplement-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_supplement_audit.py")
WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _download_bounded(url: str, maximum_bytes: int) -> tuple[bytes, int]:
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, headers={"User-Agent": "AutoDrive-Fly-v7-audit/1"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > maximum_bytes:
                    raise ValueError("supplement bundle exceeds download budget")
                raw = response.read(maximum_bytes + 1)
                if len(raw) > maximum_bytes:
                    raise ValueError("supplement bundle exceeded download budget while streaming")
                return raw, int(response.status)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(1)
    raise ValueError("official supplement download failed after three attempts") from last_error


def _safe_zip_member(raw: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        member = archive.getinfo(name)
        if member.is_dir() or member.file_size > 2_000_000:
            raise ValueError("unexpected supplement member size or type")
        return archive.read(member)


def _docx_content(raw: bytes) -> tuple[list[str], list[list[str]]]:
    document = _safe_zip_member(raw, "word/document.xml")
    root = ElementTree.fromstring(document)
    paragraphs = [
        text
        for paragraph in root.findall(".//w:p", WORD_NAMESPACE)
        if (
            text := "".join(
                node.text or "" for node in paragraph.findall(".//w:t", WORD_NAMESPACE)
            ).strip()
        )
    ]
    rows = []
    for table in root.findall(".//w:tbl", WORD_NAMESPACE):
        for row in table.findall("./w:tr", WORD_NAMESPACE):
            rows.append(
                [
                    " ".join(
                        "".join(
                            node.text or "" for node in cell.findall(".//w:t", WORD_NAMESPACE)
                        ).split()
                    )
                    for cell in row.findall("./w:tc", WORD_NAMESPACE)
                ]
            )
    return paragraphs, rows


def _normalized_text(paragraphs: list[str]) -> str:
    return " ".join(" ".join(paragraphs).replace("–", "-").split())


def _contains_bounds(text: str, low: float, high: float) -> bool:
    text = text.replace("–", "-")
    pattern = rf"(?<![0-9.-]){re.escape(f'{low:g}')}\s*-\s*{re.escape(f'{high:g}')}(?![0-9.])"
    return re.search(pattern, text) is not None


def evaluate_v7_t5_supplement_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source = config["source"]
    raw, status = _download_bounded(source["bundle_url"], int(source["maximum_download_bytes"]))
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        member_count = len(archive.infolist())
    if member_count != int(source["expected_member_count"]):
        raise ValueError("supplement bundle member count mismatch")
    member = _safe_zip_member(raw, source["member"])
    if len(member) != int(source["member_size_bytes"]):
        raise ValueError("supplement member size mismatch")
    if hashlib.sha256(member).hexdigest() != source["member_sha256"]:
        raise ValueError("supplement member SHA-256 mismatch")
    paragraphs, table_rows = _docx_content(member)
    _normalized_text(paragraphs)
    expected_columns = ["Parameter", "Description", "Units", "Bounds"]
    if len(table_rows) != 14 or table_rows[0] != expected_columns:
        raise ValueError("unexpected Supplementary 1 table structure")
    rows_by_description = {row[1]: row for row in table_rows[1:]}
    free_parameters = []
    for name, description, unit, low, high in config["expected_free_parameters"]:
        row = rows_by_description.get(description)
        if row is None or row[2] != unit or not _contains_bounds(row[3], float(low), float(high)):
            raise ValueError(f"missing expected bounds for {name}")
        free_parameters.append(
            {
                "name": name,
                "description": description,
                "unit": unit,
                "minimum": float(low),
                "maximum": float(high),
            }
        )
    fixed = {}
    for name, description, unit, value in config["expected_fixed_parameters"]:
        row = rows_by_description.get(description)
        normalized_bound = " ".join(row[3].split()) if row else ""
        if row is None or row[2] != unit or normalized_bound != f"{float(value):g} (fixed)":
            raise ValueError(f"missing expected fixed parameter {name}")
        fixed[name] = float(value)
    fitted_cell_values_present = len(table_rows[0]) > 4 or len(table_rows) > 14
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "safe_parsing": "nested ZIP member plus word/document.xml only",
            "office_content_executed": False,
            "parameter_fitting": False,
            "raw_files_committed": False,
        },
        "source": {
            "pmc_id": source["pmc_id"],
            "doi": source["doi"],
            "http_status": status,
            "bundle_size_bytes": len(raw),
            "bundle_sha256": hashlib.sha256(raw).hexdigest(),
            "bundle_sha256_is_identity_constraint": False,
            "bundle_is_dynamically_repacked": source["bundle_is_dynamically_repacked"],
            "bundle_member_count": member_count,
            "member": source["member"],
            "member_size_bytes": len(member),
            "member_sha256": hashlib.sha256(member).hexdigest(),
        },
        "parameter_contract": {
            "free_parameter_count": len(free_parameters),
            "free_parameters": free_parameters,
            "fixed_parameters": fixed,
            "fitted_cell_parameter_values_present": fitted_cell_values_present,
            "table_rows_including_header": len(table_rows),
            "table_columns": table_rows[0],
        },
        "replay_status": {
            "published_17_cell_parameter_values_available": False,
            "zero_fit_replay_performed": False,
            "measured_model_comparison_performed": False,
            "reason": (
                "Supplementary file 1 provides parameter definitions, units, bounds and fixed "
                "potentials, but no fitted parameter vector for any of the 17 cells."
            ),
        },
        "audit_boundary": config["audit_boundary"],
        "limitations": [
            "The Europe PMC bundle is a publication supplement package, not Figure 4 raw data.",
            "Parameter bounds cannot substitute for fitted per-cell values.",
            "No model quality or generalization metric is produced by this audit.",
        ],
        "advance_to_T5_replay": False,
        "advance_to_T5_fit": False,
        "advance_to_visual_gate": False,
        "advance_to_central_complex": False,
    }
