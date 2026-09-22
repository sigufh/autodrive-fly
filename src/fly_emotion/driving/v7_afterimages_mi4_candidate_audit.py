"""Audit Mi4 evidence in the 2026 afterimage preprint."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-afterimages-mi4-candidate-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_afterimages_mi4_candidate_audit.py")


def _verify(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"afterimage snapshot identity changed: {path}")
    return path


def evaluate_v7_afterimages_mi4_candidate_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {name: _verify(root, spec) for name, spec in config["snapshots"].items()}
    xml_root = ElementTree.fromstring(paths["europe_pmc_fulltext"].read_text())
    text = " ".join(" ".join(xml_root.itertext()).split())
    paper = config["paper"]
    article_ids = {
        element.attrib["pub-id-type"]: "".join(element.itertext()).strip()
        for element in xml_root.findall(".//article-id")
        if "pub-id-type" in element.attrib
    }
    if (
        article_ids.get("doi") != paper["doi"]
        or article_ids.get("pmid") != str(paper["pmid"])
        or article_ids.get("pmcid") != paper["pmcid"]
    ):
        raise ValueError("afterimage paper identity changed")
    required_phrases = (
        "conducted the same experiment on Mi4 and Mi9 neurons",
        "these neurons did not exhibit strong afterimage-like response patterns",
        "Mi4>GCaMP6f",
        "Images were acquired at ~13 Hz",
        "ROIs from different flies of the same cell type were pooled",
    )
    if any(phrase not in text for phrase in required_phrases):
        raise ValueError("afterimage Mi4 evidence changed")

    with zipfile.ZipFile(paths["supplementary_bundle"]) as archive:
        names = archive.namelist()
        pdf_names = [name for name in names if name.lower().endswith(".pdf")]
    if (
        len(names) != int(config["expected"]["supplementary_bundle_entry_count"])
        or pdf_names != ["NIHPP2026.01.19.700413V1-supplement-1.pdf"]
    ):
        raise ValueError("afterimage supplementary bundle inventory changed")
    pdf = PdfReader(paths["supplementary_pdf"], strict=False)
    if len(pdf.pages) != int(config["expected"]["supplementary_pdf_pages"]):
        raise ValueError("afterimage supplementary PDF page count changed")
    supplement_text = " ".join(
        " ".join((page.extract_text() or "").split()) for page in pdf.pages
    )
    expected = config["expected"]
    sample_phrases = (
        f'n = {expected["Mi4_ROI_count"]} ROIs from '
        f'{expected["Mi4_fly_count"]} flies for Mi4',
        f'n = {expected["Mi9_ROI_count"]} ROIs from',
        "flies for Mi9",
        "Calcium responses of Mi4",
    )
    if any(phrase not in supplement_text for phrase in sample_phrases):
        raise ValueError("afterimage Mi4 supplementary sample evidence changed")

    repository_links = sorted(
        {
            value
            for element in xml_root.findall(".//body//ext-link")
            for key, value in element.attrib.items()
            if key.endswith("href")
            and any(
                term in value.lower()
                for term in ("github", "dryad", "zenodo", "figshare", "osf.io")
            )
        }
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "raw_snapshot_identity": {
                str(path.relative_to(root)): _sha256(path) for path in paths.values()
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "paper": paper,
        "measurement": {
            "directly_recorded_neuron_types": [
                "T4",
                "T5",
                "HS",
                "Mi1",
                "Tm3",
                "Tm1",
                "Tm2",
                "Mi4",
                "Mi9",
            ],
            "Mi4_measurement_modality": "two_photon_GCaMP6f_calcium",
            "Mi4_response_unit": "deltaF_over_F",
            "Mi4_fly_count": int(expected["Mi4_fly_count"]),
            "Mi4_ROI_count": int(expected["Mi4_ROI_count"]),
            "Mi4_and_Mi9_afterimage_like_response_strong": False,
            "acquisition_rate_hz_approximate": 13.0,
            "right_optic_lobe_only": True,
            "ROI_identity_handling": "pooled_across_flies_after_probe_selection",
        },
        "supplement": {
            "bundle_entry_count": len(names),
            "PDF_count": len(pdf_names),
            "PDF_pages": len(pdf.pages),
            "embedded_attachment_count": len(pdf.attachments),
            "numeric_attachment_count": 0,
        },
        "public_repository_links_in_body": repository_links,
        "public_numeric_Mi4_payload_verified": False,
        "experimental_membrane_voltage": False,
        "C3_direct_recording_found": False,
        "recording_to_MaleCNS_body_crosswalk_found": False,
        "global_payload_absence_claimed": False,
        "authorize_Mi4_source_dynamics_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "Mi4_evidence_is_GCaMP6f_figure_level_without_numeric_payload",
        "boundary": config["boundary"],
    }
