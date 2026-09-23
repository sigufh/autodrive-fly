"""Bound Fendl 2021 target-receptor evidence from Mi4/C3 source dynamics."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-fendl-mi4-c3-receptor-boundary-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_fendl_mi4_c3_receptor_boundary_audit.py"
)


def _verify(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"Fendl snapshot changed: {path}")
    return path


def _term_pages(pages: list[str], term: str) -> tuple[list[int], int]:
    pattern = re.compile(rf"(?<![A-Za-z0-9]){term}(?![A-Za-z0-9])")
    counts = [len(pattern.findall(page)) for page in pages]
    return [index for index, count in enumerate(counts, 1) if count], sum(counts)


def evaluate_v7_fendl_mi4_c3_receptor_boundary_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    thesis = config["thesis"]
    landing = _verify(root, thesis["landing_page"])
    dissertation = _verify(root, thesis["dissertation"])
    if hashlib.md5(dissertation.read_bytes()).hexdigest() != thesis["dissertation"][
        "md5"
    ]:
        raise ValueError("Fendl repository MD5 changed")
    landing_text = landing.read_text(encoding="utf-8")
    if (
        thesis["title"] not in landing_text
        or 'meta content="Fendl, Sandra" name="DC.creator"' not in landing_text
        or thesis["dissertation"]["md5"] not in landing_text
    ):
        raise ValueError("Fendl repository metadata changed")

    reader = PdfReader(dissertation, strict=False)
    pages = [" ".join((page.extract_text() or "").split()) for page in reader.pages]
    expected = config["expected"]
    if (
        len(reader.pages) != int(expected["page_count"])
        or len(reader.attachments) != int(expected["embedded_attachment_count"])
    ):
        raise ValueError("Fendl dissertation inventory changed")
    term_evidence = {}
    for source in ("Mi4", "C3"):
        hit_pages, hit_count = _term_pages(pages, source)
        if (
            hit_pages != expected[f"{source}_exact_term_pages"]
            or hit_count != int(expected[f"{source}_exact_term_count"])
        ):
            raise ValueError(f"Fendl {source} term inventory changed")
        term_evidence[source] = {
            "exact_term_count": hit_count,
            "exact_term_pages": hit_pages,
        }

    text = " ".join(pages)
    phrases = (
        "used the glutamate sensor iGluSnFR to characterize the temporal dynamics "
        "of the three glutamatergic cell types of the motion vision pathway L1, "
        "Mi9 and LPi",
        "F.G.R. conducted and analyzed the imaging experiments for Mi9 and L1",
        "S.F. performed and analyzed all stainings",
        "Pure inhibitory input to T4 is provided by GABAergic Mi4, C3 and CT1 via "
        "the Rdl receptor",
        "These findings lay the foundation for future functional investigations of "
        "receptors and ion channels in T4/T5 neurons",
    )
    if any(phrase not in text for phrase in phrases):
        raise ValueError("Fendl experiment or interpretation scope changed")

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "raw_snapshot_identity": {
                thesis["landing_page"]["path"]: _sha256(landing),
                thesis["dissertation"]["path"]: _sha256(dissertation),
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "thesis": {
            "title": thesis["title"],
            "author": thesis["author"],
            "doi": thesis["doi"],
            "page_count": len(reader.pages),
            "embedded_attachment_count": len(reader.attachments),
        },
        "required_source_term_evidence": term_evidence,
        "direct_functional_imaging_cell_types": ["L1", "Mi9", "LPi4-3"],
        "target_receptor_localization_cell_types": ["T4", "T5"],
        "target_receptor_evidence": {
            "Rdl_localized_on_T4_T5_dendrites": True,
            "Mi4_C3_CT1_assigned_as_GABAergic_T4_inputs": True,
            "source_specific_Mi4_or_C3_receptor_contact_resolved": False,
        },
        "Mi4_direct_neural_recording_verified": False,
        "C3_direct_neural_recording_verified": False,
        "Mi4_C3_source_numeric_payload_verified": False,
        "experimental_Mi4_C3_membrane_voltage_verified": False,
        "classification": "target_receptor_and_pooled_input_sign_evidence_only",
        "authorize_Mi4_C3_source_dynamics_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "target_receptor_localization_is_not_source_activity",
        "boundary": config["boundary"],
    }
