"""Classify legacy C3/Mi4 literature candidates without overclaiming access."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path

import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-legacy-c3-candidate-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_legacy_c3_candidate_audit.py")


def _verify(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"legacy C3 candidate snapshot changed: {path}")
    return path


def _flat_html(path: Path) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", path.read_text()).split())


def evaluate_v7_legacy_c3_candidate_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    verified_paths = {}
    for candidate in config["candidates"].values():
        for spec in candidate["snapshots"].values():
            path = _verify(root, spec)
            verified_paths[str(path.relative_to(root))] = _sha256(path)

    tuthill_spec = config["candidates"]["Tuthill_2013"]
    tuthill_meta = json.loads(
        (root / tuthill_spec["snapshots"]["europe_pmc"]["path"]).read_text()
    )["resultList"]["result"][0]
    tuthill_text = _flat_html(
        root / tuthill_spec["snapshots"]["fulltext"]["path"]
    )
    tuthill_phrases = (
        "Two classes of feedback neurons (C2 and C3)",
        "We then depolarized and hyperpolarized each neuron type",
        "quantified fly behavioral responses",
        "will require physiological recordings from C2 and C3 neurons",
    )
    if (
        tuthill_meta["doi"] != tuthill_spec["doi"]
        or any(phrase not in tuthill_text for phrase in tuthill_phrases)
    ):
        raise ValueError("Tuthill C3 intervention boundary changed")

    maisak_spec = config["candidates"]["Maisak_2018"]
    maisak_openalex = json.loads(
        (root / maisak_spec["snapshots"]["openalex"]["path"]).read_text()
    )
    reader = PdfReader(root / maisak_spec["snapshots"]["dissertation"]["path"])
    maisak_text = " ".join(
        " ".join((page.extract_text() or "").split()) for page in reader.pages
    )
    if (
        maisak_openalex["doi"].lower()
        != f"https://doi.org/{maisak_spec['doi']}".lower()
        or "Mi4 and Mi9 cells have been added to the potential candidates"
        not in maisak_text
        or "Mi9, C3, and CT1" not in maisak_text
        or "without functional studies" not in maisak_text
    ):
        raise ValueError("Maisak dissertation scope changed")

    ramos_spec = config["candidates"]["Ramos_Traslosheros_2020"]
    ramos_openalex = json.loads(
        (root / ramos_spec["snapshots"]["openalex"]["path"]).read_text()
    )
    oai_root = ET.fromstring(
        (root / ramos_spec["snapshots"]["oai_dc"]["path"]).read_text()
    )
    oai_text = " ".join(" ".join(oai_root.itertext()).split())
    xmeta_text = (
        root / ramos_spec["snapshots"]["xMetaDissPlus"]["path"]
    ).read_text()
    ramos_phrases = (
        "we map the functional circuit organization",
        "We focus on Tm9",
        "identify Dm4, Dm12, and Dm20",
        "genetic silencing with in vivo imaging",
    )
    if (
        ramos_openalex["doi"].lower()
        != f"https://doi.org/{ramos_spec['doi']}".lower()
        or any(phrase not in oai_text for phrase in ramos_phrases)
        or ramos_spec["fulltext_url"] not in unescape(xmeta_text)
    ):
        raise ValueError("Ramos-Traslosheros thesis metadata changed")

    candidates = {
        "Tuthill_2013": {
            "doi": tuthill_spec["doi"],
            "fulltext_retrieved": True,
            "C3_role": "genetic_activation_and_silencing_target",
            "measurement_object": "flight_steering_behavior",
            "direct_C3_neural_recording_verified": False,
            "explicit_future_need_for_C3_physiology": True,
            "classification": "independent_C3_intervention_only",
        },
        "Maisak_2018": {
            "doi": maisak_spec["doi"],
            "fulltext_retrieved": True,
            "page_count": len(reader.pages),
            "embedded_attachment_count": len(reader.attachments),
            "C3_exact_term_count": len(
                re.findall(r"(?<![A-Za-z0-9])C3(?![A-Za-z0-9])", maisak_text)
            ),
            "Mi4_exact_term_count": len(
                re.findall(r"(?<![A-Za-z0-9])Mi4(?![A-Za-z0-9])", maisak_text)
            ),
            "direct_C3_or_Mi4_recording_verified": False,
            "classification": "anatomical_candidate_mentions_only",
        },
        "Ramos_Traslosheros_2020": {
            "doi": ramos_spec["doi"],
            "OAI_metadata_retrieved": True,
            "official_PDF_locator_found": True,
            "fulltext_probe_status": int(ramos_spec["fulltext_probe_status"]),
            "fulltext_retrieved": False,
            "abstract_named_direct_imaging_types": ["Tm9", "Dm4", "Dm12", "Dm20"],
            "abstract_names_C3_or_Mi4_direct_recording": False,
            "complete_fulltext_scope_resolved": False,
            "classification": "unresolved_beyond_abstract_scope",
        },
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "raw_snapshot_identity": verified_paths,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "candidates": candidates,
        "audited_candidate_count": len(candidates),
        "independent_C3_intervention_only_candidates": ["Tuthill_2013"],
        "direct_C3_or_Mi4_source_dynamics_candidates": [],
        "unresolved_fulltext_candidates": ["Ramos_Traslosheros_2020"],
        "authorize_Mi4_C3_source_dynamics_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            "legacy_candidates_add_intervention_or_bounded_scope_"
            "not_direct_source_dynamics"
        ),
        "boundary": config["boundary"],
    }
