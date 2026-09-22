"""Track the unresolved cell-type scope of the 2026 ASAP7y preprint."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-hao-asap7y-candidate-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_hao_asap7y_candidate_audit.py")


def _verify(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"ASAP7y candidate snapshot changed: {path}")
    return path


def evaluate_v7_hao_asap7y_candidate_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {name: _verify(root, spec) for name, spec in config["snapshots"].items()}
    payloads = {}
    for name in (
        "crossref",
        "openalex",
        "semanticscholar",
        "biorxiv_api",
        "europe_pmc",
        "openrxiv_response",
    ):
        payloads[name] = json.loads(paths[name].read_text(encoding="utf-8"))
    paper = config["paper"]
    crossref = payloads["crossref"]["message"]
    if crossref["DOI"] != paper["doi"]:
        raise ValueError("ASAP7y Crossref identity changed")
    biorxiv = payloads["biorxiv_api"]["collection"]
    if len(biorxiv) != 1 or biorxiv[0]["doi"] != paper["doi"]:
        raise ValueError("ASAP7y bioRxiv API identity changed")
    europe = payloads["europe_pmc"]
    if europe["hitCount"] != 1 or europe["resultList"]["result"][0]["doi"] != paper["doi"]:
        raise ValueError("ASAP7y Europe PMC identity changed")
    abstract = " ".join(re.sub(r"<[^>]+>", " ", crossref["abstract"]).split())
    required_phrases = (
        "develop ASAP7y",
        "both mice and flies",
        "two-photon random-access microscopy",
        "individual neurons in Drosophila",
        "visual system",
        "717 cell types",
    )
    if any(phrase not in abstract for phrase in required_phrases):
        raise ValueError("ASAP7y abstract evidence changed")
    reference_dois = [item.get("DOI", "").lower() for item in crossref["reference"]]
    if reference_dois.count("10.1038/s41586-022-04428-3") != 1:
        raise ValueError("ASAP7y Groschner reference changed")
    if payloads["openalex"]["open_access"]["oa_status"] != "closed":
        raise ValueError("ASAP7y OpenAlex access status changed")
    if payloads["openrxiv_response"].get("error") != "No works found":
        raise ValueError("ASAP7y openRxiv availability changed")
    if paths["biorxiv_HTML_response"].read_text().strip() != "error code: 1015":
        raise ValueError("ASAP7y bioRxiv HTML access response changed")
    if paths["biorxiv_JATS_response"].read_text().strip() != "error code: 1015":
        raise ValueError("ASAP7y bioRxiv JATS access response changed")
    if json.loads(paths["highwire_response"].read_text())["status"] != 403:
        raise ValueError("ASAP7y HighWire access response changed")
    if json.loads(paths["europe_pmc_fulltext_response"].read_text())["status"] != 500:
        raise ValueError("ASAP7y Europe PMC full-text response changed")
    thread_spec = config["public_author_thread"]
    thread_paths = {
        "resolve": _verify(root, thread_spec["resolve"]),
        "thread": _verify(root, thread_spec["thread"]),
        **{
            name: _verify(root, spec) for name, spec in thread_spec["images"].items()
        },
    }
    if json.loads(thread_paths["resolve"].read_text())["did"] != thread_spec["author_did"]:
        raise ValueError("ASAP7y author-thread DID identity changed")
    thread = json.loads(thread_paths["thread"].read_text())
    thread_text = json.dumps(thread, ensure_ascii=False)
    thread_phrases = (
        "ASAP7y recordings with analysis of the EM connectome of 717 neuron types",
        "most of the ASAP7y paper is in flies",
    )
    if any(phrase not in thread_text for phrase in thread_phrases):
        raise ValueError("ASAP7y public author-thread text changed")
    visually_verified_examples = thread_spec["visually_verified_post_11_labels"]

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
            }
            | {
                str(path.relative_to(root)): _sha256(path)
                for path in thread_paths.values()
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "paper": paper,
        "verified_scope": {
            "Drosophila_in_vivo_voltage_imaging": True,
            "measurement_modality": "two_photon_ASAP7y_voltage_imaging",
            "millisecond_subcellular_subthreshold_resolution": True,
            "visual_system_model_cell_type_count": 717,
            "cites_Groschner_2022": True,
        },
        "cell_type_resolution": {
            "full_text_retrieved": False,
            "public_author_figure_named_examples": visually_verified_examples,
            "complete_experimental_cell_type_set_resolved": False,
            "experimental_Drosophila_cell_types_named_in_paper_sources": [],
            "Mi4_direct_recording_verified": False,
            "C3_direct_recording_verified": False,
            "candidate_classification": "unresolved_high_value_candidate",
        },
        "locator_evidence": {
            "source": "public_author_Bluesky_thread",
            "author_handle": thread_spec["author_handle"],
            "post_11_visual_labels": visually_verified_examples,
            "counts_as_numeric_payload": False,
            "resolves_complete_paper_cell_type_set": False,
        },
        "access_boundaries": {
            "bioRxiv_HTML_status": 429,
            "bioRxiv_JATS_status": 429,
            "HighWire_status": 403,
            "Europe_PMC_fulltext_status": 500,
            "Europe_PMC_in_PMC": False,
            "OpenAlex_status": "closed",
            "openRxiv_mapping_found": False,
        },
        "public_numeric_payload_verified": False,
        "authorize_Mi4_C3_candidate_classification": False,
        "authorize_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "experimental_Drosophila_cell_types_not_resolved_from_accessible_sources",
        "boundary": config["boundary"],
    }
