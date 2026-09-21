"""Audit whether Yuan et al. supplies direct C3 source dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-yuan-c3-candidate-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_yuan_c3_candidate_audit.py")


def _verified_json(root: Path, spec: dict) -> dict:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"snapshot size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"snapshot SHA-256 mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_v7_yuan_c3_candidate_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    snapshots = {
        name: _verified_json(root, spec)
        for name, spec in config["snapshots"].items()
        if name not in {"publisher_fulltext_response", "publisher_supplement_response"}
    }
    for name in ("publisher_fulltext_response", "publisher_supplement_response"):
        spec = config["snapshots"][name]
        path = root / spec["path"]
        if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
            raise ValueError(f"publisher response identity changed: {name}")

    paper = config["paper"]
    crossref = snapshots["crossref_work"]["message"]
    if crossref["DOI"] != paper["doi"]:
        raise ValueError("Yuan Crossref identity changed")
    europe = snapshots["europe_pmc"]
    if europe["hitCount"] != 1:
        raise ValueError("Yuan Europe PMC exact PMID result count changed")
    record = europe["resultList"]["result"][0]
    if record["pmid"] != str(paper["pmid"]) or record["doi"] != paper["doi"]:
        raise ValueError("Yuan Europe PMC identity changed")
    openalex = snapshots["openalex_work"]
    if openalex["doi"] != f"https://doi.org/{paper['doi']}":
        raise ValueError("Yuan OpenAlex identity changed")
    semantics = snapshots["semanticscholar_work"]
    if semantics["externalIds"]["DOI"] != paper["doi"]:
        raise ValueError("Yuan Semantic Scholar identity changed")
    abstract = semantics["abstract"]
    required_phrases = (
        "specific blocking of different types of lamina feedback neurons "
        "Lawf1, Lawf2, C2, C3, and T1",
        "Thermal activation of Lawf1 neurons could suppress neural activities in L1 and L2 neurons",
    )
    if any(phrase not in abstract for phrase in required_phrases):
        raise ValueError("Yuan abstract evidence changed")

    datacite_count = snapshots["datacite_related"]["meta"]["total"]
    zenodo_count = snapshots["zenodo_doi_search"]["hits"]["total"]
    if datacite_count != 0 or zenodo_count != 0:
        raise ValueError("public index gained an unreviewed Yuan candidate")
    fulltext_response = (
        root / config["snapshots"]["publisher_fulltext_response"]["path"]
    ).read_text(encoding="utf-8")
    supplement_response = (
        root / config["snapshots"]["publisher_supplement_response"]["path"]
    ).read_text(encoding="utf-8")
    blocked = all(
        "Just a moment" in response or "Enable JavaScript and cookies" in response
        for response in (fulltext_response, supplement_response)
    )
    if not blocked:
        raise ValueError("Yuan publisher access boundary changed")

    crossref_data_links = [
        item["URL"]
        for item in crossref.get("link", [])
        if item["URL"].lower().split("?", 1)[0].endswith(
            (".csv", ".mat", ".npy", ".npz", ".zip", ".h5", ".xlsx")
        )
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "raw_snapshot_identity": {
                spec["path"]: _sha256(root / spec["path"])
                for spec in config["snapshots"].values()
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "paper": paper,
        "open_access_status": {
            "Europe_PMC_in_PMC": record.get("inPMC") == "Y",
            "Europe_PMC_has_supplement": record.get("hasSuppl") == "Y",
            "Europe_PMC_has_database_cross_references": record.get("hasDbCrossReferences") == "Y",
            "OpenAlex_status": openalex["open_access"]["oa_status"],
            "OpenAlex_repository_fulltext_available": openalex["open_access"][
                "any_repository_has_fulltext"
            ],
            "Semantic_Scholar_open_access_PDF_available": bool(semantics["openAccessPdf"]["url"]),
            "publisher_fulltext_and_supplement_retrieved": False,
            "publisher_responses_are_access_challenges": blocked,
        },
        "abstract_evidence": {
            "C3_manipulated_by_specific_blocking": True,
            "directly_recorded_neural_activity_sources": ["L1", "L2"],
            "direct_C3_source_dynamics_reported": False,
            "experimental_C3_membrane_voltage_reported": False,
        },
        "public_index_evidence": {
            "Crossref_numeric_data_links": crossref_data_links,
            "DataCite_related_result_count": datacite_count,
            "Zenodo_exact_DOI_result_count": zenodo_count,
        },
        "C3_intervention_candidate_verified": True,
        "C3_direct_recording_candidate_verified": False,
        "C3_public_numeric_source_dynamics_payload_verified": False,
        "C3_experimental_membrane_voltage_verified": False,
        "authorize_C3_source_dynamics_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "C3_is_an_intervention_target_not_a_verified_direct_source_recording",
        "boundary": config["boundary"],
    }
