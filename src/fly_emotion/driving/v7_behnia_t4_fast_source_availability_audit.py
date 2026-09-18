"""Audit public numeric-payload availability for Behnia et al. 2014."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-behnia-t4-fast-source-availability-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_behnia_t4_fast_source_availability_audit.py"
)
DATA_SUFFIXES = (
    ".npy",
    ".npz",
    ".mat",
    ".csv",
    ".xls",
    ".xlsx",
    ".zip",
    ".h5",
    ".hdf5",
    ".pickle",
    ".pkl",
)


def evaluate_v7_behnia_t4_fast_source_availability_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_path = Path(config["paper_evidence"])
    evidence = json.loads((root / evidence_path).read_text(encoding="utf-8"))
    if set(evidence["source_evidence"]) != {"Mi1", "Tm3"}:
        raise ValueError("Behnia source phenotype coverage changed")
    snapshots = {}
    for name, spec in config["metadata_snapshots"].items():
        path = root / spec["path"]
        if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
            raise ValueError(f"Behnia metadata snapshot changed: {name}")
        snapshots[name] = json.loads(path.read_text(encoding="utf-8"))

    europe = snapshots["europe_pmc"]
    if europe["hitCount"] != 1:
        raise ValueError("Europe PMC exact PMID result count changed")
    europe_record = europe["resultList"]["result"][0]
    if (
        europe_record["pmid"] != "25043016"
        or europe_record["pmcid"] != "PMC4243710"
        or europe_record["doi"] != "10.1038/nature13427"
    ):
        raise ValueError("Europe PMC paper identity changed")

    crossref = snapshots["crossref"]["message"]
    if crossref["DOI"] != "10.1038/nature13427":
        raise ValueError("Crossref paper identity changed")
    crossref_links = [item["URL"] for item in crossref.get("link", [])]
    crossref_data_links = [
        url for url in crossref_links if url.lower().split("?", 1)[0].endswith(DATA_SUFFIXES)
    ]

    github_doi = snapshots["github_doi_search"]
    github_title = snapshots["github_title_search"]
    doi_data_hits = sorted(
        (item["repository"]["full_name"], item["path"])
        for item in github_doi["items"]
        if item["name"].lower().endswith(DATA_SUFFIXES)
    )
    expected_irrelevant = sorted(
        (item["repository"], item["path"]) for item in config["github_DOI_data_like_hits"]
    )
    if doi_data_hits != expected_irrelevant:
        raise ValueError("GitHub DOI data-like search hits changed")
    title_data_hits = [
        item for item in github_title["items"] if item["name"].lower().endswith(DATA_SUFFIXES)
    ]

    observations = {
        "europe_PMC": {
            "exact_record_count": europe["hitCount"],
            "is_open_access": europe_record.get("isOpenAccess") == "Y",
            "has_supplementary_files": europe_record.get("hasSuppl") == "Y",
            "has_database_cross_references": europe_record.get("hasDbCrossReferences")
            == "Y",
        },
        "Crossref": {
            "relation_count": sum(len(items) for items in crossref.get("relation", {}).values()),
            "links": crossref_links,
            "numeric_data_links": crossref_data_links,
        },
        "DataCite": {
            "related_dataset_count": snapshots["datacite_related"]["meta"]["total"]
        },
        "Zenodo": {
            "exact_title_result_count": snapshots["zenodo_title_search"]["hits"][
                "total"
            ]
        },
        "GitHub": {
            "exact_DOI_result_count": github_doi["total_count"],
            "exact_title_result_count": github_title["total_count"],
            "exact_title_numeric_file_hits": title_data_hits,
            "exact_DOI_data_like_hits": config["github_DOI_data_like_hits"],
            "exact_DOI_neuronal_payload_hits": [],
        },
    }
    payload_found = bool(
        observations["europe_PMC"]["has_supplementary_files"]
        or crossref_data_links
        or observations["DataCite"]["related_dataset_count"]
        or observations["Zenodo"]["exact_title_result_count"]
        or title_data_hits
        or observations["GitHub"]["exact_DOI_neuronal_payload_hits"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(evidence_path): _sha256(root / evidence_path),
                **{
                    spec["path"]: _sha256(root / spec["path"])
                    for spec in config["metadata_snapshots"].values()
                },
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "paper": evidence["protocol"]["paper"],
        "audited_public_indexes": observations,
        "local_numeric_trace_payload_found_in_audited_indexes": payload_found,
        "independent_Mi1_Tm3_numeric_transfer_authorized": False,
        "authorize_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "audited_public_indexes_expose_article_metadata_but_no_numeric_trace_payload"
        ),
        "boundary": config["boundary"],
    }
