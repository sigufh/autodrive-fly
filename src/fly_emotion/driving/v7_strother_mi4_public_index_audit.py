"""Audit public indexes for numeric Mi4 data linked to Strother et al. 2018."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-strother-mi4-public-index-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_strother_mi4_public_index_audit.py"
)
DATA_SUFFIXES = (".csv", ".mat", ".npy", ".npz", ".zip", ".h5", ".xlsx")


def _verify(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"Strother index snapshot changed: {path}")
    return path


def evaluate_v7_strother_mi4_public_index_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {name: _verify(root, spec) for name, spec in config["snapshots"].items()}
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
        if not name.startswith("figshare_")
    }
    paper = config["paper"]
    crossref = payloads["crossref_work"]["message"]
    if crossref["DOI"] != paper["doi"]:
        raise ValueError("Strother Crossref identity changed")
    europe = payloads["europe_pmc"]
    if europe["hitCount"] != 1:
        raise ValueError("Strother Europe PMC result count changed")
    record = europe["resultList"]["result"][0]
    if record["pmid"] != str(paper["pmid"]) or record["pmcid"] != paper["pmcid"]:
        raise ValueError("Strother Europe PMC identity changed")
    openalex = payloads["openalex_work"]
    semantics = payloads["semanticscholar_work"]
    if openalex["doi"] != f"https://doi.org/{paper['doi']}":
        raise ValueError("Strother OpenAlex identity changed")
    if semantics["externalIds"]["DOI"] != paper["doi"]:
        raise ValueError("Strother Semantic Scholar identity changed")

    crossref_data_links = [
        item["URL"] for item in crossref.get("link", [])
        if item["URL"].lower().split("?", 1)[0].endswith(DATA_SUFFIXES)
    ]
    datacite_related = payloads["datacite_related"]["meta"]["total"]
    datacite_title = payloads["datacite_title"]["meta"]["total"]
    zenodo_doi = payloads["zenodo_doi_search"]["hits"]["total"]
    zenodo_title = payloads["zenodo_title_search"]["hits"]["total"]
    github_doi = payloads["github_doi_repositories"]["total_count"]
    github_title = payloads["github_title_repositories"]["total_count"]
    zero_counts = [
        datacite_related,
        datacite_title,
        zenodo_doi,
        zenodo_title,
        github_doi,
        github_title,
    ]
    if any(zero_counts):
        raise ValueError("public index gained an unreviewed Strother candidate")
    figshare = paths["figshare_doi_response"].read_text(encoding="utf-8")
    if "403 Forbidden" not in figshare:
        raise ValueError("Strother Figshare access boundary changed")

    source_spec = config["paper_source"]
    source_path = _verify(root, source_spec)
    source_html = source_path.read_text(encoding="utf-8")
    source_text = " ".join(re.sub(r"<[^>]+>", " ", source_html).split())
    required_phrases = (
        "GCaMP6f for Mi1, Tm3, Mi4, and Mi9",
        "slow speed of the calcium indicator GCaMP6s",
        "unlikely to accurately capture the true kinetics of the Mi4 response",
        "All data, reagents, and code used in this manuscript will be provided upon request",
    )
    if any(phrase not in source_text for phrase in required_phrases):
        raise ValueError("Strother Mi4 source text changed")

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
            | {source_spec["path"]: _sha256(source_path)},
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "paper": paper,
        "audited_indexes": {
            "Crossref_relation_count": sum(
                len(items) for items in crossref.get("relation", {}).values()
            ),
            "Crossref_numeric_data_links": crossref_data_links,
            "Europe_PMC_has_supplement": record.get("hasSuppl") == "Y",
            "Europe_PMC_has_database_cross_references": (
                record.get("hasDbCrossReferences") == "Y"
            ),
            "OpenAlex_repository_fulltext_available": openalex["open_access"][
                "any_repository_has_fulltext"
            ],
            "Semantic_Scholar_open_access_location_is_article": (
                semantics["openAccessPdf"]["url"]
                == f"https://doi.org/{paper['doi']}"
            ),
            "DataCite_related_result_count": datacite_related,
            "DataCite_exact_title_result_count": datacite_title,
            "Zenodo_exact_DOI_result_count": zenodo_doi,
            "Zenodo_exact_title_result_count": zenodo_title,
            "GitHub_exact_DOI_repository_count": github_doi,
            "GitHub_exact_title_repository_count": github_title,
            "Figshare_search_status": 403,
            "Figshare_search_interpretable": False,
        },
        "PMC_attachment_inventory": {
            "supplementary_PDF_count": 1,
            "video_count": 1,
            "numeric_data_attachment_count": 0,
        },
        "Mi4_measurement_modality": "deltaF_over_F",
        "local_numeric_Mi4_trace_payload_found_in_successful_indexes": False,
        "global_absence_claimed": False,
        "authorize_Mi4_source_dynamics_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            "successful_indexes_expose_no_linked_Mi4_trace_while_Figshare_is_"
            "inaccessible"
        ),
        "boundary": config["boundary"],
    }
