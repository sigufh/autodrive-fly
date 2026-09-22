"""Audit public indexes for numeric Mi4 data linked to Strother et al. 2018."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import yaml
from pypdf import PdfReader

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
    json_names = (
        "crossref_work",
        "europe_pmc",
        "openalex_work",
        "semanticscholar_work",
        "datacite_related",
        "datacite_title",
        "zenodo_doi_search",
        "zenodo_title_search",
        "github_doi_repositories",
        "github_title_repositories",
    )
    payloads = {
        name: json.loads(paths[name].read_text(encoding="utf-8"))
        for name in json_names
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

    with zipfile.ZipFile(paths["europe_pmc_supplementary_bundle"]) as archive:
        supplement_names = archive.namelist()
    pdf_names = [name for name in supplement_names if name.lower().endswith(".pdf")]
    video_names = [name for name in supplement_names if name.lower().endswith(".mp4")]
    numeric_names = [
        name for name in supplement_names
        if name.lower().split("?", 1)[0].endswith(DATA_SUFFIXES)
    ]
    if (
        len(supplement_names) != 16
        or pdf_names != ["pnas.201703090SI.pdf"]
        or video_names != ["pnas.1703090115.sm01.mp4"]
        or numeric_names
    ):
        raise ValueError("Strother supplementary inventory changed")
    pdf = PdfReader(paths["supplementary_pdf"], strict=False)
    pdf_text = " ".join(
        " ".join((page.extract_text() or "").split()) for page in pdf.pages
    )
    supplement_phrases = (
        "All data, reagents, and code used in this manuscript will be provided upon request",
        "Time series of axonal calcium responses (ΔF/F) of Mi1, Tm3, Mi4, Mi9, "
        "and T4 neurons to a grating moving at 90°/s",
        "n = 5 for each genotype",
        "Time series of Mi4 dendritic calcium response",
        "n = 3 for each+CDM and−CDM",
    )
    if len(pdf.pages) != 10 or any(
        phrase not in pdf_text for phrase in supplement_phrases
    ):
        raise ValueError("Strother supplementary Mi4 evidence changed")

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
            "bundle_entry_count": len(supplement_names),
            "supplementary_PDF_pages": len(pdf.pages),
            "PDF_embedded_attachment_count": len(pdf.attachments),
        },
        "supplementary_Mi4_evidence": {
            "moving_grating_axonal_trace_figure_present": True,
            "moving_grating_speed_degrees_per_second": 90.0,
            "moving_grating_temporal_frequency_hz": 3.0,
            "moving_grating_fly_count": 5,
            "L5_photoactivation_dendritic_trace_figure_present": True,
            "GCaMP6f_photoactivation_brain_count_per_condition": 3,
            "data_availability": "upon_request",
            "public_numeric_trace_attachment_verified": False,
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
