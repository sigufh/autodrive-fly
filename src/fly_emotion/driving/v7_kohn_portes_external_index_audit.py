"""Bound external-index availability for Kohn-Portes stimulus logs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-kohn-portes-external-index-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_kohn_portes_external_index_audit.py"
)


def evaluate_v7_kohn_portes_external_index_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    snapshots = {}
    for name, spec in config["snapshots"].items():
        path = root / spec["path"]
        if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
            raise ValueError(f"Kohn-Portes external-index snapshot changed: {name}")
        snapshots[name] = path

    crossref = json.loads(snapshots["crossref_work"].read_text())["message"]
    if crossref["DOI"] != config["paper_doi"]:
        raise ValueError("Kohn-Portes Crossref identity changed")
    relations = crossref.get("relation", {})
    preprints = [item["id"] for item in relations.get("has-preprint", [])]
    if preprints != [config["preprint_doi"]]:
        raise ValueError("Kohn-Portes Crossref preprint relation changed")
    crossref_links = [item["URL"] for item in crossref.get("link", [])]
    data_suffixes = (".csv", ".json", ".mat", ".npy", ".npz", ".zip")
    crossref_data_links = [
        url for url in crossref_links if url.lower().split("?", 1)[0].endswith(data_suffixes)
    ]

    preprint = json.loads(snapshots["crossref_preprint"].read_text())["message"]
    if preprint["DOI"] != config["preprint_doi"]:
        raise ValueError("Kohn-Portes preprint identity changed")
    if [item["id"] for item in preprint.get("relation", {}).get("is-preprint-of", [])] != [
        config["paper_doi"]
    ]:
        raise ValueError("Kohn-Portes preprint article relation changed")

    datacite_related = json.loads(snapshots["datacite_related"].read_text())
    datacite_title = json.loads(snapshots["datacite_title"].read_text())
    if datacite_related["meta"]["total"] != 0 or datacite_title["meta"]["total"] != 0:
        raise ValueError("DataCite gained an unreviewed Kohn-Portes candidate")

    europe = json.loads(snapshots["europe_pmc"].read_text())
    if europe["hitCount"] != 1:
        raise ValueError("Europe PMC exact DOI count changed")
    europe_record = europe["resultList"]["result"][0]
    if europe_record.get("pmcid") != "PMC8725177":
        raise ValueError("Europe PMC Kohn-Portes identity changed")

    pmc_html = snapshots["pmc_article"].read_text(encoding="utf-8")
    supplement_matches = re.findall(
        r'href="([^"]*NIHMS1750660-supplement-1\.pdf)"[^>]*>[^<]+</a>'
        r'<sup> \(([^)]+), pdf\)',
        pmc_html,
    )
    if not supplement_matches:
        raise ValueError("PMC supplementary PDF listing changed")
    supplement_paths = sorted({path for path, _size in supplement_matches})
    supplement_sizes = sorted({size for _path, size in supplement_matches})
    data_availability = (
        "Custom Python code used for modeling and analysis is freely available at "
        "GITHUB LINK TBD. All source code used for visual stimulation is available on GitLab"
    )
    if data_availability not in pmc_html:
        raise ValueError("PMC data-availability statement changed")

    supplement_response = snapshots["pmc_supplement_response"].read_text(encoding="utf-8")
    if (
        "Preparing to download" not in supplement_response
        or "POW_CHALLENGE" not in supplement_response
    ):
        raise ValueError("PMC supplement access response changed")
    elsevier = ElementTree.parse(snapshots["elsevier_minimal_response"]).getroot()
    elsevier_text = " ".join(element.text or "" for element in elsevier.iter())
    if config["paper_doi"] not in elsevier_text:
        raise ValueError("Elsevier minimal metadata identity changed")
    elsevier_headers = snapshots["elsevier_headers"].read_text(encoding="utf-8")
    if "Unauthorized request results in minimized metadata response" not in elsevier_headers:
        raise ValueError("Elsevier access boundary changed")
    figshare_headers = snapshots["figshare_headers"].read_text(encoding="utf-8")
    figshare_response = snapshots["figshare_response"].read_text(encoding="utf-8")
    if "HTTP/2 403" not in figshare_headers or "403 Forbidden" not in figshare_response:
        raise ValueError("Figshare search access boundary changed")

    history_path = Path(config["evidence"]["motyxia2_public_history"])
    history = json.loads((root / history_path).read_text(encoding="utf-8"))
    if history[
        "Kohn_Portes_record_specific_stimulus_log_found_in_audited_public_history"
    ]:
        raise ValueError("Motyxia2 history boundary changed")

    observations = {
        "Crossref": {
            "preprint_relation": preprints,
            "article_links": crossref_links,
            "numeric_data_links": crossref_data_links,
        },
        "DataCite": {
            "related_DOI_result_count": datacite_related["meta"]["total"],
            "exact_title_result_count": datacite_title["meta"]["total"],
        },
        "Europe_PMC": {
            "exact_DOI_result_count": europe["hitCount"],
            "is_open_access": europe_record.get("isOpenAccess") == "Y",
            "has_supplementary_file": europe_record.get("hasSuppl") == "Y",
            "has_database_cross_references": (
                europe_record.get("hasDbCrossReferences") == "Y"
            ),
        },
        "PMC": {
            "supplement_paths": supplement_paths,
            "supplement_reported_sizes": supplement_sizes,
            "supplement_content_type": "pdf",
            "supplement_content_retrieved": False,
            "download_response_is_proof_of_work_HTML": True,
            "data_availability_points_to_Motyxia2": True,
            "analysis_repository_link_remains_placeholder": True,
        },
        "Elsevier": {
            "minimal_metadata_only_due_to_unauthorized_request": True
        },
        "Figshare": {
            "search_endpoint_status": 403,
            "search_result_interpretable": False,
        },
    }
    positive_payload = bool(
        crossref_data_links
        or datacite_related["meta"]["total"]
        or datacite_title["meta"]["total"]
    )
    gates = {
        "successful_machine_readable_indexes_audited": True,
        "DOI_linked_numeric_stimulus_log_payload_found": positive_payload,
        "PMC_supplement_content_retrieved_and_inspected": False,
        "Figshare_search_endpoint_accessible": False,
        "record_specific_stimulus_log_recovered": False,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{
                    spec["path"]: _sha256(root / spec["path"])
                    for spec in config["snapshots"].values()
                },
                str(history_path): _sha256(root / history_path),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "audited_indexes": observations,
        "availability_gates": gates,
        "record_specific_stimulus_log_found_in_successfully_audited_external_indexes": (
            positive_payload
        ),
        "record_specific_stimulus_log_global_absence_claimed": False,
        "authorize_recording_field_recovery": False,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "successful_indexes_expose_no_linked_numeric_log_while_supplement_and_"
            "Figshare_contents_remain_uninspected"
        ),
        "boundary": config["boundary"],
    }
