"""Audit Yang et al. T4/T5 source optical-voltage evidence."""

from __future__ import annotations

import importlib.metadata
import json
import re
import zipfile
from pathlib import Path

import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-yang-t5-voltage-evidence-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_yang_t5_voltage_evidence_audit.py")


def evaluate_v7_yang_t5_voltage_evidence_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    supplement_reports = []
    combined_text = []
    for spec in config["supplements"]:
        path = root / spec["path"]
        if path.stat().st_size != int(spec["bytes"]):
            raise ValueError("Yang supplement size mismatch")
        if _sha256(path) != spec["sha256"]:
            raise ValueError("Yang supplement SHA-256 mismatch")
        reader = PdfReader(path)
        if len(reader.pages) != int(spec["pages"]):
            raise ValueError("Yang supplement page count mismatch")
        attachment_count = len(reader.attachments or {})
        if attachment_count != 0:
            raise ValueError("Yang supplement unexpectedly embeds an attachment")
        text = " ".join(" ".join((page.extract_text() or "").split()) for page in reader.pages)
        combined_text.append(text)
        supplement_reports.append(
            {
                **spec,
                "actual_sha256": _sha256(path),
                "actual_pages": len(reader.pages),
                "embedded_attachment_count": attachment_count,
            }
        )
    text = " ".join(combined_text).replace("ﬂ", "fl").replace("ﬁ", "fi")
    required_fragments = (
        "constant frame rate of 38.9 Hz",
        "resampled our data from 38.9 Hz to 120 Hz",
        "Mi1 layer M10 (green; voltage: n = 67 cells, 4 flies",
        "Tm3 layer M1 0 (red; voltage: n = 100 cells, 9 flies",
        "Tm1 layer Lo1",
        "n = 30 cells, 3 flies",
        "Tm2 (red, n = 89 cells, 4 flies)",
        "Tm2 layer M2 (green; voltage: n = 89 cells, 6 flies",
        "25 ms light and dark flashes",
    )
    missing_fragments = [fragment for fragment in required_fragments if fragment not in text]
    if missing_fragments:
        raise ValueError(f"Yang supplement evidence changed: {missing_fragments}")

    index_paths = {name: root / spec["path"] for name, spec in config["external_indexes"].items()}
    for name, path in index_paths.items():
        spec = config["external_indexes"][name]
        if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
            raise ValueError(f"Yang external-index snapshot changed: {name}")
    crossref = json.loads(index_paths["crossref"].read_text())["message"]
    if crossref["DOI"] != config["paper"]["doi"] or crossref.get("relation"):
        raise ValueError("Yang Crossref identity or relations changed")
    crossref_data_links = [
        item["URL"]
        for item in crossref.get("link", [])
        if item["URL"]
        .lower()
        .split("?", 1)[0]
        .endswith((".csv", ".json", ".mat", ".npy", ".npz", ".zip"))
    ]
    datacite_counts = {
        name: int(json.loads(index_paths[name].read_text())["meta"]["total"])
        for name in ("datacite_related", "datacite_title")
    }
    if any(datacite_counts.values()):
        raise ValueError("DataCite gained an unreviewed Yang candidate")
    europe = json.loads(index_paths["europe_pmc"].read_text())
    if europe["hitCount"] != 1:
        raise ValueError("Europe PMC Yang DOI count changed")
    europe_record = europe["resultList"]["result"][0]
    if europe_record.get("pmcid") != "PMC5606228":
        raise ValueError("Europe PMC Yang identity changed")
    pmc_html = index_paths["pmc_article"].read_text(encoding="utf-8")
    attachment_names = re.findall(
        r'href="/articles/instance/5606228/bin/(NIHMS785612-supplement-[^"]+)"',
        pmc_html,
    )
    attachment_names = list(dict.fromkeys(attachment_names))
    expected_attachment_names = [item["name"] for item in config["pmc_attachments"]]
    if attachment_names != expected_attachment_names:
        raise ValueError("PMC Yang attachment listing changed")

    attachment_reports = []
    attachment_root = root / "data/raw/t5-yang-voltage/external-index"
    for spec in config["pmc_attachments"]:
        suffix = spec["name"].rsplit(".", 1)[1]
        local_name = spec["name"].replace(f".{suffix}", f"-real.{suffix}")
        path = attachment_root / local_name
        if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
            raise ValueError(f"PMC Yang attachment changed: {spec['name']}")
        if suffix == "pdf":
            reader = PdfReader(path)
            if len(reader.pages) != int(spec["pages"]) or reader.attachments:
                raise ValueError(f"PMC Yang PDF structure changed: {spec['name']}")
            attachment_reports.append(
                {**spec, "local_path": str(path.relative_to(root)), "embedded_files": []}
            )
        else:
            with zipfile.ZipFile(path) as archive:
                members = sorted(archive.namelist())
                document_text = archive.read("word/document.xml").decode("utf-8")
            embedded = [name for name in members if name.startswith("word/embeddings/")]
            if embedded or "SUPPLEMENTAL FIGURE LEGENDS" not in document_text:
                raise ValueError("PMC Yang DOCX attachment structure changed")
            attachment_reports.append(
                {
                    **spec,
                    "local_path": str(path.relative_to(root)),
                    "member_count": len(members),
                    "embedded_files": embedded,
                }
            )
    github = json.loads(index_paths["github_exact_title"].read_text())
    zenodo = json.loads(index_paths["zenodo_exact_title"].read_text())
    if github["total_count"] != 0 or zenodo["hits"]["total"] != 0:
        raise ValueError("GitHub or Zenodo gained an unreviewed Yang candidate")
    figshare_headers = index_paths["figshare_headers"].read_text()
    figshare_response = index_paths["figshare_response"].read_text()
    if "403" not in figshare_headers or "403 Forbidden" not in figshare_response:
        raise ValueError("Figshare Yang access boundary changed")
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    required = contract["required_families"]["T5"]["source_types"]
    source_evidence = {
        source: config["evidence"][source]
        for source in ("Mi1", "Tm3", "Tm1", "Tm2")
    }
    covered = [source for source in required if source in source_evidence]
    missing = [source for source in required if source not in covered]
    payload = config["payload"]
    gates = {
        "Tm1_lobula_Lo1_optical_voltage_phenotype_published": bool(
            config["evidence"]["Tm1"]["phenotype_reported"]
        ),
        "Tm2_medulla_M2_optical_voltage_phenotype_published": bool(
            config["evidence"]["Tm2"]["phenotype_reported"]
        ),
        "Tm2_fly_count_report_consistent": bool(
            config["evidence"]["Tm2"]["fly_count_report_consistent"]
        ),
        "every_T5_source_type_covered": not missing,
        "numerical_per_fly_time_series_available": bool(
            payload["numerical_per_fly_time_series_present"]
        ),
        "stable_recording_and_individual_IDs_available": bool(
            payload["stable_recording_unit_ids_present"]
            and payload["stable_biological_individual_ids_present"]
        ),
        "baseline_window_available": bool(payload["baseline_window_seconds_present"]),
        "MaleCNS_mapping_available": bool(payload["source_type_to_MaleCNS_mapping_present"]),
        "required_split_roles_available": bool(payload["required_split_roles_present"]),
    }
    transferable = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
                **{
                    str(config["external_indexes"][name]["path"]): _sha256(path)
                    for name, path in index_paths.items()
                },
                **{item["local_path"]: item["sha256"] for item in attachment_reports},
            },
            "paper": config["paper"],
            "supplements": supplement_reports,
            "pypdf_version": importlib.metadata.version("pypdf"),
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "measurement_protocol": {
            key: value for key, value in config["evidence"].items() if key not in ("Tm1", "Tm2")
        },
        "external_index_audit": {
            "crossref_relation_count": sum(
                len(items) for items in crossref.get("relation", {}).values()
            ),
            "crossref_data_links": crossref_data_links,
            "datacite_related_count": datacite_counts["datacite_related"],
            "datacite_exact_title_count": datacite_counts["datacite_title"],
            "europe_pmc_has_supplements": europe_record.get("hasSuppl") == "Y",
            "europe_pmc_has_database_cross_references": (
                europe_record.get("hasDbCrossReferences") == "Y"
            ),
            "pmc_attachment_count": len(attachment_reports),
            "pmc_attachments": attachment_reports,
            "pmc_numeric_attachment_count": sum(
                item["name"].lower().endswith((".csv", ".json", ".mat", ".npy", ".npz", ".zip"))
                for item in attachment_reports
            ),
            "github_exact_title_repository_count": github["total_count"],
            "zenodo_exact_title_record_count": zenodo["hits"]["total"],
            "figshare_search_accessible": False,
        },
        "source_evidence": {
            **source_evidence,
        },
        "T4_source_contract": {
            "required_sources": contract["required_families"]["T4"][
                "source_types"
            ],
            "sources_with_optical_voltage_phenotype": ["Mi1", "Tm3"],
            "sources_without_optical_voltage_phenotype": ["Mi4", "C3"],
            "phenotype_coverage_fraction": 0.5,
        },
        "T5_source_contract": {
            "required_sources": required,
            "sources_with_optical_voltage_phenotype": covered,
            "sources_without_optical_voltage_phenotype": missing,
            "phenotype_coverage_fraction": len(covered) / len(required),
        },
        "payload": payload,
        "transfer_gates": gates,
        "T5_partial_optical_voltage_phenotype_verified": True,
        "T5_numerical_voltage_payload_verified": bool(
            gates["numerical_per_fly_time_series_available"]
        ),
        "T5_source_dynamics_transfer_authorized": transferable,
        "authorize_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else "Tm1_Tm2_voltage_phenotypes_lack_numerical_individual_payload_and_full_T5_coverage"
        ),
        "boundary": config["boundary"],
    }
