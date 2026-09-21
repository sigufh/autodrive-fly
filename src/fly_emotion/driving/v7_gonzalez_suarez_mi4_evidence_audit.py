"""Audit independent Mi4 dynamics evidence from Gonzalez-Suarez et al. 2022."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import h5py
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-gonzalez-suarez-mi4-evidence-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_gonzalez_suarez_mi4_evidence_audit.py"
)


def _verified_path(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"snapshot size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"snapshot SHA-256 mismatch: {path}")
    return path


def evaluate_v7_gonzalez_suarez_mi4_evidence_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    snapshot_paths = {
        name: _verified_path(root, spec)
        for name, spec in config["snapshots"].items()
    }
    paper = config["paper"]
    html = snapshot_paths["pmc_article"].read_text(encoding="utf-8")
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    required_phrases = (
        "Mi4 (Mi4 &gt; GC6f, n = 15 flies)",
        "Using in vivo two-photon microscopy",
        "expressing the calcium indicator GCaMP6f",
        "using Arclight",
        "slo -RNAi in Tm3 and NaChBac in Mi1",
        "All original modeling code has been deposited at GitHub",
        "https://github.com/ClarkLabCode/TimingModels",
    )
    if any(phrase not in text for phrase in required_phrases):
        raise ValueError("Gonzalez-Suarez paper evidence text changed")

    crossref = json.loads(snapshot_paths["crossref_work"].read_text())["message"]
    if crossref["DOI"] != paper["doi"]:
        raise ValueError("Gonzalez-Suarez Crossref identity changed")
    preprints = [
        item["id"] for item in crossref.get("relation", {}).get("has-preprint", [])
    ]
    if preprints != [paper["preprint_doi"]]:
        raise ValueError("Gonzalez-Suarez preprint relation changed")
    crossref_links = [item["URL"] for item in crossref.get("link", [])]
    data_suffixes = (".csv", ".mat", ".npy", ".npz", ".zip", ".h5")
    crossref_data_links = [
        url for url in crossref_links if url.lower().split("?", 1)[0].endswith(data_suffixes)
    ]

    europe = json.loads(snapshot_paths["europe_pmc"].read_text())
    if europe["hitCount"] != 1:
        raise ValueError("Europe PMC exact PMID result count changed")
    europe_record = europe["resultList"]["result"][0]
    if (
        europe_record["pmid"] != str(paper["pmid"])
        or europe_record["pmcid"] != paper["pmcid"]
        or europe_record["doi"] != paper["doi"]
    ):
        raise ValueError("Europe PMC paper identity changed")
    datacite = json.loads(snapshot_paths["datacite_related"].read_text())
    zenodo = json.loads(snapshot_paths["zenodo_doi_search"].read_text())
    if datacite["meta"]["total"] != 0 or zenodo["hits"]["total"] != 0:
        raise ValueError("public index gained an unreviewed DOI-linked candidate")

    supplement_response = snapshot_paths["pmc_supplement_response"].read_text()
    supplement_inaccessible = (
        "Preparing to download" in supplement_response
        and "POW_CHALLENGE" in supplement_response
    )
    if not supplement_inaccessible:
        raise ValueError("PMC supplement access boundary changed")

    repository = root / config["repository"]["path"]
    local_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    remote_head = subprocess.run(
        ["git", "rev-parse", "refs/remotes/origin/main"], cwd=repository,
        check=True, capture_output=True, text=True
    ).stdout.strip()
    if local_head != config["repository"]["current_remote_head"] or remote_head != local_head:
        raise ValueError("TimingModels repository revision changed")
    commit_count = int(subprocess.run(
        ["git", "rev-list", "--all", "--count"], cwd=repository, check=True,
        capture_output=True, text=True
    ).stdout.strip())
    tree_files = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=repository,
        check=True, capture_output=True, text=True
    ).stdout.splitlines()
    if commit_count != int(config["repository"]["commit_count"]):
        raise ValueError("TimingModels commit count changed")
    if len(tree_files) != int(config["repository"]["tree_file_count"]):
        raise ValueError("TimingModels file count changed")
    numerical_files = sorted(
        path for path in tree_files
        if Path(path).suffix.lower() in {".csv", ".mat", ".npy", ".npz", ".h5", ".hdf5", ".xlsx"}
    )
    if numerical_files != sorted(config["repository"]["numerical_files"]):
        raise ValueError("TimingModels numerical file inventory changed")

    timing_path = Path(config["timing_models_evidence"])
    timing = json.loads((root / timing_path).read_text())
    if timing["protocol"]["repository_commit"] != local_head:
        raise ValueError("TimingModels evidence revision mismatch")
    filter_specs = [
        item
        for item in timing["protocol"]["verified_files"]
        if "filterData_" in item["path"]
    ]
    dataset_fields = {}
    for spec in filter_specs:
        with h5py.File(root / spec["path"], "r") as source:
            dataset_fields[Path(spec["path"]).name] = sorted(
                name for name in source if name != "#refs#"
            )
    expected_fields = [
        "dtFilter",
        "filterLabel",
        "filterList",
        "filterMat",
        "filterSem",
        "tSec",
    ]
    if any(fields != expected_fields for fields in dataset_fields.values()):
        raise ValueError("TimingModels public filter MAT structure changed")

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(timing_path): _sha256(root / timing_path),
            },
            "raw_snapshot_identity": {
                str(path.relative_to(root)): _sha256(path)
                for path in snapshot_paths.values()
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "paper": paper,
        "paper_measurement_evidence": {
            "Mi4_GCaMP6f_fly_count": 15,
            "Mi4_measurement_modality": "two_photon_GCaMP6f_calcium_imaging",
            "ArcLight_voltage_source_types": ["Mi1", "Tm3"],
            "Mi4_experimental_membrane_voltage_measured": False,
            "C3_measured": False,
        },
        "public_indexes": {
            "Crossref_preprint_relation": preprints,
            "Crossref_numeric_data_links": crossref_data_links,
            "Europe_PMC_has_supplement": europe_record.get("hasSuppl") == "Y",
            "Europe_PMC_has_database_cross_references": (
                europe_record.get("hasDbCrossReferences") == "Y"
            ),
            "DataCite_related_result_count": datacite["meta"]["total"],
            "Zenodo_exact_DOI_result_count": zenodo["hits"]["total"],
            "PMC_supplement_content_retrieved": False,
            "PMC_supplement_response_is_proof_of_work_HTML": supplement_inaccessible,
        },
        "repository_evidence": {
            "url": config["repository"]["url"],
            "local_and_remote_head": local_head,
            "commit_count": commit_count,
            "tree_file_count": len(tree_files),
            "numerical_files": numerical_files,
            "filter_dataset_fields": dataset_fields,
            "filter_source_types": timing["filter_data"]["filter_list"],
            "filter_sample_interval_seconds": timing["filter_data"]["sample_interval_seconds"],
            "Mi4_type_average_filter_available": "Mi4" in timing["filter_data"]["filter_list"],
            "C3_filter_available": "C3" in timing["filter_data"]["filter_list"],
            "individual_cell_axis_available": False,
            "stable_biological_individual_ID_available": False,
        },
        "independent_Mi4_calcium_type_average_available": True,
        "independent_Mi4_experimental_membrane_voltage_available": False,
        "independent_Mi4_individual_numeric_dynamics_available": False,
        "independent_C3_source_dynamics_available": False,
        "authorize_Mi4_C3_voltage_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "authorize_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "independent_Mi4_evidence_is_type_average_GCaMP6f_and_C3_is_absent",
        "boundary": config["boundary"],
    }
