"""Audit a bounded C3 citation graph and a high-relevance exclusion candidate."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-c3-citation-graph-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_c3_citation_graph_audit.py")


def _verify(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"citation-graph snapshot identity changed: {path}")
    return path


def _json(root: Path, spec: dict) -> dict:
    return json.loads(_verify(root, spec).read_text(encoding="utf-8"))


def evaluate_v7_c3_citation_graph_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    roots = {
        name: _json(root, spec["snapshot"])
        for name, spec in config["roots"].items()
    }
    for name, item in roots.items():
        spec = config["roots"][name]
        if item["id"].rsplit("/", 1)[-1] != spec["openalex_id"]:
            raise ValueError(f"OpenAlex root identity changed: {name}")
        if item["doi"].lower() != f"https://doi.org/{spec['doi']}".lower():
            raise ValueError(f"OpenAlex root DOI changed: {name}")

    graph = {
        name: _json(root, spec) for name, spec in config["graph_snapshots"].items()
    }
    reference_ids = {
        item for root_item in roots.values() for item in root_item["referenced_works"]
    }
    resolved_references = [
        item
        for name, payload in graph.items()
        if name.startswith("references_page_")
        for item in payload["results"]
    ]
    citing = [
        item
        for name, payload in graph.items()
        if name.endswith("_citing")
        for item in payload["results"]
    ]
    resolved_ids = {item["id"] for item in resolved_references}
    citing_ids = {item["id"] for item in citing}
    root_ids = {item["id"] for item in roots.values()}

    candidate = config["high_relevance_candidate"]
    html_path = _verify(root, candidate["pmc_source"])
    html = html_path.read_text(encoding="utf-8")
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    required_phrases = (
        "two-photon voltage imaging",
        "temporal response properties of L1 and L2",
        "voltage indicator ASAP2f",
        "All data reported in this paper will be shared by the lead contact upon request",
        "https://github.com/ClandininLab/L1L2-deblur",
    )
    if any(phrase not in text for phrase in required_phrases):
        raise ValueError("Pang measurement or availability evidence changed")
    source_term_hits = {
        term: len(re.findall(rf"(?<![A-Za-z0-9]){term}(?![A-Za-z0-9])", text, re.I))
        for term in ("L1", "L2", "C2", "C3", "Lawf1", "Lawf2")
    }
    if source_term_hits["C3"] or source_term_hits["C2"]:
        raise ValueError("Pang full text gained a C2/C3 term hit")

    crossref = _json(root, candidate["crossref_final"])["message"]
    if crossref["DOI"] != candidate["final_doi"]:
        raise ValueError("Pang final DOI identity changed")
    dryad_specs = candidate["dryad_snapshots"]
    dryad_dataset = _json(root, dryad_specs["dataset"])
    if dryad_dataset["identifier"] != f"doi:{candidate['dryad_doi']}":
        raise ValueError("Pang Dryad identity changed")
    dryad_files = []
    for name, spec in dryad_specs.items():
        if name == "dataset":
            continue
        payload = _json(root, spec)
        dryad_files.extend(payload["_embedded"]["stash:files"])
    names = [item["path"] for item in dryad_files]
    if len(names) != 75 or len(set(names)) != 75:
        raise ValueError("Pang Dryad file inventory changed")
    filename_hits = {
        term: sorted(
            name
            for name in names
            if re.search(rf"(?<![A-Za-z0-9]){term}(?![A-Za-z0-9])", name, re.I)
        )
        for term in ("L1", "L2", "C2", "C3", "Lawf")
    }
    if filename_hits["C2"] or filename_hits["C3"] or filename_hits["Lawf"]:
        raise ValueError("Pang Dryad gained an unreviewed feedback-source filename")

    repository = root / candidate["repository"]["path"]
    local_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    remote_head = subprocess.run(
        ["git", "rev-parse", "refs/remotes/origin/main"], cwd=repository,
        check=True, capture_output=True, text=True
    ).stdout.strip()
    file_count = len(subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=repository,
        check=True, capture_output=True, text=True
    ).stdout.splitlines())
    commit_count = int(subprocess.run(
        ["git", "rev-list", "--all", "--count"], cwd=repository, check=True,
        capture_output=True, text=True
    ).stdout.strip())
    repo_spec = candidate["repository"]
    if (
        local_head != repo_spec["current_remote_head"]
        or remote_head != local_head
        or file_count != int(repo_spec["tree_file_count"])
        or commit_count != int(repo_spec["commit_count"])
    ):
        raise ValueError("Pang code repository identity changed")

    snapshot_specs = [
        *(item["snapshot"] for item in config["roots"].values()),
        *config["graph_snapshots"].values(),
        candidate["pmc_source"],
        candidate["crossref_final"],
        *candidate["dryad_snapshots"].values(),
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
                for spec in snapshot_specs
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "citation_graph": {
            "root_count": len(roots),
            "root_reference_ID_count": len(reference_ids),
            "resolved_reference_count": len(resolved_ids),
            "unresolved_reference_ID_count": len(reference_ids - resolved_ids),
            "citing_record_count": len(citing),
            "unique_citing_work_count": len(citing_ids),
            "union_unique_work_count": len(root_ids | resolved_ids | citing_ids),
        },
        "high_relevance_candidate": {
            "name": candidate["name"],
            "preprint_doi": candidate["preprint_doi"],
            "final_doi": candidate["final_doi"],
            "measurement_modality": "two_photon_ASAP2f_voltage_imaging",
            "directly_recorded_neuron_types": ["L1", "L2"],
            "full_text_source_term_hits": source_term_hits,
            "C3_direct_recording_found": False,
            "C3_intervention_found": False,
            "data_availability": "upon_request",
        },
        "Dryad": {
            "doi": candidate["dryad_doi"],
            "version_number": dryad_dataset["versionNumber"],
            "file_count": len(names),
            "total_declared_bytes": sum(int(item["size"]) for item in dryad_files),
            "all_files_have_SHA256": all(len(item["digest"]) == 64 for item in dryad_files),
            "filename_hits": filename_hits,
            "bulk_download_performed": False,
            "bulk_download_authorized_for_C3_audit": False,
        },
        "repository": {
            "url": repo_spec["url"],
            "local_and_remote_head": local_head,
            "commit_count": commit_count,
            "tree_file_count": file_count,
        },
        "new_independent_C3_direct_recording_candidate_found": False,
        "authorize_C3_source_dynamics_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "citation_graph_candidate_is_L1_L2_voltage_not_C3",
        "boundary": config["boundary"],
    }
