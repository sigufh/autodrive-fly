"""Bound Shomar C3 behavior from the paper's direct LC15 imaging."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-shomar-c3-distance-boundary-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_shomar_c3_distance_boundary_audit.py")


def _verify(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"Shomar snapshot identity changed: {path}")
    return path


def _term_hits(text: str, term: str) -> int:
    return len(re.findall(rf"(?<![A-Za-z0-9]){term}(?![A-Za-z0-9])", text, re.I))


def evaluate_v7_shomar_c3_distance_boundary_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    snapshots = {
        name: _verify(root, spec) for name, spec in config["snapshots"].items()
    }
    paper = config["paper"]
    html = snapshots["full_text"].read_text(encoding="utf-8")
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    phrases = (
        "Silencing C3 and LC15 shifted the gap crossing frequency curve toward larger gaps",
        "We therefore chose to focus our investigations on LC15",
        "recording calcium responses in LC15 neurons with GCaMP6f",
        "male flies expressing GCaMP6f in LC15",
        "Code and data for this study are available on GitHub and Dryad",
    )
    if any(phrase not in text for phrase in phrases):
        raise ValueError("Shomar C3 or LC15 experimental scope changed")
    term_hits = {
        term: _term_hits(text, term) for term in config["expected"]["full_text_term_hits"]
    }
    if term_hits != config["expected"]["full_text_term_hits"]:
        raise ValueError("Shomar full-text term inventory changed")

    crossref = json.loads(snapshots["crossref"].read_text(encoding="utf-8"))[
        "message"
    ]
    if crossref["DOI"] != paper["doi"] or crossref["title"] != [paper["title"]]:
        raise ValueError("Shomar Crossref identity changed")
    github = json.loads(snapshots["github_tree"].read_text(encoding="utf-8"))
    if github["truncated"]:
        raise ValueError("Shomar GitHub tree is truncated")
    entries = github["tree"]
    paths = [item["path"] for item in entries if item["type"] == "blob"]
    expected_tree = config["expected"]["github_tree_filename_hits"]
    tree_hits = {term: sum(_term_hits(path, term) > 0 for path in paths) for term in expected_tree}
    if tree_hits != expected_tree:
        raise ValueError("Shomar GitHub filename scope changed")
    readme = snapshots["github_readme"].read_text(encoding="utf-8")
    if (
        "all analyses done in the manuscript" not in readme
        or "neural imaging experiments" not in readme
        or "Visual circuitry for distance estimation" not in readme
    ):
        raise ValueError("Shomar GitHub README scope changed")

    dryad_dataset = json.loads(
        snapshots["dryad_dataset"].read_text(encoding="utf-8")
    )
    if (
        dryad_dataset["identifier"] != f"doi:{config['dryad']['doi']}"
        or dryad_dataset["versionNumber"] != int(config["dryad"]["version_number"])
    ):
        raise ValueError("Shomar Dryad identity changed")
    dryad_page = json.loads(snapshots["dryad_files"].read_text(encoding="utf-8"))
    dryad_files = dryad_page["_embedded"]["stash:files"]
    if (
        len(dryad_files) != int(config["dryad"]["file_count"])
        or sum(int(item["size"]) for item in dryad_files)
        != int(config["dryad"]["total_declared_bytes"])
        or not all(item["digestType"] == "sha-256" for item in dryad_files)
    ):
        raise ValueError("Shomar Dryad inventory changed")
    dryad_names = sorted(item["path"] for item in dryad_files)
    imaging_archives = [name for name in dryad_names if "Imaging" in name]
    if imaging_archives != [
        "LC15_Imaging_Data.zip",
        "LC15_Imaging_and_Behavior_Data.zip",
    ]:
        raise ValueError("Shomar Dryad imaging archive scope changed")
    headers = snapshots["dryad_readme_download_headers"].read_text(encoding="utf-8")
    body = snapshots["dryad_readme_download_body"].read_text(encoding="utf-8")
    if "HTTP/2 401" not in headers or "Unauthorized" not in body:
        raise ValueError("Shomar Dryad README access boundary changed")

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
        "full_text_term_hits": term_hits,
        "experimental_scope": {
            "C3_manipulation": "shibire_ts_silencing_in_gap_crossing_behavior",
            "direct_neural_imaging_cell_types": ["LC15"],
            "imaging_modality": "two_photon_GCaMP6f_calcium",
            "C3_direct_neural_recording_verified": False,
            "Mi4_direct_neural_recording_verified": False,
        },
        "GitHub": {
            "url": config["repository"]["url"],
            "head_commit": config["repository"]["head_commit"],
            "commit_count": config["repository"]["commit_count"],
            "tree_entry_count": len(entries),
            "tracked_file_count": len(paths),
            "filename_term_hits": tree_hits,
            "C3_imaging_script_found": False,
            "LC15_imaging_script_count": tree_hits["LC15"],
        },
        "Dryad": {
            "doi": config["dryad"]["doi"],
            "version_number": dryad_dataset["versionNumber"],
            "file_count": len(dryad_files),
            "total_declared_bytes": sum(int(item["size"]) for item in dryad_files),
            "filenames": dryad_names,
            "all_files_have_SHA256": all(
                item["digestType"] == "sha-256" and len(item["digest"]) == 64
                for item in dryad_files
            ),
            "imaging_archives": imaging_archives,
            "C3_named_archive_found": any(_term_hits(name, "C3") for name in dryad_names),
            "README_download_status": config["dryad"]["readme_download_status"],
            "bulk_download_performed": False,
        },
        "classification": "C3_behavioral_silencing_with_LC15_calcium_imaging",
        "C3_public_numeric_source_dynamics_verified": False,
        "authorize_C3_source_dynamics_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "C3_is_manipulated_in_behavior_but_only_LC15_is_imaged",
        "boundary": config["boundary"],
    }
