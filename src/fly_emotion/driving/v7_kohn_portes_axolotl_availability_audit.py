"""Bound public-source availability for Kohn--Portes ``axolotl.tmodel``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-kohn-portes-axolotl-availability-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_kohn_portes_axolotl_availability_audit.py")
TEXT_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
}


def _verified_path(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"snapshot size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"snapshot SHA-256 mismatch: {path}")
    return path


def _verified_status(root: Path, spec: dict) -> int:
    path = root / spec["headers_path"]
    if _sha256(path) != spec["headers_sha256"]:
        raise ValueError(f"response-header SHA-256 mismatch: {path}")
    status = int(path.read_text().splitlines()[0].split()[1])
    if status != int(spec["status"]):
        raise ValueError(f"response status mismatch: {path}")
    return status


def _directory_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(bytes([0]))
        digest.update(path.read_bytes())
        digest.update(bytes([0]))
    return digest.hexdigest()


def evaluate_v7_kohn_portes_axolotl_availability_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    identity_path = Path(config["figure6_identity_evidence"])
    identity = json.loads((root / identity_path).read_text())
    if not identity["gates"]["Figure6_axolotl_import_verified"]:
        raise ValueError("Figure 6 axolotl import is not verified")

    snapshot_paths = {
        name: _verified_path(root, spec) for name, spec in config["snapshots"].items()
    }
    snapshots = {
        name: json.loads(path.read_text())
        for name, path in snapshot_paths.items()
        if name != "project_page"
    }
    project = snapshots["project"]
    if project["id"] != int(config["gitlab_project_id"]):
        raise ValueError("axolotl GitLab project ID changed")
    if project["path_with_namespace"] != config["gitlab_project_path"]:
        raise ValueError("axolotl GitLab project path changed")
    expected = config["expected"]
    if project["description"] != expected["project_description"]:
        raise ValueError("axolotl GitLab project description changed")
    if project["visibility"] != expected["project_visibility"]:
        raise ValueError("axolotl GitLab project visibility changed")
    if project["created_at"] != expected["project_created_at"]:
        raise ValueError("axolotl GitLab project creation timestamp changed")
    if project["last_activity_at"] != expected["project_last_activity_at"]:
        raise ValueError("axolotl GitLab project activity timestamp changed")
    group_projects = snapshots["group_projects"]
    group_match = [
        item
        for item in group_projects
        if item["id"] == int(config["gitlab_project_id"])
        and item["path_with_namespace"] == config["gitlab_project_path"]
    ]
    if len(group_match) != 1:
        raise ValueError("axolotl project is not uniquely identified in group index")

    graphql_project = snapshots["graphql_project"]["data"]["project"]
    project_page = snapshot_paths["project_page"].read_text()
    page_marks_nonempty = 'data-is-project-empty="false"' in project_page
    statuses = {
        name: _verified_status(root, spec)
        for name, spec in config["snapshots"].items()
        if "status" in spec
    }
    response_bodies = {name: snapshots[name] for name in statuses}
    if response_bodies["forks"] != []:
        raise ValueError("axolotl public fork inventory changed")
    for name in ("branches", "commits"):
        if response_bodies[name] != {"message": "404 Repository Not Found"}:
            raise ValueError(f"axolotl {name} access response changed")
    for name in ("tree", "tags", "releases", "packages"):
        if response_bodies[name] != {"message": "403 Forbidden"}:
            raise ValueError(f"axolotl {name} access response changed")

    history = config["flexible_filtering_history"]
    history_paths = {
        name: _verified_path(root, spec) for name, spec in history.items() if isinstance(spec, dict)
    }
    commits = json.loads(history_paths["commits"].read_text())
    tree_dir = root / history["commit_tree_directory"]
    commit_trees = sorted(tree_dir.glob("*.json"))
    if len(commits) != int(history["expected_commit_count"]):
        raise ValueError("flexible-filtering commit count changed")
    if len(commit_trees) != len(commits):
        raise ValueError("flexible-filtering commit tree coverage is incomplete")
    if _directory_digest(commit_trees) != history["expected_commit_tree_digest"]:
        raise ValueError("flexible-filtering commit tree digest changed")

    blobs: dict[str, set[str]] = {}
    for path in commit_trees:
        for entry in json.loads(path.read_text()):
            if entry["type"] == "blob":
                blobs.setdefault(entry["id"], set()).add(entry["path"])
    text_blobs = {
        blob: paths
        for blob, paths in blobs.items()
        if Path(next(iter(paths))).suffix.lower() in TEXT_EXTENSIONS
    }
    blob_dir = root / "data/raw/kohn-portes-axolotl-search/blobs"
    blob_paths = [blob_dir / blob for blob in text_blobs]
    if len(blobs) != int(history["expected_unique_blob_count"]):
        raise ValueError("flexible-filtering unique blob inventory changed")
    if len(blob_paths) != int(history["expected_unique_text_blob_count"]):
        raise ValueError("flexible-filtering text blob inventory changed")
    if _directory_digest(blob_paths) != history["expected_text_blob_digest"]:
        raise ValueError("flexible-filtering text blob digest changed")
    all_text = "\n".join(path.read_text(errors="ignore") for path in blob_paths)
    definition_hits = {token: token in all_text for token in expected["source_definition_tokens"]}

    pypi = snapshots["pypi_axolotl"]
    upload_years = sorted(
        int(item["upload_time_iso_8601"][:4])
        for files in pypi["releases"].values()
        for item in files
    )
    conda = snapshots["anaconda_axolotl"]
    conda_exact = [item for item in conda if item["name"] == "axolotl"]
    package_indexes_are_unrelated = bool(
        pypi["info"]["summary"] == expected["pypi_summary"]
        and min(upload_years) == int(expected["pypi_first_upload_year"])
        and conda_exact
        and all("AI models" in item["summary"] for item in conda_exact)
    )
    wayback_hits = len(snapshots["wayback_project"]) + len(snapshots["wayback_archive"])

    gates = {
        "related_GitLab_project_metadata_found": True,
        "related_GitLab_repository_anonymously_readable": (
            graphql_project["repository"] is not None
        ),
        "related_GitLab_public_fork_found": bool(response_bodies["forks"]),
        "related_GitLab_release_or_package_endpoint_accessible": (
            statuses["releases"] == 200 or statuses["packages"] == 200
        ),
        "Wayback_repository_snapshot_found": wayback_hits > 0,
        "public_package_index_candidate_matches_research_code": not package_indexes_are_unrelated,
        "flexible_filtering_all_commit_trees_scanned": True,
        "flexible_filtering_contains_axolotl_definitions": any(definition_hits.values()),
        "Figure6_axolotl_source_recovered": False,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(identity_path): _sha256(root / identity_path),
                **{
                    spec["path"]: _sha256(path)
                    for spec, path in zip(
                        config["snapshots"].values(),
                        snapshot_paths.values(),
                        strict=True,
                    )
                },
                **{
                    spec["headers_path"]: _sha256(root / spec["headers_path"])
                    for spec in config["snapshots"].values()
                    if "headers_path" in spec
                },
                **{
                    spec["path"]: _sha256(path)
                    for spec, path in zip(
                        (value for value in history.values() if isinstance(value, dict)),
                        history_paths.values(),
                        strict=True,
                    )
                },
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "related_project": {
            "id": project["id"],
            "path": project["path_with_namespace"],
            "description": project["description"],
            "visibility": project["visibility"],
            "created_at": project["created_at"],
            "last_activity_at": project["last_activity_at"],
            "graphql_repository": graphql_project["repository"],
            "page_marks_project_nonempty": page_marks_nonempty,
            "endpoint_statuses": statuses,
            "public_fork_count": len(response_bodies["forks"]),
        },
        "flexible_filtering_history": {
            "branch_tree_entry_counts": {
                name: len(json.loads(path.read_text()))
                for name, path in history_paths.items()
                if name.endswith("tree")
            },
            "commit_count": len(commits),
            "commit_tree_count": len(commit_trees),
            "unique_blob_count": len(blobs),
            "unique_text_blob_count": len(text_blobs),
            "Figure6_import_occurrence_count": all_text.count(expected["exact_import"]),
            "source_definition_hits": definition_hits,
        },
        "package_indexes": {
            "PyPI_name": pypi["info"]["name"],
            "PyPI_summary": pypi["info"]["summary"],
            "PyPI_first_upload_year": min(upload_years),
            "Conda_exact_name_packages": [item["full_name"] for item in conda_exact],
            "same_name_packages_are_unrelated_LLM_software": package_indexes_are_unrelated,
        },
        "wayback_snapshot_count": wayback_hits,
        "gates": gates,
        "source_global_absence_claimed": False,
        "authorize_Tm_to_T5_model_transfer_to_v7": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "related_project_found_but_repository_content_not_anonymously_accessible",
        "boundary": config["boundary"],
    }
