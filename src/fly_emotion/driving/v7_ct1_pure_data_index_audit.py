"""Audit unresolved CT1 component archives exposed by the official MPG.PuRe index."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-ct1-pure-data-index-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_ct1_pure_data_index_audit.py")


def _verified_path(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"frozen CT1 evidence changed: {spec['path']}")
    return path


def evaluate_v7_ct1_pure_data_index_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    snapshot_spec = config["index_snapshot"]
    snapshot_path = root / snapshot_spec["path"]
    if snapshot_path.stat().st_size != int(snapshot_spec["bytes"]):
        raise ValueError("MPG.PuRe CT1 index snapshot size changed")
    if _sha256(snapshot_path) != snapshot_spec["sha256"]:
        raise ValueError("MPG.PuRe CT1 index snapshot SHA-256 changed")
    retrieval = config["retrieval_observation"]
    retrieval_path = root / retrieval["path"]
    if retrieval_path.stat().st_size != int(retrieval["bytes"]):
        raise ValueError("MPG.PuRe CT1 retrieval observation size changed")
    if _sha256(retrieval_path) != retrieval["sha256"]:
        raise ValueError("MPG.PuRe CT1 retrieval observation SHA-256 changed")
    retrieval_text = retrieval_path.read_text(encoding="utf-8")
    if any(
        phrase not in retrieval_text
        for phrase in (
            "file_3247642,file_3247643",
            "curl_exit_28_connection_timeout",
            "two_200_application_zip_captures_retrieved",
            "component_contents_verified_no_new_numeric_payload",
        )
    ):
        raise ValueError("MPG.PuRe CT1 retrieval observation content changed")

    html = snapshot_path.read_text(encoding="utf-8")
    paper = config["paper"]
    if paper["title"] not in html or paper["doi"] not in html:
        raise ValueError("MPG.PuRe CT1 paper identity changed")

    components = {}
    for name, spec in config["components"].items():
        pattern = (
            rf'<a class="{re.escape(spec["link_class"])}"[^>]*'
            rf'href="([^"]*{re.escape(paper["pure_item_id"])}'
            rf"/component/{re.escape(spec['file_id'])}/"
            rf'{re.escape(spec["filename"])}\?mode=download)"[^>]*>.*?'
            rf"{re.escape(spec['link_label'])}"
        )
        matches = re.findall(pattern, html)
        if len(matches) != 1:
            raise ValueError(f"MPG.PuRe CT1 component identity changed: {name}")
        local_path = spec.get("local_path")
        local_exists = bool(local_path and (root / local_path).is_file())
        if local_exists != bool(spec["locally_retrieved"]):
            raise ValueError(f"MPG.PuRe CT1 local retrieval state changed: {name}")
        component = {
            **spec,
            "url": matches[0].replace("&amp;", "&"),
            "local_file_exists": local_exists,
        }
        if local_exists:
            path = root / local_path
            if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
                raise ValueError(f"MPG.PuRe CT1 component changed: {name}")
            component["actual_sha256"] = _sha256(path)
        handle_spec = spec.get("handle_snapshot")
        if handle_spec:
            handle_path = root / handle_spec["path"]
            if (
                handle_path.stat().st_size != int(handle_spec["bytes"])
                or _sha256(handle_path) != handle_spec["sha256"]
            ):
                raise ValueError(f"MPG.PuRe Handle snapshot changed: {name}")
            handle = json.loads(handle_path.read_text())
            urls = [
                value["data"]["value"]
                for value in handle["values"]
                if value["type"] == "URL"
            ]
            if handle["handle"] != spec["handle"] or len(urls) != 1:
                raise ValueError(f"MPG.PuRe Handle identity changed: {name}")
            if spec["file_id"] not in urls[0] or spec["filename"] not in urls[0]:
                raise ValueError(f"MPG.PuRe Handle target changed: {name}")
            component["handle_target_url"] = urls[0]
        components[name] = component

    archives = [item for item in components.values() if item["link_class"] == "zip"]
    unresolved = [item for item in archives if not item["local_file_exists"]]
    contents_verified = bool(archives) and not unresolved
    if contents_verified != bool(config["retrieval_observation"]["archive_contents_verified"]):
        raise ValueError("MPG.PuRe CT1 archive verification state changed")

    expected_members = config["expected_archive_members"]
    expected_by_name = {item["filename"]: item for item in expected_members}
    archive_inspection = {}
    if contents_verified:
        for archive_spec in archives:
            path = root / archive_spec["local_path"]
            with zipfile.ZipFile(path) as zipped:
                bad_member = zipped.testzip()
                names = sorted(name for name in zipped.namelist() if not name.endswith("/"))
                members = []
                for filename in names:
                    payload = zipped.read(filename)
                    expected_member = expected_by_name.get(filename)
                    if expected_member is None:
                        raise ValueError(f"unexpected PuRe archive member: {filename}")
                    if (
                        len(payload) != int(expected_member["bytes"])
                        or _sha256(root / expected_member["local_path"])
                        != expected_member["sha256"]
                        or payload != (root / expected_member["local_path"]).read_bytes()
                    ):
                        raise ValueError(f"PuRe member differs from official PDF: {filename}")
                    members.append(
                        {
                            "filename": filename,
                            "bytes": len(payload),
                            "sha256": expected_member["sha256"],
                            "matches_already_audited_official_PDF": True,
                        }
                    )
            archive_inspection[archive_spec["file_id"]] = {
                "zip_integrity_passed": bad_member is None,
                "member_count": len(members),
                "members": members,
                "contains_only_already_audited_PDFs": names
                == sorted(expected_by_name),
                "new_numeric_or_code_member_count": 0,
            }

    aggregated_spec = config["aggregated_index"]
    aggregated_path = root / aggregated_spec["path"]
    if (
        aggregated_path.stat().st_size != int(aggregated_spec["bytes"])
        or _sha256(aggregated_path) != aggregated_spec["sha256"]
    ):
        raise ValueError("OpenAIRE CT1 snapshot changed")
    aggregated = json.loads(aggregated_path.read_text())
    serialized_aggregated = json.dumps(aggregated)
    if any(handle not in serialized_aggregated for handle in aggregated_spec["expected_handles"]):
        raise ValueError("OpenAIRE CT1 component Handle set changed")

    history = config["author_repository_history"]
    history_paths = {}
    for key in ("commits_snapshot", "initial_tree", "v1_0_tree"):
        history_paths[key] = _verified_path(root, history[key])
    commits = json.loads(history_paths["commits_snapshot"].read_text())
    trees = [
        json.loads(history_paths[key].read_text()) for key in ("initial_tree", "v1_0_tree")
    ]
    all_paths = sorted({item["path"] for tree in trees for item in tree["tree"]})
    if len(commits) != int(history["commit_count"]) or all_paths != sorted(
        history["expected_all_paths"]
    ):
        raise ValueError("CT1 author repository history changed")

    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(snapshot_spec["path"]): _sha256(snapshot_path),
        str(retrieval["path"]): _sha256(retrieval_path),
        str(aggregated_spec["path"]): _sha256(aggregated_path),
        **{str(path.relative_to(root)): _sha256(path) for path in history_paths.values()},
    }
    for spec in config["components"].values():
        for path_key in ("local_path",):
            if spec.get(path_key) and (root / spec[path_key]).is_file():
                dependencies[spec[path_key]] = _sha256(root / spec[path_key])
        if spec.get("handle_snapshot"):
            handle_path = spec["handle_snapshot"]["path"]
            dependencies[handle_path] = _sha256(root / handle_path)
    for member in expected_members:
        dependencies[member["local_path"]] = _sha256(root / member["local_path"])

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "paper": paper,
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "official_index": {
            "snapshot": snapshot_spec,
            "components": components,
            "archive_candidate_count": len(archives),
            "unresolved_archive_count": len(unresolved),
            "aggregated_index": aggregated_spec,
        },
        "archive_inspection": archive_inspection,
        "author_repository_history": {
            "repository": history["repository"],
            "commit_count": len(commits),
            "all_paths_across_history": all_paths,
            "experimental_data_file_count": 0,
            "only_morphology_and_model_code_present": True,
        },
        "retrieval_observation": retrieval,
        "archive_contents_verified": contents_verified,
        "new_CT1_numerical_payload_verified": False,
        "new_CT1_Lo1_time_series_verified": False,
        "new_stable_biological_individual_ids_verified": False,
        "new_complete_stimulus_and_baseline_fields_verified": False,
        "authorize_CT1_source_dynamics_transfer": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "verified_PuRe_archives_only_duplicate_the_two_PDFs_and_add_no_numeric_payload"
        ),
        "boundary": config["boundary"],
    }
