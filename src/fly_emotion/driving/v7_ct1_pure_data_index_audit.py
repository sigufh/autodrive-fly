"""Audit unresolved CT1 component archives exposed by the official MPG.PuRe index."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-ct1-pure-data-index-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_ct1_pure_data_index_audit.py")


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
            "component_contents_unresolved_not_absent",
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
        components[name] = {
            **spec,
            "url": matches[0].replace("&amp;", "&"),
            "local_file_exists": local_exists,
        }

    archives = [item for item in components.values() if item["link_class"] == "zip"]
    unresolved = [item for item in archives if not item["local_file_exists"]]
    contents_verified = bool(archives) and not unresolved
    if contents_verified != bool(config["retrieval_observation"]["archive_contents_verified"]):
        raise ValueError("MPG.PuRe CT1 archive verification state changed")

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(snapshot_spec["path"]): _sha256(snapshot_path),
                str(retrieval["path"]): _sha256(retrieval_path),
            },
            "paper": paper,
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "official_index": {
            "snapshot": snapshot_spec,
            "components": components,
            "archive_candidate_count": len(archives),
            "unresolved_archive_count": len(unresolved),
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
        "stop_reason": "official_PuRe_archive_candidates_exist_but_contents_remain_unresolved",
        "boundary": config["boundary"],
    }
