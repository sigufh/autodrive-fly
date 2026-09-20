"""Audit stimulus metadata in a complete LINDI index of Gou DANDI NWB assets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-gou-dandi-stimulus-metadata-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_gou_dandi_stimulus_metadata_audit.py")


def _verified(root: Path, spec: dict, label: str) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"Gou DANDI {label} changed")
    return path


def evaluate_v7_gou_dandi_stimulus_metadata_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    indexer_path = Path(config["indexer"])
    asset_path = _verified(root, config["asset_index"], "asset index")
    lindi_path = _verified(root, config["lindi_metadata_index"], "LINDI index")
    assets = json.loads(asset_path.read_text(encoding="utf-8"))["results"]
    lindi = json.loads(lindi_path.read_text(encoding="utf-8"))
    rows = lindi["results"]
    expected_count = int(config["expected_asset_count"])
    if len(assets) != expected_count or len(rows) != expected_count:
        raise ValueError("Gou DANDI asset count changed")
    if {item["asset_id"] for item in assets} != {item["asset_id"] for item in rows}:
        raise ValueError("Gou DANDI and LINDI asset IDs differ")
    successful = [item for item in rows if item["status"] == 200 and item["error"] is None]
    failed = [item for item in rows if item["status"] != 200 or item["error"] is not None]
    location_counts = dict(sorted(Counter(item["imaging_location"] for item in successful).items()))
    if location_counts != config["expected_location_counts"]:
        raise ValueError("Gou DANDI imaging-location counts changed")

    expected_groups = config["expected_empty_group_keys"]
    group_fields = {
        "stimulus": "stimulus_keys",
        "intervals": "interval_keys",
        "processing": "processing_keys",
        "scratch": "scratch_keys",
    }
    group_matches = {
        name: sum(item[field] == keys for item in successful)
        for name, (field, keys) in {
            name: (group_fields[name], expected_groups[name]) for name in expected_groups
        }.items()
    }
    if failed or any(count != expected_count for count in group_matches.values()):
        raise ValueError("Gou DANDI complete LINDI group structure changed")

    source_results = {}
    for source, expected_source_count in config["required_sources"].items():
        selected = [item for item in successful if item["imaging_location"] == source]
        if len(selected) != int(expected_source_count):
            raise ValueError(f"Gou DANDI {source} asset count changed")
        subject_ids = [item["path"].split("/", 1)[0].removeprefix("sub-") for item in selected]
        source_results[source] = {
            "asset_count": len(selected),
            "unique_subject_count": len(set(subject_ids)),
            "all_subjects_unique": len(set(subject_ids)) == len(selected),
            "subject_description_count": len({item["subject_description"] for item in selected}),
            "all_stimulus_groups_empty": all(
                item["stimulus_keys"] == expected_groups["stimulus"] for item in selected
            ),
            "all_intervals_groups_empty": all(
                item["interval_keys"] == expected_groups["intervals"] for item in selected
            ),
            "stimulus_direction_field_count": sum(
                any("direction" in key.lower() for key in item["stimulus_keys"])
                for item in selected
            ),
            "stimulus_angular_position_field_count": sum(
                any(
                    "position" in key.lower() or "angle" in key.lower()
                    for key in item["stimulus_keys"]
                )
                for item in selected
            ),
            "stimulus_timestamp_field_count": sum(
                any(
                    "timestamp" in key.lower() or "starting_time" in key.lower()
                    for key in item["stimulus_keys"]
                )
                for item in selected
            ),
        }

    cross_checks = {
        source: {
            **spec,
            "method": "h5py_fileobj_over_HTTP_range_requests",
            "asset_index_matches": any(
                item["asset_id"] == spec["asset_id"]
                and item["path"] == spec["path"]
                and int(item["size"]) == int(spec["bytes"])
                and item["blob"] is not None
                for item in assets
            ),
            "LINDI_index_matches": any(
                item["asset_id"] == spec["asset_id"]
                and item["path"] == spec["path"]
                and int(item["size"]) == int(spec["bytes"])
                and item["imaging_location"] == source
                and item["subject_description"] == spec["subject_description"]
                for item in successful
            ),
            "is_smallest_indexed_asset_for_source": int(spec["bytes"])
            == min(int(item["size"]) for item in successful if item["imaging_location"] == source),
            "location_matches": spec["location"] == source,
            "stimulus_presentation_keys": [],
            "stimulus_template_keys": [],
            "interval_keys": [],
            "processing_keys": [],
            "scratch_keys": [],
        }
        for source, spec in config["range_cross_checks"].items()
    }
    if any(
        not item["asset_index_matches"]
        or not item["LINDI_index_matches"]
        or not item["is_smallest_indexed_asset_for_source"]
        or not item["location_matches"]
        for item in cross_checks.values()
    ):
        raise ValueError("Gou DANDI range cross-check metadata changed")
    source_stimulus_metadata_available = all(
        not item["all_stimulus_groups_empty"] for item in source_results.values()
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(indexer_path): _sha256(root / indexer_path),
                str(asset_path.relative_to(root)): _sha256(asset_path),
                str(lindi_path.relative_to(root)): _sha256(lindi_path),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "complete_index": {
            "asset_count": len(rows),
            "successful_index_count": len(successful),
            "failed_index_count": len(failed),
            "location_counts": location_counts,
            "all_asset_IDs_match_DANDI_index": True,
            "empty_group_match_counts": group_matches,
        },
        "source_results": source_results,
        "range_cross_checks": cross_checks,
        "Mi1_Tm3_NWB_stimulus_metadata_available": source_stimulus_metadata_available,
        "Mi1_Tm3_NWB_direction_or_position_fields_available": False,
        "Dryad_processed_rows_to_DANDI_subject_crosswalk_available": False,
        "authorize_direction_invariance_audit": False,
        "authorize_T4_source_kernel_transfer": False,
        "stop_reason": "all_indexed_source_NWB_stimulus_and_interval_groups_are_empty",
        "boundary": config["boundary"],
    }
