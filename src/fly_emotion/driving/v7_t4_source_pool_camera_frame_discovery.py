"""Discover T4 source-pool readouts in the declared camera coordinate frame."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t4_source_pool_local import (
    evaluate_v7_t4_source_pool_local,
)

CONFIG = Path("configs/driving-v7-t4-source-pool-camera-frame-discovery.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_source_pool_camera_frame_discovery.py"
)


def evaluate_v7_t4_source_pool_camera_frame_discovery(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    legacy_path = Path(audit["legacy_evidence"])
    source_config_path = Path(audit["source_pool_config"])
    source_implementation_path = Path(audit["source_pool_implementation"])
    coordinate_path = Path(audit["coordinate_contract"])
    coordinate_implementation_path = Path(audit["coordinate_implementation"])
    legacy = json.loads((root / legacy_path).read_text(encoding="utf-8"))
    if legacy["authorize_new_target_formula"]:
        raise ValueError("camera-frame discovery requires preserved legacy failure")
    if legacy["protocol"]["source_coordinate_frame"] != "legacy_common_optic_hex":
        raise ValueError("legacy source coordinate frame changed")
    coordinate = json.loads((root / coordinate_path).read_text(encoding="utf-8"))
    grid = coordinate["engineering_angular_grid"]
    if not (
        grid["projection"] == "equiangular_raster"
        and grid["elevation_sign"] == "top_positive_bottom_negative"
    ):
        raise ValueError("camera coordinate contract changed")
    result = evaluate_v7_t4_source_pool_local(
        root, source_coordinate_frame=audit["source_coordinate_frame"]
    )
    if list(result["direction_pass_counts"]) != audit["expected_readout_order"]:
        raise ValueError("source-pool readout order changed")
    discovered = [
        name for name, values in result["bilateral_passing_subtypes"].items() if values
    ]
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(legacy_path): _sha256(root / legacy_path),
        str(source_config_path): _sha256(root / source_config_path),
        str(source_implementation_path): _sha256(root / source_implementation_path),
        str(coordinate_path): _sha256(root / coordinate_path),
        str(coordinate_implementation_path): _sha256(
            root / coordinate_implementation_path
        ),
    }
    return {
        "protocol": {
            "name": audit["name"],
            "observed_on": audit["observed_on"],
            "dependencies_sha256": dependencies,
            "condition_id": result["protocol"]["condition_id"],
            "source_coordinate_frame": result["protocol"][
                "source_coordinate_frame"
            ],
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "coordinate_transform": audit["coordinate_transform"],
        "direction_pass_counts": result["direction_pass_counts"],
        "bilateral_passing_subtypes": result["bilateral_passing_subtypes"],
        "population_scores": result["population_scores"],
        "discovered_bilateral_readouts": discovered,
        "discovered_bilateral_subtypes": {
            name: result["bilateral_passing_subtypes"][name] for name in discovered
        },
        "post_hoc_candidate_discovered": bool(discovered),
        "independent_replication_preregistered": False,
        "independent_replication_evaluated": False,
        "authorize_new_target_formula": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "post_hoc_camera_frame_candidate_requires_preregistered_replication",
        "boundary": audit["boundary"],
    }
