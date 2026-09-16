from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-navigation-nested.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_navigation_nested.py")


def _environment_manifest(seed: int) -> dict:
    environment = DrivingEnvironment()
    environment.reset(seed)
    obstacles = [
        {"x": float(item.x), "y": float(item.y), "radius": float(item.radius)}
        for item in environment.obstacles
    ]
    canonical = json.dumps(
        {
            "seed": seed,
            "pair_seed": environment.pair_seed,
            "mirror": environment.mirror,
            "obstacles": obstacles,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "seed": seed,
        "pair_seed": environment.pair_seed,
        "mirror": environment.mirror,
        "obstacle_count": len(obstacles),
        "environment_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def evaluate_v7_navigation_nested(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
    }
    for path in config["frozen_architecture_evidence"]:
        dependencies[path] = _sha256(root / path)
    conditions = config["conditions"]
    role_counts = {
        role: sum(item["role"] == role for item in conditions)
        for role in ("tuning", "calibration", "external_final")
    }
    if role_counts != {"tuning": 3, "calibration": 1, "external_final": 1}:
        raise ValueError("navigation nested protocol role cardinality changed")
    local = []
    seen_hashes = set()
    for condition in conditions:
        if condition["role"] == "external_final":
            if "mirror_pair_seeds" in condition or "generator_parameters" in condition:
                raise ValueError("external navigation final cannot have a local generator")
            continue
        seeds = [int(seed) for seed in condition["mirror_pair_seeds"]]
        if len(seeds) != 2 or seeds[1] != seeds[0] + 1 or seeds[0] % 2:
            raise ValueError("navigation conditions must be one even/odd mirror pair")
        manifests = [_environment_manifest(seed) for seed in seeds]
        if manifests[0]["pair_seed"] != manifests[1]["pair_seed"]:
            raise ValueError("navigation mirror pair does not share a pair seed")
        if manifests[0]["mirror"] == manifests[1]["mirror"]:
            raise ValueError("navigation mirror pair has equal mirror signs")
        hashes = {item["environment_sha256"] for item in manifests}
        if seen_hashes & hashes:
            raise ValueError("navigation conditions share environment manifests")
        seen_hashes |= hashes
        local.append(
            {
                "condition_id": condition["condition_id"],
                "role": condition["role"],
                "manifests": manifests,
            }
        )
    final = next(item for item in conditions if item["role"] == "external_final")
    custody = final["external_manifest"]
    committed = bool(
        custody["custodian"]
        and custody["commitment_sha256"]
        and custody["committed_at_utc"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": dependencies,
            "architecture_frozen_before_new_tuning": True,
            "model_evaluated": False,
            "parameters_fitted": False,
            "runtime_modified": False,
        },
        "architecture": config["architecture"],
        "role_counts": role_counts,
        "local_conditions": local,
        "historical_regression": config["historical_regression"],
        "external_final": {
            "condition_id": final["condition_id"],
            "committed": committed,
            "retrieval_authorized": custody["retrieval_authorized"],
            "consumed": custody["consumed"],
            "local_seed_available": False,
            "local_generator_available": False,
        },
        "execution_policy": config["execution_policy"],
        "advance_to_tuning": True,
        "advance_to_calibration": False,
        "advance_to_final": False,
        "advance_to_navigation_release": False,
    }
