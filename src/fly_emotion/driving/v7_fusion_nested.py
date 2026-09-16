from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-fusion-nested.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_fusion_nested.py")


def _manifest(seed: int) -> dict:
    environment = DrivingEnvironment()
    environment.reset(seed)
    payload = {
        "seed": seed,
        "pair_seed": environment.pair_seed,
        "mirror": environment.mirror,
        "obstacles": [
            [float(item.x), float(item.y), float(item.radius)] for item in environment.obstacles
        ],
    }
    return {
        "seed": seed,
        "pair_seed": environment.pair_seed,
        "mirror": environment.mirror,
        "obstacle_count": len(environment.obstacles),
        "environment_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def evaluate_v7_fusion_nested(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    architecture_path = Path(config["architecture_evidence"])
    architecture = json.loads((root / architecture_path).read_text())
    if architecture["selected_lamina_weight"] != config["frozen_architecture"][
        "lamina_weight"
    ]:
        raise ValueError("fusion protocol differs from frozen architecture evidence")
    if not architecture["cross_validation_passed"]:
        raise ValueError("fusion architecture did not pass its development CV")
    role_counts = {
        role: sum(item["role"] == role for item in config["conditions"])
        for role in ("tuning", "calibration", "external_final")
    }
    if role_counts != {"tuning": 3, "calibration": 1, "external_final": 1}:
        raise ValueError("fusion nested role cardinality changed")
    local = []
    hashes = set()
    for condition in config["conditions"]:
        if condition["role"] == "external_final":
            if "mirror_pair_seeds" in condition:
                raise ValueError("fusion external final cannot expose local seeds")
            continue
        seeds = [int(seed) for seed in condition["mirror_pair_seeds"]]
        if len(seeds) != 2 or seeds[0] % 2 or seeds[1] != seeds[0] + 1:
            raise ValueError("fusion condition is not an even/odd mirror pair")
        manifests = [_manifest(seed) for seed in seeds]
        current = {item["environment_sha256"] for item in manifests}
        if hashes & current:
            raise ValueError("fusion nested conditions share an environment")
        hashes |= current
        local.append(
            {
                "condition_id": condition["condition_id"],
                "role": condition["role"],
                "manifests": manifests,
            }
        )
    final = next(item for item in config["conditions"] if item["role"] == "external_final")
    custody = final["external_manifest"]
    committed = bool(
        custody["custodian"] and custody["commitment_sha256"] and custody["committed_at_utc"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(architecture_path): _sha256(root / architecture_path),
            },
            "architecture_frozen_before_new_tuning": True,
            "model_evaluated": False,
            "parameters_fitted": False,
            "runtime_modified": False,
        },
        "frozen_architecture": config["frozen_architecture"],
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
