"""Freeze T5 increment-order replication before observing its outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path(
    "configs/driving-v7-t5-increment-order-control-replication-preregistration.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_increment_order_control_replication_preregistration.py"
)


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def evaluate_v7_t5_increment_order_control_replication_preregistration(
    root: Path,
) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    revision = config["frozen_discovery_commit"]
    resolved = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != revision:
        raise ValueError("frozen discovery commit did not resolve exactly")
    frozen_inputs = []
    for spec in config["frozen_inputs"]:
        payload = _git_bytes(root, revision, spec["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"frozen discovery input mismatch: {spec['path']}")
        if _sha256(root / spec["path"]) != digest:
            raise ValueError(f"working tree differs from frozen discovery input: {spec['path']}")
        frozen_inputs.append({**spec, "bytes": len(payload)})

    discovery = json.loads(
        (root / "artifacts/v7-t5-increment-order-control-discovery-audit.json").read_text(
            encoding="utf-8"
        )
    )
    if discovery["protocol"]["discovery_condition_id"] != config[
        "discovery_condition_id"
    ]:
        raise ValueError("discovery condition changed")
    if discovery["increment_order"][0] != 0:
        raise ValueError("initialization increment is no longer fixed")
    if discovery["existing_temporal_gate_replaced"]:
        raise ValueError("discovery unexpectedly replaced the temporal gate")

    stage_path = Path(
        next(
            spec["path"]
            for spec in config["frozen_inputs"]
            if spec["path"].endswith("stage1-nested.yaml")
        )
    )
    stage = yaml.safe_load((root / stage_path).read_text(encoding="utf-8"))
    conditions = {row["condition_id"]: row for row in stage["conditions"]}
    for condition_id in config["replication_condition_ids"]:
        condition = conditions[condition_id]
        if condition["role"] != "tuning":
            raise ValueError("replication may consume tuning conditions only")
        if float(condition["generator_parameters"]["edge_speed_pixels_per_frame"]) != float(
            config["expected_edge_speeds_pixels_per_frame"][condition_id]
        ):
            raise ValueError("replication condition speed changed")
    inherited_path = Path(config["inherited_output_thresholds"]["source"])
    inherited = yaml.safe_load((root / inherited_path).read_text(encoding="utf-8"))
    for target, source in (
        (
            "maximum_control_to_ordered_residual_energy_ratio",
            "maximum_shuffle_to_ordered_residual_energy_ratio",
        ),
        (
            "maximum_static_to_ordered_energy_ratio",
            "maximum_static_to_ordered_energy_ratio",
        ),
    ):
        if float(config["inherited_output_thresholds"][target]) != float(
            inherited["controls"][source]
        ):
            raise ValueError("replication threshold differs from inherited gate")

    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        **{spec["path"]: spec["sha256"] for spec in config["frozen_inputs"]},
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "frozen_discovery_commit": revision,
            "frozen_inputs": frozen_inputs,
            "replication_outputs_observed": False,
            "runtime_modified": False,
        },
        "discovery_condition_id": config["discovery_condition_id"],
        "replication_condition_ids": config["replication_condition_ids"],
        "tested_brain_updates_per_frame": config["tested_brain_updates_per_frame"],
        "settle_frames": config["settle_frames"],
        "control": config["control"],
        "input_validity": config["input_validity"],
        "candidate_order": config["candidate_order"],
        "inherited_output_thresholds": config["inherited_output_thresholds"],
        "replication_gate": config["replication_gate"],
        "replication_protocol_frozen": True,
        "replication_evaluated": False,
        "authorize_direction_scoring": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "boundary": config["boundary"],
    }
