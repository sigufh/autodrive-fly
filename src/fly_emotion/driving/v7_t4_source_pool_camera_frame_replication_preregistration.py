"""Freeze T4 camera-frame source-pool replication before evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path(
    "configs/driving-v7-t4-source-pool-camera-frame-replication-preregistration.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_source_pool_camera_frame_replication_preregistration.py"
)


def _git_sha256(root: Path, revision: str, path: str) -> tuple[str, int]:
    payload = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest(), len(payload)


def evaluate_v7_t4_source_pool_camera_frame_replication_preregistration(
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
    frozen = []
    for spec in config["frozen_inputs"]:
        digest, size = _git_sha256(root, revision, spec["path"])
        if digest != spec["sha256"]:
            raise ValueError(f"frozen discovery input mismatch: {spec['path']}")
        if _sha256(root / spec["path"]) != digest:
            raise ValueError(f"working tree differs from frozen discovery input: {spec['path']}")
        frozen.append({**spec, "bytes": size})
    discovery = json.loads(
        (root / "artifacts/v7-t4-source-pool-camera-frame-discovery.json").read_text()
    )
    candidate = config["candidate"]
    if discovery["discovered_bilateral_readouts"] != [candidate["readout"]]:
        raise ValueError("preregistered readout differs from discovery")
    if discovery["discovered_bilateral_subtypes"][candidate["readout"]] != ["d"]:
        raise ValueError("preregistered subtype differs from discovery")
    if discovery["authorize_new_target_formula"]:
        raise ValueError("discovery unexpectedly authorized target formula")
    typed = yaml.safe_load((root / "configs/driving-v7-lplc-typed-screen.yaml").read_text())
    conditions = {item["condition_id"]: item for item in typed["conditions"]}
    for name in config["replication_condition_ids"]:
        if conditions[name]["role"] != "tuning":
            raise ValueError("replication may consume tuning conditions only")
        if int(conditions[name]["duration_frames"]) != int(
            config["expected_duration_frames"][name]
        ):
            raise ValueError("replication duration changed")
    if float(config["replication_variation"]["local_edge_noise_standard_deviation"]) != 0.0:
        raise ValueError("local-edge replication noise changed")
    scoring_path = Path(config["ordered_gate"]["scoring_threshold_source"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{item["path"]: item["sha256"] for item in config["frozen_inputs"]},
            },
            "frozen_discovery_commit": revision,
            "frozen_inputs": frozen,
            "replication_outputs_observed": False,
            "runtime_modified": False,
        },
        "discovery_condition_id": config["discovery_condition_id"],
        "replication_condition_ids": config["replication_condition_ids"],
        "expected_duration_frames": config["expected_duration_frames"],
        "replication_variation": config["replication_variation"],
        "candidate": candidate,
        "ordered_gate": {
            **config["ordered_gate"],
            "thresholds": scoring["thresholds"],
        },
        "controls_after_ordered_gate": config["controls_after_ordered_gate"],
        "replication_gate": config["replication_gate"],
        "replication_protocol_frozen": True,
        "replication_evaluated": False,
        "authorize_new_target_formula": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "boundary": config["boundary"],
    }
