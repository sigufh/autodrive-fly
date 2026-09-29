"""Freeze a diagnostic T4 fast-by-delayed source-pair protocol."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t4-source-pair-preregistration.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_source_pair_preregistration.py"
)


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def evaluate_v7_t4_source_pair_preregistration(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    revision = config["frozen_LOO_commit"]
    resolved = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != revision:
        raise ValueError("T4 source-pair frozen commit did not resolve exactly")
    frozen = []
    for specification in config["frozen_inputs"]:
        payload = _git_bytes(root, revision, specification["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != specification["sha256"]:
            raise ValueError(f"frozen T4 source-pair input mismatch: {specification['path']}")
        if _sha256(root / specification["path"]) != digest:
            raise ValueError(
                f"working tree differs from frozen source-pair input: {specification['path']}"
            )
        frozen.append({**specification, "bytes": len(payload)})

    expected_pairs = [
        {"name": f"{fast}_x_{delayed}", "fast": fast, "delayed": delayed}
        for fast in config["fast_sources"]
        for delayed in config["delayed_sources"]
    ]
    if config["pair_conditions"] != expected_pairs:
        raise ValueError("T4 source-pair grid is not the fixed Cartesian product")
    if config["modes"] != ["ordered", "temporal_shuffle", "static_sham"]:
        raise ValueError("T4 source-pair temporal controls changed")
    if config["temporal_reductions"] != ["positive_peak", "signed_mean"]:
        raise ValueError("T4 source-pair temporal reductions changed")

    scoring_path = Path(config["scoring_contract"]["source"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{item["path"]: item["sha256"] for item in config["frozen_inputs"]},
            },
            "frozen_LOO_commit": revision,
            "frozen_inputs": frozen,
            "pair_outputs_observed": False,
            "runtime_modified": False,
        },
        "condition_id": config["condition_id"],
        "fast_sources": config["fast_sources"],
        "delayed_sources": config["delayed_sources"],
        "pair_conditions": expected_pairs,
        "modes": config["modes"],
        "temporal_reductions": config["temporal_reductions"],
        "temporal_shuffle_seed": config["temporal_shuffle_seed"],
        "scoring_contract": {
            **config["scoring_contract"],
            "thresholds": scoring["thresholds"],
        },
        "analysis_outputs": config["analysis_outputs"],
        "interpretation": config["interpretation"],
        "pair_protocol_frozen": True,
        "pair_evaluated": False,
        "authorize_source_pair_candidate": False,
        "authorize_source_removal": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "boundary": config["boundary"],
    }
