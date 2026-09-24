"""Freeze a Tm4 column-offset replication before retrieving its outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path(
    "configs/driving-v7-malecns-tm4-offset-replication-preregistration.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_malecns_tm4_offset_replication_preregistration.py"
)


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _ids_sha256(values: list[int]) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<i8").tobytes()
    ).hexdigest()


def evaluate_v7_malecns_tm4_offset_replication_preregistration(root: Path) -> dict:
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
        raise ValueError("Tm4 discovery commit did not resolve exactly")
    frozen_inputs = []
    for spec in config["frozen_inputs"]:
        payload = _git_bytes(root, revision, spec["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"frozen Tm4 discovery input mismatch: {spec['path']}")
        frozen_inputs.append({**spec, "bytes": len(payload)})

    discovery = json.loads(
        _git_bytes(
            root,
            revision,
            "artifacts/v7-malecns-tm4-synapse-column-boundary-audit.json",
        )
    )
    excluded = sorted(
        int(body_id)
        for body_id in discovery["sample_results"]["R"]["sample_body_ids"]
    )
    if _ids_sha256(excluded) != config["discovery_excluded_body_ids_sha256"]:
        raise ValueError("Tm4 excluded discovery body IDs changed")
    offsets = [
        (
            record["candidate_hex"][0] - record["native_hex"][0],
            record["candidate_hex"][1] - record["native_hex"][1],
        )
        for record in discovery["sample_results"]["R"]["records"]
        if not record["candidate_matches_native"]
    ]
    offset_mode = max(set(offsets), key=lambda value: offsets.count(value))
    correction = [-offset_mode[0], -offset_mode[1]]
    if correction != config["candidate_rules"]["discovery_offset_mode_correction"]:
        raise ValueError("Tm4 discovery offset correction changed")

    annotation_spec = config["external_snapshots"]["annotations"]
    annotation_path = root / annotation_spec["path"]
    if (
        annotation_path.stat().st_size != int(annotation_spec["bytes"])
        or _sha256(annotation_path) != annotation_spec["sha256"]
    ):
        raise ValueError("Tm4 preregistration annotation snapshot changed")
    annotations = feather.read_table(
        annotation_path,
        columns=[
            "bodyId",
            "type",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    ).to_pandas()
    sampling = config["replication_sampling"]
    universe = annotations.loc[
        annotations["type"].eq(sampling["source_type"])
        & annotations["somaSide"].eq(sampling["side"])
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
        & ~annotations["bodyId"].isin(excluded)
    ].sort_values(sampling["sort_columns"])
    count = int(sampling["deterministic_evenly_spaced_sample_count"])
    indices = sorted(
        {round(index * (len(universe) - 1) / (count - 1)) for index in range(count)}
    )
    if len(indices) != count:
        raise ValueError("Tm4 replication sample contains duplicate indices")
    sample_ids = universe.iloc[indices]["bodyId"].astype(int).tolist()
    if set(sample_ids) & set(excluded):
        raise ValueError("Tm4 replication overlaps discovery sample")
    if _ids_sha256(sample_ids) != sampling["sample_body_ids_sha256"]:
        raise ValueError("Tm4 replication sample identities changed")

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{spec["path"]: spec["sha256"] for spec in config["frozen_inputs"]},
                annotation_spec["path"]: annotation_spec["sha256"],
            },
            "frozen_discovery_commit": revision,
            "frozen_inputs": frozen_inputs,
            "replication_outputs_observed": False,
            "runtime_modified": False,
        },
        "discovery_excluded_body_ids": excluded,
        "discovery_excluded_body_ids_sha256": _ids_sha256(excluded),
        "replication_universe_count": int(len(universe)),
        "replication_sample_body_ids": sample_ids,
        "replication_sample_body_ids_sha256": _ids_sha256(sample_ids),
        "candidate_rules": config["candidate_rules"],
        "primary_metric": config["primary_metric"],
        "secondary_metric": config["secondary_metric"],
        "replication_gate": config["replication_gate"],
        "replication_protocol_frozen": True,
        "replication_evaluated": False,
        "authorize_left_Tm4_coordinate_writeback": False,
        "authorize_source_mapping_gate_change": False,
        "advance_to_T4_T5_functional_precheck": False,
        "boundary": config["boundary"],
    }
