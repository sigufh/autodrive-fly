"""Audit pretrained FlyVis C3 time constants without executing checkpoint pickle."""

from __future__ import annotations

import hashlib
import io
import json
import pickletools
import zipfile
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-flyvis-c3-time-constant-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_flyvis_c3_time_constant_audit.py"
)


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"FlyVis file size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"FlyVis file SHA-256 mismatch: {path}")
    return path


def _checkpoint_time_constants(payload: bytes, expected_count: int) -> np.ndarray:
    with zipfile.ZipFile(io.BytesIO(payload)) as checkpoint:
        if checkpoint.testzip() is not None:
            raise ValueError("FlyVis checkpoint ZIP failed integrity check")
        members = checkpoint.namelist()
        prefixes = {name.split("/", 1)[0] for name in members}
        if len(prefixes) != 1:
            raise ValueError("FlyVis checkpoint has ambiguous root prefix")
        prefix = prefixes.pop()
        pickle_payload = checkpoint.read(f"{prefix}/data.pkl")
        string_tokens = [
            value
            for opcode, value, _ in pickletools.genops(pickle_payload)
            if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE"}
        ]
        if "nodes_time_const" not in string_tokens:
            raise ValueError("FlyVis checkpoint lacks nodes_time_const")
        storage = checkpoint.read(f"{prefix}/data/1")
    values = np.frombuffer(storage, dtype="<f4")
    if values.shape != (expected_count,) or not np.all(np.isfinite(values)):
        raise ValueError("FlyVis time-constant storage has unexpected shape or values")
    return values.copy()


def _summary(values: np.ndarray, solver_dt: float) -> dict:
    return {
        "minimum_seconds": float(np.min(values)),
        "q25_seconds": float(np.quantile(values, 0.25)),
        "median_seconds": float(np.median(values)),
        "q75_seconds": float(np.quantile(values, 0.75)),
        "maximum_seconds": float(np.max(values)),
        "mean_seconds": float(np.mean(values)),
        "standard_deviation_seconds": float(np.std(values, ddof=1)),
        "models_at_or_below_solver_dt": int(np.count_nonzero(values <= solver_dt)),
        "fraction_at_or_below_solver_dt": float(np.mean(values <= solver_dt)),
    }


def evaluate_v7_flyvis_c3_time_constant_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    head_path = root / config["repository"]["checkout_head_path"]
    checkout_commit = head_path.read_text(encoding="utf-8").strip()
    if checkout_commit != config["repository"]["commit"]:
        raise ValueError("FlyVis checkout commit mismatch")
    for spec in config["code_files"]:
        _verify_file(root, spec)
    archive_path = _verify_file(root, config["pretrained_archive"])
    connectome_path = root / config["code_files"][-1]["path"]
    connectome = json.loads(connectome_path.read_text(encoding="utf-8"))
    cell_types = [item["name"] for item in connectome["nodes"]]
    expected_count = int(config["checkpoint_contract"]["expected_cell_type_count"])
    if len(cell_types) != expected_count or len(set(cell_types)) != expected_count:
        raise ValueError("FlyVis cell-type list differs from checkpoint contract")
    required = config["checkpoint_contract"]["required_source_types"]
    if not set(required) <= set(cell_types):
        raise ValueError("FlyVis connectome lacks required source type")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("FlyVis pretrained archive failed integrity check")
        prefix = config["pretrained_archive"]["ensemble_path"].rstrip("/")
        checkpoints = sorted(
            name
            for name in archive.namelist()
            if name.startswith(f"{prefix}/")
            and name.endswith("/best_chkpt")
            and "/chkpts/" not in name
        )
        metadata_files = sorted(
            name
            for name in archive.namelist()
            if name.startswith(f"{prefix}/") and name.endswith("/_meta.yaml")
        )
        expected_models = int(config["pretrained_archive"]["expected_model_count"])
        if len(checkpoints) != expected_models or len(metadata_files) != expected_models:
            raise ValueError("FlyVis pretrained ensemble model count mismatch")
        metadata_hashes = {
            hashlib.sha256(archive.read(name)).hexdigest() for name in metadata_files
        }
        if len(metadata_hashes) != 1:
            raise ValueError("FlyVis model metadata contracts differ")
        metadata = yaml.safe_load(archive.read(metadata_files[0]))
        dataset = metadata["config"]["task"]["dataset"]
        if dataset["tasks"] != ["flow"] or float(dataset["dt"]) != float(
            config["training_contract"]["dataset_dt_seconds"]
        ):
            raise ValueError("FlyVis training task or dt differs from frozen audit")
        matrix = np.stack(
            [
                _checkpoint_time_constants(archive.read(name), expected_count)
                for name in checkpoints
            ]
        )
    solver_dt = float(config["training_contract"]["dataset_dt_seconds"])
    source_values = {
        name: matrix[:, cell_types.index(name)] for name in required
    }
    summaries = {
        name: _summary(values, solver_dt) for name, values in source_values.items()
    }
    c3 = summaries["C3"]
    identifiability = {
        "all_models_above_solver_dt": c3["models_at_or_below_solver_dt"] == 0,
        "C3_physiology_supervision_available": False,
        "independent_C3_holdout_available": False,
        "MaleCNS_topology_used": False,
        "stable_FlyVis_type_to_MaleCNS_body_mapping_available": False,
    }
    transferable = all(identifiability.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "paper_doi": config["paper"]["doi"],
            "repository_url": config["repository"]["url"],
            "repository_commit": checkout_commit,
            "pretrained_archive": {
                "path": config["pretrained_archive"]["path"],
                "google_drive_file_id": config["pretrained_archive"][
                    "google_drive_file_id"
                ],
                "bytes": archive_path.stat().st_size,
                "sha256": _sha256(archive_path),
                "entry_count": 353,
                "integrity_verified": True,
            },
            "code_files": [
                {
                    "path": spec["path"],
                    "bytes": (root / spec["path"]).stat().st_size,
                    "sha256": _sha256(root / spec["path"]),
                }
                for spec in config["code_files"]
            ],
            "checkpoint_pickle_executed": False,
            "runtime_modified": False,
        },
        "training_contract": {
            **config["training_contract"],
            "model_count": len(checkpoints),
            "cell_type_count": len(cell_types),
            "metadata_contract_sha256": next(iter(metadata_hashes)),
        },
        "source_time_constants": summaries,
        "C3_identifiability": identifiability,
        "C3_time_constant_transfer_authorized": transferable,
        "advance_to_T4_functional_precheck": transferable,
        "blocking_requirements": [
            name for name, passed in identifiability.items() if not passed
        ],
        "stop_reason": (
            None
            if transferable
            else "FlyVis_C3_time_constant_not_physiologically_or_MaleCNS_identified"
        ),
        "boundary": config["boundary"],
    }
