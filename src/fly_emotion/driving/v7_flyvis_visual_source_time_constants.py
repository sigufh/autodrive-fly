"""Audit FlyVis time constants for all required T4/T5 source types."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_flyvis_c3_time_constant_audit import (
    IMPLEMENTATION as C3_AUDIT_IMPLEMENTATION,
)
from fly_emotion.driving.v7_flyvis_c3_time_constant_audit import (
    _checkpoint_time_constants,
)
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-flyvis-visual-source-time-constants.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_flyvis_visual_source_time_constants.py"
)


def _summary(values: np.ndarray, dt: float, maximum_iqr_ratio: float) -> dict:
    q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
    ratio = float((q75 - q25) / median)
    gates = {
        "all_finite_and_positive": bool(
            np.all(np.isfinite(values)) and np.all(values > 0)
        ),
        "every_model_above_solver_dt": bool(np.all(values > dt)),
        "interquartile_to_median_ratio": ratio <= maximum_iqr_ratio,
    }
    return {
        "model_count": len(values),
        "minimum_seconds": float(np.min(values)),
        "q25_seconds": float(q25),
        "median_seconds": float(median),
        "q75_seconds": float(q75),
        "maximum_seconds": float(np.max(values)),
        "models_at_or_below_solver_dt": int(np.count_nonzero(values <= dt)),
        "interquartile_to_median_ratio": ratio,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_v7_flyvis_visual_source_time_constants(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    head_path = root / config["repository"]["checkout_head_path"]
    if head_path.read_text().strip() != config["repository"]["commit"]:
        raise ValueError("FlyVis checkout commit mismatch")
    archive_path = root / config["pretrained_archive"]["path"]
    if archive_path.stat().st_size != int(config["pretrained_archive"]["bytes"]):
        raise ValueError("FlyVis archive size mismatch")
    if _sha256(archive_path) != config["pretrained_archive"]["sha256"]:
        raise ValueError("FlyVis archive SHA-256 mismatch")
    connectome_path = root / config["connectome"]["path"]
    if connectome_path.stat().st_size != int(config["connectome"]["bytes"]):
        raise ValueError("FlyVis connectome size mismatch")
    if _sha256(connectome_path) != config["connectome"]["sha256"]:
        raise ValueError("FlyVis connectome SHA-256 mismatch")
    connectome = json.loads(connectome_path.read_text(encoding="utf-8"))
    cell_types = [item["name"] for item in connectome["nodes"]]
    required = {
        family: list(config["checkpoint_contract"][f"{family}_required_sources"])
        for family in ("T4", "T5")
    }
    requested = sorted(set(required["T4"] + required["T5"]))
    expected_count = int(config["checkpoint_contract"]["expected_cell_type_count"])
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
        if len(checkpoints) != int(config["pretrained_archive"]["expected_model_count"]):
            raise ValueError("FlyVis pretrained ensemble model count mismatch")
        matrix = np.stack(
            [
                _checkpoint_time_constants(archive.read(name), expected_count)
                for name in checkpoints
            ]
        )
    dt = float(config["training_contract"]["dataset_dt_seconds"])
    summaries = {
        name: _summary(
            matrix[:, cell_types.index(name)],
            dt,
            float(config["gates"]["maximum_interquartile_to_median_ratio"]),
        )
        for name in requested
        if name in cell_types
    }
    coverage = {
        family: {
            "required": names,
            "present": [name for name in names if name in cell_types],
            "missing": [name for name in names if name not in cell_types],
        }
        for family, names in required.items()
    }
    gates = {
        "every_source_in_connectome": all(
            not item["missing"] for item in coverage.values()
        ),
        "all_models_finite_and_positive": all(
            item["gates"]["all_finite_and_positive"] for item in summaries.values()
        ),
        "every_source_above_solver_dt_in_every_model": all(
            item["gates"]["every_model_above_solver_dt"] for item in summaries.values()
        ),
        "every_source_cross_model_IQR_stable": all(
            item["gates"]["interquartile_to_median_ratio"]
            for item in summaries.values()
        ),
        "source_physiology_supervision_available": False,
        "MaleCNS_topology_and_body_mapping_available": False,
        "independent_dynamic_validation_available": False,
    }
    transferable = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(C3_AUDIT_IMPLEMENTATION): _sha256(root / C3_AUDIT_IMPLEMENTATION),
                config["pretrained_archive"]["path"]: _sha256(archive_path),
                config["connectome"]["path"]: _sha256(connectome_path),
            },
            "repository_commit": config["repository"]["commit"],
            "archive": {
                "path": config["pretrained_archive"]["path"],
                "bytes": archive_path.stat().st_size,
                "sha256": _sha256(archive_path),
                "integrity_verified": True,
            },
            "connectome": {
                "path": config["connectome"]["path"],
                "bytes": connectome_path.stat().st_size,
                "sha256": _sha256(connectome_path),
            },
            "checkpoint_pickle_executed": False,
            "runtime_modified": False,
        },
        "training_contract": {
            **config["training_contract"],
            "model_count": len(checkpoints),
            "cell_type_count": len(cell_types),
        },
        "source_coverage": coverage,
        "source_time_constants": summaries,
        "gates": gates,
        "FlyVis_visual_source_time_constants_transferable": transferable,
        "authorize_source_dynamics_response_audit": transferable,
        "advance_to_T4_or_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "blocking_requirements": [
            name for name, passed in gates.items() if not passed
        ],
        "stop_reason": (
            None
            if transferable
            else "FlyVis_source_time_constants_not_physiologically_transferable_to_MaleCNS"
        ),
        "boundary": config["boundary"],
    }
