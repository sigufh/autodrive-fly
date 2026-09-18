"""Audit public T4 models for independent v7 source-type dynamics."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-public-t4-model-source-coverage-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_public_t4_model_source_coverage_audit.py"
)


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"public T4 model file size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"public T4 model file SHA-256 mismatch: {path}")
    return path


def evaluate_v7_public_t4_model_source_coverage_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    synaptic = config["synaptic_model"]
    checkout = root / synaptic["checkout_commit_path"]
    if checkout.read_text(encoding="utf-8").strip() != synaptic["commit"]:
        raise ValueError("SynapticModel checkout commit mismatch")
    synaptic_paths = [_verify_file(root, spec) for spec in synaptic["files"]]
    synaptic_text = "\n".join(path.read_text(encoding="utf-8") for path in synaptic_paths)
    for snippet in ("Input 1 ~ Mi9", "Input 2 ~ Mi1", "Input 3 ~ Mi4"):
        if snippet not in synaptic_text:
            raise ValueError("SynapticModel source-arm annotation mismatch")
    if "p.dt = 1/240" not in synaptic_text:
        raise ValueError("SynapticModel time step mismatch")
    modeldb = config["modeldb_239435"]
    archive_path = root / modeldb["archive_path"]
    if archive_path.stat().st_size != int(modeldb["archive_bytes"]):
        raise ValueError("ModelDB archive size mismatch")
    if _sha256(archive_path) != modeldb["archive_sha256"]:
        raise ValueError("ModelDB archive SHA-256 mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ModelDB archive failed integrity check")
        archive_entries = len(archive.infolist())
    modeldb_paths = [_verify_file(root, spec) for spec in modeldb["files"]]
    modeldb_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in modeldb_paths
    )
    if "taur_e = params.x[0]" not in modeldb_text or "taur_i = params.x[2]" not in modeldb_text:
        raise ValueError("ModelDB excitatory/inhibitory timing semantics mismatch")
    if not re.search(r"dt\s*=\s*0\.1", modeldb_text):
        raise ValueError("ModelDB solver step mismatch")
    required = set(config["required_sources"])
    models = {
        "Clark_SynapticModel": {
            "model_url": synaptic["repository_url"],
            "version": synaptic["commit"],
            "source_arm_annotations": synaptic["source_arm_annotations"],
            "independently_parameterized_source_types": synaptic[
                "independent_source_types"
            ],
            "required_source_coverage_fraction": len(
                required & set(synaptic["independent_source_types"])
            )
            / len(required),
            "missing_required_sources": sorted(
                required - set(synaptic["independent_source_types"])
            ),
            "time_step_seconds": float(synaptic["time_step_seconds"]),
            "time_parameters_seconds": {
                "low_pass": float(synaptic["low_pass_seconds"]),
                "high_pass": float(synaptic["high_pass_seconds"]),
            },
        },
        "ModelDB_239435": {
            "model_url": modeldb["model_url"],
            "archive_url": modeldb["archive_url"],
            "archive_entries": archive_entries,
            "source_arm_annotations": modeldb["source_arm_annotations"],
            "independently_parameterized_source_types": modeldb[
                "independent_source_types"
            ],
            "required_source_coverage_fraction": 0.0,
            "missing_required_sources": sorted(required),
            "solver_step_milliseconds": float(modeldb["solver_step_milliseconds"]),
            "moving_bar_durations_milliseconds": modeldb[
                "moving_bar_durations_milliseconds"
            ],
        },
    }
    gates = {
        "every_required_source_independently_parameterized": all(
            not item["missing_required_sources"] for item in models.values()
        ),
        "C3_specific_parameters_available": any(
            "C3" in item["independently_parameterized_source_types"]
            for item in models.values()
        ),
        "source_type_to_MaleCNS_body_mapping_available": False,
        "independent_cell_holdout_available": False,
    }
    transferable = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "required_sources": config["required_sources"],
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "models": models,
        "transfer_gates": gates,
        "complete_source_dynamics_transfer_authorized": transferable,
        "advance_to_T4_functional_precheck": transferable,
        "blocking_requirements": [
            name for name, passed in gates.items() if not passed
        ],
        "stop_reason": (
            None
            if transferable
            else "public_T4_models_use_incomplete_or_abstract_source_arms"
        ),
        "boundary": config["boundary"],
    }
