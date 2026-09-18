"""Audit availability of the frozen Edmond Fig. 3 1-kHz source arrays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-edmond-fig3-retrieval-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_edmond_fig3_retrieval_audit.py")


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_candidate(path: Path, expected: dict) -> dict:
    if not path.is_file():
        return {"present": False, "fully_verified": False}
    actual = {
        "present": True,
        "bytes": path.stat().st_size,
        "md5": _digest(path, "md5"),
        "sha256": _digest(path, "sha256"),
    }
    payload_hash_verified = all(
        actual[key] == expected[key] for key in ("bytes", "md5", "sha256")
    )
    actual["payload_hash_verified"] = payload_hash_verified
    if not payload_hash_verified:
        actual.update({"safe_array_loaded": False, "fully_verified": False})
        return actual
    values = np.load(path, allow_pickle=False)
    actual.update(
        {
            "safe_array_loaded": True,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "finite_fraction": float(np.isfinite(values).mean()),
        }
    )
    actual["fully_verified"] = (
        actual["shape"] == expected["shape"] and actual["dtype"] == expected["dtype"]
    )
    return actual


def evaluate_v7_edmond_fig3_retrieval_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_paths = {name: Path(path) for name, path in config["evidence"].items()}
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        if path.suffix == ".json"
        else yaml.safe_load((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }
    ephys_config = evidence["electrophysiology_config"]
    ephys_report = evidence["electrophysiology_report"]
    required = config["required_files"]
    for name, expected in required.items():
        configured = ephys_config["files"][name]
        verified = ephys_report["verified_files"][name]
        header = ephys_report["array_headers"][name]
        cross_check = {
            "id": int(configured["id"]),
            "bytes": int(configured["size"]),
            "md5": configured["md5"],
            "sha256": configured["sha256"],
            "shape": header["shape"],
            "dtype": header["dtype"],
        }
        if any(cross_check[key] != expected[key] for key in cross_check):
            raise ValueError(f"frozen Edmond manifest mismatch for {name}")
        if verified != {key: expected[key] for key in ("id", "bytes", "md5", "sha256")}:
            raise ValueError(f"prior verified Edmond payload mismatch for {name}")
    positions = config["datacite_observation"]["required_size_zero_based_positions"]
    if set(positions) != set(required) or len(set(positions.values())) != len(required):
        raise ValueError("DataCite size positions do not cover the four required files exactly")
    kernel = evidence["workbook_kernel_report"]
    workbook_steps = {
        round(float(item["response_sample_interval_milliseconds"]), 9)
        for condition in kernel["conditions"].values()
        for source, item in condition["source_results"].items()
        if source in {"Tm3", "Mi1", "Mi4", "C3"}
    }
    if workbook_steps != {10.0}:
        raise ValueError("frozen workbook is no longer a 10-ms source")
    payload_dir = root / config["dataset"]["local_payload_directory"]
    helper_paths = [
        Path(config["retrieval_helper"]["path"]),
        Path(config["retrieval_helper"]["workflow"]),
    ]
    if any(not (root / path).is_file() for path in helper_paths):
        raise ValueError("Edmond retrieval helper or manual workflow is missing")
    candidates = {
        name: _inspect_candidate(payload_dir / name, expected)
        for name, expected in required.items()
    }
    all_verified = all(item["fully_verified"] for item in candidates.values())
    split = evidence["individual_split_report"]
    prior_passes = sorted(
        f"{condition}:{source}"
        for condition, details in split["conditions"].items()
        for source, result in details["sources"].items()
        if result["passed"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in evidence_paths.values()},
                **{str(path): _sha256(root / path) for path in helper_paths},
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "dataset": config["dataset"],
        "frozen_file_manifest": required,
        "historical_manifest_observation": config["historical_manifest_observation"],
        "manifest_cross_check": {
            "all_four_files_match_prior_config_report_and_headers": True,
            "prior_payload_verification_existed": True,
            "current_payload_retained_by_prior_audit": False,
        },
        "datacite_observation": config["datacite_observation"],
        "retrieval_observations": config["retrieval_observations"],
        "mirror_search": config["mirror_search"],
        "retrieval_helper": config["retrieval_helper"],
        "local_candidates": candidates,
        "workbook_resolution_boundary": {
            "sample_interval_milliseconds": 10.0,
            "repository_array_interval_milliseconds": 1.0,
            "workbook_is_repository_array": False,
            "interpolation_authorized_as_repository_array": False,
        },
        "prior_fixed_individual_split": {
            "passing_source_conditions": prior_passes,
            "all_source_conditions_passed": split[
                "all_source_condition_individual_split_gates_passed"
            ],
            "rules_may_change_for_full_resolution_recompute": False,
        },
        "gates": {
            "frozen_four_file_manifest_complete": True,
            "all_four_local_payloads_hash_and_structure_verified": all_verified,
            "one_khz_cell_order_identity_verified": False,
            "full_resolution_fixed_split_recompute_authorized": False,
            "scientific_source_rejected": False,
            "workbook_interpolation_authorized": False,
        },
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "verified_1khz_payload_not_currently_available_and_array_to_workbook_identity_order_unverified"
            if not all_verified
            else "array_to_workbook_identity_order_still_requires_notebook_verification"
        ),
        "boundary": config["boundary"],
    }
