"""Audit availability of the frozen Edmond Fig. 3 1-kHz source arrays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def _canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    notebook_spec = config["ordering_notebook"]
    notebook_path = root / notebook_spec["path"]
    notebook_verified = (
        notebook_path.is_file()
        and notebook_path.stat().st_size == int(notebook_spec["bytes"])
        and _digest(notebook_path, "md5") == notebook_spec["md5"]
        and _sha256(notebook_path) == notebook_spec["sha256"]
    )
    manifest_spec = config["complete_dataset_manifest"]
    manifest_path = root / manifest_spec["path"]
    if (
        manifest_path.stat().st_size != int(manifest_spec["bytes"])
        or _sha256(manifest_path) != manifest_spec["sha256"]
    ):
        raise ValueError("complete Edmond dataset manifest payload changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _canonical_json_sha256(manifest) != manifest_spec["canonical_json_sha256"]:
        raise ValueError("complete Edmond dataset manifest semantics changed")
    dataset = manifest["data"]
    version = dataset["latestVersion"]
    files = version["files"]
    if (
        dataset["id"] != int(manifest_spec["dataset_id"])
        or version["id"] != int(manifest_spec["dataset_version_id"])
        or len(files) != int(manifest_spec["file_count"])
    ):
        raise ValueError("complete Edmond dataset manifest identity changed")
    manifest_files = {item["label"]: item["dataFile"] for item in files}
    for name, expected in required.items():
        item = manifest_files[name]
        if any(
            item[key] != expected[expected_key]
            for key, expected_key in (
                ("id", "id"),
                ("filesize", "bytes"),
                ("md5", "md5"),
                ("storageIdentifier", "storage_identifier"),
            )
        ):
            raise ValueError(f"complete Edmond manifest file entry changed for {name}")
    identity_tokens = (
        "fly",
        "animal",
        "recording",
        "specimen",
        "individual",
        "subject",
        "meta",
        "manifest",
        "index",
    )
    identity_sidecars = sorted(
        name for name in manifest_files if any(token in name.lower() for token in identity_tokens)
    )
    workbook_path = root / evidence["workbook_kernel_report"]["protocol"]["source_file"][
        "path"
    ]
    identity_matches = {}
    condition_sheets = {"on": "Fig. 3a, left (on)", "off": "Fig. 3a, right (off)"}
    source_names = ("Tm3", "Mi1", "Mi4", "C3")
    for condition_index, (condition, sheet) in enumerate(condition_sheets.items()):
        frame = pd.read_excel(workbook_path, sheet_name=sheet)
        for source in source_names:
            name = f"fig3_{source}.npy"
            key = f"{condition}:{source}"
            if not candidates[name]["fully_verified"]:
                identity_matches[key] = {"verified": False}
                continue
            columns = [column for column in frame if str(column).startswith(f"{source}-")]
            workbook_values = frame[columns].to_numpy(dtype=np.float64).T
            array_values = np.load(payload_dir / name, allow_pickle=False)[condition_index, :, ::10]
            errors = np.max(
                np.abs(workbook_values[:, None, :] - array_values[None, :, :]), axis=2
            )
            nearest = np.argmin(errors, axis=1)
            diagonal = np.diag(errors)
            second_best = np.partition(errors, 1, axis=1)[:, 1]
            identity_matches[key] = {
                "verified": bool(
                    np.array_equal(nearest, np.arange(len(columns)))
                    and np.max(diagonal) <= 1e-12
                    and np.min(second_best) > 1e-6
                ),
                "cell_count": len(columns),
                "workbook_column_labels": columns,
                "array_downsample_phase_samples": 0,
                "array_downsample_stride_samples": 10,
                "maximum_diagonal_absolute_error_millivolts": float(np.max(diagonal)),
                "minimum_second_best_absolute_error_millivolts": float(
                    np.min(second_best)
                ),
                "every_workbook_column_unique_nearest_array_row_at_same_ordinal": bool(
                    np.array_equal(nearest, np.arange(len(columns)))
                ),
            }
    identity_verified = all(item["verified"] for item in identity_matches.values())
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
        "complete_dataset_manifest": {
            **manifest_spec,
            "actual_file_count": len(files),
            "required_file_entries_verified": True,
            "identity_sidecar_candidates": identity_sidecars,
        },
        "manifest_cross_check": {
            "all_four_files_match_prior_config_report_and_headers": True,
            "prior_payload_verification_existed": True,
            "current_payload_retained_by_prior_audit": False,
            "current_payload_recovered_during_this_audit": all_verified,
        },
        "datacite_observation": config["datacite_observation"],
        "retrieval_observations": config["retrieval_observations"],
        "mirror_search": config["mirror_search"],
        "retrieval_result": {
            **config["retrieval_result"],
            "verified_official_payload_count": sum(
                item["fully_verified"] for item in candidates.values()
            ),
        },
        "retrieval_helper": config["retrieval_helper"],
        "local_candidates": candidates,
        "ordering_notebook": {**notebook_spec, "fully_verified": notebook_verified},
        "array_to_workbook_identity": {
            "comparison": "workbook column against every array row after exact [::10] sampling",
            "conditions": identity_matches,
            "all_four_sources_both_conditions_verified": identity_verified,
        },
        "workbook_resolution_boundary": {
            "sample_interval_milliseconds": 10.0,
            "repository_array_interval_milliseconds": 1.0,
            "workbook_is_repository_array": False,
            "interpolation_authorized_as_repository_array": False,
        },
        "gates": {
            "frozen_four_file_manifest_complete": True,
            "all_four_local_payloads_hash_and_structure_verified": all_verified,
            "one_khz_cell_order_identity_verified": identity_verified,
            "full_resolution_fixed_split_recompute_authorized": (
                all_verified and identity_verified and notebook_verified
            ),
            "scientific_source_rejected": False,
            "workbook_interpolation_authorized": False,
        },
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "verified_1khz_payload_not_currently_available"
            if not all_verified
            else (
                "array_to_workbook_identity_order_or_notebook_unverified"
                if not identity_verified or not notebook_verified
                else None
            )
        ),
        "boundary": config["boundary"],
    }
