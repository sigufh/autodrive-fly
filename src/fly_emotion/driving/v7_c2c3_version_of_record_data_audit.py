"""Audit source-data changes in the C2/C3 eLife Version of Record."""

from __future__ import annotations

import hashlib
import io
import subprocess
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-c2c3-version-of-record-data-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_c2c3_version_of_record_data_audit.py"
)


def _git_bytes(repository: Path, revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _tree(repository: Path, revision: str) -> dict[str, str]:
    output = subprocess.run(
        ["git", "ls-tree", "-r", revision],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line.split("\t", 1)[1]: line.split()[2] for line in output.splitlines()}


def _mat_inventory(payload: bytes, spec: dict) -> dict:
    data = scipy.io.loadmat(io.BytesIO(payload), struct_as_record=False, squeeze_me=True)
    records = np.atleast_1d(data["RF_DATA"])
    groups = {}
    fly_names = {}
    axes = {}
    axis_fields = set()
    for group in records:
        name = str(group.name)
        rows = np.atleast_1d(group.DATA)
        groups[name] = len(rows)
        fly_names[name] = [str(row.Flyname) for row in rows]
        axes[name] = {
            axis: sum(hasattr(row, axis) for row in rows) for axis in ("Az", "El")
        }
        for row in rows:
            for axis in ("Az", "El"):
                if hasattr(row, axis):
                    axis_fields.update(getattr(row, axis)._fieldnames or [])
    if groups != spec["groups"]:
        raise ValueError(f"unexpected groups in {spec['path']}")
    return {
        "groups": groups,
        "unique_Flyname_count_by_group": {
            name: len(set(values)) for name, values in fly_names.items()
        },
        "axis_record_count_by_group": axes,
        "axis_fields": sorted(axis_fields),
        "voltage_field_found": any(
            "volt" in field.lower() for field in axis_fields
        ),
        "current_clamp_field_found": any(
            "current" in field.lower() for field in axis_fields
        ),
        "MaleCNS_body_field_found": any(
            "body" in field.lower() for field in axis_fields
        ),
    }


def evaluate_v7_c2c3_version_of_record_data_audit(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    repository = root / audit["repository"]["path"]
    revision = audit["repository"]["current_remote_head"]
    local_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if local_head != revision:
        raise ValueError("local C2/C3 repository revision changed")
    commit_count = int(
        subprocess.run(
            [
                "git",
                "rev-list",
                "--count",
                f"{audit['repository']['pre_revision_baseline']}..{revision}",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if commit_count != int(audit["repository"]["revision_commit_count"]):
        raise ValueError("C2/C3 revision commit count changed")
    old_tree = _tree(repository, audit["repository"]["pre_revision_baseline"])
    new_tree = _tree(repository, revision)
    old_blobs = set(old_tree.values())
    new_blob_paths = {path: blob for path, blob in new_tree.items() if blob not in old_blobs}
    payloads = {}
    for spec in audit["new_numerical_blobs"]:
        payload = _git_bytes(repository, revision, spec["path"])
        if len(payload) != int(spec["bytes"]):
            raise ValueError(f"source-data size mismatch: {spec['path']}")
        if new_tree[spec["path"]] != spec["git_blob_sha1"]:
            raise ValueError(f"source-data Git blob mismatch: {spec['path']}")
        if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
            raise ValueError(f"source-data SHA-256 mismatch: {spec['path']}")
        payloads[spec["path"]] = {
            "bytes": len(payload),
            "git_blob_sha1": spec["git_blob_sha1"],
            "sha256": spec["sha256"],
            "source_types": spec["source_types"],
            "inventory": _mat_inventory(payload, spec),
        }
    source_code = {}
    for spec in audit["source_code"]:
        payload = _git_bytes(repository, revision, spec["path"])
        if (
            len(payload) != int(spec["bytes"])
            or new_tree[spec["path"]] != spec["git_blob_sha1"]
            or hashlib.sha256(payload).hexdigest() != spec["sha256"]
        ):
            raise ValueError(f"source-code identity mismatch: {spec['path']}")
        source_code[spec["path"]] = {
            "bytes": len(payload),
            "git_blob_sha1": spec["git_blob_sha1"],
            "sha256": spec["sha256"],
            "mentions_fluorescence": b"Fluorescence responses" in payload,
            "defines_delta_F_over_F": b"(dsignal[i]-F)/np.abs(F)" in payload,
        }
    required_blobs = {spec["git_blob_sha1"] for spec in audit["new_numerical_blobs"]}
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
    }
    return {
        "protocol": {
            "name": audit["name"],
            "observed_on": audit["observed_on"],
            "dependencies_sha256": dependencies,
            "repository_url": audit["repository"]["url"],
            "local_checked_out_revision": local_head,
            "remote_head_observed": revision,
            "network_metadata_frozen_in_config": True,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "version_of_record": audit["paper"],
        "supplementary_relation": audit["paper"]["supplementary_relation"],
        "repository_release_count": int(audit["repository"]["release_count"]),
        "repository_tag_count": int(audit["repository"]["tag_count"]),
        "revision_commit_count": commit_count,
        "tree_path_count_before_revision": len(old_tree),
        "tree_path_count_at_version_of_record": len(new_tree),
        "new_blob_count": len(set(new_tree.values()) - set(old_tree.values())),
        "removed_blob_count": len(set(old_tree.values()) - set(new_tree.values())),
        "declared_new_numerical_blobs_present": required_blobs.issubset(
            set(new_blob_paths.values())
        ),
        "new_numerical_payloads": payloads,
        "source_code_evidence": source_code,
        "C3_unique_Flyname_count": payloads[
            "Data/Figure3-SourceData3/RF_DATA_C2C3_posTime.mat"
        ]["inventory"]["unique_Flyname_count_by_group"]["C3"],
        "Mi1_control_unique_Flyname_count": payloads[
            "Data/Figure4-SourceData2/RF_DATA_Mi1_posTime.mat"
        ]["inventory"]["unique_Flyname_count_by_group"]["Control"],
        "new_payload_source_types": ["C3", "Mi1"],
        "new_payload_measurement_modality": "calcium_fluorescence_STRF",
        "new_C3_or_Mi4_membrane_voltage_payload_found": False,
        "new_Mi4_numerical_payload_found": False,
        "new_recording_to_MaleCNS_body_crosswalk_found": False,
        "source_dynamics_transfer_gate_changed": False,
        "authorize_new_T4_functional_candidate": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "version_of_record_adds_calcium_STRFs_but_no_transferable_C3_or_Mi4_voltage",
        "boundary": audit["boundary"],
    }
