"""Audit all frozen public Motyxia2 refs for Kohn-Portes record logs."""

from __future__ import annotations

import io
import json
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-motyxia2-public-history-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_motyxia2_public_history_audit.py"
)


def _git(repository: Path, *arguments: str, allow_no_match: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 or (allow_no_match and result.returncode == 1):
        return result.stdout
    raise ValueError(f"Motyxia2 Git audit failed: {' '.join(arguments)}")


def _batch_check(repository: Path, object_ids: list[str]) -> dict[str, tuple[str, int]]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ],
        input="\n".join(object_ids) + "\n",
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("Motyxia2 Git batch-check failed")
    rows = {}
    for line in result.stdout.splitlines():
        object_id, object_type, object_size = line.split()
        rows[object_id] = (object_type, int(object_size))
    if set(rows) != set(object_ids):
        raise ValueError("Motyxia2 Git batch-check object set changed")
    return rows


def _scan_blobs(
    repository: Path,
    blob_ids: list[str],
    tokens: list[str],
    paths_by_object: dict[str, list[str]],
) -> list[dict]:
    result = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "--batch"],
        input=("\n".join(blob_ids) + "\n").encode(),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError("Motyxia2 Git blob scan failed")
    stream = io.BytesIO(result.stdout)
    hits = []
    for expected_id in blob_ids:
        header = stream.readline().decode("ascii").strip().split()
        if len(header) != 3:
            raise ValueError("Motyxia2 Git blob header changed")
        object_id, object_type, object_size = header
        if object_id != expected_id or object_type != "blob":
            raise ValueError("Motyxia2 Git blob order or type changed")
        payload = stream.read(int(object_size))
        if stream.read(1) != b"\n":
            raise ValueError("Motyxia2 Git blob delimiter changed")
        for token in tokens:
            if token.encode() in payload:
                hits.append(
                    {
                        "token": token,
                        "object_id": object_id,
                        "paths": paths_by_object.get(object_id, []),
                    }
                )
    if stream.read():
        raise ValueError("Motyxia2 Git blob stream has trailing bytes")
    return hits


def evaluate_v7_motyxia2_public_history_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    bundle_spec = config["bundle"]
    bundle = root / bundle_spec["path"]
    if bundle.stat().st_size != int(bundle_spec["bytes"]):
        raise ValueError("Motyxia2 all-refs bundle size changed")
    if _sha256(bundle) != bundle_spec["sha256"]:
        raise ValueError("Motyxia2 all-refs bundle SHA-256 changed")
    identity_path = Path(config["evidence"]["kohn_portes_identity"])
    identity = json.loads((root / identity_path).read_text(encoding="utf-8"))
    recording_ids = sorted(
        {
            recording_id
            for item in identity["source_capacity"].values()
            for recording_id in item["full_repository_recording_ids"]
        }
    )
    recording_tokens = [f"JRK{recording_id}" for recording_id in recording_ids]
    search_tokens = [*recording_tokens, *config["expected_stimulus_tokens"]]

    with tempfile.TemporaryDirectory() as temporary:
        mirror = Path(temporary) / "motyxia2.git"
        clone = subprocess.run(
            ["git", "clone", "--quiet", "--mirror", str(bundle), str(mirror)],
            check=False,
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            raise ValueError("Motyxia2 all-refs bundle could not be cloned")
        show_ref = _git(mirror, "show-ref").splitlines()
        public_branches = sorted(
            line.split(" ", 1)[1]
            for line in show_ref
            if line.split(" ", 1)[1].startswith("refs/remotes/origin/")
            and not line.endswith("refs/remotes/origin/HEAD")
        )
        tag_refs = [line for line in show_ref if " refs/tags/" in line]
        commits = _git(mirror, "rev-list", "--all").splitlines()
        history_paths = sorted(
            {
                line
                for line in _git(
                    mirror, "log", "--all", "--format=", "--name-only"
                ).splitlines()
                if line
            }
        )
        object_rows = [
            line.split(" ", 1)
            for line in _git(mirror, "rev-list", "--objects", "--all").splitlines()
        ]
        object_ids = sorted({row[0] for row in object_rows})
        object_metadata = _batch_check(mirror, object_ids)
        blob_ids = sorted(
            object_id
            for object_id, (object_type, _size) in object_metadata.items()
            if object_type == "blob"
        )
        unique_blob_count = len(blob_ids)
        stimulus_cache_rows = [
            row
            for row in object_rows
            if len(row) == 2
            and re.fullmatch(
                r"motyxia/allen/stimuli/data/[^/]+\.(?:json|npy|pkl)", row[1]
            )
        ]
        stimulus_cache_paths = sorted({row[1] for row in stimulus_cache_rows})
        stimulus_cache_blobs = sorted({row[0] for row in stimulus_cache_rows})
        log_paths = sorted(path for path in history_paths if path.startswith("logs/"))

        paths_by_object: dict[str, list[str]] = {}
        for row in object_rows:
            if len(row) == 2:
                paths_by_object.setdefault(row[0], []).append(row[1])
        exact_hits = _scan_blobs(
            mirror, blob_ids, search_tokens, paths_by_object
        )

    expected = config["repository"]
    observed_counts = {
        "public_branch_count": len(public_branches),
        "reachable_commit_count": len(commits),
        "historical_unique_path_count": len(history_paths),
        "unique_blob_count": unique_blob_count,
        "stimulus_cache_historical_path_count": len(stimulus_cache_paths),
        "stimulus_cache_unique_blob_version_count": len(stimulus_cache_blobs),
    }
    for name, value in observed_counts.items():
        if value != int(expected[name]):
            raise ValueError(f"Motyxia2 public-history count changed: {name}")
    if tag_refs:
        raise ValueError("Motyxia2 public history gained an unreviewed tag")
    if log_paths != config["expected_public_log_paths"]:
        raise ValueError("Motyxia2 public log path inventory changed")
    if exact_hits:
        raise ValueError("Motyxia2 history gained an exact Kohn-Portes record-token hit")
    gates = {
        "all_frozen_public_branch_tips_present": True,
        "complete_reachable_history_enumerated": True,
        "exact_Kohn_Portes_recording_token_hit_found": False,
        "exact_Kohn_Portes_stimulus_name_hit_found": False,
        "recording_key_to_UUID_stimulus_cache_join_found": False,
        "record_specific_stimulus_log_recovered": False,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                bundle_spec["path"]: _sha256(bundle),
                str(identity_path): _sha256(root / identity_path),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "repository": {
            "url": config["repository"]["url"],
            "public_branches": public_branches,
            "public_branch_count": len(public_branches),
            "public_tag_count": 0,
            "reachable_commit_count": len(commits),
            "historical_unique_path_count": len(history_paths),
            "unique_blob_count": unique_blob_count,
        },
        "search": {
            "recording_id_count": len(recording_ids),
            "recording_tokens": recording_tokens,
            "stimulus_tokens": config["expected_stimulus_tokens"],
            "exact_content_hits": exact_hits,
        },
        "candidate_artifacts": {
            "public_log_paths": log_paths,
            "public_log_scope": "fischerfritz_test_only",
            "stimulus_cache_historical_path_count": len(stimulus_cache_paths),
            "stimulus_cache_unique_blob_version_count": len(stimulus_cache_blobs),
            "stimulus_cache_paths_are_UUID_or_generic_fmp": True,
            "stimulus_cache_recording_identity_keys_present": False,
        },
        "availability_gates": gates,
        "Kohn_Portes_record_specific_stimulus_log_found_in_audited_public_history": False,
        "authorize_recording_field_recovery": False,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "audited_public_Motyxia2_history_has_only_unlinked_test_logs_and_UUID_caches"
        ),
        "boundary": config["boundary"],
    }
