"""Audit the independent Mi4 calcium payload from Tanaka et al. 2023."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-tanaka-mi4-calcium-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_tanaka_mi4_calcium_audit.py")


def _verify(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"Tanaka Mi4 snapshot identity changed: {path}")
    return path


def _zip_entries(path: Path) -> dict[str, dict]:
    payload = path.read_bytes()
    offset = 0
    entries = {}
    while offset + 46 <= len(payload) and payload[offset : offset + 4] == b"PK\x01\x02":
        values = struct.unpack_from("<4s6H3L5H2L", payload, offset)
        name_size, extra_size, comment_size = values[10:13]
        start = offset + 46
        name = payload[start : start + name_size].decode("utf-8", errors="replace")
        extra = payload[start + name_size : start + name_size + extra_size]
        uncompressed, compressed, local_offset = values[9], values[8], values[16]
        extra_offset = 0
        while extra_offset + 4 <= len(extra):
            field_id, field_size = struct.unpack_from("<HH", extra, extra_offset)
            field = extra[extra_offset + 4 : extra_offset + 4 + field_size]
            position = 0
            if field_id == 1:
                if uncompressed == 0xFFFFFFFF:
                    uncompressed = struct.unpack_from("<Q", field, position)[0]
                    position += 8
                if compressed == 0xFFFFFFFF:
                    compressed = struct.unpack_from("<Q", field, position)[0]
                    position += 8
                if local_offset == 0xFFFFFFFF:
                    local_offset = struct.unpack_from("<Q", field, position)[0]
            extra_offset += 4 + field_size
        entries[name] = {
            "compressed_bytes": int(compressed),
            "uncompressed_bytes": int(uncompressed),
            "local_header_offset": int(local_offset),
            "crc32": f"0x{values[7]:08x}",
        }
        offset = start + name_size + extra_size + comment_size
    if offset != len(payload):
        raise ValueError("Tanaka ZIP central-directory parsing incomplete")
    return entries


def _payload_inventory(path: Path, expected: dict) -> dict:
    payload = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    if set(key for key in payload if not key.startswith("__")) != {
        "data",
        "data_nonselected",
    }:
        raise ValueError("Tanaka Mi4 MAT top-level variables changed")
    selected = payload["data"]
    fly_ids = [int(value) for value in np.atleast_1d(selected.fliesUsed)]
    flies = np.atleast_1d(selected.indFly)
    roi_counts = []
    repetition_counts = set()
    all_finite = True
    for fly in flies:
        cells = np.asarray(fly.p6_baselineSubtracted.snipMat, dtype=object)
        if cells.ndim == 1:
            cells = cells[:, None]
        roi_counts.append(int(cells.shape[1]))
        if cells.shape[0] != int(expected["selected_epoch_count"]):
            raise ValueError("Tanaka selected epoch count changed")
        for epoch in range(cells.shape[0]):
            for roi in range(cells.shape[1]):
                trace = np.asarray(cells[epoch, roi], dtype=float)
                if trace.shape[0] != int(expected["time_sample_count"]):
                    raise ValueError("Tanaka Mi4 time dimension changed")
                repetition_counts.add(int(trace.shape[1]))
                all_finite = all_finite and bool(np.isfinite(trace).all())
    time = np.asarray(selected.timeX, dtype=float)
    observed = {
        "top_level_variables": ["data", "data_nonselected"],
        "fly_ids": fly_ids,
        "fly_count": len(fly_ids),
        "selected_roi_counts_by_fly": roi_counts,
        "selected_roi_count": sum(roi_counts),
        "selected_epoch_labels": [str(value) for value in np.atleast_1d(selected.figLeg)],
        "selected_epoch_count": int(expected["selected_epoch_count"]),
        "time_sample_count": len(time),
        "time_start_milliseconds": float(time[0]),
        "time_end_milliseconds": float(time[-1]),
        "median_sample_interval_milliseconds": float(np.median(np.diff(time))),
        "repetition_counts": sorted(repetition_counts),
        "all_selected_response_values_finite": all_finite,
        "stable_within_dataset_fly_IDs_available": len(fly_ids) == len(set(fly_ids)),
        "individual_fly_axis_available": len(flies) == len(fly_ids),
        "trial_and_ROI_axes_available": True,
    }
    exact_checks = {
        "fly_ids": expected["fly_ids"],
        "selected_roi_counts_by_fly": expected["selected_roi_counts"],
        "selected_epoch_labels": expected["selected_epoch_labels"],
        "time_sample_count": expected["time_sample_count"],
        "repetition_counts": [expected["repetitions_per_selected_epoch_and_roi"]],
    }
    if any(observed[key] != value for key, value in exact_checks.items()):
        raise ValueError("Tanaka Mi4 payload structure changed")
    for key in (
        "time_start_milliseconds",
        "time_end_milliseconds",
        "median_sample_interval_milliseconds",
    ):
        if not np.isclose(observed[key], float(expected[key])):
            raise ValueError(f"Tanaka Mi4 {key} changed")
    return observed


def _crc32(path: Path) -> str:
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            checksum = zlib.crc32(chunk, checksum)
    return f"0x{checksum & 0xFFFFFFFF:08x}"


def _local_zip_members(payload: bytes, expected_names: set[str]) -> dict[str, dict]:
    offset = 0
    members = {}
    while offset + 30 <= len(payload) and payload[offset : offset + 4] == b"PK\x03\x04":
        values = struct.unpack_from("<4s5H3L2H", payload, offset)
        method = values[3]
        crc32, compressed, uncompressed = values[6:9]
        name_size, extra_size = values[9:11]
        name_start = offset + 30
        name = payload[name_start : name_start + name_size].decode("utf-8")
        data_start = name_start + name_size + extra_size
        compressed_payload = payload[data_start : data_start + compressed]
        if len(compressed_payload) != compressed:
            raise ValueError("Tanaka Figure 6 script range is truncated")
        if method == 8:
            content = zlib.decompress(compressed_payload, -15)
        elif method == 0:
            content = compressed_payload
        else:
            raise ValueError(f"unsupported ZIP compression method: {method}")
        if len(content) != uncompressed or zlib.crc32(content) & 0xFFFFFFFF != crc32:
            raise ValueError(f"Tanaka Figure 6 script CRC changed: {name}")
        members[name] = {
            "compressed_bytes": compressed,
            "uncompressed_bytes": uncompressed,
            "crc32": f"0x{crc32:08x}",
            "sha256": hashlib.sha256(content).hexdigest(),
            "text": content.decode("utf-8"),
        }
        offset = data_start + compressed
        if expected_names <= set(members):
            break
    return members


def evaluate_v7_tanaka_mi4_calcium_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {name: _verify(root, spec) for name, spec in config["snapshots"].items()}
    html = paths["pmc_article"].read_text(encoding="utf-8")
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    paper = config["paper"]
    required_phrases = (
        "recorded the axonal calcium activity of Mi4 at layer 10 of medulla",
        "using jGCaMP7b",
        "images were acquired at 8.46 Hz",
        "each fly was treated as an independent sample",
        "N = 10 flies",
        "All data presented in the figures are deposited on Dryad",
    )
    if any(phrase not in text for phrase in required_phrases):
        raise ValueError("Tanaka Mi4 paper evidence changed")
    if f'name="citation_doi" content="{paper["doi"]}"' not in html:
        raise ValueError("Tanaka paper DOI changed")

    dryad = json.loads(paths["dryad_dataset"].read_text())
    dryad_files = json.loads(paths["dryad_files"].read_text())
    zenodo = json.loads(paths["zenodo_mirror"].read_text())
    spec = config["dryad"]
    files = {item["path"]: item for item in dryad_files["_embedded"]["stash:files"]}
    archive = files[spec["archive_name"]]
    zenodo_archive = {item["key"]: item for item in zenodo["files"]}[
        spec["archive_name"]
    ]
    if (
        dryad["identifier"] != f'doi:{spec["doi"]}'
        or dryad["versionNumber"] != spec["version_number"]
        or dryad["storageSize"] != spec["dataset_bytes"]
        or archive["size"] != spec["archive_bytes"]
        or archive["digest"] != spec["archive_sha256"]
        or zenodo["doi"] != spec["doi"]
        or zenodo_archive["size"] != spec["archive_bytes"]
        or zenodo_archive["checksum"] != f'md5:{spec["archive_md5"]}'
    ):
        raise ValueError("Tanaka Dryad/Zenodo archive identity changed")

    entries = _zip_entries(paths["zip_central_directory"])
    if len(entries) != spec["zip_entry_count"]:
        raise ValueError("Tanaka ZIP entry count changed")
    figure7_entry = entries[spec["figure7_payload_path"]]
    if (
        figure7_entry["compressed_bytes"] != spec["figure7_payload_compressed_bytes"]
        or figure7_entry["uncompressed_bytes"]
        != spec["figure7_payload_uncompressed_bytes"]
        or figure7_entry["crc32"] != spec["figure7_payload_crc32"]
        or spec["figure7_script_path"] not in entries
    ):
        raise ValueError("Tanaka Figure 7 ZIP member identity changed")
    if _crc32(paths["figure7_payload"]) != figure7_entry["crc32"]:
        raise ValueError("Tanaka extracted Figure 7 payload CRC changed")

    behavior_entries = {}
    for expected_entry in spec["figure6_behavior_members"]:
        entry = entries[expected_entry["path"]]
        if any(
            entry[key] != expected_entry[key]
            for key in ("compressed_bytes", "uncompressed_bytes", "crc32")
        ):
            raise ValueError("Tanaka Figure 6 behavioral member identity changed")
        behavior_entries[expected_entry["path"]] = entry

    expected_scripts = {item["path"]: item for item in spec["figure6_script_members"]}
    extracted_scripts = _local_zip_members(
        paths["figure6_scripts_range"].read_bytes(), set(expected_scripts)
    )
    if set(extracted_scripts) != set(expected_scripts):
        raise ValueError("Tanaka Figure 6 script range inventory changed")
    for name, expected_entry in expected_scripts.items():
        observed = extracted_scripts[name]
        central = entries[name]
        if any(
            observed[key] != expected_entry[key]
            for key in ("compressed_bytes", "uncompressed_bytes", "crc32", "sha256")
        ) or any(
            central[key] != expected_entry[key]
            for key in ("compressed_bytes", "uncompressed_bytes", "crc32")
        ):
            raise ValueError("Tanaka Figure 6 script identity changed")
    screen_script = extracted_scripts[
        "counterevidence_data_upload/counterevidence_dryad_data/scripts/fig6_01_screen_distplot.m"
    ]["text"]
    replication_script = extracted_scripts[
        "counterevidence_data_upload/counterevidence_dryad_data/scripts/fig6_02_replication.m"
    ]["text"]
    if any(
        phrase not in screen_script
        for phrase in (
            "'C2';'C3';'Mi1';'Tm3(a)';'Tm3(b)'",
            "cells_b = {'Dm1';'Dm2';'Dm3';'Dm4'",
            "'Dm16';'Dm17';'Dm9Dm13Dm18';'Mi4';'Mi9';'Tm1';'Tm2';'Tm4';'Tm9'}",
            "meanResps_a{gg,1}(ff,:)= mean(mat(isStim,:,1))",
            "Gal4/shi fractional turn",
        )
    ) or any(
        phrase not in replication_script
        for phrase in (
            "cellnames  = {'C3';'Mi4';'Mi9';'Tm3(a)'",
            "meanResps{gg,1}(ff,:)= mean(mat(isStim,:,1))",
            "angular velocity (deg/s)",
        )
    ):
        raise ValueError("Tanaka Figure 6 behavioral semantics changed")

    script = paths["figure7_script"].read_text(encoding="utf-8")
    script_phrases = (
        "Recordings from Mi4 expressing jGCaMP7b",
        "Each cell of indFly corresponds to individual fly",
        "dimension of time x repetitions",
        "ROIs were selected for their consistent response to flash probe",
        "01/03/05/07/09: Full field bright flash",
        "02/04/06/08/10: Full field dark flash",
        "12/13: full-field stationary/flickering checkerboards",
    )
    if any(phrase not in script for phrase in script_phrases):
        raise ValueError("Tanaka Figure 7 analysis semantics changed")
    payload = _payload_inventory(paths["figure7_payload"], config["expected_payload"])

    repository = root / config["repository"]["path"]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_head = subprocess.run(
        ["git", "rev-parse", "refs/remotes/origin/main"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit_count = int(
        subprocess.run(
            ["git", "rev-list", "--all", "--count"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    tree_files = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    repo_spec = config["repository"]
    if (
        head != repo_spec["revision"]
        or remote_head != head
        or commit_count != repo_spec["commit_count"]
        or len(tree_files) != repo_spec["tree_file_count"]
    ):
        raise ValueError("Tanaka Mi4Decoder repository identity changed")
    if "tau=0.300 # seconds" not in (repository / "utility.py").read_text():
        raise ValueError("Mi4Decoder temporal constant changed")

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "raw_snapshot_identity": {
                str(path.relative_to(root)): _sha256(path) for path in paths.values()
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "paper": paper,
        "measurement": {
            "source_type": "Mi4",
            "compartment": "medulla_layer_M10_axons",
            "modality": "two_photon_jGCaMP7b_calcium",
            "response_unit": "deltaF_over_F",
            "acquisition_rate_hz": 8.46,
            "stimulus_conditions": [
                "stationary_checkerboard",
                "flickering_checkerboard_15_Hz",
            ],
            "stimulus_duration_seconds": 5.0,
            "baseline_window_seconds": [-0.5, 0.0],
            "right_optic_lobe_only": True,
        },
        "Dryad": {
            "doi": spec["doi"],
            "version_id": spec["version_id"],
            "version_number": dryad["versionNumber"],
            "dataset_bytes": dryad["storageSize"],
            "top_level_file_count": dryad_files["count"],
            "archive_bytes": archive["size"],
            "declared_archive_sha256": archive["digest"],
            "declared_Zenodo_archive_md5": zenodo_archive["checksum"],
            "full_archive_sha256_locally_verified": False,
            "ZIP_entry_count": len(entries),
            "Figure_7_payload_entry": figure7_entry,
            "Figure_7_extracted_payload_crc32_verified": True,
            "full_archive_downloaded": False,
            "Figure_7_member_range_extracted": True,
        },
        "payload_inventory": payload,
        "Figure_6_named_Mi4_C3_members": {
            "members": behavior_entries,
            "combined_uncompressed_bytes": sum(
                item["uncompressed_bytes"] for item in behavior_entries.values()
            ),
            "measurement_object": "walking_turning_angular_velocity",
            "manipulation": "Mi4_or_C3_targeted_shibire_ts_silencing",
            "neural_activity_recording": False,
            "source_dynamics_payload": False,
            "large_MAT_members_downloaded": False,
            "classification_supported_by_author_scripts": True,
        },
        "ROI_selection": {
            "selected_by_flash_probe_response_consistency": True,
            "response_consistency_threshold": 0.4,
            "unselected_payload_retained": True,
        },
        "model_boundary": {
            "repository_url": repo_spec["url"],
            "repository_revision": head,
            "temporal_filter_tau_seconds": 0.3,
            "tau_source": "Arenz_2017_not_Figure_7_fit",
            "Figure_7_payload_used_to_fit_tau": False,
        },
        "independent_Mi4_numerical_calcium_dynamics_verified": True,
        "experimental_membrane_voltage": False,
        "C3_direct_recording_found": False,
        "recording_to_MaleCNS_body_crosswalk_found": False,
        "authorize_Mi4_source_dynamics_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "Mi4_payload_is_individual_calcium_not_allowed_membrane_voltage",
        "boundary": config["boundary"],
    }
