from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.io import loadmat

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t5-conductance-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_conductance_audit.py")
PROTOCOL_CODES = {
    0: "single_bar",
    1: "moving_grating",
    2: "static_grating",
    3: "moving_bar",
    4: "minimal_motion",
}
PREFIXES = {
    "single_bar": "sb",
    "moving_bar": "mb",
    "minimal_motion": "mm",
    "moving_grating": "mg",
    "static_grating": "sg",
}


def _run_git(arguments: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _clone_fixed_repository(config: dict, target: Path) -> None:
    _run_git(["init", "--quiet", str(target)])
    _run_git(["remote", "add", "origin", config["repository"]["url"]], target)
    _run_git(
        ["fetch", "--quiet", "--depth", "1", "origin", config["repository"]["commit"]],
        target,
    )
    _run_git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], target)


def _vector(value: object) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def _segments(data: object, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _vector(getattr(data, f"vm_all_{prefix}"))
    starts = _vector(getattr(data, f"ind_ref1_{prefix}")).astype(int)
    stops = _vector(getattr(data, f"ind_ref2_{prefix}")).astype(int)
    if starts.size != stops.size or starts[0] != 1 or stops[-1] != values.size:
        raise ValueError(f"invalid {prefix} segment bounds")
    if not np.all(starts[1:] == stops[:-1] + 1):
        raise ValueError(f"non-contiguous {prefix} segment bounds")
    return values, starts, stops


def _load_cell(source: Path, cell_id: int, suffix: str) -> tuple[object, object]:
    payload = loadmat(
        source / f"data_cell_{cell_id}_{suffix}.mat",
        squeeze_me=True,
        struct_as_record=False,
    )
    if set(key for key in payload if not key.startswith("__")) != {"d", "p"}:
        raise ValueError(f"unexpected MAT roots for cell {cell_id} {suffix}")
    return payload["d"], payload["p"]


def _trace_matches(
    source_values: np.ndarray,
    source_start: int,
    source_stop: int,
    target_values: np.ndarray,
    target_start: int,
    target_stop: int,
) -> bool:
    source = source_values[source_start - 1 : source_stop]
    target = target_values[target_start - 1 : target_stop]
    return source.shape == target.shape and bool(np.array_equal(source, target))


def _file_manifest(source: Path) -> list[dict]:
    blobs = {}
    for line in _run_git(["ls-tree", "-r", "HEAD"], source).splitlines():
        metadata, path = line.split("\t", 1)
        blobs[path] = metadata.split()[2]
    entries = []
    for path in sorted(item for item in source.iterdir() if item.is_file()):
        entries.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "git_blob": blobs[path.name],
                "sha256": _sha256(path),
            }
        )
    return entries


def audit_t5_conductance_repository(source: Path, config: dict) -> dict:
    commit = _run_git(["rev-parse", "HEAD"], source)
    if commit != config["repository"]["commit"]:
        raise ValueError(f"repository commit mismatch: {commit}")
    files = _file_manifest(source)
    digest_input = [
        {key: entry[key] for key in ("path", "size_bytes", "sha256")} for entry in files
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected = config["repository"]
    byte_total = sum(item["size_bytes"] for item in files)
    if len(files) != expected["file_count"] or byte_total != expected["total_file_bytes"]:
        raise ValueError("repository file count or byte total mismatch")
    if manifest_sha256 != expected["manifest_sha256"]:
        raise ValueError("repository SHA-256 manifest mismatch")
    license_text = (source / "LICENSE").read_text(encoding="utf-8")
    if "GNU GENERAL PUBLIC LICENSE" not in license_text or "Version 3" not in license_text:
        raise ValueError("expected GPL-3.0 license text was not found")
    script = (source / "optimize_model.m").read_text(encoding="utf-8")
    required_script_fragments = (
        "attempts_num = 10",
        "err_cutoff = 3.5",
        "for cell_id =  1:17",
        "if(wid==2)",
        "fmincon(fun",
        "p.vi =  -74",
        "p.vl = -65",
        "p.ve =  0",
    )
    if any(fragment not in script for fragment in required_script_fragments):
        raise ValueError("optimization script no longer matches the frozen protocol")

    cells = []
    interval_counts: Counter[str] = Counter()
    aggregate: Counter[str] = Counter()
    grating_cells = set(config["model_protocol"]["grating_cell_ids"])
    for cell_id in range(1, int(config["paper"]["recorded_cells"]) + 1):
        spfr_data, spfr_protocol = _load_cell(source, cell_id, "spfr")
        all_data, all_protocol = _load_cell(source, cell_id, "all")
        spfr_values, spfr_starts, spfr_stops = _segments(spfr_data, "sb")
        all_values, all_starts, all_stops = _segments(all_data, "sb")
        spfr_widths = _vector(spfr_protocol.width_sb).astype(int)
        protocols = _vector(all_protocol.protocol).astype(int)
        times = list(_vector(all_protocol.t))
        if len(times) != protocols.size:
            raise ValueError(f"protocol/time count mismatch for cell {cell_id}")
        for time in times:
            vector = _vector(time).astype(float)
            if vector.size < 2 or not np.all(np.diff(vector) > 0):
                raise ValueError(f"invalid time vector for cell {cell_id}")
            interval = float(np.median(np.diff(vector)))
            if not np.allclose(np.diff(vector), interval, rtol=0, atol=1e-9):
                raise ValueError(f"non-uniform native time vector for cell {cell_id}")
            interval_counts[f"{interval:g}"] += 1
        condition_keys = lambda p: list(  # noqa: E731
            zip(_vector(p.duration_sb), _vector(p.location_sb), _vector(p.width_sb), strict=True)
        )
        spfr_keys, all_keys = condition_keys(spfr_protocol), condition_keys(all_protocol)
        exact_matches = 0
        for index, key in enumerate(spfr_keys):
            candidates = [j for j, candidate in enumerate(all_keys) if candidate == key]
            if not candidates:
                raise ValueError(f"spfr condition absent from all data for cell {cell_id}")
            if any(
                _trace_matches(
                    spfr_values,
                    spfr_starts[index],
                    spfr_stops[index],
                    all_values,
                    all_starts[j],
                    all_stops[j],
                )
                for j in candidates
            ):
                exact_matches += 1
        modality = {}
        finite = True
        voltage_min, voltage_max = np.inf, -np.inf
        for name, prefix in PREFIXES.items():
            protocol_code = next(key for key, value in PROTOCOL_CODES.items() if value == name)
            count = int(np.count_nonzero(protocols == protocol_code))
            if hasattr(all_data, f"vm_all_{prefix}"):
                values, starts, stops = _segments(all_data, prefix)
                if count != starts.size:
                    raise ValueError(f"{name} protocol count mismatch for cell {cell_id}")
                finite = finite and bool(np.all(np.isfinite(values)))
                voltage_min = min(voltage_min, float(np.min(values)))
                voltage_max = max(voltage_max, float(np.max(values)))
                modality[name] = {"conditions": int(starts.size), "samples": int(values.size)}
            elif count:
                raise ValueError(f"missing {name} values for cell {cell_id}")
            else:
                modality[name] = {"conditions": 0, "samples": 0}
            aggregate[f"{name}_conditions"] += modality[name]["conditions"]
            aggregate[f"{name}_samples"] += modality[name]["samples"]
        has_gratings = modality["moving_grating"]["conditions"] > 0
        if has_gratings != (cell_id in grating_cells):
            raise ValueError(f"grating cell-set mismatch for cell {cell_id}")
        training_mask = spfr_widths == 2
        training_samples = int(
            sum(
                stop - start + 1
                for start, stop, use in zip(spfr_starts, spfr_stops, training_mask, strict=True)
                if use
            )
        )
        aggregate["spfr_single_bar_conditions"] += spfr_starts.size
        aggregate["training_width2_conditions"] += int(np.count_nonzero(training_mask))
        aggregate["training_width2_samples"] += training_samples
        cells.append(
            {
                "cell_id": cell_id,
                "spfr_single_bar_conditions": int(spfr_starts.size),
                "training_width2_conditions": int(np.count_nonzero(training_mask)),
                "training_width2_samples": training_samples,
                "spfr_conditions_exactly_present_in_all": exact_matches,
                "all_modalities": modality,
                "all_values_finite": finite,
                "response_delta_voltage_millivolts_minimum": voltage_min,
                "response_delta_voltage_millivolts_maximum": voltage_max,
            }
        )
    if not all(
        cell["spfr_single_bar_conditions"] == cell["spfr_conditions_exactly_present_in_all"]
        for cell in cells
    ):
        raise ValueError("spfr traces are not an exact subset of all traces")
    return {
        "repository": {
            "url": expected["url"],
            "commit": commit,
            "license": expected["license"],
            "file_count": len(files),
            "total_file_bytes": byte_total,
            "manifest_sha256": manifest_sha256,
            "files": files,
        },
        "paper": config["paper"],
        "model_protocol": config["model_protocol"],
        "data_semantics": {
            "response_quantity": "baseline_subtracted_membrane_voltage",
            "response_unit": "millivolts",
            "model_comparison_expression": "vm_minus_vl",
            "time_unit": "milliseconds",
            "native_sample_intervals_milliseconds": {
                key: interval_counts[key] for key in sorted(interval_counts, key=float)
            },
            "ragged_trace_lengths": True,
        },
        "cells": cells,
        "aggregate": dict(sorted(aggregate.items())),
        "subset_verification": {
            "spfr_conditions_found_in_all": int(
                sum(cell["spfr_conditions_exactly_present_in_all"] for cell in cells)
            ),
            "all_spfr_traces_byte_exact_with_condition_matched_all_trace": True,
        },
    }


def evaluate_v7_t5_conductance_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if not config["exploratory"] or config["advance_allowed"]:
        raise ValueError("T5 conductance audit must remain exploratory and non-advancing")
    with tempfile.TemporaryDirectory(prefix="autodrive-v7-t5-conductance-") as temporary:
        source = Path(temporary) / "repository"
        _clone_fixed_repository(config, source)
        audited = audit_t5_conductance_repository(source, config)
    return {
        "protocol": {
            "name": config["name"],
            "exploratory": True,
            "advance_allowed": False,
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "parameter_fitting": False,
            "model_simulation": False,
            "raw_files_committed": False,
            "temporary_clone_deleted_after_audit": True,
        },
        **audited,
        "split_contract": {
            "fit_source": "same-cell width-2 single-bar flash traces from _spfr files",
            "prediction_conditions": [
                "other single bars",
                "moving bars",
                "minimal motion",
                "moving gratings where available",
                "static gratings where available",
            ],
            "same_cells_used_for_fit_and_prediction": True,
            "independent_cell_holdout": False,
            "untouched_final_test": False,
            "classification": "within_cell_stimulus_condition_generalization",
        },
        "optimizer_reconciliation": {
            "paper_protocol": (
                "1000 random initializations per cell; best-error 1% used for predictions"
            ),
            "repository_function_unit": (
                "one seed, up to 10 attempts, threshold early stop, one saved model per cell"
            ),
            "relationship": (
                "The repository function is a seeded job unit. Reproducing the paper's 1000-start "
                "selection requires orchestrating many seeds and ranking outputs; neither "
                "orchestration "
                "nor published result directories are present in this repository."
            ),
            "paper_1000_start_selection_reproduced": False,
        },
        "interface_boundary": config["interface_boundary"],
        "limitations": [
            (
                "The verified files are processed per-cell traces and protocol metadata, "
                "not raw acquisition files."
            ),
            (
                "Responses are baseline-subtracted membrane voltage, not absolute resting "
                "voltage traces."
            ),
            (
                "Native traces are ragged and use 2.5-ms or 5-ms intervals; they are not a "
                "1-kHz bundle."
            ),
            (
                "Prediction stimuli reuse the same 17 cells used for fitting and are not an "
                "independent cell holdout."
            ),
            (
                "The paper's 1000-start/best-1% ensemble was not reconstructed and no "
                "parameters were fitted."
            ),
            (
                "GPL-3.0 source/data files remain external and are not copied into this "
                "Apache-2.0 repository."
            ),
        ],
        "advance_to_T5_fit": False,
        "advance_to_visual_gate": False,
        "advance_to_central_complex": False,
    }
