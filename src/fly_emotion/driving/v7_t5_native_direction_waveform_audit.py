"""Audit native-time T5 PD-minus-ND waveforms without fitting a model."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t5_conductance_audit import _load_cell, _segments, _vector

CONFIG = Path("configs/driving-v7-t5-native-direction-waveform-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_native_direction_waveform_audit.py"
)


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _unit(values: np.ndarray, time: np.ndarray) -> np.ndarray | None:
    baseline = values[time < 0]
    centered = values - (float(np.mean(baseline)) if len(baseline) else float(values[0]))
    norm = float(np.linalg.norm(centered))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        return None
    return centered / norm


def _correlation(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _summary(values: list[float | None]) -> dict:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    output = {
        "fixed_denominator": len(values),
        "finite_count": len(finite),
        "undefined_count": len(values) - len(finite),
    }
    if len(finite):
        output.update(
            minimum=float(np.min(finite)),
            q25=float(np.quantile(finite, 0.25)),
            median=float(np.median(finite)),
            q75=float(np.quantile(finite, 0.75)),
            maximum=float(np.max(finite)),
        )
    return output


def _pairs(source: Path, recorded_files: int) -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for cell_id in range(1, recorded_files + 1):
        data, protocol = _load_cell(source, cell_id, "all")
        values, starts, stops = _segments(data, "mb")
        protocol_codes = _vector(protocol.protocol).astype(int)
        times = list(_vector(protocol.t)[protocol_codes == 3])
        widths = _vector(protocol.width_mb).astype(int)
        durations = _vector(protocol.duration_mb).astype(int)
        directions = _vector(protocol.direction_mb).astype(int)
        records: dict[tuple[int, int], dict[int, tuple[np.ndarray, np.ndarray]]] = (
            defaultdict(dict)
        )
        for index, (start, stop, width, duration, direction) in enumerate(
            zip(starts, stops, widths, durations, directions, strict=True)
        ):
            key = (int(width), int(duration))
            time = _vector(times[index]).astype(np.float64)
            trace = values[start - 1 : stop].astype(np.float64)
            if len(time) != len(trace) or not np.all(np.diff(time) > 0):
                raise ValueError(f"invalid moving-bar time axis for file {cell_id}")
            records[key][int(direction)] = (time, trace)
        for key, directions_by_code in records.items():
            if set(directions_by_code) != {0, 1}:
                raise ValueError(f"unpaired T5 moving-bar directions for file {cell_id}")
            nd_time, nd = directions_by_code[0]
            pd_time, pd = directions_by_code[1]
            interval = max(
                float(np.median(np.diff(nd_time))),
                float(np.median(np.diff(pd_time))),
            )
            lower = max(float(nd_time[0]), float(pd_time[0]))
            upper = min(float(nd_time[-1]), float(pd_time[-1]))
            grid = np.arange(np.ceil(lower / interval) * interval, upper + interval / 2, interval)
            waveform = np.interp(grid, pd_time, pd) - np.interp(grid, nd_time, nd)
            grouped[key].append(
                {
                    "recording_file": cell_id,
                    "interval_milliseconds": interval,
                    "time": grid,
                    "waveform": waveform,
                }
            )
    return grouped


def evaluate_v7_t5_native_direction_waveform_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    repository = root / config["repository"]["path"]
    if _git_head(repository) != config["repository"]["commit"]:
        raise ValueError("T5 native waveform repository commit mismatch")
    evidence_path = Path(config["repository"]["evidence"])
    evidence = json.loads((root / evidence_path).read_text())
    if evidence["repository"]["commit"] != config["repository"]["commit"]:
        raise ValueError("T5 native waveform evidence revision mismatch")
    label_path = Path(config["label_evidence"])
    label = json.loads((root / label_path).read_text())
    expected_mapping = {
        str(key): value
        for key, value in config["analysis"]["direction_mapping"].items()
    }
    if label["label_status"]["direction_code_to_PD_ND"] != expected_mapping:
        raise ValueError("T5 native waveform direction mapping mismatch")
    phenotype_path = Path(config["phenotype_evidence"])
    phenotype = json.loads((root / phenotype_path).read_text())
    if phenotype["label_boundary"]["biological_PD_code_assigned"] != 1:
        raise ValueError("T5 phenotype has not adopted verified PD code")
    recorded_files = int(config["analysis"]["recorded_file_count"])
    grouped = _pairs(repository, recorded_files)
    threshold = float(config["gates"]["minimum_held_out_correlation"])
    median_threshold = float(config["gates"]["minimum_median_held_out_correlation"])
    condition_results = {}
    total_pairs = 0
    for (width, duration), rows in sorted(grouped.items()):
        total_pairs += len(rows)
        lower = max(float(row["time"][0]) for row in rows)
        upper = min(float(row["time"][-1]) for row in rows)
        interval = max(float(row["interval_milliseconds"]) for row in rows)
        grid = np.arange(np.ceil(lower / interval) * interval, upper + interval / 2, interval)
        waveforms = [
            _unit(np.interp(grid, row["time"], row["waveform"]), grid) for row in rows
        ]
        folds = []
        for index, row in enumerate(rows):
            training = [
                value
                for j, value in enumerate(waveforms)
                if j != index and value is not None
            ]
            template = (
                _unit(np.mean(training, axis=0), grid)
                if len(training) == len(rows) - 1
                else None
            )
            folds.append(
                {
                    "held_out_recording_file": row["recording_file"],
                    "correlation": _correlation(waveforms[index], template),
                }
            )
        correlations = [item["correlation"] for item in folds]
        summary = _summary(correlations)
        gates = {
            "every_held_out_file": all(
                value is not None and value >= threshold for value in correlations
            ),
            "median_held_out_correlation": bool(
                summary.get("median") is not None
                and summary["median"] >= median_threshold
            ),
        }
        condition_results[f"width={width}:duration={duration}ms"] = {
            "width_pixels": width,
            "step_duration_milliseconds": duration,
            "recording_file_count": len(rows),
            "recording_file_ids": [row["recording_file"] for row in rows],
            "common_interval_milliseconds": interval,
            "common_sample_count": len(grid),
            "leave_one_recording_file_out": folds,
            "correlation_summary": summary,
            "gates": gates,
            "passed": all(gates.values()),
        }
    all_passed = all(item["passed"] for item in condition_results.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(evidence_path): _sha256(root / evidence_path),
                str(label_path): _sha256(root / label_path),
                str(phenotype_path): _sha256(root / phenotype_path),
            },
            "repository_commit": _git_head(repository),
            "recording_file_count": recorded_files,
            "paired_condition_count": total_pairs,
            "parameter_fit": False,
            "model_simulation": False,
            "runtime_modified": False,
        },
        "analysis_contract": config["analysis"],
        "conditions": condition_results,
        "condition_count": len(condition_results),
        "passing_condition_count": int(sum(item["passed"] for item in condition_results.values())),
        "all_native_direction_waveform_gates_passed": all_passed,
        "authorize_T5_direction_template": all_passed,
        "advance_to_T5_fit": False,
        "advance_to_visual_gate": False,
        "stop_reason": (
            None
            if all_passed
            else "T5_PD_minus_ND_waveform_failed_leave_one_recording_file_out_robustness"
        ),
        "boundary": config["boundary"],
    }
