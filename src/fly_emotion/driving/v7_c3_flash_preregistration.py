"""Freeze the C3 flash comparator before evaluating FlyVis responses."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-c3-flash-preregistration.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_c3_flash_preregistration.py")


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324


def _verify_file(root: Path, spec: dict) -> dict:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"C3 flash source size mismatch: {spec['path']}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"C3 flash source SHA-256 mismatch: {spec['path']}")
    if _git_blob_sha1(path) != spec["git_blob_sha1"]:
        raise ValueError(f"C3 flash source git blob mismatch: {spec['path']}")
    return {
        "path": spec["path"],
        "bytes": path.stat().st_size,
        "git_blob_sha1": _git_blob_sha1(path),
        "sha256": _sha256(path),
    }


def _author_aggregate(records: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decrement_periods = []
    for record in records:
        stimulus = np.asarray(record.istim, dtype=np.int16)
        decrements = np.flatnonzero(np.diff(stimulus) < 0) + 1
        if len(decrements) > 1:
            decrement_periods.append(float(np.mean(np.diff(decrements))))
    epoch_length = int(round(float(np.mean(decrement_periods))))
    duration = int(round(epoch_length * 1.3))
    traces, stimuli, fly_ids = [], [], []
    for record in records:
        stimulus = np.asarray(record.istim, dtype=np.int16)
        anchors = np.flatnonzero(np.diff(stimulus[:-duration]) < 0) + 1
        signal = np.asarray(record.idSignal1, dtype=np.float64)
        signal = signal / np.mean(signal) - 1.0
        traces.append(np.mean([signal[i : i + duration] for i in anchors], axis=0))
        stimuli.append(np.mean([stimulus[i : i + duration] for i in anchors], axis=0))
        fly_ids.append(int(record.flyID))
    traces = np.stack(traces)
    stimuli = np.stack(stimuli)
    fly_ids = np.asarray(fly_ids)
    eligible = (np.sum(traces, axis=1) != 0) & np.all(np.isfinite(traces), axis=1)
    traces, stimuli, fly_ids = traces[eligible], stimuli[eligible], fly_ids[eligible]
    mean_stimulus = np.mean(stimuli, axis=0)
    correlations = np.corrcoef(mean_stimulus[None, :], traces)[0, 1:]
    selected = correlations > 0
    traces, fly_ids = traces[selected], fly_ids[selected]
    unique_flies = np.unique(fly_ids)
    fly_means = np.stack([np.mean(traces[fly_ids == fly], axis=0) for fly in unique_flies])
    return fly_means, mean_stimulus, correlations[selected]


def _window(time: np.ndarray, bounds: list[float]) -> np.ndarray:
    return (time >= float(bounds[0])) & (time < float(bounds[1]))


def _summary(values: np.ndarray) -> dict:
    return {
        "minimum": float(np.min(values)),
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "q975": float(np.quantile(values, 0.975)),
        "maximum": float(np.max(values)),
    }


def evaluate_v7_c3_flash_preregistration(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    repository = config["repository"]
    head_path = root / repository["checkout_head_path"]
    revision = head_path.read_text(encoding="utf-8").strip()
    if revision != repository["archived_revision"]:
        raise ValueError("C3 flash repository revision mismatch")
    source = _verify_file(root, config["source_data"])
    source_code = [_verify_file(root, spec) for spec in config["source_code"]]
    payload = scipy.io.loadmat(root / source["path"], struct_as_record=False, squeeze_me=True)
    records = np.atleast_1d(payload[config["author_analysis_contract"]["source_struct"]])
    fly_means, mean_stimulus, selected_correlations = _author_aggregate(records)
    sample_interval = 1.0 / float(config["author_analysis_contract"]["interpolated_rate_hz"])
    onset = int(np.flatnonzero(mean_stimulus >= 0.5)[0])
    offset = int(np.flatnonzero((np.arange(len(mean_stimulus)) > onset) & (mean_stimulus < 0.5))[0])
    time = (np.arange(len(mean_stimulus)) - onset) * sample_interval
    contract = config["comparison_contract"]
    baseline = np.mean(fly_means[:, _window(time, contract["baseline_window_seconds"])], axis=1)
    centered = fly_means - baseline[:, None]
    peak_mask = _window(time, contract["onset_peak_window_seconds"])
    plateau_mask = _window(time, contract["late_plateau_window_seconds"])
    recovery_mask = _window(time, contract["recovery_window_seconds"])
    peaks = np.max(centered[:, peak_mask], axis=1)
    peak_indices = np.argmax(centered[:, peak_mask], axis=1)
    peak_latencies = time[peak_mask][peak_indices]
    plateau_ratios = np.mean(centered[:, plateau_mask], axis=1) / peaks
    recovery_ratios = np.abs(np.mean(centered[:, recovery_mask], axis=1)) / peaks
    empirical = {
        "record_count": len(records),
        "selected_ROI_count": len(selected_correlations),
        "fly_count": len(fly_means),
        "sample_interval_seconds": sample_interval,
        "epoch_samples": len(time),
        "ON_onset_sample": onset,
        "OFF_onset_sample": offset,
        "ON_duration_seconds": float((offset - onset) * sample_interval),
        "selected_stimulus_correlation": _summary(selected_correlations),
        "onset_peak_amplitude_delta_F_over_F": _summary(peaks),
        "onset_peak_latency_seconds": _summary(peak_latencies),
        "late_plateau_to_peak_ratio": _summary(plateau_ratios),
        "absolute_recovery_to_peak_ratio": _summary(recovery_ratios),
        "mean_baseline_centered_trace": [float(value) for value in np.mean(centered, axis=0)],
        "time_seconds": [float(value) for value in time],
    }
    inherited_path = root / contract["inherited_threshold_evidence"]
    inherited = yaml.safe_load(inherited_path.read_text(encoding="utf-8"))
    inherited_threshold = float(contract["inherited_minimum_correlation"])
    if inherited_threshold != float(inherited["gates"]["minimum_held_out_correlation"]):
        raise ValueError("C3 flash preregistration threshold differs from inherited gate")
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(contract["inherited_threshold_evidence"]): _sha256(inherited_path),
            },
            "repository_url": repository["url"],
            "archived_revision": revision,
            "source_data": source,
            "source_code": source_code,
            "pilot_model_response_observed": True,
            "ensemble_outputs_observed": False,
            "runtime_modified": False,
        },
        "author_analysis_contract": config["author_analysis_contract"],
        "comparison_contract": contract,
        "official_reproduction_contract": config["official_reproduction_contract"],
        "empirical_C3_flash": empirical,
        "comparison_metrics_frozen": True,
        "boundary": config["boundary"],
    }
