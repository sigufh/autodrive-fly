from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7 import (
    V7Contract,
    V7VisualProbe,
    VisualStimulus,
    build_controlled_stimuli,
)

AUDIT_IMPLEMENTATION = Path("src/fly_emotion/driving/v7_temporal_audit.py")
SOURCE_TYPES = ("L1", "L2", "L3", "Mi1", "Tm3", "Mi4", "Mi9", "C3", "Tm1", "Tm2", "Tm4", "Tm9")
ON_RESPONSE_SIGN = {"Mi1": 1, "Tm3": 1, "Mi4": 1, "Mi9": -1, "C3": 1}


def build_step_stimuli(
    width: int, height: int, *, pre_frames: int = 16, post_frames: int = 32
) -> list[VisualStimulus]:
    stimuli = []
    for name, level in (("on", 0.8), ("off", 0.2), ("sham", 0.5)):
        frames = np.full((pre_frames + post_frames, height, width), 0.5, dtype=np.float32)
        frames[pre_frames:] = level
        stimuli.append(VisualStimulus(f"step_{name}", "uniform_step", name, "none", frames))
    return stimuli


def independent_pixel_filter(
    frames: np.ndarray, *, leak: float, substeps: int, encoding: str = "linear_luminance"
) -> np.ndarray:
    state = np.zeros(frames.shape[1:], dtype=np.float64)
    baseline = frames[0].astype(np.float64)
    previous = baseline.copy()
    trace = []
    for frame in frames:
        if encoding == "linear_luminance":
            drive = frame - baseline
        elif encoding == "signed_frame_difference":
            drive = np.clip((frame - previous) / np.maximum(baseline, 0.05), -1.0, 1.0)
        else:
            raise ValueError(f"unsupported pixel-filter encoding: {encoding}")
        previous = frame
        for _ in range(substeps):
            state = (1.0 - leak) * state + leak * drive
        trace.append(state.copy())
    return np.asarray(trace)


def audit_edge_window(
    width: int, height: int, frames: int, *, encoding: str = "linear_luminance"
) -> dict:
    stimuli = {
        item.name: item
        for item in build_controlled_stimuli(width=width, height=height, frames=frames)
    }
    results = {}
    for positive, negative, axis in (("right", "left", 1), ("down", "up", 0)):
        pair = [stimuli[f"on_edge_{direction}"].frames for direction in (positive, negative)]
        signals = [
            independent_pixel_filter(item, leak=0.24, substeps=4, encoding=encoding)
            for item in pair
        ]
        common = np.logical_and.reduce([item[-1] != item[0] for item in pair])
        energies = [np.maximum(signal, 0.0).mean(axis=0) for signal in signals]
        contrast = (energies[0] - energies[1]) / (energies[0] + energies[1] + 1e-12)
        aligned = []
        arrival = []
        for image_sequence in pair:
            onset = np.argmax(image_sequence != image_sequence[0], axis=0)
            arrival.append(onset)
            windows = []
            padded = np.concatenate((image_sequence, np.repeat(image_sequence[-1:], 16, axis=0)))
            for y, x in np.argwhere(common):
                t = int(onset[y, x])
                local = padded[t - 1 : t + 16, y : y + 1, x : x + 1]
                windows.append(
                    float(
                        np.maximum(
                            independent_pixel_filter(
                                local, leak=0.24, substeps=4, encoding=encoding
                            ),
                            0,
                        ).mean()
                    )
                )
            aligned.append(np.asarray(windows))
        aligned_contrast = (aligned[0] - aligned[1]) / (aligned[0] + aligned[1] + 1e-12)
        coordinate = np.indices(common.shape)[axis]
        midpoint = (common.shape[axis] - 1) / 2
        results[f"{positive}_versus_{negative}"] = {
            "common_changing_pixels": int(common.sum()),
            "excluded_pixels": int((~common).sum()),
            "global_window_median_contrast": float(np.median(contrast[common])),
            "global_window_median_absolute_contrast": float(np.median(np.abs(contrast[common]))),
            "global_window_maximum_absolute_contrast": float(np.max(np.abs(contrast[common]))),
            "lower_coordinate_half_median_contrast": float(
                np.median(contrast[common & (coordinate < midpoint)])
            ),
            "upper_coordinate_half_median_contrast": float(
                np.median(contrast[common & (coordinate > midpoint)])
            ),
            "maximum_arrival_difference_frames": int(
                np.max(np.abs(arrival[0][common] - arrival[1][common]))
            ),
            "aligned_local_window_maximum_absolute_contrast": float(
                np.max(np.abs(aligned_contrast))
            ),
        }
    return {
        "model": "independent pixel filters; no spatial connections or direction labels",
        "encoding": encoding,
        "leak": 0.24,
        "substeps": 4,
        "aligned_window_frames": 16,
        "aligned_window_role": "diagnostic only; not a replacement scoring or runtime model",
        "pairs": results,
    }


def _step_trace(probe: V7VisualProbe, stimulus: VisualStimulus, nodes: np.ndarray) -> np.ndarray:
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    baseline = probe._sample_retina(stimulus.frames[0])
    baseline_drive = probe._retinal_code(baseline, baseline, baseline)
    for _ in range(probe.baseline_frames * probe.brain_substeps):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        state = probe._advance(state, drive, history)
        state[probe.retina.node_indices] = baseline_drive
    previous = baseline.copy()
    trace = []
    for image in stimulus.frames:
        sampled = probe._sample_retina(image)
        values = probe._retinal_code(sampled, baseline, previous)
        previous = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = values
        trace.append(state[nodes].copy())
    return np.asarray(trace)


def summarize_step_response(delta: np.ndarray, *, onset: int, expected_sign: int | None) -> dict:
    post = delta[onset:]
    peak_index = np.argmax(np.abs(post), axis=0)
    peak = post[peak_index, np.arange(post.shape[1])]
    active = np.abs(peak) > 1e-6
    return {
        "cells": int(post.shape[1]),
        "active_fraction": float(active.mean()),
        "median_signed_peak": float(np.median(peak)),
        "positive_peak_fraction": float(np.mean(peak > 1e-6)),
        "negative_peak_fraction": float(np.mean(peak < -1e-6)),
        "expected_peak_sign": expected_sign,
        "fraction_all_cells_expected_peak_sign": float(np.mean(peak * expected_sign > 1e-6))
        if expected_sign
        else None,
        "median_peak_latency_frames_active": float(np.median(peak_index[active]))
        if active.any()
        else None,
        "fraction_active_peaks_at_window_end": float(np.mean(peak_index[active] == len(post) - 1))
        if active.any()
        else None,
        "median_early_response": float(np.median(post[:4].mean(axis=0))),
        "median_late_response": float(np.median(post[-8:].mean(axis=0))),
        "maximum_pre_step_difference": float(np.max(np.abs(delta[:onset]))),
        "population_mean_trace": delta.mean(axis=1).tolist(),
    }


def evaluate_v7_temporal_input_audit(root: Path) -> dict:
    contract = V7Contract.load(root)
    visual = contract.payload["controlled_vision"]
    pre_frames, post_frames = 16, 32
    stimuli = build_step_stimuli(
        visual["width"], visual["height"], pre_frames=pre_frames, post_frames=post_frames
    )
    backends = {}
    for backend in visual["retinal_backends"]:
        probe = V7VisualProbe(
            root,
            retinal_backend=backend,
            dynamics_backend="typed_visual_subgraph_v1",
            brain_substeps=4,
            baseline_frames=32,
        )
        sides = probe._node_labels(
            root / "data/raw/malecns-v1.0/body-annotations.feather", "somaSide"
        )
        nodes = np.flatnonzero(np.isin(probe.node_types, SOURCE_TYPES))
        traces = {item.polarity: _step_trace(probe, item, nodes) for item in stimuli}
        groups = {}
        for cell_type in SOURCE_TYPES:
            for side in ("L", "R"):
                mask = (probe.node_types[nodes] == cell_type) & (sides[nodes] == side)
                if not mask.any():
                    continue
                groups[f"{cell_type}_{side}"] = {
                    polarity: summarize_step_response(
                        traces[polarity][:, mask] - traces["sham"][:, mask],
                        onset=pre_frames,
                        expected_sign=ON_RESPONSE_SIGN.get(cell_type) if polarity == "on" else None,
                    )
                    for polarity in ("on", "off")
                }
        backends[backend] = {
            "populations": groups,
            "trace_sha256": hashlib.sha256(
                b"".join(traces[name].tobytes() for name in ("on", "off", "sham"))
            ).hexdigest(),
        }
    dependencies = [
        AUDIT_IMPLEMENTATION,
        Path("src/fly_emotion/driving/v7.py"),
        Path("src/fly_emotion/connectome/graph.py"),
        Path("src/fly_emotion/driving/retina.py"),
        Path("src/fly_emotion/driving/engine.py"),
        Path("configs/driving-v7.yaml"),
        Path("data/processed/malecns-v1.0/body_ids.npy"),
        Path("data/processed/malecns-v1.0/adjacency_raw.npz"),
        Path("data/processed/malecns-v1.0/adjacency_target_norm.npz"),
        Path("data/processed/malecns-v1.0/retina_map.npz"),
        Path("data/raw/malecns-v1.0/body-annotations.feather"),
        Path("data/raw/malecns-v1.0/body-neurotransmitters.feather"),
    ]
    dependency_hashes = {}
    for path in dependencies:
        with (root / path).open("rb") as source:
            dependency_hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-temporal-input-audit-v1",
            "exploratory": True,
            "advance_allowed": False,
            "parameter_fitting": False,
            "driving_data_used": False,
            "dependencies_sha256": dependency_hashes,
            "dynamics": "typed_visual_subgraph_v1",
            "geometry": "legacy_proxy_v2",
            "direct_input": "R1-R6 only",
            "baseline_frames": 32,
            "substeps_per_frame": 4,
            "pre_step_frames": pre_frames,
            "post_step_frames": post_frames,
            "response_reference": "same-frame sham trajectory with identical prehistory",
            "activity_threshold": 1e-6,
            "reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8891015/",
            "reference_scope": (
                "ON sign qualitative comparison only; uniform screen steps are not "
                "matched local electrophysiology stimuli"
            ),
            "latency_units": "simulation frames, not biological milliseconds",
        },
        "stimuli": [{"name": item.name, "sha256": item.sha256} for item in stimuli],
        "edge_window_negative_controls": {
            encoding: audit_edge_window(
                visual["width"], visual["height"], visual["frames_per_stimulus"], encoding=encoding
            )
            for encoding in ("linear_luminance", "signed_frame_difference")
        },
        "step_responses": backends,
        "limitations": [
            "Pixel-filter control demonstrates a scoring confound, not the entire neural failure.",
            "Peak signs depend on encoding, stimulus and proxy dynamics, not just transmitters.",
            "Existing battery, thresholds, models and default runtime remain unchanged.",
        ],
        "advance_to_central_complex": False,
    }
