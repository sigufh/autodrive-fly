from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import (
    HORIZONTAL_PREFERENCE,
    VERTICAL_PREFERENCE,
    V7Contract,
    VisualStimulus,
)
from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    V7PublishedConductanceProbe,
    _collect_source_normalization,
)
from fly_emotion.driving.v7_stability import TypedBackgroundProbe
from fly_emotion.driving.v7_temporal_audit import _step_trace, independent_pixel_filter

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_phase_motion.py")
PERIOD = 8
CYCLES = 4
PRE_FRAMES = 8
BASELINE_FRAMES = 32
SCORED_CYCLES = (2, 3)
CONDITIONS = {
    "right": ("x", 1),
    "left": ("x", -1),
    "down": ("y", 1),
    "up": ("y", -1),
    "static_x": ("x", 0),
    "static_y": ("y", 0),
}
OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}


def phase_grating(
    condition: str, phase: int, *, width: int = 48, height: int = 24
) -> VisualStimulus:
    axis, sign = CONDITIONS[condition]
    yy, xx = np.mgrid[:height, :width]
    coordinate = xx if axis == "x" else yy
    c = np.sqrt(0.5)
    wave = np.asarray([1, c, 0, -c, -1, -c, 0, c], dtype=np.float32)
    frames = np.full((PRE_FRAMES + CYCLES * PERIOD, height, width), 0.5, dtype=np.float32)
    for t in range(CYCLES * PERIOD):
        frames[PRE_FRAMES + t] = 0.5 + 0.3 * wave[(coordinate - sign * t + phase) % PERIOD]
    return VisualStimulus(f"phase_{condition}_{phase}", "phase_grating", "mixed", condition, frames)


def cycle_energy(trace: np.ndarray, sham: np.ndarray) -> np.ndarray:
    delta = np.asarray(trace, dtype=np.float64) - np.asarray(sham, dtype=np.float64)
    return np.stack(
        [
            np.mean(
                np.maximum(
                    delta[PRE_FRAMES + cycle * PERIOD : PRE_FRAMES + (cycle + 1) * PERIOD], 0
                ),
                axis=0,
            )
            for cycle in SCORED_CYCLES
        ]
    )


def phase_average_contrast(preferred: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    p = np.asarray(preferred, dtype=np.float64).mean(axis=0)
    o = np.asarray(opposite, dtype=np.float64).mean(axis=0)
    return (p - o) / (np.abs(p) + np.abs(o) + 1e-12)


def motion_controls() -> dict:
    negative = {}
    for encoding in ("linear_luminance", "signed_frame_difference"):
        values = {}
        for condition in ("right", "left", "down", "up"):
            values[condition] = np.stack(
                [
                    cycle_energy(
                        independent_pixel_filter(
                            phase_grating(condition, phase).frames,
                            leak=0.24,
                            substeps=4,
                            encoding=encoding,
                        ),
                        0,
                    )
                    for phase in range(PERIOD)
                ]
            )
        negative[encoding] = {
            f"{a}_{b}": float(np.max(np.abs(phase_average_contrast(values[a], values[b]))))
            for a, b in (("right", "left"), ("down", "up"))
        }
    positive = {}
    for axis, pair in (("x", ("right", "left")), ("y", ("down", "up"))):
        energies = []
        for condition in pair:
            traces = []
            for phase in range(PERIOD):
                frames = phase_grating(condition, phase).frames.astype(float) - 0.5
                a = frames[:, :, :-1] if axis == "x" else frames[:, :-1, :]
                b = frames[:, :, 1:] if axis == "x" else frames[:, 1:, :]
                response = np.zeros_like(a)
                response[1:] = a[:-1] * b[1:] - b[:-1] * a[1:]
                traces.append(cycle_energy(response, 0))
            energies.append(np.stack(traces))
        positive[axis] = float(np.median(phase_average_contrast(*energies)))
    return {
        "independent_pixel_maximum_absolute_contrast": negative,
        "two_pixel_correlator_median_contrast": positive,
        "scope": "synthetic assay controls, never injected into neural populations",
        "controls_pass": all(v < 1e-6 for x in negative.values() for v in x.values())
        and all(v > 0.5 for v in positive.values()),
    }


def evaluate_v7_phase_motion(root: Path) -> dict:
    contract = V7Contract.load(root)
    controls = motion_controls()
    if not controls["controls_pass"]:
        raise ValueError("phase-motion assay failed synthetic controls")
    config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text())
    results = {}
    input_hashes = {}
    sham_stimulus = VisualStimulus(
        "phase_gray_sham",
        "sham",
        "mixed",
        "none",
        np.full((PRE_FRAMES + CYCLES * PERIOD, 24, 48), 0.5, dtype=np.float32),
    )
    stimulus_hashes = {
        f"{condition}:{phase}": phase_grating(condition, phase).sha256
        for condition in CONDITIONS
        for phase in range(PERIOD)
    }
    for backend in ("linear_luminance", "signed_frame_difference"):
        normalization = _collect_source_normalization(
            root,
            retinal_backend=backend,
            retinal_geometry=config["primary_retinal_geometry"],
            config=config,
        )
        models = {}
        for cls in (TypedBackgroundProbe, V7PublishedConductanceProbe):
            kwargs = {"retinal_backend": backend}
            if cls is V7PublishedConductanceProbe:
                kwargs["normalization"] = normalization
            probe = cls(root, **kwargs)
            probe.baseline_frames = BASELINE_FRAMES
            nodes = np.flatnonzero(
                np.isin(probe.node_types, ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"))
            )
            sides = probe._node_labels(
                root / "data/raw/malecns-v1.0/body-annotations.feather", "somaSide"
            )
            if not np.all(np.isin(sides[nodes], ("L", "R"))):
                raise ValueError("motion readout requires an explicit eye annotation")
            if np.intersect1d(nodes, probe.retina.node_indices).size:
                raise ValueError("target population overlaps receptor inputs")
            sham = _step_trace(probe, sham_stimulus, nodes)
            energies = {}
            response_digest = hashlib.sha256()
            for condition in CONDITIONS:
                phase_energies = []
                for phase in range(PERIOD):
                    stimulus = phase_grating(condition, phase)
                    trace = _step_trace(probe, stimulus, nodes)
                    if not np.array_equal(trace[:PRE_FRAMES], sham[:PRE_FRAMES]):
                        raise ValueError("phase trial changed the common gray prehistory")
                    phase_energies.append(cycle_energy(trace, sham))
                    response_digest.update(trace.tobytes())
                    sampled = np.stack([probe._sample_retina(frame) for frame in stimulus.frames])
                    baseline = sampled[0]
                    codes = np.stack(
                        [
                            probe._retinal_code(frame, baseline, sampled[max(i - 1, 0)])
                            for i, frame in enumerate(sampled)
                        ]
                    )
                    key = f"{backend}:{condition}:{phase}"
                    digest = hashlib.sha256(codes.tobytes()).hexdigest()
                    if key in input_hashes and input_hashes[key] != digest:
                        raise ValueError("candidate comparison changed retinal encoding")
                    input_hashes[key] = digest
                energies[condition] = np.stack(phase_energies)
            populations = {}
            for kind in np.unique(probe.node_types[nodes]):
                for eye in ("L", "R"):
                    mask = (probe.node_types[nodes] == kind) & (sides[nodes] == eye)
                    subtype = kind[-1]
                    preferred = (
                        HORIZONTAL_PREFERENCE[(subtype, eye)]
                        if subtype in "ab"
                        else VERTICAL_PREFERENCE[subtype]
                    )
                    p, o = (
                        energies[preferred][:, :, mask],
                        energies[OPPOSITE[preferred]][:, :, mask],
                    )
                    pcycle, ocycle = p.mean(axis=0), o.mean(axis=0)
                    cycle_contrasts = (pcycle - ocycle) / (pcycle + ocycle + 1e-12)
                    contrast = phase_average_contrast(pcycle, ocycle)
                    static = energies["static_x" if subtype in "ab" else "static_y"][:, :, mask]
                    populations[f"{kind}_{eye}"] = {
                        "cells": int(mask.sum()),
                        "body_ids": probe.graph.body_ids[nodes[mask]].tolist(),
                        "expected_direction": preferred,
                        "median_direction_contrast": float(np.median(contrast)),
                        "direction_positive_fraction": float(np.mean(contrast > 0)),
                        "cycle_median_direction_contrasts": np.median(
                            cycle_contrasts, axis=1
                        ).tolist(),
                        "fraction_response_sum_above_1e_6": float(
                            np.mean(p.mean(axis=(0, 1)) + o.mean(axis=(0, 1)) > 1e-6)
                        ),
                        "median_preferred_response": float(np.median(p.mean(axis=(0, 1)))),
                        "median_opposite_response": float(np.median(o.mean(axis=(0, 1)))),
                        "median_static_response": float(np.median(static.mean(axis=(0, 1)))),
                        "median_absolute_cycle_contrast_change": float(
                            np.median(np.abs(cycle_contrasts[1] - cycle_contrasts[0]))
                        ),
                        "per_cell_preferred_cycle_means": pcycle.tolist(),
                        "per_cell_opposite_cycle_means": ocycle.tolist(),
                    }
            models[cls.__name__] = {
                "populations": populations,
                "target_count": len(nodes),
                "response_sha256": response_digest.hexdigest(),
                "T4_population_median": float(
                    np.median(
                        [
                            v["median_direction_contrast"]
                            for k, v in populations.items()
                            if k.startswith("T4")
                        ]
                    )
                ),
                "T5_population_median": float(
                    np.median(
                        [
                            v["median_direction_contrast"]
                            for k, v in populations.items()
                            if k.startswith("T5")
                        ]
                    )
                ),
                "sham_peak_to_peak_max": float(
                    np.max(np.ptp(sham[PRE_FRAMES + 2 * PERIOD :], axis=0))
                ),
            }
        results[backend] = {"normalization_sha256": normalization.sha256, "models": models}
    paths = [
        IMPLEMENTATION,
        Path("src/fly_emotion/driving/v7.py"),
        Path("src/fly_emotion/driving/v7_temporal_audit.py"),
        Path("src/fly_emotion/driving/v7_stability.py"),
        Path("src/fly_emotion/driving/v7_branched.py"),
        Path("src/fly_emotion/driving/v7_conductance.py"),
        Path("src/fly_emotion/driving/v7_disinhibition.py"),
        Path("src/fly_emotion/driving/v7_source_audit.py"),
        Path("src/fly_emotion/driving/v7_retina_audit.py"),
        Path("src/fly_emotion/driving/retina.py"),
        Path("src/fly_emotion/connectome/graph.py"),
        Path("src/fly_emotion/driving/engine.py"),
        Path("configs/driving-v7.yaml"),
        CONDUCTANCE_CONFIG,
        Path("configs/driving-v7-branched-t4.yaml"),
        Path("configs/driving-v7-t4-source-audit.yaml"),
        Path("artifacts/v7-t4-source-audit.json"),
        Path("data/raw/malecns-v1.0/body-annotations.feather"),
        Path("data/raw/malecns-v1.0/body-neurotransmitters.feather"),
        Path("data/processed/malecns-v1.0/body_ids.npy"),
        Path("data/processed/malecns-v1.0/retina_map.npz"),
        Path("data/processed/malecns-v1.0/adjacency_raw.npz"),
        Path("data/processed/malecns-v1.0/adjacency_target_norm.npz"),
    ]
    hashes = {}
    for path in paths:
        with (root / path).open("rb") as source:
            hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-phase-balanced-motion-v1",
            "exploratory": True,
            "advance_allowed": False,
            "dependencies_sha256": hashes,
            "v7_config_sha256": contract.sha256,
            "phases": list(range(PERIOD)),
            "period_frames": PERIOD,
            "cycles": CYCLES,
            "scored_cycles_zero_based": list(SCORED_CYCLES),
            "gray_pre_frames": PRE_FRAMES,
            "gray_adaptation_frames": BASELINE_FRAMES,
            "substeps_per_frame": 4,
            "spatial_period_pixels": PERIOD,
            "contrast_amplitude": 0.3,
            "aggregation": "float64 phase-average then cycle-average before per-cell contrast",
            "scoring_cycles_retained": True,
            "reference": "same-frame gray sham; static grating reported separately",
            "parameter_fitting": False,
            "driving_data_used": False,
        },
        "synthetic_controls": controls,
        "stimulus_sha256": stimulus_hashes,
        "retinal_drive_sha256": input_hashes,
        "results": results,
        "limitations": [
            "Post-inspection diagnostic, not a preregistered final test.",
            "Mixed ON/OFF gratings cannot prove contrast specialization.",
            "Gray sham removes shared drift, not nonlinear stimulus-dependent feedback.",
            "Four cycles do not establish a periodic steady state; cycle differences are retained.",
            "Synthetic positive control is not part of the MaleCNS circuit.",
            "No existing visual gate, model parameter or default policy is modified.",
        ],
        "advance_to_central_complex": False,
    }
