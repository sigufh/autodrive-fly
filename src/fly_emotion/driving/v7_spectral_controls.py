from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_phase_motion import PERIOD, PRE_FRAMES, SCORED_CYCLES, phase_grating
from fly_emotion.driving.v7_pixel_sampling import (
    eye_local_coordinates,
    periodic_pixel_filter,
    sample_eye_centres,
)
from fly_emotion.driving.v7_temporal_audit import independent_pixel_filter

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_spectral_controls.py")
REFERENCE = Path("artifacts/v7-pixel-sampling.json")


def spectral_summary(response: np.ndarray, drive: np.ndarray) -> dict:
    response, drive = np.asarray(response, dtype=float), np.asarray(drive, dtype=float)
    if response.shape != drive.shape or response.ndim != 4 or response.shape[2] != PERIOD:
        raise ValueError("expected aligned phase, cycle, eight-frame, cell arrays")
    coefficients = np.fft.rfft(response, axis=2) / PERIOD
    input_coefficients = np.fft.rfft(drive, axis=2) / PERIOD
    f1, i1 = coefficients[:, :, 1], input_coefficients[:, :, 1]
    input_power = np.mean(np.abs(i1) ** 2, axis=0)
    valid = input_power > 1e-12
    transfer = np.zeros_like(input_power, dtype=complex)
    np.divide(np.mean(f1 * np.conj(i1), axis=0), input_power, out=transfer, where=valid)
    weights = np.array([1, 2, 2, 2, 1])[None, None, :, None]
    spectral_power = np.sum(weights * np.abs(coefficients) ** 2, axis=2)
    temporal_power = np.mean(response**2, axis=2)
    return {
        "dc": coefficients[:, :, 0].real.mean(axis=0).tolist(),
        "f1_power": (2 * np.abs(f1) ** 2).mean(axis=0).tolist(),
        "total_power": temporal_power.mean(axis=0).tolist(),
        "input_f1_power": (2 * input_power).tolist(),
        "transfer_real": transfer.real.tolist(),
        "transfer_imaginary": transfer.imag.tolist(),
        "transfer_valid": valid.tolist(),
        "parseval_max_error": float(np.max(np.abs(spectral_power - temporal_power))),
    }


def compare_spectra(forward: dict, backward: dict) -> dict:
    result = {}
    for metric in ("f1_power", "total_power"):
        p, o = np.asarray(forward[metric]), np.asarray(backward[metric])
        contrast = (p - o) / (p + o + 1e-12)
        result[f"{metric}_contrast"] = contrast.tolist()
        result[f"maximum_absolute_{metric}_contrast"] = float(np.max(np.abs(contrast)))
    p, o = np.asarray(forward["dc"]), np.asarray(backward["dc"])
    result["dc_difference"] = (p - o).tolist()
    result["maximum_absolute_dc_difference"] = float(np.max(np.abs(p - o)))
    result["signed_dc_contrast"] = ((p - o) / (np.abs(p) + np.abs(o) + 1e-12)).tolist()
    return result


def _codes(frames: np.ndarray, backend: str, *, periodic: bool) -> np.ndarray:
    if backend == "linear_luminance":
        return frames - 0.5
    previous = np.roll(frames, 1, axis=0) if periodic else np.concatenate((frames[:1], frames[:-1]))
    return np.clip((frames - previous) / 0.5, -1, 1)


def negative_controls(x: np.ndarray, y: np.ndarray, side: np.ndarray) -> dict:
    summaries = {
        backend: {mode: {} for mode in ("finite", "periodic")}
        for backend in ("linear_luminance", "signed_frame_difference")
    }
    for condition in ("right", "left", "down", "up"):
        samples = [
            np.stack(
                [
                    sample_eye_centres(image, x, y, side)
                    for image in phase_grating(condition, phase).frames
                ]
            ).astype(float)
            for phase in range(PERIOD)
        ]
        for backend, modes in summaries.items():
            for mode in modes:
                responses, drives = [], []
                for frames in samples:
                    if mode == "periodic":
                        code = _codes(
                            frames[PRE_FRAMES : PRE_FRAMES + PERIOD], backend, periodic=True
                        )
                        trace = periodic_pixel_filter(code)
                        responses.append(trace[None])
                        drives.append(code[None])
                    else:
                        trace = independent_pixel_filter(
                            frames, leak=0.24, substeps=4, encoding=backend
                        )
                        code = _codes(frames, backend, periodic=False)
                        windows = [
                            slice(PRE_FRAMES + c * PERIOD, PRE_FRAMES + (c + 1) * PERIOD)
                            for c in SCORED_CYCLES
                        ]
                        responses.append(np.stack([trace[w] for w in windows]))
                        drives.append(np.stack([code[w] for w in windows]))
                modes[mode][condition] = spectral_summary(np.stack(responses), np.stack(drives))
    comparisons = {
        backend: {
            mode: {
                f"{p}_{o}": compare_spectra(values[p], values[o])
                for p, o in (("right", "left"), ("down", "up"))
            }
            for mode, values in modes.items()
        }
        for backend, modes in summaries.items()
    }
    passed = all(
        result["maximum_absolute_f1_power_contrast"] < 1e-6
        and result["maximum_absolute_total_power_contrast"] < 1e-6
        and result["maximum_absolute_dc_difference"] < 1e-6
        for modes in comparisons.values()
        for pairs in modes.values()
        for result in pairs.values()
    )
    return {
        "cells": len(x),
        "spectra": summaries,
        "comparisons": comparisons,
        "negative_controls_pass": passed,
    }


def directional_controls() -> dict:
    fractions = np.linspace(0, 1, 33)
    results = {}
    for axis, pair in (("x", ("right", "left")), ("y", ("down", "up"))):
        signals = {}
        for condition in pair:
            traces, drives, null_traces = [], [], []
            for phase in range(PERIOD):
                frames = phase_grating(condition, phase).frames[PRE_FRAMES : PRE_FRAMES + PERIOD]
                first = (
                    np.stack(
                        [
                            sample_eye_centres(
                                image,
                                fractions / 23 if axis == "x" else np.full(33, 0.5),
                                fractions / 23 if axis == "y" else np.full(33, 0.5),
                                np.full(33, -1),
                            )
                            for image in frames
                        ]
                    ).astype(float)
                    - 0.5
                )
                second = (
                    np.stack(
                        [
                            sample_eye_centres(
                                image,
                                (fractions + 1) / 23 if axis == "x" else np.full(33, 0.5),
                                (fractions + 1) / 23 if axis == "y" else np.full(33, 0.5),
                                np.full(33, -1),
                            )
                            for image in frames
                        ]
                    ).astype(float)
                    - 0.5
                )
                traces.append(
                    (np.roll(first, 1, axis=0) * second - np.roll(second, 1, axis=0) * first)[None]
                )
                null_traces.append(
                    (np.roll(first, 1, axis=0) * first - np.roll(first, 1, axis=0) * first)[None]
                )
                drives.append(first[None])
            traces, drives = np.stack(traces), np.stack(drives)
            signals[condition] = {
                "opponent": spectral_summary(traces, drives),
                "inverted_opponent": spectral_summary(-traces, drives),
                "coincident_inputs": spectral_summary(np.stack(null_traces), drives),
            }
        comparisons = {
            kind: compare_spectra(signals[pair[0]][kind], signals[pair[1]][kind])
            for kind in ("opponent", "inverted_opponent", "coincident_inputs")
        }
        results[axis] = {"spectra": signals, "comparisons": comparisons}
    passed = all(
        np.min(item["comparisons"]["opponent"]["signed_dc_contrast"]) > 0.99
        and np.max(item["comparisons"]["inverted_opponent"]["signed_dc_contrast"]) < -0.99
        and np.max(np.abs(item["comparisons"]["coincident_inputs"]["dc_difference"])) == 0
        for item in results.values()
    )
    return {
        "fractional_offsets": fractions.tolist(),
        "results": results,
        "directional_controls_pass": bool(passed),
        "scope": "synthetic two-receptor opponent correlator, never injected into neurons",
    }


def evaluate_v7_spectral_controls(root: Path) -> dict:
    reference = json.loads((root / REFERENCE).read_text())
    dependencies = reference["protocol"]["dependencies_sha256"]
    for path, expected in dependencies.items():
        if _sha256(root / path) != expected:
            raise ValueError(f"spectral control has stale dependency: {path}")
    geometry = reference["geometry"]
    u, v = np.asarray(geometry["uv"], dtype=np.float32).T
    side = np.asarray(geometry["side"])
    receptors = negative_controls(eye_local_coordinates(u, side), v, side)
    fractions = np.linspace(0, 1, 33)
    grid = negative_controls(
        np.tile(fractions / 23, 2), np.tile(fractions[::-1] / 23, 2), np.repeat([-1, 1], 33)
    )
    positive = directional_controls()
    return {
        "protocol": {
            "name": "v7-spectral-response-controls-v1",
            "exploratory": True,
            "advance_allowed": False,
            "dependencies_sha256": {
                **dependencies,
                str(REFERENCE): _sha256(root / REFERENCE),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "period_frames": PERIOD,
            "phases": list(range(PERIOD)),
            "scored_cycles_zero_based": list(SCORED_CYCLES),
            "negative_tolerance": 1e-6,
            "summary_axes": (
                "phase average, retaining each cycle and cell; no complex phase cancellation"
            ),
            "protocol_revision": "post-failure diagnostic; old rectified scores not overwritten",
            "direction_gate_replaced": False,
            "parameter_fitting": False,
        },
        "geometry": geometry,
        "receptor_negative_controls": receptors,
        "fractional_grid_negative_controls": grid,
        "directional_controls": positive,
        "measurement_controls_pass": bool(
            receptors["negative_controls_pass"]
            and grid["negative_controls_pass"]
            and positive["directional_controls_pass"]
        ),
        "neural_evaluation_performed": False,
        "limitations": [
            "Synthetic control success is not neural or biological validation.",
            "F1 power and total power discard sign and can miss opponent direction encoded in DC.",
            "DC, F1 power and input-referenced complex transfer must be reported separately.",
            "A downstream cell has no unique receptor reference without independent mapping.",
            "Power cannot establish ON/OFF selectivity; a polarity inversion preserves power.",
            "Finite cycle spectra do not establish steady state or remove recurrent drift.",
            "Single frequency, amplitude and fixed filter; no universal metric validity claim.",
            "Old release thresholds are not transferable to these diagnostic summaries.",
        ],
        "advance_to_central_complex": False,
    }
