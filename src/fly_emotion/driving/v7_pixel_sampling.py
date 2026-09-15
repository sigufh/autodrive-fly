from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_phase_motion import (
    PERIOD,
    PRE_FRAMES,
    cycle_energy,
    motion_controls,
    phase_average_contrast,
    phase_grating,
)
from fly_emotion.driving.v7_temporal_audit import independent_pixel_filter

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_pixel_sampling.py")
REFERENCE = Path("artifacts/v7-geometry-sign.json")


def eye_local_coordinates(u: np.ndarray, side: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(u, dtype=np.float64) - (np.asarray(side) > 0)


def sample_eye_centres(
    image: np.ndarray, local_x: np.ndarray, local_y: np.ndarray, side: np.ndarray
) -> np.ndarray:
    height, width = image.shape
    if width < 4 or width % 2 or height < 2:
        raise ValueError("eye-centre sampling requires even width >= 4 and height >= 2")
    x = np.asarray(local_x, dtype=np.float64) * (width // 2 - 1)
    y = np.asarray(local_y, dtype=np.float64) * (height - 1)
    offset = (np.asarray(side) > 0) * (width // 2)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, width // 2 - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    a = image[y0, x0 + offset].astype(np.float64)
    b = image[y0, x1 + offset].astype(np.float64)
    c = image[y1, x0 + offset].astype(np.float64)
    d = image[y1, x1 + offset].astype(np.float64)
    upper = a + wx * (b - a)
    lower = c + wx * (d - c)
    return (upper + wy * (lower - upper)).astype(np.float32)


def sampling_controls(u: np.ndarray, v: np.ndarray, side: np.ndarray) -> dict:
    local_x = eye_local_coordinates(u, side)
    if not (
        np.all(np.isin(side, (-1, 1)))
        and np.all(np.isfinite(local_x))
        and np.all(np.isfinite(v))
        and np.all((local_x >= 0) & (local_x <= 1))
        and np.all((v >= 0) & (v <= 1))
    ):
        raise ValueError("retinal coordinates must be finite, within each eye and explicitly sided")
    rng = np.random.default_rng(20260915)
    frame = rng.random((24, 48), dtype=np.float32)
    half = 24
    mirrored = np.concatenate((frame[::-1, :half][:, ::-1], frame[::-1, half:][:, ::-1]), axis=1)
    error = np.max(
        np.abs(
            sample_eye_centres(mirrored, local_x, v, side)
            - sample_eye_centres(frame, 1.0 - local_x, 1.0 - v.astype(np.float64), side)
        )
    )
    isolated = {}
    for eye, column in ((-1, half), (1, 0)):
        changed = frame.copy()
        changed[:, column : column + half] = 1.0 - changed[:, column : column + half]
        mask = side == eye
        isolated[str(eye)] = float(
            np.max(
                np.abs(
                    sample_eye_centres(frame, local_x, v, side)[mask]
                    - sample_eye_centres(changed, local_x, v, side)[mask]
                )
            )
        )
    sampled = {
        condition: [
            np.stack(
                [
                    sample_eye_centres(image, local_x, v, side)
                    for image in phase_grating(condition, phase).frames
                ]
            )
            for phase in range(PERIOD)
        ]
        for condition in ("right", "left", "down", "up")
    }
    negative, retained = {}, {}
    for backend in ("linear_luminance", "signed_frame_difference"):
        energies = {
            condition: np.stack(
                [
                    cycle_energy(
                        independent_pixel_filter(frames, leak=0.24, substeps=4, encoding=backend), 0
                    )
                    for frames in sequences
                ]
            )
            for condition, sequences in sampled.items()
        }
        negative[backend], retained[backend] = {}, {}
        for p, o in (("right", "left"), ("down", "up")):
            key = f"{p}_{o}"
            contrast = phase_average_contrast(energies[p], energies[o])
            negative[backend][key] = float(np.max(np.abs(contrast)))
            retained[backend][key] = {
                "positive_phase_mean_by_cycle": energies[p].mean(axis=0).tolist(),
                "negative_phase_mean_by_cycle": energies[o].mean(axis=0).tolist(),
                "contrast_by_cycle": contrast.tolist(),
                "fraction_receptors_any_cycle_above_1e_6": float(
                    np.mean(np.any(np.abs(contrast) > 1e-6, axis=0))
                ),
            }
    return {
        "seed": 20260915,
        "within_eye_xy_mirror_max_error": float(error),
        "other_eye_change_max_error": isolated,
        "sampled_independent_pixel_max_absolute_contrast": negative,
        "per_receptor_negative_control": retained,
        "sampled_luminance_sha256": {
            f"{condition}:{phase}": hashlib.sha256(frames.tobytes()).hexdigest()
            for condition, sequences in sampled.items()
            for phase, frames in enumerate(sequences)
        },
        "controls_pass": bool(
            error <= 1e-6
            and max(isolated.values()) == 0
            and max(v for pair in negative.values() for v in pair.values()) < 1e-6
        ),
        "scope": "sampler algebra and receptor-only controls, not neural mirror equivalence",
    }


def periodic_pixel_filter(drive: np.ndarray) -> np.ndarray:
    a = (1.0 - 0.24) ** 4
    state = (
        (1.0 - a)
        * np.sum(a ** np.arange(len(drive) - 1, -1, -1)[:, None] * drive, axis=0)
        / (1.0 - a ** len(drive))
    )
    response = []
    for frame in drive:
        state = a * state + (1.0 - a) * frame
        response.append(state.copy())
    return np.asarray(response)


def fractional_phase_diagnostic() -> dict:
    fractions = np.linspace(0.0, 1.0, 33)
    local_x, local_y = fractions / 23, np.zeros_like(fractions)
    side = np.full(len(fractions), -1)
    results = {}
    for backend in ("linear_luminance", "signed_frame_difference"):
        values, periods = {}, {}
        for direction in ("right", "left"):
            responses, samples = [], []
            for phase in range(PERIOD):
                frames = phase_grating(direction, phase).frames[PRE_FRAMES : PRE_FRAMES + PERIOD]
                sampled = np.stack(
                    [sample_eye_centres(frame, local_x, local_y, side) for frame in frames]
                ).astype(np.float64)
                drive = (
                    sampled - 0.5
                    if backend == "linear_luminance"
                    else np.clip((sampled - np.roll(sampled, 1, axis=0)) / 0.5, -1, 1)
                )
                responses.append(periodic_pixel_filter(drive))
                samples.append(sampled)
            values[direction], periods[direction] = np.stack(responses), np.stack(samples)
        p, o = values["right"], values["left"]
        positive = phase_average_contrast(
            np.maximum(p, 0).mean(axis=1), np.maximum(o, 0).mean(axis=1)
        )
        square = phase_average_contrast((p**2).mean(axis=1), (o**2).mean(axis=1))
        results[backend] = {
            "periodic_positive_mean_contrast": positive.tolist(),
            "periodic_mean_square_contrast": square.tolist(),
            "maximum_absolute_positive_mean_contrast": float(np.max(np.abs(positive))),
            "maximum_absolute_mean_square_contrast": float(np.max(np.abs(square))),
            "sampled_periods": {key: value.tolist() for key, value in periods.items()},
            "periodic_response_periods": {key: value.tolist() for key, value in values.items()},
        }
    return {
        "fractional_pixel_offsets": fractions.tolist(),
        "filter": "independent linear filter, leak=0.24, four substeps/frame",
        "initial_state": "analytic periodic fixed point, not finite warmup",
        "results": results,
        "scope": "post-failure scoring diagnostic; mean-square not adopted as neural metric",
    }


def evaluate_v7_pixel_sampling(root: Path) -> dict:
    reference = json.loads((root / REFERENCE).read_text())
    dependencies = reference["protocol"]["dependencies_sha256"]
    for path, expected in dependencies.items():
        if _sha256(root / path) != expected:
            raise ValueError(f"pixel-sampling diagnostic has stale dependency: {path}")
    geometry = reference["geometry"]
    u, v = np.asarray(geometry["reversed_uv"], dtype=np.float32).T
    side = np.asarray(geometry["side"])
    controls = sampling_controls(u, v, side)
    local_x = eye_local_coordinates(u, side)
    x, y = local_x * 23, v.astype(np.float64) * 23
    return {
        "protocol": {
            "name": "v7-eye-centre-sampling-preflight-v1",
            "exploratory": True,
            "advance_allowed": False,
            "dependencies_sha256": {
                **dependencies,
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(REFERENCE): _sha256(root / REFERENCE),
            },
            "reference_results": str(REFERENCE),
            "phase_motion_protocol": reference["protocol"]["phase_motion_protocol"],
            "sampler": "local_x=2*u-(side>0); x=local_x*(half_width-1); y=v*(height-1); "
            "bilinear within same eye only",
            "coordinates_changed": False,
            "physiological_labels_changed": False,
            "dynamics_changed": False,
            "normalization_refit": False,
            "parameter_fitting": False,
            "driving_data_used": False,
            "protocol_revision": "After input-control failure, neural evaluation was removed; "
            "analytic periodic-state diagnostic added without changing gates.",
        },
        "geometry": {
            "receptor_body_ids": geometry["receptor_body_ids"],
            "side": side.tolist(),
            "uv": geometry["reversed_uv"],
            "local_pixel_centre_xy": np.stack((x, y), axis=1).tolist(),
            "half_image_width": 24,
            "reference_cross_eye_sample_cells": geometry["left_half_rounds_to_right_half_cells"],
            "cross_eye_support_cells": int(np.count_nonzero((np.floor(x) < 0) | (np.ceil(x) > 23))),
            "interpolated_cells": int(np.count_nonzero((x != np.floor(x)) | (y != np.floor(y)))),
        },
        "sampling_controls": controls,
        "synthetic_controls": motion_controls(),
        "fractional_phase_diagnostic": fractional_phase_diagnostic(),
        "stimulus_sha256": {
            f"{direction}:{phase}": phase_grating(direction, phase).sha256
            for direction in ("right", "left", "down", "up")
            for phase in range(PERIOD)
        },
        "neural_evaluation": {
            "performed": False,
            "results": {},
            "reason": "Input-only preflight; neural comparison requires valid sampled controls.",
        },
        "limitations": [
            "Post-inspection diagnostic, not independent final validation.",
            "Scaling, interpolation and boundaries change together; no boundary-only attribution.",
            "Sampler reflection identity does not establish neural symmetry or retinotopy.",
            "Bilinear interpolation is an engineering rule, not a measured receptive field.",
            "Fractional sampling interacts with rectified discrete cycle scoring; "
            "periodic-state negative control still fails without any neural spatial computation.",
            "Mean-square control does not establish a valid general neural direction metric.",
            "No neural model was fitted or evaluated; default services and policies unchanged.",
        ],
        "advance_to_central_complex": False,
    }
