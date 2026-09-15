from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE, VisualStimulus
from fly_emotion.driving.v7_conductance import CONDUCTANCE_CONFIG, _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import (
    MotionSignConductanceProbe,
    MotionSignTypedProbe,
    _sha256,
)
from fly_emotion.driving.v7_phase_motion import (
    BASELINE_FRAMES,
    CYCLES,
    OPPOSITE,
    PERIOD,
    PRE_FRAMES,
    SCORED_CYCLES,
    phase_grating,
)
from fly_emotion.driving.v7_pixel_sampling import eye_local_coordinates, sample_eye_centres
from fly_emotion.driving.v7_temporal_audit import _step_trace

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_spectra.py")
CONTROL_REFERENCE = Path("artifacts/v7-spectral-controls.json")
GEOMETRY_REFERENCE = Path("artifacts/v7-geometry-sign.json")
TARGET_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


class EyeCentreSampling:
    def _sample_retina(self, image: np.ndarray) -> np.ndarray:
        return sample_eye_centres(
            image,
            eye_local_coordinates(self.retinal_u, self.retina.side),
            self.retinal_v,
            self.retina.side,
        )


class EyeCentreTypedProbe(EyeCentreSampling, MotionSignTypedProbe):
    pass


class EyeCentreConductanceProbe(EyeCentreSampling, MotionSignConductanceProbe):
    pass


def response_spectrum(response: np.ndarray) -> dict[str, np.ndarray | float]:
    values = np.asarray(response, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] != PERIOD or values.shape[2] != PERIOD:
        raise ValueError("expected phase, cycle, eight-frame, cell response array")
    coefficients = np.fft.rfft(values, axis=2) / PERIOD
    weights = np.asarray((1, 2, 2, 2, 1), dtype=float)[None, None, :, None]
    spectral_power = np.sum(weights * np.abs(coefficients) ** 2, axis=2)
    temporal_power = np.mean(values**2, axis=2)
    return {
        "dc": coefficients[:, :, 0].real.mean(axis=0),
        "f1_power": (2 * np.abs(coefficients[:, :, 1]) ** 2).mean(axis=0),
        "total_power": temporal_power.mean(axis=0),
        "parseval_max_error": float(np.max(np.abs(spectral_power - temporal_power))),
    }


def _windows(trace: np.ndarray, sham: np.ndarray) -> np.ndarray:
    delta = np.asarray(trace, dtype=np.float64) - np.asarray(sham, dtype=np.float64)
    return np.stack(
        [
            delta[PRE_FRAMES + cycle * PERIOD : PRE_FRAMES + (cycle + 1) * PERIOD]
            for cycle in SCORED_CYCLES
        ]
    )


def _contrast(preferred: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    p, o = preferred.mean(axis=0), opposite.mean(axis=0)
    return (p - o) / (np.abs(p) + np.abs(o) + 1e-12)


def _population_summary(
    probe, nodes: np.ndarray, sides: np.ndarray, spectra: dict[str, dict]
) -> dict:
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
            opposite = OPPOSITE[preferred]
            metrics = {}
            for metric in ("dc", "f1_power", "total_power"):
                p = np.asarray(spectra[preferred][metric])[:, mask]
                o = np.asarray(spectra[opposite][metric])[:, mask]
                contrast = _contrast(p, o)
                cycle_contrast = (p - o) / (np.abs(p) + np.abs(o) + 1e-12)
                denominator = np.abs(p.mean(axis=0)) + np.abs(o.mean(axis=0))
                active = denominator > 1e-6
                metrics[metric] = {
                    "preferred_by_cycle": p.tolist(),
                    "opposite_by_cycle": o.tolist(),
                    "phase_and_cycle_averaged_contrast": contrast.tolist(),
                    "median_contrast": float(np.median(contrast)),
                    "response_denominator": denominator.tolist(),
                    "median_response_denominator": float(np.median(denominator)),
                    "fraction_response_denominator_above_1e_6": float(np.mean(active)),
                    "fraction_response_denominator_above_1e_4": float(np.mean(denominator > 1e-4)),
                    "active_median_contrast_at_1e_6": (
                        float(np.median(contrast[active])) if np.any(active) else None
                    ),
                    "positive_contrast_fraction": float(np.mean(contrast > 0)),
                    "cycle_median_contrast": np.median(cycle_contrast, axis=1).tolist(),
                    "median_absolute_cycle_change": float(np.median(np.abs(p[1] - p[0]))),
                }
            populations[f"{kind}_{eye}"] = {
                "cells": int(mask.sum()),
                "body_ids": probe.graph.body_ids[nodes[mask]].tolist(),
                "expected_direction": preferred,
                "metrics": metrics,
            }
    family_summary = {}
    for family in ("T4", "T5"):
        family_summary[family] = {
            metric: {
                "eight_population_median_contrast": float(
                    np.median(
                        [
                            value["metrics"][metric]["median_contrast"]
                            for name, value in populations.items()
                            if name.startswith(family)
                        ]
                    )
                ),
                "eight_population_median_active_fraction": float(
                    np.median(
                        [
                            value["metrics"][metric]["fraction_response_denominator_above_1e_6"]
                            for name, value in populations.items()
                            if name.startswith(family)
                        ]
                    )
                ),
            }
            for metric in ("dc", "f1_power", "total_power")
        }
    return {
        "populations": populations,
        "target_count": len(nodes),
        "family_medians": family_summary,
    }


def _paired_sampler_summary(nearest: dict, interpolated: dict) -> dict:
    result = {}
    for population, first in nearest["populations"].items():
        second = interpolated["populations"][population]
        if first["body_ids"] != second["body_ids"]:
            raise ValueError("sampler comparison changed target body IDs")
        metrics = {}
        for metric in ("dc", "f1_power", "total_power"):
            a, b = first["metrics"][metric], second["metrics"][metric]
            ap = np.asarray(a["preferred_by_cycle"])
            bp = np.asarray(b["preferred_by_cycle"])
            ao = np.asarray(a["opposite_by_cycle"])
            bo = np.asarray(b["opposite_by_cycle"])
            ac = np.asarray(a["phase_and_cycle_averaged_contrast"])
            bc = np.asarray(b["phase_and_cycle_averaged_contrast"])
            metrics[metric] = {
                "median_preferred_change": float(np.median(bp.mean(axis=0) - ap.mean(axis=0))),
                "median_opposite_change": float(np.median(bo.mean(axis=0) - ao.mean(axis=0))),
                "median_contrast_change": float(np.median(bc - ac)),
                "median_absolute_contrast_change": float(np.median(np.abs(bc - ac))),
            }
        result[population] = {"cells": first["cells"], "metrics": metrics}
    return result


def evaluate_v7_neural_spectra(root: Path) -> dict:
    controls = json.loads((root / CONTROL_REFERENCE).read_text())
    geometry = json.loads((root / GEOMETRY_REFERENCE).read_text())
    if not controls.get("measurement_controls_pass"):
        raise ValueError("neural spectral comparison requires passed measurement controls")
    for path, expected in controls["protocol"]["dependencies_sha256"].items():
        if _sha256(root / path) != expected:
            raise ValueError(f"neural spectral comparison has stale dependency: {path}")
    config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text())
    results, retinal_hashes = {}, {}
    sham_stimulus = VisualStimulus(
        "phase_gray_sham",
        "sham",
        "mixed",
        "none",
        np.full((PRE_FRAMES + CYCLES * PERIOD, 24, 48), 0.5, dtype=np.float32),
    )
    samplers = {
        "nearest_reversed_axis": (MotionSignTypedProbe, MotionSignConductanceProbe),
        "eye_centre_bilinear": (EyeCentreTypedProbe, EyeCentreConductanceProbe),
    }
    for backend in ("linear_luminance", "signed_frame_difference"):
        normalization = _collect_source_normalization(
            root,
            retinal_backend=backend,
            retinal_geometry=config["primary_retinal_geometry"],
            config=config,
        )
        expected_normalization = geometry["results"][backend]["normalization_sha256"]
        if normalization.sha256 != expected_normalization:
            raise ValueError("neural spectral comparison changed frozen normalization")
        backend_results = {}
        for sampler, classes in samplers.items():
            model_results = {}
            for cls, name in zip(
                classes,
                ("TypedBackgroundProbe", "V7PublishedConductanceProbe"),
                strict=True,
            ):
                kwargs = {"retinal_backend": backend}
                if "Conductance" in cls.__name__:
                    kwargs["normalization"] = normalization
                probe = cls(root, **kwargs)
                probe.baseline_frames = BASELINE_FRAMES
                nodes = np.flatnonzero(np.isin(probe.node_types, TARGET_TYPES))
                sides = probe._node_labels(
                    root / "data/raw/malecns-v1.0/body-annotations.feather", "somaSide"
                )
                if not np.all(np.isin(sides[nodes], ("L", "R"))):
                    raise ValueError("neural spectra require explicit target eye labels")
                if np.intersect1d(nodes, probe.retina.node_indices).size:
                    raise ValueError("neural targets overlap external receptor inputs")
                sham = _step_trace(probe, sham_stimulus, nodes)
                spectra, response_digest = {}, hashlib.sha256()
                for condition in ("right", "left", "down", "up"):
                    windows = []
                    input_digest = hashlib.sha256()
                    for phase in range(PERIOD):
                        stimulus = phase_grating(condition, phase)
                        trace = _step_trace(probe, stimulus, nodes)
                        if not np.array_equal(trace[:PRE_FRAMES], sham[:PRE_FRAMES]):
                            raise ValueError("neural spectral trial changed common gray prehistory")
                        windows.append(_windows(trace, sham))
                        response_digest.update(trace.tobytes())
                        sampled = np.stack(
                            [probe._sample_retina(frame) for frame in stimulus.frames]
                        )
                        codes = np.stack(
                            [
                                probe._retinal_code(frame, sampled[0], sampled[max(index - 1, 0)])
                                for index, frame in enumerate(sampled)
                            ]
                        )
                        input_digest.update(codes.tobytes())
                    key = f"{backend}:{sampler}:{condition}"
                    digest = input_digest.hexdigest()
                    if key in retinal_hashes and retinal_hashes[key] != digest:
                        raise ValueError("models disagree on receptor drive")
                    retinal_hashes[key] = digest
                    spectra[condition] = response_spectrum(np.stack(windows))
                summary = _population_summary(probe, nodes, sides, spectra)
                summary.update(
                    {
                        "response_sha256": response_digest.hexdigest(),
                        "sham_sha256": hashlib.sha256(sham.tobytes()).hexdigest(),
                        "sham_peak_to_peak_max": float(
                            np.max(np.ptp(sham[PRE_FRAMES + 2 * PERIOD :], axis=0))
                        ),
                        "maximum_parseval_error": float(
                            max(value["parseval_max_error"] for value in spectra.values())
                        ),
                    }
                )
                model_results[name] = summary
                t4 = summary["family_medians"]["T4"]
                print(
                    f"neural-spectra {backend} {sampler} {name}: "
                    f"T4_DC={t4['dc']['eight_population_median_contrast']:.6f}, "
                    f"T4_F1={t4['f1_power']['eight_population_median_contrast']:.6f}",
                    flush=True,
                )
            backend_results[sampler] = model_results
        paired = {
            name: _paired_sampler_summary(
                backend_results["nearest_reversed_axis"][name],
                backend_results["eye_centre_bilinear"][name],
            )
            for name in ("TypedBackgroundProbe", "V7PublishedConductanceProbe")
        }
        results[backend] = {
            "normalization_sha256": normalization.sha256,
            "samplers": backend_results,
            "paired_sampler_change": paired,
        }
    for backend in results.values():
        for name in ("TypedBackgroundProbe", "V7PublishedConductanceProbe"):
            first = backend["samplers"]["nearest_reversed_axis"][name]
            second = backend["samplers"]["eye_centre_bilinear"][name]
            if first["sham_sha256"] != second["sham_sha256"]:
                raise ValueError("sampler changed uniform gray neural trajectory")
    dependencies = {
        **controls["protocol"]["dependencies_sha256"],
        str(CONTROL_REFERENCE): _sha256(root / CONTROL_REFERENCE),
        str(GEOMETRY_REFERENCE): _sha256(root / GEOMETRY_REFERENCE),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
    }
    return {
        "protocol": {
            "name": "v7-frozen-neural-spectral-sampler-comparison-v1",
            "exploratory": True,
            "advance_allowed": False,
            "dependencies_sha256": dependencies,
            "samplers": list(samplers),
            "models": ["TypedBackgroundProbe", "V7PublishedConductanceProbe"],
            "retinal_backends": ["linear_luminance", "signed_frame_difference"],
            "phases": list(range(PERIOD)),
            "scored_cycles_zero_based": list(SCORED_CYCLES),
            "metrics": ["signed_dc", "f1_power", "total_power"],
            "activity_denominator_thresholds": [1e-6, 1e-4],
            "downstream_phase_reference_defined": False,
            "old_direction_gate_replaced": False,
            "parameter_fitting": False,
            "normalization_refit": False,
            "dynamics_changed": False,
            "physiological_labels_changed": False,
            "driving_data_used": False,
        },
        "retinal_drive_sha256": retinal_hashes,
        "results": results,
        "limitations": [
            "Post-inspection diagnostic on reused stimuli and models; not independent validation.",
            "DC, F1 and total-power results remain separate; no metric is selected by outcome.",
            "Power loses response sign and cannot establish ON/OFF specialization.",
            "No downstream input phase reference is defined; neural phase lag is not interpreted.",
            "Finite cycles and sham subtraction do not remove recurrent dynamics.",
            "Bilinear interpolation is an engineering comparator, not a measured receptive field.",
            "No old visual threshold is transferred and no default geometry or policy is modified.",
        ],
        "advance_to_central_complex": False,
    }
