from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE, VisualStimulus
from fly_emotion.driving.v7_branched import SOURCE_AUDIT
from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    V7PublishedConductanceProbe,
    _collect_source_normalization,
)
from fly_emotion.driving.v7_phase_motion import (
    BASELINE_FRAMES,
    CONDITIONS,
    CYCLES,
    OPPOSITE,
    PERIOD,
    PRE_FRAMES,
    SCORED_CYCLES,
    cycle_energy,
    motion_controls,
    phase_average_contrast,
    phase_grating,
)
from fly_emotion.driving.v7_source_audit import (
    CARDINAL_VECTORS,
    _load_tables,
    _t4_records,
    _weighted_centroid,
)
from fly_emotion.driving.v7_stability import TypedBackgroundProbe
from fly_emotion.driving.v7_temporal_audit import _step_trace

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_geometry_sign.py")
REFERENCE = Path("artifacts/v7-phase-motion.json")


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def reverse_eye_coordinates(
    u: np.ndarray, v: np.ndarray, side: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    horizontal = np.where(side < 0, 0.5, 1.5) - u.astype(np.float64)
    return horizontal.astype(np.float32), (1.0 - v.astype(np.float64)).astype(np.float32)


class MotionSignGeometry:
    def _nested_t4_retinal_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        u, v = super()._nested_t4_retinal_coordinates()
        self.legacy_coordinates = (u.copy(), v.copy())
        return reverse_eye_coordinates(u, v, self.retina.side)


class MotionSignTypedProbe(MotionSignGeometry, TypedBackgroundProbe):
    pass


class MotionSignConductanceProbe(MotionSignGeometry, V7PublishedConductanceProbe):
    pass


def anatomical_sign_audit(records: list[dict], transforms: dict) -> dict:
    groups = {}
    for eye in ("L", "R"):
        rows = [record for record in records if record["side"] == eye]
        expected = np.stack([CARDINAL_VECTORS[row["expected"]] for row in rows])
        centre = np.stack([_weighted_centroid(row, ("Mi1",)) for row in rows])
        proximal = np.stack([_weighted_centroid(row, ("Mi4", "C3")) for row in rows])
        distal = np.stack([_weighted_centroid(row, ("Mi9",)) for row in rows])
        axes = {}
        for name, offset in (
            ("centre_to_proximal", proximal - centre),
            ("distal_to_centre", centre - distal),
        ):
            projected = offset @ np.asarray(transforms[eye])
            valid = np.all(np.isfinite(projected), axis=1) & (
                np.linalg.norm(projected, axis=1) > 1e-12
            )
            alignment = np.full(len(rows), np.nan)
            alignment[valid] = np.sum(projected[valid] * expected[valid], axis=1) / np.linalg.norm(
                projected[valid], axis=1
            )
            axes[name] = {
                "valid_cells": int(valid.sum()),
                "missing_or_zero_cells": int((~valid).sum()),
                "legacy_cosine": [
                    float(value) if np.isfinite(value) else None for value in alignment
                ],
                "reversed_cosine": [
                    float(-value) if np.isfinite(value) else None for value in alignment
                ],
                "legacy_positive_fraction_all_cells": float(np.mean(alignment > 0)),
                "reversed_positive_fraction_all_cells": float(np.mean(alignment < 0)),
            }
        groups[eye] = {
            "cells": len(rows),
            "body_ids": [row["body_id"] for row in rows],
            "axes": axes,
        }
    return {
        "groups": groups,
        "legacy_transforms_by_eye": transforms,
        "reversed_transforms_by_eye": {
            eye: (-np.asarray(transform)).tolist() for eye, transform in transforms.items()
        },
        "legacy_fit_vector": "centroid(Mi1) - centroid(Mi4+C3)",
        "preferred_motion_vector": "centroid(Mi4+C3) - centroid(Mi1)",
        "preferred_stimulus_order": ["distal Mi9", "central Mi1/Tm3", "proximal Mi4/C3"],
        "literature": {
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8891015/",
            "doi": "10.1038/s41586-022-04428-3",
            "title": "A biophysical account of multiplication by a single neuron",
            "location": "Main text following Fig. 1; Fig. 3 input-sequence caption",
            "distal_quote": (
                "where stimuli moving in the T4 cell’s PD first affect its membrane potential"
            ),
        },
        "scope": (
            "reused anatomy, frozen Mi1 versus Mi4+C3 selection; no independent axis validation"
        ),
    }


def _pixel_indices(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack(
        (np.clip(np.rint(u * 47), 0, 47), np.clip(np.rint(v * 23), 0, 23)), axis=1
    ).astype(np.int32)


def geometry_record(probe) -> dict:
    u, v = probe.legacy_coordinates
    reversed_u, reversed_v = probe.retinal_u, probe.retinal_v
    legacy = _pixel_indices(u, v)
    reversed_pixels = _pixel_indices(reversed_u, reversed_v)
    ideal_pixel_mirror = legacy.copy()
    ideal_pixel_mirror[:, 0] = np.where(probe.retina.side < 0, 23, 71) - legacy[:, 0]
    ideal_pixel_mirror[:, 1] = 23 - legacy[:, 1]
    return {
        "receptor_body_ids": probe.retina.body_ids.tolist(),
        "side": probe.retina.side.tolist(),
        "legacy_uv": np.stack((u, v), axis=1).tolist(),
        "reversed_uv": np.stack((reversed_u, reversed_v), axis=1).tolist(),
        "legacy_pixels": legacy.tolist(),
        "reversed_pixels": reversed_pixels.tolist(),
        "receptors": probe.retina.size,
        "changed_pixel_fraction": float(np.mean(np.any(legacy != reversed_pixels, axis=1))),
        "difference_from_discrete_half_image_mirror_cells": int(
            np.count_nonzero(np.any(reversed_pixels != ideal_pixel_mirror, axis=1))
        ),
        "left_half_rounds_to_right_half_cells": int(
            np.count_nonzero((probe.retina.side < 0) & (reversed_pixels[:, 0] >= 24))
        ),
        "sampling": "unchanged round(u*(width-1)), round(v*(height-1)); width=48, height=24",
        "rounding_scope": "continuous per-eye reversal is not exact discrete half-image reversal",
    }


def _summarize(probe, nodes: np.ndarray, sides: np.ndarray, energies: dict, original: dict) -> dict:
    populations = {}
    for kind in np.unique(probe.node_types[nodes]):
        for eye in ("L", "R"):
            key = f"{kind}_{eye}"
            mask = (probe.node_types[nodes] == kind) & (sides[nodes] == eye)
            subtype = kind[-1]
            preferred = (
                HORIZONTAL_PREFERENCE[(subtype, eye)]
                if subtype in "ab"
                else VERTICAL_PREFERENCE[subtype]
            )
            p = energies[preferred][:, :, mask].mean(axis=0)
            o = energies[OPPOSITE[preferred]][:, :, mask].mean(axis=0)
            contrast = phase_average_contrast(p, o)
            cycles = (p - o) / (p + o + 1e-12)
            old = original["populations"][key]
            ids = probe.graph.body_ids[nodes[mask]].tolist()
            if ids != old["body_ids"] or preferred != old["expected_direction"]:
                raise ValueError("geometry comparison changed target IDs or physiological labels")
            old_contrast = phase_average_contrast(
                old["per_cell_preferred_cycle_means"], old["per_cell_opposite_cycle_means"]
            )
            populations[key] = {
                "cells": int(mask.sum()),
                "body_ids": ids,
                "expected_direction": preferred,
                "per_cell_preferred_cycle_means": p.tolist(),
                "per_cell_opposite_cycle_means": o.tolist(),
                "median_direction_contrast": float(np.median(contrast)),
                "direction_positive_fraction": float(np.mean(contrast > 0)),
                "cycle_median_direction_contrasts": np.median(cycles, axis=1).tolist(),
                "median_absolute_cycle_contrast_change": float(
                    np.median(np.abs(cycles[1] - cycles[0]))
                ),
                "fraction_response_sum_above_1e_6": float(
                    np.mean(p.mean(axis=0) + o.mean(axis=0) > 1e-6)
                ),
                "median_static_response": float(
                    np.median(
                        energies["static_x" if subtype in "ab" else "static_y"][:, :, mask].mean(
                            axis=(0, 1)
                        )
                    )
                ),
                "legacy_median_direction_contrast": old["median_direction_contrast"],
                "paired_median_contrast_change": float(np.median(contrast - old_contrast)),
                "median_absolute_residual_to_exact_sign_flip": float(
                    np.median(np.abs(contrast + old_contrast))
                ),
            }
    return {
        "populations": populations,
        "target_count": len(nodes),
        **{
            f"{family}_population_median": float(
                np.median(
                    [
                        value["median_direction_contrast"]
                        for key, value in populations.items()
                        if key.startswith(family)
                    ]
                )
            )
            for family in ("T4", "T5")
        },
    }


def evaluate_v7_geometry_sign(root: Path) -> dict:
    original = json.loads((root / REFERENCE).read_text())
    dependencies = original["protocol"]["dependencies_sha256"]
    for path, expected in dependencies.items():
        if _sha256(root / path) != expected:
            raise ValueError(f"geometry comparison has stale phase-motion dependency: {path}")
    controls = motion_controls()
    if not controls["controls_pass"]:
        raise ValueError("phase-motion assay failed synthetic controls")
    config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text())
    audit = json.loads((root / SOURCE_AUDIT).read_text())
    calibration = audit["frozen_axis_calibration"]
    if calibration["direct_sources"] != ["Mi1"] or set(calibration["offset_sources"]) != {
        "Mi4",
        "C3",
    }:
        raise ValueError("anatomical sign convention requires frozen Mi1 versus Mi4+C3 sources")
    graph, annotations, _ = _load_tables(root)
    records = _t4_records(graph, annotations, ("Mi1", "Mi4", "C3", "Mi9"))
    anatomy = anatomical_sign_audit(records, calibration["transforms_by_eye"])
    del graph, annotations, records
    results = {}
    input_hashes = {}
    coordinates = None
    sham_stimulus = VisualStimulus(
        "phase_gray_sham",
        "sham",
        "mixed",
        "none",
        np.full((PRE_FRAMES + CYCLES * PERIOD, 24, 48), 0.5, dtype=np.float32),
    )
    for backend in ("linear_luminance", "signed_frame_difference"):
        normalization = _collect_source_normalization(
            root,
            retinal_backend=backend,
            retinal_geometry=config["primary_retinal_geometry"],
            config=config,
        )
        if normalization.sha256 != original["results"][backend]["normalization_sha256"]:
            raise ValueError("geometry comparison changed frozen source normalization")
        models = {}
        for cls, legacy_name in (
            (MotionSignTypedProbe, "TypedBackgroundProbe"),
            (MotionSignConductanceProbe, "V7PublishedConductanceProbe"),
        ):
            kwargs = {"retinal_backend": backend}
            if cls is MotionSignConductanceProbe:
                kwargs["normalization"] = normalization
            probe = cls(root, **kwargs)
            probe.baseline_frames = BASELINE_FRAMES
            current_geometry = geometry_record(probe)
            if coordinates is not None and current_geometry != coordinates:
                raise ValueError("models disagree on retinal geometry")
            coordinates = current_geometry
            nodes = np.flatnonzero(
                np.isin(probe.node_types, ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"))
            )
            sides = probe._node_labels(
                root / "data/raw/malecns-v1.0/body-annotations.feather", "somaSide"
            )
            if (
                not np.all(np.isin(sides[nodes], ("L", "R")))
                or np.intersect1d(nodes, probe.retina.node_indices).size
            ):
                raise ValueError("target readout must retain eye labels and exclude receptors")
            sham = _step_trace(probe, sham_stimulus, nodes)
            energies = {}
            digest = hashlib.sha256()
            for condition in CONDITIONS:
                values = []
                for phase in range(PERIOD):
                    stimulus = phase_grating(condition, phase)
                    if stimulus.sha256 != original["stimulus_sha256"][f"{condition}:{phase}"]:
                        raise ValueError("geometry comparison changed stimulus")
                    trace = _step_trace(probe, stimulus, nodes)
                    if not np.array_equal(trace[:PRE_FRAMES], sham[:PRE_FRAMES]):
                        raise ValueError("geometry trial changed common gray prehistory")
                    values.append(cycle_energy(trace, sham))
                    digest.update(trace.tobytes())
                    sampled = np.stack([probe._sample_retina(frame) for frame in stimulus.frames])
                    codes = np.stack(
                        [
                            probe._retinal_code(frame, sampled[0], sampled[max(i - 1, 0)])
                            for i, frame in enumerate(sampled)
                        ]
                    )
                    key = f"{backend}:{condition}:{phase}"
                    code_hash = hashlib.sha256(codes.tobytes()).hexdigest()
                    if key in input_hashes and input_hashes[key] != code_hash:
                        raise ValueError("geometry candidates disagree on retinal drive")
                    input_hashes[key] = code_hash
                energies[condition] = np.stack(values)
            models[legacy_name] = {
                **_summarize(
                    probe,
                    nodes,
                    sides,
                    energies,
                    original["results"][backend]["models"][legacy_name],
                ),
                "response_sha256": digest.hexdigest(),
                "sham_peak_to_peak_max": float(
                    np.max(np.ptp(sham[PRE_FRAMES + 2 * PERIOD :], axis=0))
                ),
            }
            if (
                models[legacy_name]["sham_peak_to_peak_max"]
                != original["results"][backend]["models"][legacy_name]["sham_peak_to_peak_max"]
            ):
                raise ValueError("geometry changed uniform-background dynamics")
            print(
                f"geometry-sign {backend} {legacy_name}: "
                f"T4={models[legacy_name]['T4_population_median']:.6f}, "
                f"T5={models[legacy_name]['T5_population_median']:.6f}",
                flush=True,
            )
        results[backend] = {"normalization_sha256": normalization.sha256, "models": models}
    dependencies = {
        **dependencies,
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(REFERENCE): _sha256(root / REFERENCE),
    }
    return {
        "protocol": {
            "name": "v7-anatomical-motion-sign-v1",
            "exploratory": True,
            "advance_allowed": False,
            "dependencies_sha256": dependencies,
            "phase_motion_protocol": original["protocol"],
            "reference_results": str(REFERENCE),
            "reference_reuse": (
                "all recorded dependency hashes verified; no legacy result relabeling"
            ),
            "geometry_operation": (
                "u'=0.5-u (L), u'=1.5-u (R), v'=1-v; equivalent to negating both frozen axes "
                "before min-max normalization, up to float32 rounding"
            ),
            "normalization_refit": False,
            "dynamics_changed": False,
            "physiological_labels_changed": False,
            "parameter_fitting": False,
            "driving_data_used": False,
            "scored_cycles_zero_based": list(SCORED_CYCLES),
        },
        "anatomy": anatomy,
        "geometry": coordinates,
        "synthetic_controls": controls,
        "stimulus_sha256": original["stimulus_sha256"],
        "retinal_drive_sha256": input_hashes,
        "results": results,
        "limitations": [
            "Sign selected from anatomical stimulus order after inspecting prior failures; "
            "not a pristine final test.",
            "Frozen subtype-based axis calibration is not independent validation of retinotopy.",
            "Continuous coordinate reversal preserves historical min-max scaling and rounding, "
            "including midpoint boundary ambiguity.",
            "Reused centroids exclude unknown coordinates; all target IDs and missing/zero-vector "
            "denominators retained.",
            "Geometry also affects T5 input but the sign rationale comes from T4 anatomy, "
            "not T5 functional validation.",
            "Phase-balanced contrast at one frequency cannot establish ON/OFF, looming, mirror "
            "or topology gates.",
            "Gray dynamics and frozen normalization are unchanged; no learning or deployment.",
        ],
        "advance_to_central_complex": False,
    }
