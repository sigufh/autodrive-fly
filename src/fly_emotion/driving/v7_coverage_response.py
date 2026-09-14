from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7 import (
    HORIZONTAL_PREFERENCE,
    VERTICAL_PREFERENCE,
    V7Contract,
    build_controlled_stimuli,
)
from fly_emotion.driving.v7_branched import V7BranchedT4Probe

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_coverage_response.py")
COVERAGE_REPORT = Path("artifacts/v7-t4-input-coverage.json")
REFERENCE_REPORT = Path("artifacts/v7-branched-t4-candidate.json")
BINS = ("absent", "unknown_only", "zero_covered", "below_half", "at_least_half")
CATEGORIES = ("covered", "uncovered", "missing_coordinate", "missing_side")
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}


def coverage_bins(weights: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = {name: np.asarray(weights[name], dtype=np.int64) for name in CATEGORIES}
    if len({value.shape for value in values.values()}) != 1 or any(
        value.ndim != 1 or np.any(value < 0) for value in values.values()
    ):
        raise ValueError("coverage weights must be aligned nonnegative vectors")
    total = sum(values.values())
    covered = values["covered"]
    unknown = values["missing_coordinate"] + values["missing_side"]
    labels = np.full(len(total), "absent", dtype=object)
    present = total > 0
    labels[present & (unknown == total)] = "unknown_only"
    labels[present & (covered == 0) & (unknown < total)] = "zero_covered"
    labels[(covered > 0) & (2 * covered < total)] = "below_half"
    labels[(covered > 0) & (2 * covered >= total)] = "at_least_half"
    fraction = np.divide(covered, total, out=np.zeros(len(total), dtype=float), where=present)
    return labels, fraction, total


def align_by_body_id(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    if len(np.unique(source_ids)) != len(source_ids) or len(np.unique(target_ids)) != len(
        target_ids
    ):
        raise ValueError("body IDs must be unique")
    order = np.argsort(source_ids)
    indices = np.searchsorted(source_ids[order], target_ids)
    if np.any(indices >= len(source_ids)):
        raise ValueError("missing response body IDs")
    if not np.array_equal(source_ids[order[indices]], target_ids):
        raise ValueError("missing response body IDs")
    return order[indices]


def summarize_direction(
    preferred: np.ndarray, opposite: np.ndarray, off: np.ndarray, mask: np.ndarray
) -> dict:
    count = int(np.count_nonzero(mask))
    if not count:
        return {
            "cells": 0,
            "median_direction_contrast": None,
            "direction_positive_fraction": None,
            "median_on_off_contrast": None,
            "median_response_sum": None,
            "fraction_direction_pair_above_1e_6": None,
        }
    p, o, n = preferred[mask], opposite[mask], off[mask]
    contrast = (p - o) / (np.abs(p) + np.abs(o) + 1e-12)
    polarity = (p - n) / (np.abs(p) + np.abs(n) + 1e-12)
    return {
        "cells": count,
        "median_direction_contrast": float(np.median(contrast)),
        "direction_positive_fraction": float(np.mean(contrast > 0)),
        "median_on_off_contrast": float(np.median(polarity)),
        "median_response_sum": float(np.median(np.abs(p) + np.abs(o))),
        "fraction_direction_pair_above_1e_6": float(np.mean(np.abs(p) + np.abs(o) > 1e-6)),
    }


def evaluate_v7_coverage_response(root: Path) -> dict:
    contract = V7Contract.load(root)
    coverage = json.loads((root / COVERAGE_REPORT).read_text())
    dependency_hashes = dict(coverage["protocol"]["dependencies_sha256"])
    for relative, expected in dependency_hashes.items():
        with (root / relative).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"stale coverage dependency: {relative}")
    reference = json.loads((root / REFERENCE_REPORT).read_text())
    reference_dependencies = {
        "v7_config_sha256": "configs/driving-v7.yaml",
        "v7_implementation_sha256": "src/fly_emotion/driving/v7.py",
        "candidate_config_sha256": "configs/driving-v7-branched-t4.yaml",
        "candidate_implementation_sha256": "src/fly_emotion/driving/v7_branched.py",
        "source_audit_sha256": "artifacts/v7-t4-source-audit.json",
    }
    for key, relative in reference_dependencies.items():
        with (root / relative).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if reference["protocol"].get(key) != actual:
            raise ValueError(f"stale response reference: {relative}")
    targets = coverage["targets"]
    ids = np.asarray(targets["body_ids"], dtype=np.int64)
    populations = np.asarray(
        [
            f"{kind}_{'L' if side == -1 else 'R' if side == 1 else 'unknown'}"
            for kind, side in zip(targets["types"], targets["sides"], strict=True)
        ]
    )
    source_weights = targets["synapse_weights_by_source_and_coverage"]
    groups = {source: [source] for source in source_weights} | coverage["branches"]
    strata = {}
    for group, members in groups.items():
        weights = {
            category: sum(
                np.asarray(source_weights[member][category], dtype=np.int64) for member in members
            )
            for category in CATEGORIES
        }
        labels, fraction, total = coverage_bins(weights)
        strata[group] = {"labels": labels, "fraction": fraction, "total": total}
    visual = contract.payload["controlled_vision"]
    stimuli = [
        item
        for item in build_controlled_stimuli(
            width=visual["width"], height=visual["height"], frames=visual["frames_per_stimulus"]
        )
        if item.family == "moving_edge"
    ]
    results = {}
    for backend in ("linear_luminance", "signed_frame_difference"):
        probe = V7BranchedT4Probe(
            root,
            retinal_backend=backend,
            retinal_geometry="nested_t4_axis_v1",
            brain_substeps=4,
            baseline_frames=8,
        )
        responses = {stimulus.name: probe.run(stimulus) for stimulus in stimuli}
        preferred = np.zeros(len(ids))
        opposite = np.zeros(len(ids))
        off = np.zeros(len(ids))
        assigned = np.zeros(len(ids), dtype=np.int32)
        replay_checks = {}
        baseline = reference["retinal_results"][f"nested_t4_axis_v1:{backend}"]
        stimulus_replay = {
            name: all(
                response[key] == baseline["responses"][name][key]
                for key in ("stimulus_sha256", "retinal_drive_sha256")
            )
            for name, response in responses.items()
        }
        if not all(stimulus_replay.values()):
            raise ValueError("stimulus or retinal drive differs from frozen reference")
        for population in np.unique(populations):
            if population not in probe.populations:
                raise ValueError(f"missing runtime population {population}")
            mask = populations == population
            response_ids = probe.graph.body_ids[probe.populations[population]]
            permutation = align_by_body_id(response_ids, ids[mask])
            subtype, side = population[2], population[-1]
            direction = (
                HORIZONTAL_PREFERENCE[(subtype, side)]
                if subtype in "ab"
                else VERTICAL_PREFERENCE[subtype]
            )
            for output, name in (
                (preferred, f"on_edge_{direction}"),
                (opposite, f"on_edge_{OPPOSITE[direction]}"),
                (off, f"off_edge_{direction}"),
            ):
                output[mask] = np.asarray(
                    responses[name]["_population_neuron_positive_mean"][population]
                )[permutation]
            assigned[mask] += 1
            contrast = summarize_direction(preferred, opposite, off, mask)[
                "median_direction_contrast"
            ]
            expected = baseline["scores"]["direction_selectivity"][population]["contrast"]
            replay_checks[population] = bool(np.isclose(contrast, expected, atol=1e-12, rtol=1e-10))
        if not np.all(assigned == 1) or not all(replay_checks.values()):
            raise ValueError("T4 alignment or frozen response replay failed")
        bin_reports = {}
        for group, info in strata.items():
            bin_reports[group] = {
                population: {
                    label: summarize_direction(
                        preferred,
                        opposite,
                        off,
                        (populations == population) & (info["labels"] == label),
                    )
                    for label in BINS
                }
                for population in np.unique(populations)
            }
        results[backend] = {
            "replay_checks": replay_checks,
            "stimulus_replay_checks": stimulus_replay,
            "population_median_direction_contrast": float(
                np.median(
                    [
                        summarize_direction(preferred, opposite, off, populations == population)[
                            "median_direction_contrast"
                        ]
                        for population in np.unique(populations)
                    ]
                )
            ),
            "all_populations": {
                population: summarize_direction(preferred, opposite, off, populations == population)
                for population in np.unique(populations)
            },
            "by_source_coverage": bin_reports,
            "all_cells": summarize_direction(
                preferred, opposite, off, np.ones(len(ids), dtype=bool)
            ),
            "per_cell": {
                "preferred_on_response": preferred.tolist(),
                "opposite_on_response": opposite.tolist(),
                "preferred_off_response": off.tolist(),
                "branched_equation_applied": np.isin(
                    ids, probe.graph.body_ids[probe.branched["targets"]]
                ).tolist(),
            },
            "stimuli": {
                name: {
                    "sha256": response["stimulus_sha256"],
                    "retinal_drive_sha256": response["retinal_drive_sha256"],
                }
                for name, response in responses.items()
            },
        }
    dependencies = [
        IMPLEMENTATION,
        COVERAGE_REPORT,
        REFERENCE_REPORT,
        Path("src/fly_emotion/driving/v7_branched.py"),
        Path("configs/driving-v7-branched-t4.yaml"),
        Path("src/fly_emotion/driving/v7_source_audit.py"),
        Path("configs/driving-v7-t4-source-audit.yaml"),
        Path("artifacts/v7-t4-source-audit.json"),
        Path("src/fly_emotion/driving/engine.py"),
        Path("data/raw/malecns-v1.0/body-neurotransmitters.feather"),
        Path("data/processed/malecns-v1.0/adjacency_target_norm.npz"),
    ]
    for path in dependencies:
        with (root / path).open("rb") as source:
            dependency_hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-t4-coverage-response-v1",
            "exploratory": True,
            "advance_allowed": False,
            "dependencies_sha256": dependency_hashes,
            "geometry": "nested_t4_axis_v1",
            "dynamics": "columnar-branched-t4-v2",
            "strata": (
                "covered=0, 0<covered/total<0.5, covered/total>=0.5; "
                "unknown-only and absent separate"
            ),
            "unknown_weights_in_denominator": True,
            "cells_filtered_from_overall_result": False,
            "direction_score": "unchanged whole-window mean positive response contrast",
            "activity_threshold_role": "descriptive only; no silent cells removed",
            "prior_data_exposure": (
                "same cells and edge battery used previously; not independent validation"
            ),
            "parameter_fitting": False,
            "driving_data_used": False,
        },
        "targets": {"body_ids": ids.tolist(), "populations": populations.tolist()},
        "coverage_strata": {
            group: {
                "labels": info["labels"].tolist(),
                "total_synapse_weight": info["total"].tolist(),
                "covered_fraction": [
                    float(value) if total else None
                    for value, total in zip(info["fraction"], info["total"], strict=True)
                ],
            }
            for group, info in strata.items()
        },
        "results": results,
        "limitations": [
            "Coverage is anatomical association, not intervention or source functional status.",
            "Within-subtype and eye comparisons must accompany any pooled contrast.",
            "Original direction scores retain known finite-window confounds.",
            "All-cell median and median of population medians are distinct summaries.",
            "A covered subset cannot authorize deployment; original all-cell gates remain failed.",
        ],
        "advance_to_central_complex": False,
    }
