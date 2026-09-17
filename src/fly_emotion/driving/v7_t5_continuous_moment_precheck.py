"""Tuning-only continuous input-moment diagnostic for T5 direction dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import _node_annotations
from fly_emotion.driving.v7_source_audit import CARDINAL_VECTORS
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE
from fly_emotion.driving.v7_t5_lamina_split import (
    IMPLEMENTATION as LAMINA_SPLIT_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-continuous-moment-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_continuous_moment_precheck.py")
RAW_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_raw.npz")
ANNOTATIONS = Path("data/raw/malecns-v1.0/body-annotations.feather")


def _row_matrix(rows: list[tuple[np.ndarray, np.ndarray]], node_count: int):
    row_indices: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_index, (nodes, weights) in enumerate(rows):
        row_indices.extend([row_index] * len(nodes))
        columns.extend(nodes.tolist())
        values.extend(weights.tolist())
    return sparse.csr_matrix(
        (np.asarray(values), (np.asarray(row_indices), np.asarray(columns))),
        shape=(len(rows), node_count),
        dtype=np.float64,
    )


def _centroid(types, sources, weights, positions, node_types) -> np.ndarray:
    keep = np.isin(node_types[sources], types) & np.all(np.isfinite(positions), axis=1)
    return (
        np.average(positions[keep], axis=0, weights=weights[keep])
        if np.any(keep)
        else np.full(2, np.nan)
    )


def _build_projection(root: Path, probe, config: dict, targets: np.ndarray) -> dict:
    raw = load_graph(root / "data/processed/malecns-v1.0", normalized=False).adjacency
    sides, coordinates = _node_annotations(root, probe)
    rows: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name in ("mass", "x_moment", "y_moment")
    }
    axes = []
    valid = []
    for target in targets:
        row = raw.getrow(int(target))
        sources = row.indices
        weights = np.abs(row.data).astype(np.float64)
        positions = coordinates[sources].copy()
        if sides[target] == "L":
            positions[:, 0] *= -1.0
        source_mask = np.isin(probe.node_types[sources], config["source_types"])
        source_mask &= np.all(np.isfinite(positions), axis=1)
        denominator = float(np.sum(weights[source_mask]))
        selected_nodes = sources[source_mask]
        normalized = (
            weights[source_mask] / denominator if denominator > 0.0 else weights[source_mask]
        )
        rows["mass"].append((selected_nodes, normalized))
        rows["x_moment"].append((selected_nodes, normalized * positions[source_mask, 0]))
        rows["y_moment"].append((selected_nodes, normalized * positions[source_mask, 1]))

        centroids = {
            source_type: _centroid(
                (source_type,), sources, weights, positions, probe.node_types
            )
            for source_type in ("Tm1", "Tm2", "Tm9")
        }
        axis = np.asarray(
            (
                centroids["Tm2"][0] - centroids["Tm9"][0],
                centroids["Tm9"][1] - centroids["Tm1"][1],
            )
        )
        norm = float(np.linalg.norm(axis))
        axis_valid = bool(np.isfinite(norm) and norm > 1e-12)
        axes.append(axis / norm if axis_valid else np.full(2, np.nan))
        valid.append(denominator > 0.0 and axis_valid)
    return {
        "matrices": {
            name: _row_matrix(entries, probe.graph.node_count)
            for name, entries in rows.items()
        },
        "axes": np.asarray(axes),
        "valid": np.asarray(valid, dtype=bool),
    }


def _moment_trace(probe, stimulus, projection: dict) -> np.ndarray:
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    baseline_image = stimulus.frames[0]
    baseline_values = probe._sample_retina(baseline_image)[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(baseline_values, baseline_values, baseline_values)
    for _ in range(probe.baseline_frames):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = baseline_drive
    baseline_state = state.astype(np.float64)

    previous_retina = baseline_values.copy()
    previous_mass = previous_x = previous_y = None
    projected = []
    matrices = projection["matrices"]
    axes = projection["axes"]
    for image in stimulus.frames[probe.baseline_frames :]:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous_retina)
        previous_retina = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        positive = np.maximum(state.astype(np.float64) - baseline_state, 0.0)
        mass = matrices["mass"] @ positive
        moment_x = matrices["x_moment"] @ positive
        moment_y = matrices["y_moment"] @ positive
        if previous_mass is not None:
            cross_x = moment_x * previous_mass - mass * previous_x
            cross_y = moment_y * previous_mass - mass * previous_y
            denominator = (
                np.abs(moment_x * previous_mass)
                + np.abs(mass * previous_x)
                + np.abs(moment_y * previous_mass)
                + np.abs(mass * previous_y)
                + 1e-9
            )
            projected.append((axes[:, 0] * cross_x + axes[:, 1] * cross_y) / denominator)
        previous_mass, previous_x, previous_y = mass, moment_x, moment_y
    return np.stack(projected)


def _reduce(values: np.ndarray, name: str) -> np.ndarray:
    if name == "signed_mean":
        return np.mean(values, axis=0)
    if name == "maximum":
        return np.max(values, axis=0)
    if name == "positive_mean":
        return np.mean(np.maximum(values, 0.0), axis=0)
    if name == "terminal_sum":
        return np.sum(values, axis=0)
    raise ValueError(f"unknown temporal reduction: {name}")


def _compact(score: dict) -> dict:
    return {
        key: score[key]
        for key in (
            "cell_count",
            "valid_cell_count",
            "valid_cell_fraction",
            "median_signed_contrast",
            "positive_cell_fraction",
            "passed",
        )
    }


def evaluate_v7_t5_continuous_moment_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    lamina_path = Path(config["lamina_split_protocol"])
    lamina = yaml.safe_load((root / lamina_path).read_text(encoding="utf-8"))
    lamina_evidence_path = Path(config["lamina_split_evidence"])
    lamina_evidence = json.loads((root / lamina_evidence_path).read_text(encoding="utf-8"))
    if not all(
        item["polarity"]["passing_condition_count"] == 3
        for item in lamina_evidence["population_consistency"].values()
    ):
        raise ValueError("continuous-moment precheck requires passed T5 OFF component")
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text(encoding="utf-8"))
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    source_path = Path(config["source_position_protocol"])
    condition = next(
        row for row in stage1["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("continuous-moment precheck may consume tuning only")

    probe = LaminaSplitProbe(root, lamina)
    populations = {
        f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    names = np.concatenate([np.full(len(nodes), name) for name, nodes in populations.items()])
    projection = _build_projection(root, probe, config, targets)
    structural = {}
    median_axes = {}
    for population, nodes in populations.items():
        group = np.flatnonzero(names == population)
        axes = projection["axes"][group]
        valid = projection["valid"][group]
        expected = CARDINAL_VECTORS[scoring["direction_populations"][population]]
        cosine = axes @ expected
        center = np.median(axes[valid], axis=0)
        center /= np.linalg.norm(center)
        gates = {
            "valid_cell_fraction": float(np.mean(valid))
            >= float(scoring["thresholds"]["minimum_valid_cell_fraction"]),
            "median_expected_cosine": float(np.median(cosine[valid]))
            >= float(scoring["thresholds"]["minimum_median_signed_contrast"]),
            "positive_alignment_fraction": float(np.mean(valid & (cosine > 0.0)))
            >= float(scoring["thresholds"]["minimum_positive_cell_fraction"]),
        }
        median_axes[population] = center
        structural[population] = {
            "cell_count": int(len(nodes)),
            "valid_axis_count": int(np.count_nonzero(valid)),
            "valid_axis_fraction": float(np.mean(valid)),
            "median_expected_cosine": float(np.median(cosine[valid])),
            "positive_alignment_fraction_all_cells": float(np.mean(valid & (cosine > 0.0))),
            "median_unit_axis": center.tolist(),
            "gates": gates,
            "passed": bool(all(gates.values())),
        }
    structural_mirror = {}
    for subtype in "abcd":
        left = median_axes[f"T5{subtype}_L"]
        right = median_axes[f"T5{subtype}_R"]
        reflected = np.asarray((-right[0], right[1]))
        error = float(np.linalg.norm(left - reflected))
        structural_mirror[subtype] = {
            "error": error,
            "passed": error
            <= float(scoring["thresholds"]["maximum_energy_weighted_mirror_error"]),
        }
    structural_gate = all(item["passed"] for item in structural.values()) and all(
        item["passed"] for item in structural_mirror.values()
    )

    local_path = Path(lamina["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text(encoding="utf-8"))
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    responses = {}
    for xi, center_x in enumerate(x_centers):
        for yi, center_y in enumerate(y_centers):
            for direction in ("left", "right", "up", "down"):
                stimulus = _local_step_edge(
                    float(center_x),
                    float(center_y),
                    float(local["stimulus"]["aperture_radius_pixels"]),
                    speed,
                    "off",
                    direction,
                    int(local["common_background_frames"]),
                )
                responses[(xi, yi, direction)] = _moment_trace(probe, stimulus, projection)
    reductions = {}
    for reduction in config["activity_moment"]["temporal_reductions"]:
        scores = {}
        for population, nodes in populations.items():
            group = np.flatnonzero(names == population)
            subtype, side = population[2], population[-1]
            preferred = (
                HORIZONTAL_PREFERENCE[(subtype, side)]
                if subtype in {"a", "b"}
                else VERTICAL_PREFERENCE[subtype]
            )

            def value(
                direction: str,
                *,
                selected_group: np.ndarray = group,
                selected_reduction: str = reduction,
            ) -> np.ndarray:
                grid = [
                    _reduce(responses[(xi, yi, direction)], selected_reduction)
                    for yi in range(len(y_centers))
                    for xi in range(len(x_centers))
                ]
                output = np.max(np.stack(grid), axis=0)[selected_group]
                output[~projection["valid"][selected_group]] = np.nan
                return output

            score = strict_contrast_summary(
                probe.graph.body_ids[nodes],
                value(preferred),
                value(OPPOSITE[preferred]),
                scoring["thresholds"],
            )
            scores[population] = _compact(score)
        bilateral = [
            subtype
            for subtype in "abcd"
            if scores[f"T5{subtype}_L"]["passed"]
            and scores[f"T5{subtype}_R"]["passed"]
        ]
        reductions[reduction] = {
            "direction_pass_count": int(sum(item["passed"] for item in scores.values())),
            "bilateral_direction_subtypes": bilateral,
            "population_scores": scores,
        }
    minimum_bilateral = int(
        config["stop_gate"]["minimum_bilateral_direction_subtypes_to_run_controls"]
    )
    control_eligible = any(
        len(result["bilateral_direction_subtypes"]) >= minimum_bilateral
        for result in reductions.values()
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(lamina_path): _sha256(root / lamina_path),
                str(LAMINA_SPLIT_IMPLEMENTATION): _sha256(root / LAMINA_SPLIT_IMPLEMENTATION),
                str(lamina_evidence_path): _sha256(root / lamina_evidence_path),
                str(stage1_path): _sha256(root / stage1_path),
                str(scoring_path): _sha256(root / scoring_path),
                str(source_path): _sha256(root / source_path),
                str(local_path): _sha256(root / local_path),
                str(RAW_ADJACENCY): _sha256(root / RAW_ADJACENCY),
                str(ANNOTATIONS): _sha256(root / ANNOTATIONS),
            },
            "condition_id": config["condition_id"],
            "position_count": int(len(x_centers) * len(y_centers)),
            "stimulus_count": len(responses),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "structural_axis": {
            "definition": config["structural_axis"],
            "population_results": structural,
            "mirror": structural_mirror,
            "all_population_and_mirror_gates_passed": bool(structural_gate),
        },
        "ordered_activity_moment": reductions,
        "control_eligibility_gate_passed": bool(control_eligible),
        "controls_requested": config["controls_after_ordered_gate"],
        "controls_evaluated": False,
        "three_condition_evaluation_performed": False,
        "advance_to_three_tuning_conditions": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            None
            if control_eligible
            else "continuous_moment_produced_no_bilateral_direction_selective_T5_subtype"
        ),
        "boundary": config["boundary"],
    }
