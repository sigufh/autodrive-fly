"""Read-only local-edge direction audit of real T4 source pools."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import (
    _infer_t4_t5_positions,
    _node_annotations,
)
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4_source_resolved import _frozen_axis_coordinates
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    CONFIG as LOCAL_EDGE_CONFIG,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    IMPLEMENTATION as LOCAL_EDGE_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    OPPOSITE,
    _local_edge,
)

CONFIG = Path("configs/driving-v7-t4-source-pool-local.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_source_pool_local.py")


def _projection_rows(
    probe: MassBalancedVisualProbe,
    targets: np.ndarray,
    source_types: tuple[str, ...],
    *,
    coordinates: np.ndarray | None = None,
    moment: int | None = None,
) -> sparse.csr_matrix:
    matrix = (
        probe.adjacency[targets, :]
        .multiply(
            (
                np.isin(probe.node_types, source_types)
                & (np.all(np.isfinite(coordinates), axis=1) if coordinates is not None else True)
            ).astype(np.float32)
        )
        .tocsr()
    )
    totals = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    scale = np.zeros_like(totals, dtype=np.float64)
    np.divide(1.0, totals, out=scale, where=totals > 0.0)
    normalized = (sparse.diags(scale) @ matrix).tocsr()
    if moment is not None:
        if coordinates is None:
            raise ValueError("coordinate moment requires source coordinates")
        normalized = normalized.multiply(coordinates[:, moment]).tocsr()
    return normalized


def _projected_traces(probe, stimulus, matrices: dict[str, sparse.csr_matrix]) -> dict:
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    baseline_values = probe._sample_retina(stimulus.frames[0])[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(baseline_values, baseline_values, baseline_values)
    for _ in range(probe.baseline_frames):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = baseline_drive
    baselines = {name: matrix @ state.astype(np.float64) for name, matrix in matrices.items()}
    output = {name: [] for name in matrices}
    previous = baseline_values.copy()
    for image in stimulus.frames:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous)
        previous = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        state64 = state.astype(np.float64)
        for name, matrix in matrices.items():
            output[name].append(matrix @ state64 - baselines[name])
    return {name: np.stack(values) for name, values in output.items()}


def _compact(score: dict) -> dict:
    return {
        key: score[key]
        for key in (
            "cell_count",
            "valid_cell_count",
            "valid_cell_fraction",
            "median_signed_contrast",
            "positive_cell_fraction",
            "invalid_cell_ids",
            "gates",
            "passed",
        )
    }


def evaluate_v7_t4_source_pool_local(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    local = yaml.safe_load((root / LOCAL_EDGE_CONFIG).read_text())
    source_path = Path(config["source_protocol"])
    source = yaml.safe_load((root / source_path).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    audit_path = Path(config["source_audit_evidence"])
    audit = json.loads((root / audit_path).read_text())
    probe = MassBalancedVisualProbe(root, config)
    populations = {
        f"T4{subtype}_{side}": probe.populations[f"T4{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    target_population = np.concatenate(
        [np.full(len(nodes), name) for name, nodes in populations.items()]
    )
    target_ids = probe.graph.body_ids[targets]
    matrices = {
        name: _projection_rows(probe, targets, tuple(types))
        for name, types in config["source_groups"].items()
    }
    source_sides, source_coordinates = _node_annotations(root, probe)
    source_coordinates[source_sides == "L", 0] *= -1.0
    for name, types in config["source_groups"].items():
        matrices[f"pair_{name}"] = _projection_rows(
            probe, targets, tuple(types), coordinates=source_coordinates
        )
        for moment, suffix in ((0, "x"), (1, "y")):
            matrices[f"pair_{name}_{suffix}"] = _projection_rows(
                probe,
                targets,
                tuple(types),
                coordinates=source_coordinates,
                moment=moment,
            )
    positions, _ = _infer_t4_t5_positions(root, probe, source)
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = finite.min(axis=0), finite.max(axis=0)
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    camera = (positions[targets] - low) / np.maximum(high - low, 1e-12) * (47, 23)
    assigned = np.stack(
        (
            np.argmin(np.abs(x_centers[:, None] - camera[:, 0]), axis=0),
            np.argmin(np.abs(y_centers[:, None] - camera[:, 1]), axis=0),
        ),
        axis=1,
    )
    coordinates = _frozen_axis_coordinates(
        root, probe, audit["frozen_axis_calibration"]["transforms_by_eye"]
    )
    orientation = np.zeros(len(targets), dtype=np.float64)
    for index, target in enumerate(targets):
        row = probe.adjacency.getrow(int(target))
        sources, weights = row.indices, np.abs(row.data).astype(np.float64)
        horizontal = coordinates[sources, 0]
        centroids = []
        for types in (("Mi1", "Tm3"), ("Mi4", "C3")):
            keep = np.isin(probe.node_types[sources], types) & np.isfinite(horizontal)
            centroids.append(
                float(np.average(horizontal[keep], weights=weights[keep]))
                if np.any(keep)
                else np.nan
            )
        orientation[index] = np.sign(centroids[0] - centroids[1])
    responses = {}
    duration = int(
        next(
            row
            for row in yaml.safe_load((root / local["typed_stimulus_protocol"]).read_text())[
                "conditions"
            ]
            if row["condition_id"] == config["condition_id"]
        )["duration_frames"]
    )
    for x_index, center_x in enumerate(x_centers):
        for y_index, center_y in enumerate(y_centers):
            for direction in ("left", "right", "up", "down"):
                stimulus = _local_edge(
                    duration,
                    float(center_x),
                    float(center_y),
                    float(local["stimulus"]["aperture_radius_pixels"]),
                    float(local["stimulus"]["bar_half_width_pixels"]),
                    "on",
                    direction,
                    int(local["common_background_frames"]),
                )
                traces = _projected_traces(probe, stimulus, matrices)
                previous_center = np.concatenate(
                    (np.zeros_like(traces["center"][:1]), traces["center"][:-1])
                )
                previous_proximal = np.concatenate(
                    (np.zeros_like(traces["proximal"][:1]), traces["proximal"][:-1])
                )
                forward = traces["proximal"] * previous_center
                reverse = traces["center"] * previous_proximal
                correlation = (
                    orientation[None]
                    * (forward - reverse)
                    / (np.abs(forward) + np.abs(reverse) + 1e-6)
                )
                pairwise_vectors = {}
                for delayed_name in ("proximal", "distal"):
                    pair_center = traces["pair_center"]
                    pair_delayed = traces[f"pair_{delayed_name}"]
                    pair_center_previous = np.concatenate(
                        (np.zeros_like(pair_center[:1]), pair_center[:-1])
                    )
                    delayed_previous = np.concatenate(
                        (
                            np.zeros_like(pair_delayed[:1]),
                            pair_delayed[:-1],
                        )
                    )
                    components = []
                    denominators = []
                    for suffix in ("x", "y"):
                        center_moment = traces[f"pair_center_{suffix}"]
                        delayed_moment = traces[f"pair_{delayed_name}_{suffix}"]
                        center_moment_previous = np.concatenate(
                            (np.zeros_like(center_moment[:1]), center_moment[:-1])
                        )
                        delayed_moment_previous = np.concatenate(
                            (np.zeros_like(delayed_moment[:1]), delayed_moment[:-1])
                        )
                        pair_forward = (
                            delayed_moment * pair_center_previous
                            - pair_delayed * center_moment_previous
                        )
                        pair_reverse = (
                            delayed_moment_previous * pair_center - delayed_previous * center_moment
                        )
                        components.append(pair_forward - pair_reverse)
                        denominators.append(np.abs(pair_forward) + np.abs(pair_reverse))
                    denominator = denominators[0] + denominators[1] + 1e-6
                    pairwise_vectors[delayed_name] = (
                        components[0] / denominator,
                        components[1] / denominator,
                    )
                responses[(x_index, y_index, direction)] = {
                    **{
                        name: np.max(values, axis=0)
                        for name, values in traces.items()
                        if name in config["source_groups"]
                    },
                    "center_proximal_correlation": np.max(correlation, axis=0),
                    "pairwise_proximal_vector": pairwise_vectors["proximal"],
                    "pairwise_distal_vector": pairwise_vectors["distal"],
                }
    scores = {}
    for population in populations:
        group = np.flatnonzero(target_population == population)
        subtype, side = population[2], population[-1]
        preferred = (
            HORIZONTAL_PREFERENCE[(subtype, side)]
            if subtype in "ab"
            else VERTICAL_PREFERENCE[subtype]
        )
        rows = assigned[group, 1] * len(x_centers) + assigned[group, 0]
        columns = group
        scores[population] = {}
        for readout in config["readouts"]:

            def gather(
                direction: str,
                *,
                selected_readout: str = readout,
                selected_rows: np.ndarray = rows,
                selected_columns: np.ndarray = columns,
                selected_preferred: str = preferred,
            ) -> np.ndarray:
                items = [
                    responses[(x, y, direction)][selected_readout]
                    for y in range(len(y_centers))
                    for x in range(len(x_centers))
                ]
                if selected_readout.startswith("pairwise_"):
                    vector = {
                        "left": (-1.0, 0.0),
                        "right": (1.0, 0.0),
                        "up": (0.0, -1.0),
                        "down": (0.0, 1.0),
                    }[selected_preferred]
                    grid = np.stack(
                        [
                            np.max(vector[0] * item[0] + vector[1] * item[1], axis=0)
                            for item in items
                        ]
                    )
                else:
                    grid = np.stack(items)
                return grid[selected_rows, selected_columns]

            score = strict_contrast_summary(
                target_ids[group],
                gather(preferred),
                gather(OPPOSITE[preferred]),
                scoring["thresholds"],
            )
            scores[population][readout] = _compact(score)
    pass_counts = {
        readout: sum(scores[population][readout]["passed"] for population in scores)
        for readout in config["readouts"]
    }
    bilateral_pairs = {
        readout: [
            subtype
            for subtype in "abcd"
            if scores[f"T4{subtype}_L"][readout]["passed"]
            and scores[f"T4{subtype}_R"][readout]["passed"]
        ]
        for readout in config["readouts"]
    }
    maximum = max(map(len, bilateral_pairs.values()))
    authorize = maximum >= int(
        config["stop_gate"]["minimum_bilateral_direction_pairs_to_authorize_target_formula"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(LOCAL_EDGE_CONFIG): _sha256(root / LOCAL_EDGE_CONFIG),
                str(LOCAL_EDGE_IMPLEMENTATION): _sha256(root / LOCAL_EDGE_IMPLEMENTATION),
                str(source_path): _sha256(root / source_path),
                str(audit_path): _sha256(root / audit_path),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_id": config["condition_id"],
            "target_count": len(targets),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "population_scores": scores,
        "direction_pass_counts": pass_counts,
        "bilateral_passing_subtypes": bilateral_pairs,
        "maximum_bilateral_direction_pair_count": int(maximum),
        "authorize_new_target_formula": bool(authorize),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
