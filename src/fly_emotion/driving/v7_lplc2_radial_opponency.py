"""Tuning-only, label-informed LPLC2 radial-opponency diagnostic.

This module does not alter the MaleCNS dynamics or inject LPLC2 activity.  It
reads the visually driven T4/T5 states and projects them through the real
T4/T5 -> LPLC2 edges.  T4/T5 positions are inferred only from their own
position-annotated direct inputs; frozen subtype direction labels are used to
classify each edge as outward- or inward-aligned.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml
from scipy import sparse

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7 import (
    HORIZONTAL_PREFERENCE,
    VERTICAL_PREFERENCE,
    V7VisualProbe,
    VisualStimulus,
)
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc_typed_screen import (
    HEIGHT,
    WIDTH,
    _motion_free_darkening,
    _radial_ring,
    _translation,
)
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_source_audit import CARDINAL_VECTORS
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary

CONFIG = Path("configs/driving-v7-lplc2-radial-opponency.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_lplc2_radial_opponency.py")
T4_T5_TYPES = tuple(f"{family}{subtype}" for family in ("T4", "T5") for subtype in "abcd")
SIDES = ("L", "R")


def _matched_noise_stimuli(condition: dict, typed: dict, *, zero_noise: bool) -> dict:
    duration = int(condition["duration_frames"])
    start = float(condition["start_radius_pixels"])
    terminal = float(condition["terminal_radius_pixels"])
    period = float(typed["stimulus"]["translation_period_pixels"])
    raw = {
        "lplc2_outward": _radial_ring(duration, start, terminal, True),
        "lplc2_inward": _radial_ring(duration, start, terminal, False),
        "lplc2_motion_free": _motion_free_darkening(duration, terminal),
        "lplc2_translation": _translation(duration, period),
    }
    if zero_noise:
        perturbation = np.zeros((duration, HEIGHT, WIDTH), dtype=np.float64)
        suffix = "zero-noise"
    else:
        rng = np.random.default_rng(int(condition["seed"]))
        perturbation = rng.normal(
            0.0, float(condition["noise_standard_deviation"]), (duration, HEIGHT, WIDTH)
        )
        perturbation = (perturbation + perturbation[:, :, ::-1]) / 2.0
        suffix = "matched-noise"
    return {
        name: VisualStimulus(
            f"{condition['condition_id']}:{name}:{suffix}",
            name,
            "off",
            name.rsplit("_", 1)[-1],
            np.clip(frames + perturbation, 0.0, 1.0).astype(np.float32),
        )
        for name, frames in raw.items()
    }


def _node_annotations(root: Path, probe: MassBalancedVisualProbe) -> tuple[np.ndarray, np.ndarray]:
    table = feather.read_table(
        root / "data/raw/malecns-v1.0/body-annotations.feather",
        columns=[
            "bodyId",
            "type",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
        memory_map=True,
    ).to_pandas()
    body_ids = table["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(probe.graph.body_ids, body_ids)
    valid = nodes < probe.graph.node_count
    valid[valid] &= probe.graph.body_ids[nodes[valid]] == body_ids[valid]
    rows = table.iloc[np.flatnonzero(valid)]
    nodes = nodes[valid]
    sides = np.full(probe.graph.node_count, "", dtype=object)
    sides[nodes] = rows["somaSide"].fillna("").to_numpy(dtype=object)
    coordinates = np.full((probe.graph.node_count, 2), np.nan, dtype=np.float64)
    hex1 = rows["assignedOlHex1"].to_numpy(dtype=np.float64)
    hex2 = rows["assignedOlHex2"].to_numpy(dtype=np.float64)
    coordinates[nodes, 0] = 1.5 * hex2
    coordinates[nodes, 1] = np.sqrt(3.0) * (hex1 + hex2 / 2.0)
    return sides, coordinates


def _infer_t4_t5_positions(
    root: Path, probe: MassBalancedVisualProbe, config: dict
) -> tuple[np.ndarray, dict]:
    """Infer each T4/T5 RF centre from its own coordinate-bearing inputs."""
    raw = load_graph(root / "data/processed/malecns-v1.0", normalized=False).adjacency
    sides, upstream_coordinates = _node_annotations(root, probe)
    positions = np.full((probe.graph.node_count, 2), np.nan, dtype=np.float64)
    family_counts = {}
    source_records = []
    for family in ("T4", "T5"):
        source_types = tuple(config["source_coordinate_types"][family])
        candidates = np.flatnonzero(np.isin(probe.node_types, [f"{family}{x}" for x in "abcd"]))
        mapped = 0
        for node in candidates:
            row = raw.getrow(int(node))
            upstream = row.indices
            weights = np.abs(row.data).astype(np.float64)
            keep = np.isin(probe.node_types[upstream], source_types)
            keep &= np.all(np.isfinite(upstream_coordinates[upstream]), axis=1)
            if np.any(keep):
                position = np.average(
                    upstream_coordinates[upstream[keep]], axis=0, weights=weights[keep]
                )
                # The left optic-lobe horizontal axis is mirrored into the common image frame.
                if sides[node] == "L":
                    position[0] *= -1.0
                positions[node] = position
                mapped += 1
            source_records.append(
                (
                    int(probe.graph.body_ids[node]),
                    str(probe.node_types[node]),
                    str(sides[node]),
                    int(np.count_nonzero(keep)),
                    float(np.sum(weights[keep])),
                )
            )
        family_counts[family] = {
            "source_count": int(len(candidates)),
            "mapped_source_count": int(mapped),
            "mapped_source_fraction": float(mapped / len(candidates)) if len(candidates) else 0.0,
            "coordinate_source_types": list(source_types),
        }
    digest = hashlib.sha256()
    mapped_nodes = np.flatnonzero(np.all(np.isfinite(positions), axis=1))
    digest.update(probe.graph.body_ids[mapped_nodes].astype(np.int64).tobytes())
    digest.update(positions[mapped_nodes].astype(np.float64).tobytes())
    return positions, {
        "families": family_counts,
        "mapped_source_count": int(len(mapped_nodes)),
        "source_record_count": int(len(source_records)),
        "inferred_positions_sha256": digest.hexdigest(),
        "position_rule": (
            "raw-edge-weighted centroid of the source cell's coordinate-bearing direct inputs; "
            "left-eye horizontal coordinate reflected into common image coordinates"
        ),
        "target_label_used_for_position": False,
    }


def _direction_vector(cell_type: str, side: str) -> np.ndarray:
    subtype = cell_type[2]
    direction = (
        HORIZONTAL_PREFERENCE[(subtype, side)]
        if subtype in {"a", "b"}
        else VERTICAL_PREFERENCE[subtype]
    )
    return CARDINAL_VECTORS[direction].astype(np.float64, copy=False)


def _matrix_sha256(matrix: sparse.csr_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(matrix.indptr.astype(np.int64).tobytes())
    digest.update(matrix.indices.astype(np.int64).tobytes())
    digest.update(matrix.data.astype(np.float64).tobytes())
    return digest.hexdigest()


def _build_radial_projection(
    root: Path, probe: MassBalancedVisualProbe, config: dict
) -> tuple[dict[str, sparse.csr_matrix], dict]:
    raw = load_graph(root / "data/processed/malecns-v1.0", normalized=False).adjacency
    sides, _ = _node_annotations(root, probe)
    positions, position_metadata = _infer_t4_t5_positions(root, probe, config)
    minimum_radius = float(config["alignment"]["minimum_radius"])
    matrices = {}
    populations = {}
    for side in SIDES:
        targets = probe.populations[f"LPLC2_{side}"]
        outward_rows = []
        inward_rows = []
        target_records = []
        for target in targets:
            row = raw.getrow(int(target))
            sources = row.indices
            weights = np.abs(row.data).astype(np.float64)
            direct = np.isin(probe.node_types[sources], T4_T5_TYPES)
            direct_sources = sources[direct]
            direct_weights = weights[direct]
            mapped = np.all(np.isfinite(positions[direct_sources]), axis=1)
            total_weight = float(np.sum(direct_weights))
            mapped_weight = float(np.sum(direct_weights[mapped]))
            outward_nodes = np.empty(0, dtype=np.int32)
            inward_nodes = np.empty(0, dtype=np.int32)
            outward_values = np.empty(0, dtype=np.float64)
            inward_values = np.empty(0, dtype=np.float64)
            aligned_count = 0
            if total_weight > 0.0 and np.any(mapped):
                mapped_sources = direct_sources[mapped]
                mapped_weights = direct_weights[mapped]
                centroid = np.average(positions[mapped_sources], axis=0, weights=mapped_weights)
                radial = positions[mapped_sources] - centroid
                radii = np.linalg.norm(radial, axis=1)
                usable = radii > minimum_radius
                alignment = np.zeros(len(mapped_sources), dtype=np.float64)
                for index in np.flatnonzero(usable):
                    source = mapped_sources[index]
                    direction = _direction_vector(str(probe.node_types[source]), str(sides[source]))
                    alignment[index] = float(direction @ (radial[index] / radii[index]))
                aligned_count = int(np.count_nonzero(usable))
                outward_coefficients = mapped_weights * np.maximum(alignment, 0.0) / total_weight
                inward_coefficients = mapped_weights * np.maximum(-alignment, 0.0) / total_weight
                outward_keep = outward_coefficients > 0.0
                inward_keep = inward_coefficients > 0.0
                outward_nodes = mapped_sources[outward_keep].astype(np.int32)
                inward_nodes = mapped_sources[inward_keep].astype(np.int32)
                outward_values = outward_coefficients[outward_keep]
                inward_values = inward_coefficients[inward_keep]
            outward_rows.append((outward_nodes, outward_values))
            inward_rows.append((inward_nodes, inward_values))
            target_records.append(
                {
                    "body_id": int(probe.graph.body_ids[target]),
                    "direct_t4_t5_source_count": int(np.count_nonzero(direct)),
                    "mapped_direct_source_count": int(np.count_nonzero(mapped)),
                    "radially_usable_source_count": aligned_count,
                    "direct_t4_t5_weight": total_weight,
                    "mapped_direct_weight": mapped_weight,
                    "mapped_direct_weight_fraction": (
                        mapped_weight / total_weight if total_weight > 0.0 else 0.0
                    ),
                    "outward_edge_count": int(len(outward_nodes)),
                    "inward_edge_count": int(len(inward_nodes)),
                }
            )

        def make_matrix(rows: list[tuple[np.ndarray, np.ndarray]]) -> sparse.csr_matrix:
            row_indices = []
            columns = []
            data = []
            for index, (nodes, values) in enumerate(rows):
                row_indices.extend([index] * len(nodes))
                columns.extend(nodes.tolist())
                data.extend(values.tolist())
            return sparse.csr_matrix(
                (
                    np.asarray(data, dtype=np.float64),
                    (np.asarray(row_indices, dtype=np.int32), np.asarray(columns, dtype=np.int32)),
                ),
                shape=(len(rows), probe.graph.node_count),
            )

        outward = make_matrix(outward_rows)
        inward = make_matrix(inward_rows)
        matrices[side] = {"outward": outward, "inward": inward}
        ids = probe.graph.body_ids[targets].astype(np.int64)
        populations[side] = {
            "target_count": int(len(targets)),
            "target_body_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "targets_with_any_direct_t4_t5": int(
                sum(row["direct_t4_t5_source_count"] > 0 for row in target_records)
            ),
            "targets_with_mapped_direct_source": int(
                sum(row["mapped_direct_source_count"] > 0 for row in target_records)
            ),
            "targets_with_both_radial_pools": int(
                sum(
                    row["outward_edge_count"] > 0 and row["inward_edge_count"] > 0
                    for row in target_records
                )
            ),
            "mean_mapped_direct_weight_fraction": float(
                np.mean([row["mapped_direct_weight_fraction"] for row in target_records])
            ),
            "outward_projection_sha256": _matrix_sha256(outward),
            "inward_projection_sha256": _matrix_sha256(inward),
            "target_records": target_records,
        }
    return matrices, {"source_positions": position_metadata, "populations": populations}


def _radial_traces(
    probe: MassBalancedVisualProbe, stimulus, matrices: dict[str, dict[str, sparse.csr_matrix]]
) -> tuple[dict[str, np.ndarray], str]:
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history_length = max(1, int(probe.source_delays.max()))
    history = [state.copy() for _ in range(history_length)]
    baseline_image = stimulus.frames[0]
    baseline_values = probe._sample_retina(baseline_image)[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(baseline_values, baseline_values, baseline_values)
    for _ in range(probe.baseline_frames):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = baseline_drive
    signed = {side: pools["outward"] - pools["inward"] for side, pools in matrices.items()}
    baselines = {side: matrix @ state.astype(np.float64) for side, matrix in signed.items()}
    traces = {side: [] for side in signed}
    previous = baseline_values.copy()
    retinal_hash = hashlib.sha256()
    for image in stimulus.frames:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous)
        previous = sampled
        retinal_hash.update(receptor_values.tobytes())
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        state64 = state.astype(np.float64)
        for side, matrix in signed.items():
            response = matrix @ state64 - baselines[side]
            traces[side].append(response)
    return {side: np.stack(values) for side, values in traces.items()}, retinal_hash.hexdigest()


def _layer_nodes(probe: MassBalancedVisualProbe, config: dict) -> dict[str, np.ndarray]:
    populations = {}
    for name in config["populations"]:
        nodes = (
            probe.retina.node_indices
            if name == "mapped_R1-R6"
            else np.flatnonzero(probe.node_types == name)
        )
        if len(nodes) == 0:
            raise ValueError(f"layer-localization population is empty: {name}")
        populations[name] = np.asarray(nodes, dtype=np.int32)
    return populations


def _layer_traces(
    probe: MassBalancedVisualProbe, stimulus, populations: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history_length = max(
        1,
        int(probe.source_delays.max()),
        int(probe.correlator["history_substeps"]) if probe.correlator is not None else 0,
    )
    history = [state.copy() for _ in range(history_length)]
    baseline_image = stimulus.frames[0]
    baseline_values = probe._sample_retina(baseline_image)[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(baseline_values, baseline_values, baseline_values)
    for _ in range(probe.baseline_frames):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = baseline_drive
    baselines = {name: state[nodes].astype(np.float64) for name, nodes in populations.items()}
    traces = {name: [] for name in populations}
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
        for name, nodes in populations.items():
            traces[name].append(state[nodes].astype(np.float64) - baselines[name])
    return {name: np.stack(values) for name, values in traces.items()}


def _score(
    cell_ids: np.ndarray, preferred: np.ndarray, comparator: np.ndarray, thresholds: dict
) -> dict:
    score = strict_contrast_summary(cell_ids, preferred, comparator, thresholds)
    contrast = np.asarray(
        [value if value is not None else np.nan for value in score["contrast"]],
        dtype=np.float64,
    )
    all_target_positive_fraction = float(np.mean(np.isfinite(contrast) & (contrast > 0.0)))
    score["all_target_positive_fraction"] = all_target_positive_fraction
    score["gates"]["all_target_positive_fraction"] = all_target_positive_fraction >= float(
        thresholds["minimum_positive_cell_fraction"]
    )
    score["passed"] = all(score["gates"].values())
    return score


def _compact(score: dict) -> dict:
    return {
        key: score[key]
        for key in (
            "cell_count",
            "valid_cell_count",
            "valid_cell_fraction",
            "median_signed_contrast",
            "positive_cell_fraction",
            "all_target_positive_fraction",
            "invalid_cell_ids",
            "cell_ids",
            "contrast",
            "gates",
            "passed",
        )
    }


def _compact_scalar_score(score: dict) -> dict:
    return {
        key: score[key]
        for key in (
            "cell_count",
            "valid_cell_count",
            "valid_cell_fraction",
            "median_signed_contrast",
            "positive_cell_fraction",
            "all_target_positive_fraction",
            "gates",
            "passed",
        )
    }


def _coverage(scores: list[dict], thresholds: dict) -> dict:
    body_ids = np.asarray(scores[0]["cell_ids"], dtype=np.int64)
    contrasts = []
    for score in scores:
        ids = np.asarray(score["cell_ids"], dtype=np.int64)
        if not np.array_equal(ids, body_ids):
            raise ValueError("LPLC2 target IDs changed across tuning conditions")
        contrasts.append(
            np.asarray(
                [value if value is not None else np.nan for value in score["contrast"]],
                dtype=np.float64,
            )
        )
    values = np.stack(contrasts)
    joint_valid = np.all(np.isfinite(values), axis=0)
    minimum_contrast = float(thresholds["minimum_median_signed_contrast"])
    all_success = np.all(np.isfinite(values) & (values >= minimum_contrast), axis=0)
    joint_fraction = float(np.mean(joint_valid))
    success_fraction = float(np.mean(all_success))
    gates = {
        "joint_valid_fraction": joint_fraction >= float(thresholds["minimum_joint_valid_fraction"]),
        "all_condition_success_fraction": success_fraction
        >= float(thresholds["minimum_all_condition_success_fraction"]),
    }
    return {
        "target_count": int(len(body_ids)),
        "joint_valid_count": int(np.count_nonzero(joint_valid)),
        "joint_valid_fraction": joint_fraction,
        "all_condition_success_count": int(np.count_nonzero(all_success)),
        "all_condition_success_fraction": success_fraction,
        "joint_invalid_body_ids": body_ids[~joint_valid].tolist(),
        "all_condition_failure_body_ids": body_ids[~all_success].tolist(),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _temporal_score(
    cell_ids: np.ndarray, outward: np.ndarray, inward: np.ndarray, config: dict
) -> dict:
    ids = np.asarray(cell_ids, dtype=np.int64)
    outward = np.asarray(outward, dtype=np.float64)
    inward = np.asarray(inward, dtype=np.float64)
    if outward.shape != inward.shape or outward.ndim != 2 or outward.shape[1] != len(ids):
        raise ValueError("temporal traces must be equal time-by-target arrays")
    scale = np.mean(np.abs(outward), axis=0) + np.mean(np.abs(inward), axis=0)
    finite = np.all(np.isfinite(outward), axis=0) & np.all(np.isfinite(inward), axis=0)
    valid = finite & (scale >= float(config["minimum_valid_scale"]))
    forward_error = np.full(len(ids), np.nan, dtype=np.float64)
    reverse_error = np.full(len(ids), np.nan, dtype=np.float64)
    forward_error[valid] = (
        np.mean(np.abs(outward[:, valid] - inward[:, valid]), axis=0) / scale[valid]
    )
    reverse_error[valid] = (
        np.mean(np.abs(outward[:, valid] - inward[::-1, valid]), axis=0) / scale[valid]
    )
    threshold = float(config["minimum_median_normalized_trace_separation"])
    separated = valid & (forward_error >= threshold)
    valid_fraction = float(np.mean(valid))
    median_separation = float(np.median(forward_error[valid])) if np.any(valid) else None
    separated_fraction = float(np.mean(separated))
    gates = {
        "valid_cell_fraction": valid_fraction >= float(config["minimum_joint_valid_fraction"]),
        "median_normalized_trace_separation": median_separation is not None
        and median_separation >= threshold,
        "all_target_separated_fraction": separated_fraction
        >= float(config["minimum_all_target_separated_fraction"]),
    }
    return {
        "target_count": int(len(ids)),
        "valid_target_count": int(np.count_nonzero(valid)),
        "valid_target_fraction": valid_fraction,
        "median_normalized_forward_trace_error": median_separation,
        "median_normalized_time_reversed_trace_error": (
            float(np.median(reverse_error[valid])) if np.any(valid) else None
        ),
        "all_target_separated_count": int(np.count_nonzero(separated)),
        "all_target_separated_fraction": separated_fraction,
        "cell_ids": ids.tolist(),
        "normalized_forward_trace_error": [
            float(value) if np.isfinite(value) else None for value in forward_error
        ],
        "normalized_time_reversed_trace_error": [
            float(value) if np.isfinite(value) else None for value in reverse_error
        ],
        "gates": gates,
        "passed": all(gates.values()),
    }


def _temporal_coverage(scores: list[dict], config: dict) -> dict:
    body_ids = np.asarray(scores[0]["cell_ids"], dtype=np.int64)
    values = []
    for score in scores:
        if score["cell_ids"] != body_ids.tolist():
            raise ValueError("temporal target IDs changed across tuning conditions")
        values.append(
            np.asarray(
                [
                    value if value is not None else np.nan
                    for value in score["normalized_forward_trace_error"]
                ],
                dtype=np.float64,
            )
        )
    matrix = np.stack(values)
    joint_valid = np.all(np.isfinite(matrix), axis=0)
    threshold = float(config["minimum_median_normalized_trace_separation"])
    all_separated = np.all(np.isfinite(matrix) & (matrix >= threshold), axis=0)
    joint_fraction = float(np.mean(joint_valid))
    separated_fraction = float(np.mean(all_separated))
    gates = {
        "joint_valid_fraction": joint_fraction >= float(config["minimum_joint_valid_fraction"]),
        "all_condition_separated_fraction": separated_fraction
        >= float(config["minimum_all_condition_separated_fraction"]),
    }
    return {
        "target_count": int(len(body_ids)),
        "joint_valid_count": int(np.count_nonzero(joint_valid)),
        "joint_valid_fraction": joint_fraction,
        "all_condition_separated_count": int(np.count_nonzero(all_separated)),
        "all_condition_separated_fraction": separated_fraction,
        "joint_invalid_body_ids": body_ids[~joint_valid].tolist(),
        "all_condition_unseparated_body_ids": body_ids[~all_separated].tolist(),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _layer_score(
    body_ids: np.ndarray, outward: np.ndarray, inward: np.ndarray, config: dict
) -> dict:
    ids = np.asarray(body_ids, dtype=np.int64)
    outward = np.asarray(outward, dtype=np.float64)
    inward = np.asarray(inward, dtype=np.float64)
    if outward.shape != inward.shape or outward.ndim != 2 or outward.shape[1] != len(ids):
        raise ValueError("layer traces must be equal time-by-cell arrays")
    scale = np.mean(np.abs(outward), axis=0) + np.mean(np.abs(inward), axis=0)
    finite = np.all(np.isfinite(outward), axis=0) & np.all(np.isfinite(inward), axis=0)
    valid = finite & (scale >= float(config["minimum_valid_scale"]))
    separation = np.full(len(ids), np.nan, dtype=np.float64)
    separation[valid] = np.mean(np.abs(outward[:, valid] - inward[:, valid]), axis=0) / scale[valid]
    threshold = float(config["minimum_normalized_trace_separation"])
    separated = valid & (separation >= threshold)
    valid_fraction = float(np.mean(valid))
    median = float(np.median(separation[valid])) if np.any(valid) else None
    separated_fraction = float(np.mean(separated))
    gates = {
        "valid_cell_fraction": valid_fraction >= float(config["minimum_joint_valid_fraction"]),
        "median_normalized_trace_separation": median is not None and median >= threshold,
        "all_target_separated_fraction": separated_fraction
        >= float(config["minimum_all_target_separated_fraction"]),
    }
    return {
        "cell_count": int(len(ids)),
        "valid_cell_count": int(np.count_nonzero(valid)),
        "valid_cell_fraction": valid_fraction,
        "median_normalized_trace_separation": median,
        "all_cell_separated_count": int(np.count_nonzero(separated)),
        "all_cell_separated_fraction": separated_fraction,
        "cell_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "normalized_trace_separation": [
            float(value) if np.isfinite(value) else None for value in separation
        ],
        "gates": gates,
        "passed": all(gates.values()),
    }


def _layer_consistency(scores: list[dict], config: dict) -> dict:
    values = np.stack(
        [
            np.asarray(
                [
                    value if value is not None else np.nan
                    for value in score["normalized_trace_separation"]
                ]
            )
            for score in scores
        ]
    )
    joint_valid = np.all(np.isfinite(values), axis=0)
    threshold = float(config["minimum_normalized_trace_separation"])
    all_separated = np.all(np.isfinite(values) & (values >= threshold), axis=0)
    joint_fraction = float(np.mean(joint_valid))
    separated_fraction = float(np.mean(all_separated))
    gates = {
        "joint_valid_fraction": joint_fraction >= float(config["minimum_joint_valid_fraction"]),
        "all_condition_separated_fraction": separated_fraction
        >= float(config["minimum_all_condition_separated_fraction"]),
    }
    return {
        "cell_count": int(values.shape[1]),
        "passing_condition_count": int(sum(score["passed"] for score in scores)),
        "required_condition_count": int(len(scores)),
        "joint_valid_count": int(np.count_nonzero(joint_valid)),
        "joint_valid_fraction": joint_fraction,
        "all_condition_separated_count": int(np.count_nonzero(all_separated)),
        "all_condition_separated_fraction": separated_fraction,
        "gates": gates,
        "passed": bool(all(score["passed"] for score in scores) and all(gates.values())),
    }


def _compact_layer_score(score: dict) -> dict:
    return {key: value for key, value in score.items() if key != "normalized_trace_separation"}


def _selected_receptor_edge_audit(
    root: Path, probe: MassBalancedVisualProbe, layer_nodes: dict[str, np.ndarray]
) -> dict:
    raw = load_graph(root / "data/processed/malecns-v1.0", normalized=False).adjacency
    selected = np.zeros(probe.graph.node_count, dtype=bool)
    selected[probe.retina.node_indices] = True
    all_receptors = probe.node_types == "R1-R6"
    unselected = all_receptors & ~selected
    populations = {}
    for name in ("L1", "L2", "L3", "L5"):
        nodes = layer_nodes[name]
        matrix = raw[nodes, :]
        selected_weight = np.asarray(matrix.multiply(selected).sum(axis=1)).ravel()
        unselected_weight = np.asarray(matrix.multiply(unselected).sum(axis=1)).ravel()
        total = selected_weight + unselected_weight
        with_any = total > 0.0
        fraction = np.zeros(len(nodes), dtype=np.float64)
        np.divide(selected_weight, total, out=fraction, where=with_any)
        populations[name] = {
            "cell_count": int(len(nodes)),
            "cells_with_selected_receptor_input": int(np.count_nonzero(selected_weight > 0.0)),
            "cells_with_unselected_receptor_input": int(np.count_nonzero(unselected_weight > 0.0)),
            "cells_with_any_R1_R6_input": int(np.count_nonzero(with_any)),
            "median_selected_weight_fraction_all_cells": float(np.median(fraction)),
            "median_selected_weight_fraction_cells_with_any_R1_R6": (
                float(np.median(fraction[with_any])) if np.any(with_any) else None
            ),
        }
    return {
        "mapped_receptor_count": int(len(probe.retina.node_indices)),
        "all_graph_R1_R6_count": int(np.count_nonzero(all_receptors)),
        "unselected_graph_R1_R6_count": int(np.count_nonzero(unselected)),
        "visual_graph_keeps_unselected_R1_R6_outputs": bool(
            np.any(probe.adjacency[:, np.flatnonzero(unselected)].data)
        ),
        "populations": populations,
        "interpretation": (
            "The mass-balanced camera drives only the selected paired receptors while the "
            "typed visual adjacency retains outputs from all annotated R1-R6 cells. This is an "
            "input-contract mismatch, but a post-failure renormalization A/B did not restore "
            "the preregistered layer separation gate."
        ),
        "renormalization_AB_parameter_search": False,
        "renormalization_AB_gate_restored": False,
    }


def _full_retina_projection_ab(root: Path, config: dict, typed: dict) -> dict:
    probe = V7VisualProbe(
        root,
        brain_substeps=int(config["brain_substeps_per_frame"]),
        baseline_frames=int(config["baseline_frames"]),
        retinal_backend=config["retinal_backend"],
        retinal_geometry="legacy_proxy_v2",
        dynamics_backend=config["dynamics_backend"],
    )
    populations = _layer_nodes(probe, config["layer_localization"])
    scores = {name: [] for name in populations}
    per_condition = {}
    conditions = {row["condition_id"]: row for row in typed["conditions"]}
    for condition_id in config["condition_ids"]:
        stimuli = _matched_noise_stimuli(conditions[condition_id], typed, zero_noise=False)
        outward = _layer_traces(probe, stimuli["lplc2_outward"], populations)
        inward = _layer_traces(probe, stimuli["lplc2_inward"], populations)
        condition_scores = {}
        for name, nodes in populations.items():
            score = _layer_score(
                probe.graph.body_ids[nodes],
                outward[name],
                inward[name],
                config["layer_localization"],
            )
            scores[name].append(score)
            condition_scores[name] = _compact_layer_score(score)
        per_condition[condition_id] = condition_scores
    consistency = {
        name: _layer_consistency(values, config["layer_localization"])
        for name, values in scores.items()
    }
    return {
        "post_failure_localization_only": True,
        "parameter_fit": False,
        "runtime_modified": False,
        "calibration_evaluated": False,
        "baseline": config["input_projection_ab"]["baseline"],
        "comparator": config["input_projection_ab"]["comparator"],
        "full_mapped_receptor_count": int(probe.retina.size),
        "full_mapping_is_exact_mirror": False,
        "same_noise_realization_within_pair": True,
        "per_condition": per_condition,
        "population_consistency": consistency,
        "passing_populations": [name for name, result in consistency.items() if result["passed"]],
    }


def _paired_noise_layer_control(
    probe: MassBalancedVisualProbe,
    matrices: dict[str, dict[str, sparse.csr_matrix]],
    config: dict,
    typed: dict,
) -> dict:
    populations = _layer_nodes(probe, config["layer_localization"])
    conditions = {row["condition_id"]: row for row in typed["conditions"]}
    controls = {}
    for control_name, zero_noise in (("matched_noise", False), ("zero_noise", True)):
        scores = {name: [] for name in populations}
        radial_scores = {side: {name: [] for name in config["comparisons"]} for side in SIDES}
        per_condition = {}
        radial_per_condition = {}
        for condition_id in config["condition_ids"]:
            stimuli = _matched_noise_stimuli(conditions[condition_id], typed, zero_noise=zero_noise)
            outward = _layer_traces(probe, stimuli["lplc2_outward"], populations)
            inward = _layer_traces(probe, stimuli["lplc2_inward"], populations)
            condition_scores = {}
            for name, nodes in populations.items():
                score = _layer_score(
                    probe.graph.body_ids[nodes],
                    outward[name],
                    inward[name],
                    config["layer_localization"],
                )
                scores[name].append(score)
                condition_scores[name] = _compact_layer_score(score)
            per_condition[condition_id] = condition_scores
            radial_traces = {
                name: _radial_traces(probe, stimulus, matrices)[0]
                for name, stimulus in stimuli.items()
            }
            radial_condition = {}
            for side in SIDES:
                ids = probe.graph.body_ids[probe.populations[f"LPLC2_{side}"]]
                radial_condition[side] = {}
                for name, comparator in config["comparisons"].items():
                    score = _score(
                        ids,
                        np.max(radial_traces["lplc2_outward"][side], axis=0),
                        np.max(radial_traces[comparator][side], axis=0),
                        config["thresholds"],
                    )
                    radial_scores[side][name].append(score)
                    radial_condition[side][name] = _compact_scalar_score(score)
            radial_per_condition[condition_id] = radial_condition
        consistency = {
            name: _layer_consistency(values, config["layer_localization"])
            for name, values in scores.items()
        }
        controls[control_name] = {
            "same_noise_realization_within_pair": True,
            "noise_standard_deviation_zero": zero_noise,
            "per_condition": per_condition,
            "population_consistency": consistency,
            "passing_populations": [
                name for name, result in consistency.items() if result["passed"]
            ],
            "radial_per_condition_scores": radial_per_condition,
            "radial_population_consistency": {
                side: {
                    name: {
                        "passing_condition_count": int(sum(score["passed"] for score in values)),
                        "required_condition_count": len(values),
                        "coverage": _coverage(values, config["thresholds"]),
                        "passed": bool(
                            all(score["passed"] for score in values)
                            and _coverage(values, config["thresholds"])["passed"]
                        ),
                    }
                    for name, values in mechanisms.items()
                }
                for side, mechanisms in radial_scores.items()
            },
        }
    return {
        "post_failure_correction": True,
        "parameter_fit": False,
        "runtime_modified": False,
        "calibration_evaluated": False,
        "primary_interpretation_source": config["paired_noise_control"][
            "primary_interpretation_source"
        ],
        "unmatched_noise_layer_localization_confounded": True,
        "controls": controls,
    }


def evaluate_v7_lplc2_radial_opponency(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    typed_path = Path(config["typed_stimulus_protocol"])
    typed = yaml.safe_load((root / typed_path).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    condition_ids = list(config["condition_ids"])
    conditions = {row["condition_id"]: row for row in typed["conditions"]}
    if condition_ids != list(typed["condition_ids"]):
        raise ValueError("radial diagnostic must use the frozen typed-screen conditions")
    if any(conditions[name]["role"] != "tuning" for name in condition_ids):
        raise ValueError("radial diagnostic may consume tuning conditions only")
    if config["thresholds"] != {
        "minimum_valid_denominator": scoring["thresholds"]["minimum_valid_denominator"],
        "minimum_valid_cell_fraction": scoring["thresholds"]["minimum_valid_cell_fraction"],
        "minimum_median_signed_contrast": scoring["thresholds"]["minimum_median_signed_contrast"],
        "minimum_positive_cell_fraction": scoring["thresholds"]["minimum_positive_cell_fraction"],
        "minimum_joint_valid_fraction": 0.80,
        "minimum_all_condition_success_fraction": 0.60,
        "maximum_mirror_error": 0.20,
    }:
        raise ValueError("radial diagnostic thresholds diverged from the frozen strict gate")
    probe = MassBalancedVisualProbe(root, config)
    matrices, anatomy = _build_radial_projection(root, probe, config)
    layer_nodes = _layer_nodes(probe, config["layer_localization"])
    receptor_edge_audit = _selected_receptor_edge_audit(root, probe, layer_nodes)
    per_condition = {}
    stimulus_manifest = []
    raw_scores = {side: {name: [] for name in config["comparisons"]} for side in SIDES}
    temporal_scores = {side: [] for side in SIDES}
    temporal_per_condition = {}
    layer_scores = {name: [] for name in layer_nodes}
    layer_per_condition = {}
    for condition_id in condition_ids:
        selected = _matched_noise_stimuli(conditions[condition_id], typed, zero_noise=False)
        responses = {}
        response_traces = {}
        for name, stimulus in selected.items():
            response_traces[name], retinal_hash = _radial_traces(probe, stimulus, matrices)
            responses[name] = {
                side: np.max(values, axis=0) for side, values in response_traces[name].items()
            }
            stimulus_manifest.append(
                {
                    "identity": stimulus.name,
                    "frame_sha256": hashlib.sha256(stimulus.frames.tobytes()).hexdigest(),
                    "retinal_drive_sha256": retinal_hash,
                }
            )
        layer_outward = _layer_traces(probe, selected["lplc2_outward"], layer_nodes)
        layer_inward = _layer_traces(probe, selected["lplc2_inward"], layer_nodes)
        condition_scores = {}
        condition_temporal = {}
        for side in SIDES:
            ids = probe.graph.body_ids[probe.populations[f"LPLC2_{side}"]]
            condition_scores[side] = {}
            for name, comparator in config["comparisons"].items():
                score = _score(
                    ids,
                    responses["lplc2_outward"][side],
                    responses[comparator][side],
                    config["thresholds"],
                )
                raw_scores[side][name].append(score)
                condition_scores[side][name] = _compact(score)
            temporal = _temporal_score(
                ids,
                response_traces["lplc2_outward"][side],
                response_traces["lplc2_inward"][side],
                config["temporal_identifiability"],
            )
            temporal_scores[side].append(temporal)
            condition_temporal[side] = temporal
        per_condition[condition_id] = condition_scores
        temporal_per_condition[condition_id] = condition_temporal
        condition_layers = {}
        for name, nodes in layer_nodes.items():
            score = _layer_score(
                probe.graph.body_ids[nodes],
                layer_outward[name],
                layer_inward[name],
                config["layer_localization"],
            )
            layer_scores[name].append(score)
            condition_layers[name] = _compact_layer_score(score)
        layer_per_condition[condition_id] = condition_layers
    consistency = {
        side: {
            name: {
                "passing_condition_count": int(sum(score["passed"] for score in scores)),
                "required_condition_count": len(condition_ids),
                "coverage": _coverage(scores, config["thresholds"]),
                "passed": bool(
                    all(score["passed"] for score in scores)
                    and _coverage(scores, config["thresholds"])["passed"]
                ),
            }
            for name, scores in mechanisms.items()
        }
        for side, mechanisms in raw_scores.items()
    }
    mirror = {}
    for condition_id in condition_ids:
        errors = {}
        for name in config["comparisons"]:
            left = per_condition[condition_id]["L"][name]["median_signed_contrast"]
            right = per_condition[condition_id]["R"][name]["median_signed_contrast"]
            error = abs(left - right) if left is not None and right is not None else None
            errors[name] = error
        finite_errors = [value for value in errors.values() if value is not None]
        maximum = max(finite_errors) if len(finite_errors) == len(errors) else None
        mirror[condition_id] = {
            "per_comparison_absolute_median_contrast_error": errors,
            "maximum_error": maximum,
            "passed": bool(
                maximum is not None
                and maximum <= float(config["thresholds"]["maximum_mirror_error"])
            ),
        }
    mechanism_passed = all(
        result["passed"] for side in consistency.values() for result in side.values()
    )
    mirror_passed = all(result["passed"] for result in mirror.values())
    temporal_consistency = {
        side: {
            "passing_condition_count": int(sum(score["passed"] for score in scores)),
            "required_condition_count": len(condition_ids),
            "coverage": _temporal_coverage(scores, config["temporal_identifiability"]),
            "passed": bool(
                all(score["passed"] for score in scores)
                and _temporal_coverage(scores, config["temporal_identifiability"])["passed"]
            ),
        }
        for side, scores in temporal_scores.items()
    }
    temporal_passed = all(result["passed"] for result in temporal_consistency.values())
    layer_consistency = {
        name: _layer_consistency(scores, config["layer_localization"])
        for name, scores in layer_scores.items()
    }
    full_retina_ab = _full_retina_projection_ab(root, config, typed)
    paired_noise = _paired_noise_layer_control(probe, matrices, config, typed)
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(typed_path): _sha256(root / typed_path),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_ids": condition_ids,
            "stimulus_count": len(stimulus_manifest),
            "paired_noise_policy": "same realization within each preferred-comparator pair",
            "fixed_target_denominators": {
                side: int(len(probe.populations[f"LPLC2_{side}"])) for side in SIDES
            },
            "direction_labels_used_by_mechanism": True,
            "may_authorize_strict_visual_gate": False,
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "mechanism": {
            "source_populations": config["source_populations"],
            "coordinate_sources": config["source_coordinate_types"],
            "alignment_formula": config["alignment"]["formula"],
            "response_formula": config["response"]["formula"],
            "direct_edge_weights_preserved": True,
            "ct1_or_other_sources_in_spatial_pool": False,
            "unmapped_sources_dropped_from_denominator": False,
            "label_informed_mechanism_diagnostic": True,
            "biological_direction_gate_claimed": False,
        },
        "anatomy": anatomy,
        "stimulus_manifest": stimulus_manifest,
        "per_condition_scores": per_condition,
        "population_consistency": consistency,
        "mirror_summary": mirror,
        "temporal_identifiability": {
            "post_failure_localization_only": True,
            "parameter_search": False,
            "metric": config["temporal_identifiability"]["metric"],
            "per_condition": temporal_per_condition,
            "population_consistency": temporal_consistency,
            "passed": bool(temporal_passed),
        },
        "layer_localization": {
            "post_failure_localization_only": True,
            "parameter_search": False,
            "selected_receptors_only": True,
            "metric": config["layer_localization"]["metric"],
            "per_condition": layer_per_condition,
            "population_consistency": layer_consistency,
            "passing_populations": [
                name for name, result in layer_consistency.items() if result["passed"]
            ],
            "selected_receptor_edge_audit": receptor_edge_audit,
            "interpretation_allowed": True,
            "paired_noise_policy": "matched_noise",
        },
        "input_projection_ab": full_retina_ab,
        "paired_noise_control": paired_noise,
        "radial_opponency_mechanism_gates_passed": bool(mechanism_passed and mirror_passed),
        "advance_to_calibration": bool(mechanism_passed and mirror_passed),
        "advance_to_runtime_integration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
        "limitations": [
            "Frozen T4/T5 subtype direction labels are part of this diagnostic mechanism.",
            (
                "A pass would localize a candidate radial pooling rule but cannot validate the "
                "upstream biological direction gate."
            ),
            (
                "Inferred receptive-field centroids are connectivity proxies, not measured "
                "receptive fields."
            ),
            "All unmapped, silent, and low-response LPLC2 targets remain in the fixed denominator.",
        ],
    }
