from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy import sparse

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7 import V7Contract, V7VisualProbe, VisualStimulus
from fly_emotion.driving.v7_retina_audit import infer_retinal_columns
from fly_emotion.driving.v7_temporal_audit import (
    ON_RESPONSE_SIGN,
    _step_trace,
    build_step_stimuli,
    summarize_step_response,
)

LOCAL_TYPES = ("Mi1", "Mi4", "Mi9", "C3")
SELECTION_SEED = "v7-local-input-20260915"
SITE_COUNT = 3
CENTRE_RADIUS = 1
OUTER_RADIUS = 3
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_local_input_audit.py")


def local_masks(width: int, height: int, x: int, y: int, side: int) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[:height, :width]
    eye = xx < width // 2 if side < 0 else xx >= width // 2
    distance_squared = (xx - x) ** 2 + (yy - y) ** 2
    return {
        "centre": eye & (distance_squared <= CENTRE_RADIUS**2),
        "annulus": eye
        & (distance_squared > CENTRE_RADIUS**2)
        & (distance_squared <= OUTER_RADIUS**2),
        "whole_eye": eye,
    }


def local_step(mask: np.ndarray, polarity: str) -> VisualStimulus:
    level = {"on": 0.8, "off": 0.2}[polarity]
    frames = np.full((48, *mask.shape), 0.5, dtype=np.float32)
    frames[16:, mask] = level
    return VisualStimulus(f"local_{polarity}", "local_step", polarity, "none", frames)


def select_local_sites(root: Path, probe: V7VisualProbe) -> tuple[list[dict], dict]:
    retina, assignments = infer_retinal_columns(root)
    if not np.array_equal(probe.retina.body_ids, retina.body_ids):
        raise ValueError("local audit requires the unchanged full retinal map")
    width, height = 48, 24
    x = np.rint(probe.retinal_u * (width - 1)).astype(np.int32)
    y = np.rint(probe.retinal_v * (height - 1)).astype(np.int32)
    columns = {
        side: set(map(tuple, assignments.coordinates[retina.side == side].tolist()))
        for side in (-1, 1)
    }
    annotations = feather.read_table(
        root / "data/raw/malecns-v1.0/body-annotations.feather",
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    available = annotations.loc[annotations["type"].isin(LOCAL_TYPES)].dropna(
        subset=["assignedOlHex1", "assignedOlHex2"]
    )
    by_column: dict[tuple, dict[str, list[int]]] = {}
    for row in available.itertuples(index=False):
        node = int(np.searchsorted(probe.graph.body_ids, row.bodyId))
        if node >= probe.graph.node_count or probe.graph.body_ids[node] != row.bodyId:
            continue
        key = (int(row.assignedOlHex1), int(row.assignedOlHex2), row.somaSide)
        by_column.setdefault(key, {}).setdefault(row.type, []).append(node)
    common = columns[-1] & columns[1]
    ordered = sorted(
        common,
        key=lambda coordinate: hashlib.sha256(
            f"{SELECTION_SEED}:{coordinate[0]}:{coordinate[1]}".encode()
        ).digest(),
    )
    eligible = []
    for coordinate in ordered:
        sides = []
        for side, eye in ((-1, "L"), (1, "R")):
            same_column = (retina.side == side) & np.all(
                assignments.coordinates == coordinate, axis=1
            )
            pixels = np.unique(np.column_stack((x[same_column], y[same_column])), axis=0)
            targets = by_column.get((*coordinate, eye), {})
            if len(pixels) != 1 or set(targets) != set(LOCAL_TYPES):
                break
            px, py = map(int, pixels[0])
            low_x, high_x = (0, width // 2 - 1) if side < 0 else (width // 2, width - 1)
            if not (
                low_x + OUTER_RADIUS <= px <= high_x - OUTER_RADIUS
                and OUTER_RADIUS <= py <= height - 1 - OUTER_RADIUS
            ):
                break
            sides.append(
                {
                    "eye": eye,
                    "side": side,
                    "pixel": [px, py],
                    "column_receptors": int(same_column.sum()),
                    "ambiguous_column_receptors": int(
                        np.count_nonzero(assignments.ambiguity[same_column] > 0)
                    ),
                    "target_nodes": {kind: sorted(nodes) for kind, nodes in targets.items()},
                    "target_body_ids": {
                        kind: probe.graph.body_ids[sorted(nodes)].tolist()
                        for kind, nodes in targets.items()
                    },
                }
            )
        if len(sides) == 2:
            eligible.append({"optic_hex": list(coordinate), "eyes": sides})
    selected = []
    for site in eligible:
        if any(
            np.linalg.norm(np.asarray(site["eyes"][index]["pixel"]) - other["eyes"][index]["pixel"])
            <= 2 * OUTER_RADIUS
            for other in selected
            for index in (0, 1)
        ):
            continue
        selected.append(site)
        if len(selected) == SITE_COUNT:
            break
    if len(selected) != SITE_COUNT:
        raise ValueError("insufficient anatomy-only nonoverlapping local sites")
    coverage = {}
    for kind in (*LOCAL_TYPES, "L3", "Tm3"):
        for eye in ("L", "R"):
            group = annotations[annotations["type"].eq(kind) & annotations["somaSide"].eq(eye)]
            coverage[f"{kind}_{eye}"] = {
                "cells": len(group),
                "coordinate_cells": int(
                    group[["assignedOlHex1", "assignedOlHex2"]].notna().all(axis=1).sum()
                ),
            }
    return selected, {
        "coordinate_coverage": coverage,
        "common_columns": len(common),
        "eligible_columns": len(eligible),
        "selected_columns": len(selected),
        "seed": SELECTION_SEED,
        "selection": "hash order; same-column targets in both eyes; pixel margin and separation",
        "functional_responses_used": False,
    }


def classify_column_coverage(
    coordinates: np.ndarray,
    sides: np.ndarray,
    receptor_coordinates: np.ndarray,
    receptor_sides: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(
        (int(side), *map(int, coordinate))
        for side, coordinate in zip(receptor_sides, receptor_coordinates, strict=True)
    )
    labels = np.full(len(coordinates), "missing_coordinate", dtype=object)
    coverage = np.full(len(coordinates), -1, dtype=np.int32)
    for index, (coordinate, side) in enumerate(zip(coordinates, sides, strict=True)):
        if side not in (-1, 1):
            labels[index] = "missing_side"
        elif np.all(np.isfinite(coordinate)):
            coverage[index] = counts[(int(side), *map(int, coordinate))]
            labels[index] = "covered" if coverage[index] > 0 else "uncovered"
    return labels, coverage


def mi9_column_coverage(root: Path, probe: V7VisualProbe) -> dict:
    retina, assignments = infer_retinal_columns(root)
    annotations = feather.read_table(
        root / "data/raw/malecns-v1.0/body-annotations.feather",
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    rows = annotations.loc[annotations["type"].eq("Mi9")].sort_values("bodyId")
    ids = rows["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(probe.graph.body_ids, ids)
    valid = nodes < probe.graph.node_count
    valid[valid] &= probe.graph.body_ids[nodes[valid]] == ids[valid]
    rows = rows.iloc[np.flatnonzero(valid)]
    nodes = nodes[valid]
    sides = rows["somaSide"].map({"L": -1, "R": 1}).fillna(0).to_numpy(dtype=np.int8)
    coordinates = rows[["assignedOlHex1", "assignedOlHex2"]].to_numpy(dtype=float)
    labels, counts = classify_column_coverage(
        coordinates, sides, assignments.coordinates, retina.side
    )
    return {
        "nodes": nodes,
        "body_ids": probe.graph.body_ids[nodes],
        "sides": sides,
        "coordinates": coordinates,
        "labels": labels,
        "receptor_counts": counts,
    }


def summarize_mi9_coverage(
    coverage: dict, traces: dict[str, np.ndarray], node_order: np.ndarray
) -> dict:
    indices = np.searchsorted(node_order, coverage["nodes"])
    delta = {name: traces[name][:, indices] - traces["sham"][:, indices] for name in ("on", "off")}
    groups = {}
    for eye, side in (("L", -1), ("R", 1), ("unknown", 0)):
        groups[eye] = {}
        for label in ("all", "covered", "uncovered", "missing_coordinate", "missing_side"):
            mask = coverage["sides"] == side
            if label != "all":
                mask &= coverage["labels"] == label
            if mask.any():
                groups[eye][label] = {
                    name: summarize_step_response(
                        values[:, mask], onset=16, expected_sign=-1 if name == "on" else None
                    )
                    for name, values in delta.items()
                }
    cell_responses = {}
    for name, values in delta.items():
        post = values[16:]
        peaks = np.argmax(np.abs(post), axis=0)
        cell_responses[name] = {
            "signed_peak": post[peaks, np.arange(len(indices))].tolist(),
            "peak_frame_after_onset": peaks.tolist(),
        }
    return {"groups": groups, "per_cell_responses": cell_responses}


def select_receptor_masks(
    coordinates: np.ndarray,
    sides: np.ndarray,
    body_ids: np.ndarray,
    target_coordinate: np.ndarray,
    target_side: int,
) -> dict[str, np.ndarray]:
    distance = np.linalg.norm(coordinates - target_coordinate, axis=1)
    same = np.flatnonzero((sides == target_side) & (distance == 0))
    if not len(same):
        raise ValueError("mask target must have same-column receptors")
    candidates = np.flatnonzero((sides == target_side) & (distance > 0))
    order = np.lexsort((body_ids[candidates], distance[candidates]))
    near = candidates[order[: len(same)]]
    far_candidates = np.setdiff1d(candidates[distance[candidates] >= 6.0], near)
    ordered_far = sorted(
        far_candidates,
        key=lambda index: hashlib.sha256(
            f"v7-mask-20260915:{int(body_ids[index])}".encode()
        ).digest(),
    )
    far = np.asarray(ordered_far[: len(same)], dtype=np.int32)
    if len(near) != len(same) or len(far) != len(same):
        raise ValueError("insufficient same-eye matched-count control receptors")
    return {"same_column": same, "nearest_other_columns": near, "far_other_columns": far}


class BaselineHeldRetinaProbe(V7VisualProbe):
    def __init__(self, root: Path, *, retinal_backend: str):
        super().__init__(
            root,
            retinal_backend=retinal_backend,
            retinal_geometry="legacy_proxy_v2",
            dynamics_backend="typed_visual_subgraph_v1",
            brain_substeps=4,
            baseline_frames=32,
        )
        self.held_receptor_positions = np.empty(0, dtype=np.int32)

    def _retinal_code(
        self, values: np.ndarray, baseline: np.ndarray, previous: np.ndarray
    ) -> np.ndarray:
        encoded = super()._retinal_code(values, baseline, previous).copy()
        baseline_code = super()._retinal_code(baseline, baseline, baseline)
        encoded[self.held_receptor_positions] = baseline_code[self.held_receptor_positions]
        return encoded


def evaluate_v7_receptor_mask_audit(root: Path) -> dict:
    contract = V7Contract.load(root)
    retina, assignment = infer_retinal_columns(root)
    stimuli = {item.polarity: item for item in build_step_stimuli(48, 24)}
    results = {}
    sites = None
    for backend in ("linear_luminance", "signed_frame_difference"):
        probe = BaselineHeldRetinaProbe(root, retinal_backend=backend)
        if sites is None:
            sites, selection = select_local_sites(root, probe)
        targets = np.unique(
            np.concatenate(
                [
                    np.asarray(eye["target_nodes"]["Mi9"], dtype=np.int32)
                    for site in sites
                    for eye in site["eyes"]
                ]
            )
        )
        if np.intersect1d(targets, probe.retina.node_indices).size:
            raise ValueError("Mi9 readout overlaps external receptor input")
        intact = {name: _step_trace(probe, stimulus, targets) for name, stimulus in stimuli.items()}
        rows = []
        for site in sites:
            for eye in site["eyes"]:
                masks = select_receptor_masks(
                    assignment.coordinates,
                    retina.side,
                    retina.body_ids,
                    np.asarray(site["optic_hex"]),
                    eye["side"],
                )
                indices = np.searchsorted(targets, eye["target_nodes"]["Mi9"])
                row = {
                    "optic_hex": site["optic_hex"],
                    "eye": eye["eye"],
                    "target_body_ids": eye["target_body_ids"]["Mi9"],
                    "conditions": {},
                }
                for condition, positions in {
                    "intact": np.empty(0, dtype=np.int32),
                    **masks,
                }.items():
                    probe.held_receptor_positions = positions
                    traces = (
                        intact
                        if condition == "intact"
                        else {
                            name: _step_trace(probe, stimulus, targets)
                            for name, stimulus in stimuli.items()
                        }
                    )
                    if not np.array_equal(traces["sham"], intact["sham"]):
                        raise ValueError("baseline-hold mask changed the sham trajectory")
                    values = {}
                    for polarity in ("on", "off"):
                        delta = (traces[polarity] - traces["sham"])[:, indices]
                        reference = (intact[polarity] - intact["sham"])[:, indices]
                        summary = summarize_step_response(
                            delta, onset=16, expected_sign=-1 if polarity == "on" else None
                        )
                        summary["mean_post_step_shift_vs_intact"] = float(
                            np.mean(delta[16:] - reference[16:])
                        )
                        summary["mean_negative_response_magnitude"] = float(
                            np.mean(np.maximum(-delta[16:], 0.0))
                        )
                        summary["negative_response_loss_vs_intact"] = float(
                            np.mean(np.maximum(-reference[16:], 0.0))
                            - summary["mean_negative_response_magnitude"]
                        )
                        values[polarity] = summary
                    distance = np.linalg.norm(
                        assignment.coordinates[positions] - site["optic_hex"], axis=1
                    )
                    row["conditions"][condition] = {
                        "held_receptor_count": int(len(positions)),
                        "held_receptor_body_ids": retina.body_ids[positions].tolist(),
                        "held_receptor_columns": assignment.coordinates[positions].tolist(),
                        "hex_index_distance": distance.tolist(),
                        "ambiguous_modal_assignments": int(
                            np.count_nonzero(assignment.ambiguity[positions] > 0)
                        ),
                        "sham_matches_intact": True,
                        "responses": values,
                    }
                probe.held_receptor_positions = np.empty(0, dtype=np.int32)
                rows.append(row)
        results[backend] = rows
    dependencies = [
        IMPLEMENTATION,
        Path("src/fly_emotion/driving/v7.py"),
        Path("src/fly_emotion/driving/v7_temporal_audit.py"),
        Path("src/fly_emotion/driving/v7_retina_audit.py"),
        Path("src/fly_emotion/driving/retina.py"),
        Path("src/fly_emotion/connectome/graph.py"),
        Path("src/fly_emotion/driving/engine.py"),
        Path("configs/driving-v7.yaml"),
        Path("data/raw/malecns-v1.0/body-annotations.feather"),
        Path("data/raw/malecns-v1.0/body-neurotransmitters.feather"),
        Path("data/processed/malecns-v1.0/body_ids.npy"),
        Path("data/processed/malecns-v1.0/adjacency_raw.npz"),
        Path("data/processed/malecns-v1.0/adjacency_target_norm.npz"),
        Path("data/processed/malecns-v1.0/retina_map.npz"),
    ]
    hashes = {}
    for path in dependencies:
        with (root / path).open("rb") as source:
            hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-receptor-baseline-hold-audit-v1",
            "exploratory": True,
            "advance_allowed": False,
            "parameter_fitting": False,
            "v7_config_sha256": contract.sha256,
            "dependencies_sha256": hashes,
            "dynamics": "typed_visual_subgraph_v1",
            "geometry": "legacy_proxy_v2",
            "baseline_frames": 32,
            "pre_step_frames": 16,
            "post_step_frames": 32,
            "substeps_per_frame": 4,
            "direct_input": "R1-R6 only",
            "intervention": "hold selected receptor external code at first-frame baseline",
            "selection": selection,
            "control_selection": (
                "same-eye equal receptor count; nearest by hex-index distance, far >=6 then hash"
            ),
            "count_matched_not_synapse_matched": True,
            "sites_previously_observed": True,
            "driving_data_used": False,
        },
        "stimuli": {name: item.sha256 for name, item in stimuli.items()},
        "sites": sites,
        "responses": results,
        "limitations": [
            "Within-model intervention on six previously observed Mi9 cells, not new animals.",
            "Baseline holding removes visual modulation, not neurons or synapses.",
            "Modal hex assignments and index distances are proxies, not measured visual geometry.",
            "Matched receptor counts do not match synapse weights or total downstream effects.",
            "No rescue of missing columns and no validation of T4 direction or driving is tested.",
        ],
        "advance_to_central_complex": False,
    }


def summarize_input_coverage(matrix: sparse.csr_matrix, labels: np.ndarray) -> dict:
    categories = ("covered", "uncovered", "missing_coordinate", "missing_side")
    if matrix.shape[1] != len(labels) or not np.all(np.isin(labels, categories)):
        raise ValueError("every source must have a declared coverage category")
    totals = np.asarray(matrix.sum(axis=1, dtype=np.int64)).ravel()
    by_category = {
        label: np.asarray(matrix[:, labels == label].sum(axis=1, dtype=np.int64)).ravel()
        for label in categories
    }
    if not np.array_equal(sum(by_category.values()), totals):
        raise ValueError("input weights do not partition by coverage")
    present = totals > 0
    fractions = np.divide(
        by_category["covered"], totals, out=np.zeros(len(totals), dtype=float), where=present
    )
    return {
        "target_count": matrix.shape[0],
        "targets_with_input": int(present.sum()),
        "targets_without_input": int((~present).sum()),
        "edge_count": matrix.nnz,
        "total_synapse_weight": int(totals.sum()),
        "synapse_weight_by_coverage": {
            label: int(values.sum()) for label, values in by_category.items()
        },
        "source_count_by_coverage": {
            label: int(np.count_nonzero(np.diff(matrix[:, labels == label].tocsc().indptr)))
            for label in categories
        },
        "covered_weight_fraction": float(by_category["covered"].sum() / totals.sum())
        if totals.sum()
        else None,
        "covered_fraction_quantiles_among_present_targets": {
            str(q): float(np.quantile(fractions[present], q)) if present.any() else None
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "targets_without_covered_input": int(
            np.count_nonzero(present & (by_category["covered"] == 0))
        ),
        "targets_with_all_weight_covered": int(
            np.count_nonzero(present & (by_category["covered"] == totals))
        ),
    }


def evaluate_v7_t4_input_coverage(root: Path) -> dict:
    contract = V7Contract.load(root)
    retina, assignments = infer_retinal_columns(root)
    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
    table = feather.read_table(
        root / "data/raw/malecns-v1.0/body-annotations.feather",
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    ids = table["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(graph.body_ids, ids)
    valid = nodes < graph.node_count
    valid[valid] &= graph.body_ids[nodes[valid]] == ids[valid]
    table = table.iloc[np.flatnonzero(valid)]
    nodes = nodes[valid]
    types = np.full(graph.node_count, "", dtype=object)
    sides = np.zeros(graph.node_count, dtype=np.int8)
    coordinates = np.full((graph.node_count, 2), np.nan)
    types[nodes] = table["type"].fillna("").to_numpy()
    sides[nodes] = table["somaSide"].map({"L": -1, "R": 1}).fillna(0).to_numpy(dtype=np.int8)
    coordinates[nodes] = table[["assignedOlHex1", "assignedOlHex2"]].to_numpy(dtype=float)
    labels, _ = classify_column_coverage(coordinates, sides, assignments.coordinates, retina.side)
    branches = {
        "centre": ("Mi1", "Tm3"),
        "proximal": ("Mi4", "C3", "CT1"),
        "distal": ("Mi9",),
    }
    source_types = ("Mi1", "Tm3", "Mi4", "C3", "CT1", "Mi9")
    source_nodes = np.flatnonzero(np.isin(types, source_types))
    targets = np.flatnonzero(np.isin(types, ("T4a", "T4b", "T4c", "T4d")))
    matrix = graph.adjacency[targets][:, source_nodes]
    source_groups = {kind: (kind,) for kind in source_types} | branches
    populations = {}
    target_counts = {}
    for subtype in ("T4a", "T4b", "T4c", "T4d"):
        for eye, side in (("L", -1), ("R", 1), ("unknown", 0)):
            mask = (types[targets] == subtype) & (sides[targets] == side)
            if not mask.any():
                continue
            name = f"{subtype}_{eye}"
            populations[name] = {}
            target_counts[name] = int(mask.sum())
            for group, members in source_groups.items():
                included = np.isin(types[source_nodes], members)
                populations[name][group] = summarize_input_coverage(
                    matrix[mask][:, included], labels[source_nodes[included]]
                )
    target_weights = {}
    for kind in source_types:
        included = types[source_nodes] == kind
        weights = matrix[:, included]
        target_weights[kind] = {
            label: np.asarray(weights[:, labels[source_nodes[included]] == label].sum(axis=1))
            .ravel()
            .astype(int)
            .tolist()
            for label in ("covered", "uncovered", "missing_coordinate", "missing_side")
        }
    dependencies = [
        IMPLEMENTATION,
        Path("src/fly_emotion/driving/v7.py"),
        Path("src/fly_emotion/driving/v7_retina_audit.py"),
        Path("src/fly_emotion/driving/retina.py"),
        Path("src/fly_emotion/connectome/graph.py"),
        Path("configs/driving-v7.yaml"),
        Path("data/raw/malecns-v1.0/body-annotations.feather"),
        Path("data/processed/malecns-v1.0/body_ids.npy"),
        Path("data/processed/malecns-v1.0/adjacency_raw.npz"),
        Path("data/processed/malecns-v1.0/retina_map.npz"),
    ]
    hashes = {}
    for path in dependencies:
        with (root / path).open("rb") as source:
            hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-t4-input-coverage-v1",
            "exploratory": True,
            "advance_allowed": False,
            "v7_config_sha256": contract.sha256,
            "dependencies_sha256": hashes,
            "input_weights": "raw synapse counts; no row renormalization or sign filtering",
            "coverage": "source soma-side and optic-hex matched to retained modal receptor columns",
            "denominator": "all edges of each declared source group, including unlocated sources",
            "parameter_fitting": False,
            "neural_responses_used": False,
            "driving_data_used": False,
        },
        "branches": {name: list(members) for name, members in branches.items()},
        "target_counts": target_counts,
        "populations": populations,
        "targets": {
            "body_ids": graph.body_ids[targets].tolist(),
            "types": types[targets].tolist(),
            "sides": sides[targets].tolist(),
            "synapse_weights_by_source_and_coverage": target_weights,
        },
        "omitted_from_declared_branches": {
            "synapse_weight": int(
                graph.adjacency[targets].sum(dtype=np.int64) - matrix.sum(dtype=np.int64)
            ),
            "edge_count": graph.adjacency[targets].nnz - matrix.nnz,
        },
        "limitations": [
            "Source-column coverage does not prove visual response or pathway reachability.",
            "Tm3 and CT1 coordinates are unknown; their edges remain in all denominators.",
            "Branch grouping is an experimental model hypothesis, not a full T4 dendritic model.",
            "All targets remain included; no covered-subset functional pass is claimed.",
            "This anatomy-only audit cannot establish direction selectivity or allow deployment.",
        ],
        "advance_to_central_complex": False,
    }


def evaluate_v7_local_input_audit(root: Path) -> dict:
    contract = V7Contract.load(root)
    results = {}
    coverage_results = {}
    coverage = None
    sites = None
    selection = None
    for backend in ("linear_luminance", "signed_frame_difference"):
        probe = V7VisualProbe(
            root,
            retinal_backend=backend,
            retinal_geometry="legacy_proxy_v2",
            dynamics_backend="typed_visual_subgraph_v1",
            brain_substeps=4,
            baseline_frames=32,
        )
        if sites is None:
            sites, selection = select_local_sites(root, probe)
            coverage = mi9_column_coverage(root, probe)
        nodes = np.unique(
            np.concatenate(
                [
                    np.asarray(group, dtype=np.int32)
                    for site in sites
                    for eye in site["eyes"]
                    for group in eye["target_nodes"].values()
                ]
            )
        )
        nodes = np.union1d(nodes, coverage["nodes"])
        if np.intersect1d(nodes, probe.retina.node_indices).size:
            raise ValueError("readout cells must not receive direct stimulus")
        uniform_stimuli = build_step_stimuli(48, 24)
        uniform = {item.polarity: _step_trace(probe, item, nodes) for item in uniform_stimuli}
        coverage_results[backend] = summarize_mi9_coverage(coverage, uniform, nodes)
        rows = []
        for site in sites:
            for eye in site["eyes"]:
                masks = local_masks(48, 24, *eye["pixel"], eye["side"])
                for region, mask in masks.items():
                    input_mask = probe._sample_retina(mask.astype(np.float32)).astype(bool)
                    for polarity in ("on", "off"):
                        stimulus = local_step(mask, polarity)
                        trace = _step_trace(probe, stimulus, nodes) - uniform["sham"]
                        groups = {}
                        for kind, target_nodes in eye["target_nodes"].items():
                            indices = np.searchsorted(nodes, target_nodes)
                            sign = (
                                ON_RESPONSE_SIGN.get(kind)
                                if polarity == "on" and region == "centre"
                                else None
                            )
                            groups[kind] = summarize_step_response(
                                trace[:, indices], onset=16, expected_sign=sign
                            )
                        rows.append(
                            {
                                "optic_hex": site["optic_hex"],
                                "eye": eye["eye"],
                                "pixel": eye["pixel"],
                                "region": region,
                                "polarity": polarity,
                                "stimulus_sha256": stimulus.sha256,
                                "stimulated_pixels": int(mask.sum()),
                                "stimulated_receptors": int(input_mask.sum()),
                                "other_eye_stimulated_receptors": int(
                                    np.count_nonzero(
                                        input_mask & (probe.retina.side != eye["side"])
                                    )
                                ),
                                "populations": groups,
                            }
                        )
                for polarity in ("on", "off"):
                    rows.append(
                        {
                            "optic_hex": site["optic_hex"],
                            "eye": eye["eye"],
                            "pixel": eye["pixel"],
                            "region": "whole_screen",
                            "polarity": polarity,
                            "stimulus_sha256": next(
                                item.sha256 for item in uniform_stimuli if item.polarity == polarity
                            ),
                            "stimulated_pixels": 48 * 24,
                            "stimulated_receptors": probe.retina.size,
                            "other_eye_stimulated_receptors": int(
                                np.count_nonzero(probe.retina.side != eye["side"])
                            ),
                            "populations": {
                                kind: summarize_step_response(
                                    (uniform[polarity] - uniform["sham"])[
                                        :, np.searchsorted(nodes, target_nodes)
                                    ],
                                    onset=16,
                                    expected_sign=None,
                                )
                                for kind, target_nodes in eye["target_nodes"].items()
                            },
                        }
                    )
        results[backend] = rows
    dependencies = [
        IMPLEMENTATION,
        Path("src/fly_emotion/driving/v7.py"),
        Path("src/fly_emotion/driving/v7_temporal_audit.py"),
        Path("src/fly_emotion/driving/v7_retina_audit.py"),
        Path("src/fly_emotion/driving/retina.py"),
        Path("src/fly_emotion/connectome/graph.py"),
        Path("src/fly_emotion/driving/engine.py"),
        Path("configs/driving-v7.yaml"),
        Path("data/raw/malecns-v1.0/body-annotations.feather"),
        Path("data/raw/malecns-v1.0/body-neurotransmitters.feather"),
        Path("data/processed/malecns-v1.0/body_ids.npy"),
        Path("data/processed/malecns-v1.0/adjacency_raw.npz"),
        Path("data/processed/malecns-v1.0/adjacency_target_norm.npz"),
        Path("data/processed/malecns-v1.0/retina_map.npz"),
    ]
    hashes = {}
    for path in dependencies:
        with (root / path).open("rb") as source:
            hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-local-input-audit-v2",
            "coverage_definition": "same-eye modal receptor column, not synaptic reachability",
            "coverage_uses_response_labels": False,
            "exploratory": True,
            "advance_allowed": False,
            "v7_config_sha256": contract.sha256,
            "dependencies_sha256": hashes,
            "geometry": "legacy_proxy_v2",
            "image_width": 48,
            "image_height": 24,
            "dynamics": "typed_visual_subgraph_v1",
            "direct_input": "R1-R6 only",
            "target_direct_input_overlap": 0,
            "baseline_frames": 32,
            "pre_step_frames": 16,
            "post_step_frames": 32,
            "brain_substeps_per_frame": 4,
            "centre_radius_pixels": CENTRE_RADIUS,
            "outer_radius_pixels": OUTER_RADIUS,
            "reference": "same-frame whole-screen sham",
            "qualitative_polarity_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8891015/",
            "expected_sign_scope": "centre ON only; no polarity expectation assigned to annulus",
            "on_level": 0.8,
            "off_level": 0.2,
            "background_level": 0.5,
            "parameter_fitting": False,
            "driving_data_used": False,
            "selection": selection,
        },
        "sites": sites,
        "responses": results,
        "mi9_coverage": {
            "body_ids": coverage["body_ids"].tolist(),
            "sides": coverage["sides"].tolist(),
            "coordinates": [
                coordinate.tolist() if np.all(np.isfinite(coordinate)) else None
                for coordinate in coverage["coordinates"]
            ],
            "labels": coverage["labels"].tolist(),
            "same_column_receptor_count": [
                int(count) if count >= 0 else None for count in coverage["receptor_counts"]
            ],
            "responses": coverage_results,
        },
        "limitations": [
            "Three anatomy-selected columns are exploratory samples, not independent animals.",
            "Optic-hex and legacy camera coordinates are proxies, not measured receptive fields.",
            "Centre and annulus have unequal areas and stimulated receptor counts.",
            "whole_eye means camera hemifield; boundary receptors may belong to the other eye.",
            "Anatomy-covered site selection is not representative of all Mi9 cells.",
            "Coverage strata are observational; uncovered cells may receive neighbouring input.",
            "Latencies are simulation frames and late peaks can be window-censored.",
            "Tm3 and left-eye L3 lack optic-hex annotations and are excluded from local readout.",
            "This audit does not modify candidate parameters, visual gates or default runtime.",
        ],
        "advance_to_central_complex": False,
    }
