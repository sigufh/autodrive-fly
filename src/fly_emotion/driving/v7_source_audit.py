"""Retrospective source-level anatomy audit for the blocked v7 T4 model."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE

AUDIT_CONFIG = Path("configs/driving-v7-t4-source-audit.yaml")
AUDIT_IMPLEMENTATION = Path("src/fly_emotion/driving/v7_source_audit.py")
CARDINAL_ORDER = ("left", "right", "up", "down")
CARDINAL_VECTORS = {
    "left": np.asarray((-1.0, 0.0)),
    "right": np.asarray((1.0, 0.0)),
    "up": np.asarray((0.0, -1.0)),
    "down": np.asarray((0.0, 1.0)),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tables(root: Path) -> tuple:
    processed = root / "data/processed/malecns-v1.0"
    raw = root / "data/raw/malecns-v1.0"
    graph = load_graph(processed, normalized=False)
    annotations = feather.read_table(
        raw / "body-annotations.feather",
        columns=[
            "bodyId",
            "type",
            "somaSide",
            "rootSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
        memory_map=True,
    ).to_pandas()
    ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(graph.body_ids, ids)
    valid = nodes < graph.node_count
    valid[valid] &= graph.body_ids[nodes[valid]] == ids[valid]
    annotations = annotations.iloc[np.flatnonzero(valid)].copy()
    annotations["node"] = nodes[valid]
    neurotransmitters = feather.read_table(
        raw / "body-neurotransmitters.feather",
        columns=["body", "consensus_nt"],
        memory_map=True,
    ).to_pandas()
    return graph, annotations, neurotransmitters


def _source_metadata(annotations, neurotransmitters, source_types: tuple[str, ...]) -> dict:
    transmitter = neurotransmitters.rename(columns={"body": "bodyId"})
    joined = annotations.merge(transmitter, on="bodyId", how="left")
    result = {}
    for source_type in source_types:
        rows = joined.loc[joined["type"].eq(source_type)]
        coordinates = rows[["assignedOlHex1", "assignedOlHex2"]].notna().all(axis=1)
        result[source_type] = {
            "cells": int(len(rows)),
            "optic_hex_coordinate_cells": int(coordinates.sum()),
            "optic_hex_coordinate_fraction": float(coordinates.mean()) if len(rows) else 0.0,
            "soma_side_counts": {
                side: int(rows["somaSide"].fillna("").eq(side).sum()) for side in ("L", "R", "")
            },
            "root_side_nonempty_fraction": float(rows["rootSide"].fillna("").ne("").mean())
            if len(rows)
            else 0.0,
            "consensus_neurotransmitters": {
                str(name): int(count)
                for name, count in rows["consensus_nt"].fillna("missing").value_counts().items()
            },
        }
    return result


def _t4_records(graph, annotations, source_types: tuple[str, ...]) -> list[dict]:
    node_types = np.full(graph.node_count, "", dtype=object)
    node_sides = np.full(graph.node_count, "", dtype=object)
    coordinates = np.full((graph.node_count, 2), np.nan, dtype=np.float64)
    nodes = annotations["node"].to_numpy(dtype=np.int32)
    node_types[nodes] = annotations["type"].fillna("").to_numpy(dtype=object)
    node_sides[nodes] = annotations["somaSide"].fillna("").to_numpy(dtype=object)
    coordinates[nodes, 0] = annotations["assignedOlHex1"].to_numpy(dtype=float)
    coordinates[nodes, 1] = annotations["assignedOlHex2"].to_numpy(dtype=float)

    records = []
    for subtype in ("a", "b", "c", "d"):
        for side in ("L", "R"):
            targets = annotations.loc[
                annotations["type"].eq(f"T4{subtype}") & annotations["somaSide"].eq(side),
                ["bodyId", "node"],
            ]
            expected = (
                HORIZONTAL_PREFERENCE[(subtype, side)]
                if subtype in {"a", "b"}
                else VERTICAL_PREFERENCE[subtype]
            )
            for body_id, target in targets.itertuples(index=False, name=None):
                row = graph.adjacency.getrow(int(target))
                sources = row.indices
                weights = np.abs(row.data).astype(np.float64)
                centroids = {}
                source_weights = {}
                opposite_side_weight = {}
                for source_type in source_types:
                    selected = node_types[sources] == source_type
                    source_weights[source_type] = float(weights[selected].sum())
                    denominator = max(source_weights[source_type], 1.0)
                    opposite_side_weight[source_type] = float(
                        weights[selected & (node_sides[sources] != side)].sum() / denominator
                    )
                    located = selected & np.all(np.isfinite(coordinates[sources]), axis=1)
                    centroids[source_type] = (
                        np.average(coordinates[sources[located]], axis=0, weights=weights[located])
                        if np.any(located)
                        else np.full(2, np.nan)
                    )
                records.append(
                    {
                        "body_id": int(body_id),
                        "subtype": subtype,
                        "side": side,
                        "expected": expected,
                        "centroids": centroids,
                        "weights": source_weights,
                        "opposite_side_weight": opposite_side_weight,
                    }
                )
    return records


def _weighted_centroid(record: dict, source_types: tuple[str, ...]) -> np.ndarray:
    points = []
    weights = []
    for source_type in source_types:
        point = record["centroids"][source_type]
        weight = record["weights"][source_type]
        if np.all(np.isfinite(point)) and weight > 0:
            points.append(point)
            weights.append(weight)
    if not points:
        return np.full(2, np.nan)
    return np.average(points, axis=0, weights=weights)


def _fold(record: dict, seed: str, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}:{record['body_id']}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _cross_validated_offset(
    records: list[dict],
    direct_sources: tuple[str, ...],
    offset_sources: tuple[str, ...],
    *,
    seed: str,
    folds: int,
) -> dict:
    offsets = np.asarray(
        [
            _weighted_centroid(record, direct_sources) - _weighted_centroid(record, offset_sources)
            for record in records
        ]
    )
    valid = np.all(np.isfinite(offsets), axis=1) & (np.linalg.norm(offsets, axis=1) > 1e-12)
    sides = np.asarray([record["side"] for record in records])
    subtypes = np.asarray([record["subtype"] for record in records])
    expected = np.stack([CARDINAL_VECTORS[record["expected"]] for record in records])
    truth = np.argmax(
        expected @ np.stack([CARDINAL_VECTORS[name] for name in CARDINAL_ORDER]).T, axis=1
    )
    fold_ids = np.asarray([_fold(record, seed, folds) for record in records])
    correct: list[bool] = []
    angles: list[float] = []
    group_correct: dict[str, list[bool]] = {}
    group_angles: dict[str, list[float]] = {}
    for held_out_fold in range(folds):
        prediction = np.full((len(records), 2), np.nan)
        for side in ("L", "R"):
            train = valid & (fold_ids != held_out_fold) & (sides == side)
            test = valid & (fold_ids == held_out_fold) & (sides == side)
            covariance = np.zeros((2, 2), dtype=np.float64)
            complete = True
            for subtype in ("a", "b", "c", "d"):
                group = train & (subtypes == subtype)
                if not np.any(group):
                    complete = False
                    break
                normalized = offsets[group] / np.linalg.norm(offsets[group], axis=1, keepdims=True)
                covariance += normalized.T @ expected[group] / np.count_nonzero(group)
            if not complete:
                continue
            left, _, right = np.linalg.svd(covariance, full_matrices=False)
            prediction[test] = offsets[test] @ (left @ right)
        test = valid & (fold_ids == held_out_fold) & np.all(np.isfinite(prediction), axis=1)
        normalized = prediction[test] / np.linalg.norm(prediction[test], axis=1, keepdims=True)
        predicted_labels = np.argmax(
            normalized @ np.stack([CARDINAL_VECTORS[name] for name in CARDINAL_ORDER]).T,
            axis=1,
        )
        test_correct = predicted_labels == truth[test]
        test_angles = np.degrees(
            np.arccos(np.clip(np.sum(normalized * expected[test], axis=1), -1.0, 1.0))
        )
        correct.extend(test_correct.tolist())
        angles.extend(test_angles.tolist())
        test_indices = np.flatnonzero(test)
        for local, index in enumerate(test_indices):
            key = f"T4{subtypes[index]}_{sides[index]}"
            group_correct.setdefault(key, []).append(bool(test_correct[local]))
            group_angles.setdefault(key, []).append(float(test_angles[local]))
    return {
        "direct_sources": list(direct_sources),
        "offset_sources": list(offset_sources),
        "valid_cells": int(np.count_nonzero(valid)),
        "five_fold_accuracy": float(np.mean(correct)) if correct else 0.0,
        "five_fold_median_angle_error_degrees": (float(np.median(angles)) if angles else 180.0),
        "by_population": {
            key: {
                "count": len(group_correct[key]),
                "accuracy": float(np.mean(group_correct[key])),
                "median_angle_error_degrees": float(np.median(group_angles[key])),
            }
            for key in sorted(group_correct)
        },
    }


def _fit_transform(
    offsets: np.ndarray, expected: np.ndarray, subtypes: np.ndarray, mask: np.ndarray
) -> np.ndarray | None:
    covariance = np.zeros((2, 2), dtype=np.float64)
    for subtype in ("a", "b", "c", "d"):
        group = mask & (subtypes == subtype)
        if not np.any(group):
            return None
        normalized = offsets[group] / np.linalg.norm(offsets[group], axis=1, keepdims=True)
        covariance += normalized.T @ expected[group] / np.count_nonzero(group)
    left, _, right = np.linalg.svd(covariance, full_matrices=False)
    return left @ right


def _evaluate_transform(
    offsets: np.ndarray, expected: np.ndarray, mask: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = offsets[mask] @ transform
    prediction /= np.linalg.norm(prediction, axis=1, keepdims=True)
    cardinal = np.stack([CARDINAL_VECTORS[name] for name in CARDINAL_ORDER])
    predicted_labels = np.argmax(prediction @ cardinal.T, axis=1)
    expected_labels = np.argmax(expected[mask] @ cardinal.T, axis=1)
    correct = predicted_labels == expected_labels
    angles = np.degrees(np.arccos(np.clip(np.sum(prediction * expected[mask], axis=1), -1.0, 1.0)))
    return correct, angles


def _nested_branch_validation(
    records: list[dict],
    anchor: tuple[str, ...],
    candidates: list[tuple[str, ...]],
    *,
    seed: str,
    outer_folds: int,
    inner_folds: int,
) -> dict:
    offsets = {
        candidate: np.asarray(
            [
                _weighted_centroid(record, anchor) - _weighted_centroid(record, candidate)
                for record in records
            ]
        )
        for candidate in candidates
    }
    valid = {
        candidate: np.all(np.isfinite(values), axis=1) & (np.linalg.norm(values, axis=1) > 1e-12)
        for candidate, values in offsets.items()
    }
    sides = np.asarray([record["side"] for record in records])
    subtypes = np.asarray([record["subtype"] for record in records])
    expected = np.stack([CARDINAL_VECTORS[record["expected"]] for record in records])
    outer = np.asarray([_fold(record, f"{seed}:outer", outer_folds) for record in records])
    inner = np.asarray([_fold(record, f"{seed}:inner", inner_folds) for record in records])
    all_correct: list[bool] = []
    all_angles: list[float] = []
    selected_by_fold = []
    outer_reports = []
    for outer_fold in range(outer_folds):
        inner_scores = []
        for candidate in candidates:
            candidate_correct: list[bool] = []
            candidate_angles: list[float] = []
            for inner_fold in range(inner_folds):
                for side in ("L", "R"):
                    train = (
                        valid[candidate]
                        & (outer != outer_fold)
                        & (inner != inner_fold)
                        & (sides == side)
                    )
                    test = (
                        valid[candidate]
                        & (outer != outer_fold)
                        & (inner == inner_fold)
                        & (sides == side)
                    )
                    transform = _fit_transform(offsets[candidate], expected, subtypes, train)
                    if transform is not None and np.any(test):
                        correct, angles = _evaluate_transform(
                            offsets[candidate], expected, test, transform
                        )
                        candidate_correct.extend(correct.tolist())
                        candidate_angles.extend(angles.tolist())
            inner_scores.append(
                (
                    float(np.mean(candidate_correct)) if candidate_correct else 0.0,
                    float(np.median(candidate_angles)) if candidate_angles else 180.0,
                    candidate,
                )
            )
        inner_scores.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = inner_scores[0][2]
        selected_by_fold.append("+".join(selected))
        fold_correct: list[bool] = []
        fold_angles: list[float] = []
        for side in ("L", "R"):
            train = valid[selected] & (outer != outer_fold) & (sides == side)
            test = valid[selected] & (outer == outer_fold) & (sides == side)
            transform = _fit_transform(offsets[selected], expected, subtypes, train)
            if transform is not None and np.any(test):
                correct, angles = _evaluate_transform(offsets[selected], expected, test, transform)
                fold_correct.extend(correct.tolist())
                fold_angles.extend(angles.tolist())
        all_correct.extend(fold_correct)
        all_angles.extend(fold_angles)
        outer_reports.append(
            {
                "fold": outer_fold,
                "selected_offset_sources": list(selected),
                "inner_selection_accuracy": inner_scores[0][0],
                "outer_cells": len(fold_correct),
                "outer_accuracy": float(np.mean(fold_correct)),
                "outer_median_angle_error_degrees": float(np.median(fold_angles)),
            }
        )
    values, counts = np.unique(selected_by_fold, return_counts=True)
    consensus_index = int(np.argmax(counts))
    consensus = str(values[consensus_index])
    return {
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "outer_cells": len(all_correct),
        "outer_accuracy": float(np.mean(all_correct)),
        "outer_median_angle_error_degrees": float(np.median(all_angles)),
        "selected_source_by_outer_fold": selected_by_fold,
        "selection_consensus": consensus.split("+"),
        "selection_consensus_fraction": float(counts[consensus_index] / outer_folds),
        "folds": outer_reports,
    }


def _frozen_consensus_transforms(
    records: list[dict], anchor: tuple[str, ...], selected: tuple[str, ...]
) -> dict[str, list[list[float]]]:
    offsets = np.asarray(
        [
            _weighted_centroid(record, anchor) - _weighted_centroid(record, selected)
            for record in records
        ]
    )
    valid = np.all(np.isfinite(offsets), axis=1) & (np.linalg.norm(offsets, axis=1) > 1e-12)
    sides = np.asarray([record["side"] for record in records])
    subtypes = np.asarray([record["subtype"] for record in records])
    expected = np.stack([CARDINAL_VECTORS[record["expected"]] for record in records])
    result = {}
    for side in ("L", "R"):
        transform = _fit_transform(offsets, expected, subtypes, valid & (sides == side))
        if transform is None:
            raise ValueError(f"cannot fit frozen T4 axis for side {side}")
        result[side] = transform.tolist()
    return result


def evaluate_v7_t4_source_audit(root: Path) -> dict:
    config_path = root / AUDIT_CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not config.get("exploratory") or config.get("advance_allowed"):
        raise ValueError("T4 source audit must remain retrospective and non-advancing")
    graph, annotations, neurotransmitters = _load_tables(root)
    source_types = tuple(
        dict.fromkeys(
            source
            for family in config["source_roles"].values()
            for role in family.values()
            for source in role
        )
    )
    metadata = _source_metadata(annotations, neurotransmitters, source_types)
    t4_sources = tuple(
        dict.fromkeys(source for role in config["source_roles"]["T4"].values() for source in role)
    )
    records = _t4_records(graph, annotations, t4_sources)
    folds = int(config["cross_validation"]["folds"])
    seed = str(config["cross_validation"]["split_seed"])
    anchor = (config["coordinate_screen"]["T4_direct_anchor"],)
    candidate_sources = tuple(config["coordinate_screen"]["T4_candidate_offset_sources"])
    candidates = []
    for count in range(1, len(candidate_sources) + 1):
        for selected in itertools.combinations(candidate_sources, count):
            candidates.append(
                _cross_validated_offset(records, anchor, selected, seed=seed, folds=folds)
            )
    candidates.sort(
        key=lambda result: (
            -result["five_fold_accuracy"],
            result["five_fold_median_angle_error_degrees"],
            result["offset_sources"],
        )
    )
    reference_sources = list(config["coordinate_screen"]["T4_reference_offset_sources"])
    reference = next(item for item in candidates if item["offset_sources"] == reference_sources)
    candidate_tuples = [tuple(item["offset_sources"]) for item in candidates]
    nested = _nested_branch_validation(
        records,
        anchor,
        candidate_tuples,
        seed=seed,
        outer_folds=int(config["cross_validation"]["nested_outer_folds"]),
        inner_folds=int(config["cross_validation"]["nested_inner_folds"]),
    )
    nested_thresholds = config["cross_validation"]["nested_gates"]
    nested_gates = {
        "outer_accuracy": nested["outer_accuracy"] >= nested_thresholds["minimum_outer_accuracy"],
        "outer_angle": nested["outer_median_angle_error_degrees"]
        <= nested_thresholds["maximum_outer_median_angle_error_degrees"],
        "selection_consensus": nested["selection_consensus_fraction"]
        >= nested_thresholds["minimum_selection_consensus_fraction"],
    }
    nested["preregistered_gates"] = nested_gates
    nested["nested_validation_pass"] = all(nested_gates.values())
    frozen_transforms = _frozen_consensus_transforms(
        records, anchor, tuple(nested["selection_consensus"])
    )

    connectivity = {}
    for source_type in t4_sources:
        weights = np.asarray([record["weights"][source_type] for record in records])
        opposite = np.asarray([record["opposite_side_weight"][source_type] for record in records])
        positive = weights > 0
        connectivity[source_type] = {
            "target_presence_fraction": float(np.mean(positive)),
            "median_synapse_weight_when_present": (
                float(np.median(weights[positive])) if np.any(positive) else 0.0
            ),
            "weighted_opposite_soma_side_fraction": float(
                np.average(opposite, weights=np.maximum(weights, 1.0))
            ),
        }

    best = candidates[0]
    improvement = best["five_fold_accuracy"] - reference["five_fold_accuracy"]
    return {
        "protocol": {
            "version": 7,
            "name": config["name"],
            "exploratory": True,
            "advance_allowed": False,
            "config_sha256": _sha256(config_path),
            "implementation_sha256": _sha256(root / AUDIT_IMPLEMENTATION),
            "driving_data_used": False,
            "visual_stimulus_responses_used": False,
            "selection_bias": (
                "all T4 cells participate across five folds and candidate branches are ranked "
                "on the same audit; scores are retrospective diagnostics, not fresh validation"
            ),
        },
        "source_roles": config["source_roles"],
        "source_metadata": metadata,
        "T4_target_connectivity": connectivity,
        "coordinate_branch_screen": {
            "folds": folds,
            "candidate_count": len(candidates),
            "ranked_candidates": candidates,
            "reference_all_coordinate_bearing_delayed_sources": reference,
            "best_exploratory_candidate": best,
            "accuracy_improvement_over_reference": float(improvement),
        },
        "nested_branch_validation": nested,
        "frozen_axis_calibration": {
            "direct_sources": list(anchor),
            "offset_sources": nested["selection_consensus"],
            "fit_cells": len(records),
            "transforms_by_eye": frozen_transforms,
            "usage_boundary": (
                "allowed only as an experimental retinal geometry; controlled visual "
                "response and mirror gates remain independent requirements"
            ),
        },
        "diagnosis": {
            "Tm3_has_no_optic_hex_coordinates": metadata["Tm3"]["optic_hex_coordinate_cells"] == 0,
            "CT1_has_no_optic_hex_coordinates": metadata["CT1"]["optic_hex_coordinate_cells"] == 0,
            "mixed_delayed_neurotransmitter_classes": len(
                {
                    transmitter
                    for source in ("Mi4", "Mi9", "C3")
                    for transmitter in metadata[source]["consensus_neurotransmitters"]
                }
            )
            > 1,
            "branch_aggregation_degrades_T4_direction_geometry": improvement > 0.20,
            "interpretation": (
                "The v1 delayed pool merges proximal GABAergic Mi4/C3, distal "
                "glutamatergic Mi9 and unlocated CT1. This destroys source-specific "
                "spatial roles and the correlator also discards transmitter sign."
            ),
        },
        "next_candidate_contract": config["next_candidate_contract"],
        "literature": config["literature"],
        "advance_to_columnar_branched_correlator_v2": False,
        "advance_to_central_complex": False,
        "stop_reason": (
            "retrospective anatomy audit supports branch separation, but a frozen "
            "candidate requires fresh controlled-vision evidence"
        ),
    }
