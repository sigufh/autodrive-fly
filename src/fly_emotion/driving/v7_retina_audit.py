"""Audit and balanced-control construction for the v7 R1--R6 retinal proxy."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.retina import RetinaMap, load_or_build_retina_map

RETINA_AUDIT_CONFIG = Path("configs/driving-v7-retina-audit.yaml")
RETINA_AUDIT_IMPLEMENTATION = Path("src/fly_emotion/driving/v7_retina_audit.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class RetinalColumnAssignment:
    coordinates: np.ndarray
    weighted_means: np.ndarray
    weighted_spreads: np.ndarray
    ambiguity: np.ndarray
    dominant_partner_types: np.ndarray


@dataclass(frozen=True)
class BalancedRetinaControl:
    retina: RetinaMap
    pair_ids: np.ndarray
    pair_coordinates: np.ndarray
    left_positions: np.ndarray
    right_positions: np.ndarray
    source_indices: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(len(self.left_positions))


def infer_retinal_columns(root: Path) -> tuple[RetinaMap, RetinalColumnAssignment]:
    processed = root / "data/processed/malecns-v1.0"
    raw = root / "data/raw/malecns-v1.0"
    normalized_graph = load_graph(processed)
    graph = load_graph(processed, normalized=False)
    retina = load_or_build_retina_map(
        normalized_graph, raw / "body-annotations.feather", processed / "retina_map.npz"
    )
    annotations = feather.read_table(
        raw / "body-annotations.feather",
        columns=["bodyId", "type", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(graph.body_ids, ids)
    valid = nodes < graph.node_count
    valid[valid] &= graph.body_ids[nodes[valid]] == ids[valid]
    rows = annotations.iloc[np.flatnonzero(valid)]
    nodes = nodes[valid]
    coordinates = np.full((graph.node_count, 2), np.nan, dtype=np.float64)
    coordinates[nodes, 0] = rows["assignedOlHex1"].to_numpy(dtype=float)
    coordinates[nodes, 1] = rows["assignedOlHex2"].to_numpy(dtype=float)
    node_types = np.full(graph.node_count, "", dtype=object)
    node_types[nodes] = rows["type"].fillna("").to_numpy(dtype=object)

    outgoing = graph.adjacency[:, retina.node_indices].tocsc()
    modes = np.empty((retina.size, 2), dtype=np.int32)
    means = np.empty((retina.size, 2), dtype=np.float64)
    spreads = np.empty(retina.size, dtype=np.float64)
    ambiguity = np.empty(retina.size, dtype=np.float64)
    dominant_types = np.empty(retina.size, dtype=object)
    for column in range(retina.size):
        start, end = outgoing.indptr[column : column + 2]
        targets = outgoing.indices[start:end]
        weights = np.abs(outgoing.data[start:end]).astype(np.float64)
        known = np.all(np.isfinite(coordinates[targets]), axis=1)
        targets = targets[known]
        weights = weights[known]
        points = coordinates[targets].astype(np.int32)
        if not len(points):
            raise ValueError("retained receptor lacks a position-annotated partner")
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
        column_weights = np.bincount(inverse, weights=weights)
        modes[column] = unique[int(np.argmax(column_weights))]
        means[column] = np.average(points, axis=0, weights=weights)
        spreads[column] = np.sqrt(
            np.average(np.sum((points - means[column]) ** 2, axis=1), weights=weights)
        )
        ambiguity[column] = 1.0 - float(column_weights.max() / column_weights.sum())
        type_weights = Counter()
        for target, weight in zip(targets, weights, strict=True):
            type_weights[str(node_types[target])] += float(weight)
        dominant_types[column] = type_weights.most_common(1)[0][0]
    return retina, RetinalColumnAssignment(modes, means, spreads, ambiguity, dominant_types)


def build_balanced_retina_control(root: Path) -> BalancedRetinaControl:
    config = yaml.safe_load((root / RETINA_AUDIT_CONFIG).read_text(encoding="utf-8"))
    retina, assignment = infer_retinal_columns(root)
    by_eye: dict[int, dict[tuple[int, int], list[int]]] = {-1: {}, 1: {}}
    for index, (side, coordinate) in enumerate(
        zip(retina.side, assignment.coordinates, strict=True)
    ):
        by_eye[int(side)].setdefault(tuple(map(int, coordinate)), []).append(index)
    common = sorted(set(by_eye[-1]) & set(by_eye[1]))
    selected: list[int] = []
    pair_ids: list[int] = []
    pair_coordinates: list[tuple[int, int]] = []
    left_positions: list[int] = []
    right_positions: list[int] = []
    for coordinate in common:
        left = sorted(by_eye[-1][coordinate], key=lambda index: int(retina.body_ids[index]))
        right = sorted(by_eye[1][coordinate], key=lambda index: int(retina.body_ids[index]))
        count = min(len(left), len(right))
        for rank in range(count):
            pair_id = len(pair_coordinates)
            left_positions.append(len(selected))
            selected.append(left[rank])
            pair_ids.append(pair_id)
            right_positions.append(len(selected))
            selected.append(right[rank])
            pair_ids.append(pair_id)
            pair_coordinates.append(coordinate)
    source_indices = np.asarray(selected, dtype=np.int32)
    pair_coordinates_array = np.asarray(pair_coordinates, dtype=np.int32)
    width = int(config["image_width"])
    height = int(config["image_height"])
    low = pair_coordinates_array.min(axis=0)
    high = pair_coordinates_array.max(axis=0)
    normalized = (pair_coordinates_array - low) / np.maximum(high - low, 1)
    local_x = np.rint(normalized[:, 0] * (width // 2 - 1)).astype(np.int32)
    pixel_y = np.rint(normalized[:, 1] * (height - 1)).astype(np.int32)
    left_pixel_x = width // 2 - 1 - local_x
    right_pixel_x = width - 1 - left_pixel_x
    u = np.empty(len(selected), dtype=np.float32)
    v = np.empty(len(selected), dtype=np.float32)
    u[np.asarray(left_positions)] = left_pixel_x / (width - 1)
    u[np.asarray(right_positions)] = right_pixel_x / (width - 1)
    v[np.asarray(left_positions)] = pixel_y / (height - 1)
    v[np.asarray(right_positions)] = pixel_y / (height - 1)
    balanced = RetinaMap(
        node_indices=retina.node_indices[source_indices],
        body_ids=retina.body_ids[source_indices],
        u=u,
        v=v,
        side=retina.side[source_indices],
        mapping_version=3,
    )
    return BalancedRetinaControl(
        balanced,
        np.asarray(pair_ids, dtype=np.int32),
        pair_coordinates_array,
        np.asarray(left_positions, dtype=np.int32),
        np.asarray(right_positions, dtype=np.int32),
        source_indices,
    )


def evaluate_v7_retina_column_audit(root: Path) -> dict:
    config_path = root / RETINA_AUDIT_CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not config.get("exploratory") or config.get("advance_allowed"):
        raise ValueError("retina audit must remain exploratory and non-advancing")
    retina, assignment = infer_retinal_columns(root)
    balanced = build_balanced_retina_control(root)
    eye_counts = {
        "left": int(np.count_nonzero(retina.side < 0)),
        "right": int(np.count_nonzero(retina.side > 0)),
    }
    unique_columns = {
        "left": int(len(np.unique(assignment.coordinates[retina.side < 0], axis=0))),
        "right": int(len(np.unique(assignment.coordinates[retina.side > 0], axis=0))),
    }
    width = int(config["image_width"])
    height = int(config["image_height"])
    rng = np.random.default_rng(20260915)
    image = rng.random((height, width), dtype=np.float32)
    mirror = image[:, ::-1]
    left_drive = balanced.retina.encode(image)[balanced.left_positions]
    right_drive = balanced.retina.encode(mirror)[balanced.right_positions]
    return {
        "protocol": {
            "version": 7,
            "name": config["name"],
            "exploratory": True,
            "advance_allowed": False,
            "config_sha256": _sha256(config_path),
            "implementation_sha256": _sha256(root / RETINA_AUDIT_IMPLEMENTATION),
            "role": config["pairing"]["role"],
        },
        "legacy_mapping": {
            "total_receptors": retina.size,
            "eye_counts": eye_counts,
            "right_to_left_receptor_ratio": eye_counts["right"] / eye_counts["left"],
            "unique_columns": unique_columns,
            "weighted_mean_differs_from_modal_column_fraction": float(
                np.mean(
                    np.linalg.norm(assignment.weighted_means - assignment.coordinates, axis=1)
                    > 1e-9
                )
            ),
            "ambiguous_column_fraction": float(np.mean(assignment.ambiguity > 0)),
            "spread_quantiles": {
                str(quantile): float(np.quantile(assignment.weighted_spreads, quantile))
                for quantile in (0.5, 0.9, 0.95, 0.99, 1.0)
            },
            "dominant_position_partner_types": dict(
                Counter(map(str, assignment.dominant_partner_types)).most_common()
            ),
        },
        "balanced_common_column_control": {
            "common_columns": int(len(np.unique(balanced.pair_coordinates, axis=0))),
            "pairs": balanced.pair_count,
            "receptors_per_eye": balanced.pair_count,
            "total_receptors": balanced.retina.size,
            "discarded_left_receptors": eye_counts["left"] - balanced.pair_count,
            "discarded_right_receptors": eye_counts["right"] - balanced.pair_count,
            "exact_mirror_drive": bool(np.array_equal(left_drive, right_drive)),
            "maximum_mirror_drive_error": float(np.max(np.abs(left_drive - right_drive))),
            "mapping_sha256": hashlib.sha256(
                balanced.retina.body_ids.tobytes()
                + balanced.retina.u.tobytes()
                + balanced.retina.v.tobytes()
                + balanced.pair_ids.tobytes()
            ).hexdigest(),
        },
        "diagnosis": {
            "contact_weighted_blur_is_common": bool(np.mean(assignment.ambiguity > 0) > 0.10),
            "legacy_eye_input_count_is_balanced": eye_counts["left"] == eye_counts["right"],
            "balanced_control_is_biological_reconstruction": False,
            "interpretation": (
                "Most retained receptors map unambiguously to one optic-hex column, but "
                "MaleCNS receptor coverage is strongly right-heavy. The paired subset is "
                "an engineering ablation that changes both input count and spatial coverage."
            ),
        },
        "advance_to_central_complex": False,
    }
