from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather

from .graph import ConnectomeGraph


def _positions_and_classes(
    annotations_path: Path, body_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    table = feather.read_table(
        annotations_path,
        columns=["bodyId", "superclass", "somaLocation", "tosomaLocation"],
        memory_map=True,
    )
    table = table.filter(pc.is_valid(table["superclass"])).sort_by("bodyId")
    if not np.array_equal(table["bodyId"].to_numpy(), body_ids):
        raise ValueError("annotations do not match canonical graph IDs")
    names = sorted(set(map(str, table["superclass"].to_pylist())))
    lookup = {name: index for index, name in enumerate(names)}
    class_ids = np.asarray([lookup[str(value)] for value in table["superclass"].to_pylist()])
    positions = np.full((body_ids.size, 3), np.nan, dtype=np.float32)
    for index, row in enumerate(table.select(["somaLocation", "tosomaLocation"]).to_pylist()):
        location = row["somaLocation"] or row["tosomaLocation"]
        if location:
            positions[index] = np.asarray(location, dtype=np.float32) * 0.008
    return positions, class_ids, names


def build_pathways(
    graph: ConnectomeGraph,
    annotations_path: Path,
    output_path: Path,
    *,
    strongest_edges: int = 40_000,
) -> dict[str, Any]:
    """Build browser LODs while accounting for every canonical graph edge."""
    raw = graph.adjacency.tocsr()
    positions, class_ids, class_names = _positions_and_classes(annotations_path, graph.body_ids)
    group_count = len(class_names)
    edge_counts = np.zeros(group_count * group_count, dtype=np.int64)
    weight_sums = np.zeros(group_count * group_count, dtype=np.int64)

    # Stream rows so peak memory stays low. Every canonical edge contributes to
    # exactly one superclass pair, including edges whose endpoints lack soma coordinates.
    row_chunk = 4096
    for start in range(0, raw.shape[0], row_chunk):
        end = min(start + row_chunk, raw.shape[0])
        begin_edge = raw.indptr[start]
        end_edge = raw.indptr[end]
        if begin_edge == end_edge:
            continue
        target = np.repeat(class_ids[start:end], np.diff(raw.indptr[start : end + 1]))
        source = class_ids[raw.indices[begin_edge:end_edge]]
        pair = source * group_count + target
        edge_counts += np.bincount(pair, minlength=edge_counts.size)
        weight_sums += np.bincount(
            pair, weights=raw.data[begin_edge:end_edge], minlength=weight_sums.size
        ).astype(np.int64)

    centroids = []
    for group in range(group_count):
        subset = positions[(class_ids == group) & np.isfinite(positions).all(axis=1)]
        centroids.append(np.median(subset, axis=0).tolist() if len(subset) else None)

    bundled_paths = []
    for pair, edge_count in enumerate(edge_counts):
        if edge_count == 0:
            continue
        source, target = divmod(pair, group_count)
        bundled_paths.append(
            {
                "source": source,
                "target": target,
                "edge_count": int(edge_count),
                "synapse_weight": int(weight_sums[pair]),
                "from": centroids[source],
                "to": centroids[target],
                "positioned": centroids[source] is not None and centroids[target] is not None,
            }
        )

    valid_nodes = np.isfinite(positions).all(axis=1)
    edge_targets = np.repeat(np.arange(raw.shape[0]), np.diff(raw.indptr))
    valid_edge = valid_nodes[raw.indices] & valid_nodes[edge_targets]
    candidate_flat = np.flatnonzero(valid_edge)
    candidate_weights = raw.data[candidate_flat]
    keep_count = min(strongest_edges, candidate_flat.size)
    if keep_count:
        chosen_local = np.argpartition(candidate_weights, -keep_count)[-keep_count:]
        chosen_flat = candidate_flat[chosen_local]
        order = np.argsort(raw.data[chosen_flat])[::-1]
        chosen_flat = chosen_flat[order]
        target_nodes = edge_targets[chosen_flat]
        source_nodes = raw.indices[chosen_flat]
    else:
        chosen_flat = np.empty(0, dtype=np.int64)
        target_nodes = source_nodes = np.empty(0, dtype=np.int64)

    strong_paths = [
        {
            "pre": int(graph.body_ids[source]),
            "post": int(graph.body_ids[target]),
            "weight": int(raw.data[flat]),
            "from": positions[source].tolist(),
            "to": positions[target].tolist(),
        }
        for flat, source, target in zip(chosen_flat, source_nodes, target_nodes, strict=True)
    ]
    payload = {
        "release": "male-cns:v1.0",
        "units": "micrometres",
        "all_edges_accounted": int(edge_counts.sum()),
        "all_synapse_weight_accounted": int(weight_sums.sum()),
        "classes": class_names,
        "class_centroids": centroids,
        "bundled_paths": bundled_paths,
        "strong_paths": strong_paths,
        "strong_path_policy": f"top {keep_count} edges with positioned endpoints",
    }
    payload["topology_layout"] = "schematic_not_anatomical"
    payload["topology_nodes"] = [
        {
            "id": i,
            "name": name,
            "neurons": int(np.count_nonzero(class_ids == i)),
            "positioned": centroids[i] is not None,
        }
        for i, name in enumerate(class_names)
    ]
    payload["exported_bundle_edges"] = sum(p["edge_count"] for p in bundled_paths)
    payload["exported_bundle_weight"] = sum(p["synapse_weight"] for p in bundled_paths)
    if payload["exported_bundle_edges"] != graph.edge_count:
        raise ValueError("exported topology omits canonical edges")
    if payload["exported_bundle_weight"] != int(raw.sum(dtype=np.int64)):
        raise ValueError("exported topology omits synapse weights")
    if payload["all_edges_accounted"] != graph.edge_count:
        raise ValueError("not every canonical edge was assigned to a pathway bundle")
    if payload["all_synapse_weight_accounted"] != int(raw.sum(dtype=np.int64)):
        raise ValueError("pathway bundle weights do not match the canonical graph")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload
