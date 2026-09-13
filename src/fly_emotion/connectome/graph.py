from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather
from scipy import sparse

from fly_emotion.data.audit import CANONICAL_NODE_RULE


@dataclass(frozen=True)
class ConnectomeGraph:
    body_ids: np.ndarray
    adjacency: sparse.csr_matrix
    metadata_path: Path

    @property
    def node_count(self) -> int:
        return int(self.body_ids.size)

    @property
    def edge_count(self) -> int:
        return int(self.adjacency.nnz)


def _map_ids(values: np.ndarray, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(body_ids, values)
    valid = indices < body_ids.size
    matched = np.zeros(values.shape, dtype=bool)
    matched[valid] = body_ids[indices[valid]] == values[valid]
    return indices, matched


def build_canonical_graph(
    annotations_path: Path,
    connections_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 4_000_000,
) -> ConnectomeGraph:
    annotations = feather.read_table(
        annotations_path, columns=["bodyId", "status", "superclass"], memory_map=True
    )
    canonical = annotations.filter(pc.is_valid(annotations["superclass"]))
    body_ids = np.sort(canonical["bodyId"].to_numpy(zero_copy_only=False).astype(np.int64))
    if np.unique(body_ids).size != body_ids.size:
        raise ValueError("canonical node IDs must be unique")

    edges = feather.read_table(
        connections_path, columns=["body_pre", "body_post", "weight"], memory_map=True
    )
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for batch in edges.to_batches(max_chunksize=batch_size):
        pre = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        post = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        weight = batch.column(2).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        pre_index, pre_valid = _map_ids(pre, body_ids)
        post_index, post_valid = _map_ids(post, body_ids)
        valid = pre_valid & post_valid
        # Matrix rows are targets and columns are sources: y = A @ x follows pre -> post.
        row_parts.append(post_index[valid].astype(np.int32))
        col_parts.append(pre_index[valid].astype(np.int32))
        weight_parts.append(weight[valid])

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    weights = np.concatenate(weight_parts)
    adjacency = sparse.coo_matrix(
        (weights, (rows, cols)), shape=(body_ids.size, body_ids.size), dtype=np.int32
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.sort_indices()

    # Normalize each target by its total incoming synapse count. Isolated rows stay zero.
    incoming = np.asarray(adjacency.sum(axis=1, dtype=np.int64)).ravel().astype(np.float32)
    scale = np.zeros_like(incoming)
    np.divide(1.0, incoming, out=scale, where=incoming > 0)
    normalized = (sparse.diags(scale, format="csr") @ adjacency).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "body_ids.npy", body_ids, allow_pickle=False)
    sparse.save_npz(output_dir / "adjacency_raw.npz", adjacency, compressed=True)
    sparse.save_npz(output_dir / "adjacency_target_norm.npz", normalized, compressed=True)
    metadata = {
        "release": "male-cns:v1.0",
        "node_rule": CANONICAL_NODE_RULE,
        "node_count": int(body_ids.size),
        "edge_count": int(adjacency.nnz),
        "synapse_weight_sum": int(adjacency.sum(dtype=np.int64)),
        "isolated_nodes": int(
            np.count_nonzero((incoming == 0) & (np.asarray(adjacency.sum(axis=0)).ravel() == 0))
        ),
        "orientation": "rows=postsynaptic targets, columns=presynaptic sources",
        "normalization": "per-target total incoming weight",
    }
    metadata_path = output_dir / "graph.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return ConnectomeGraph(
        body_ids=body_ids, adjacency=normalized.tocsr(), metadata_path=metadata_path
    )


def load_graph(output_dir: Path, *, normalized: bool = True) -> ConnectomeGraph:
    filename = "adjacency_target_norm.npz" if normalized else "adjacency_raw.npz"
    return ConnectomeGraph(
        body_ids=np.load(output_dir / "body_ids.npy", allow_pickle=False),
        adjacency=sparse.load_npz(output_dir / filename).tocsr(),
        metadata_path=output_dir / "graph.json",
    )
