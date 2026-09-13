from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy import sparse

from fly_emotion.connectome.graph import ConnectomeGraph


@dataclass(frozen=True)
class RetinaMap:
    """Proxy projection from a camera image onto mapped R1--R6 neurons.

    MaleCNS does not assign optic-lobe hex coordinates directly to most
    photoreceptors.  Coordinates here are the contact-weighted mean hex
    coordinates of each receptor's position-annotated postsynaptic partners.
    """

    node_indices: np.ndarray
    body_ids: np.ndarray
    u: np.ndarray
    v: np.ndarray
    side: np.ndarray

    @property
    def size(self) -> int:
        return int(self.node_indices.size)

    def encode(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2 or image.size == 0:
            raise ValueError("retinal stimulus must be a non-empty 2D array")
        height, width = image.shape
        x = np.clip(np.rint(self.u * (width - 1)).astype(np.int32), 0, width - 1)
        y = np.clip(np.rint(self.v * (height - 1)).astype(np.int32), 0, height - 1)
        return np.asarray(image[y, x], dtype=np.float32)


def _normalized(values: np.ndarray) -> np.ndarray:
    low, high = np.nanmin(values), np.nanmax(values)
    if high <= low:
        return np.full(values.shape, 0.5, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def build_retina_map(
    graph: ConnectomeGraph, annotations_path: Path, output_path: Path
) -> RetinaMap:
    annotations = feather.read_table(
        annotations_path,
        columns=["bodyId", "type", "rootSide", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    receptor_rows = annotations[annotations["type"].eq("R1-R6")].copy()
    receptor_ids = receptor_rows["bodyId"].to_numpy(dtype=np.int64)
    receptor_nodes = np.searchsorted(graph.body_ids, receptor_ids)
    valid_nodes = receptor_nodes < graph.node_count
    valid_nodes[valid_nodes] &= (
        graph.body_ids[receptor_nodes[valid_nodes]] == receptor_ids[valid_nodes]
    )
    receptor_rows = receptor_rows.iloc[np.flatnonzero(valid_nodes)].reset_index(drop=True)
    receptor_nodes = receptor_nodes[valid_nodes].astype(np.int32)

    coordinate_rows = annotations.dropna(subset=["assignedOlHex1", "assignedOlHex2"])
    coordinate_nodes = np.searchsorted(
        graph.body_ids, coordinate_rows["bodyId"].to_numpy(dtype=np.int64)
    )
    valid_coordinates = coordinate_nodes < graph.node_count
    valid_coordinates[valid_coordinates] &= (
        graph.body_ids[coordinate_nodes[valid_coordinates]]
        == coordinate_rows["bodyId"].to_numpy(dtype=np.int64)[valid_coordinates]
    )
    hex1 = np.full(graph.node_count, np.nan, dtype=np.float32)
    hex2 = np.full(graph.node_count, np.nan, dtype=np.float32)
    rows = coordinate_rows.iloc[np.flatnonzero(valid_coordinates)]
    nodes = coordinate_nodes[valid_coordinates]
    hex1[nodes] = rows["assignedOlHex1"].to_numpy(dtype=np.float32)
    hex2[nodes] = rows["assignedOlHex2"].to_numpy(dtype=np.float32)

    outgoing = graph.adjacency[:, receptor_nodes].tocsc()
    mapped_nodes: list[int] = []
    mapped_ids: list[int] = []
    mapped_h1: list[float] = []
    mapped_h2: list[float] = []
    mapped_side: list[int] = []
    for column, node in enumerate(receptor_nodes):
        start, end = outgoing.indptr[column : column + 2]
        targets = outgoing.indices[start:end]
        weights = np.abs(outgoing.data[start:end]).astype(np.float64)
        known = np.isfinite(hex1[targets]) & np.isfinite(hex2[targets])
        if not np.any(known):
            continue
        targets, weights = targets[known], weights[known]
        mapped_nodes.append(int(node))
        mapped_ids.append(int(graph.body_ids[node]))
        mapped_h1.append(float(np.average(hex1[targets], weights=weights)))
        mapped_h2.append(float(np.average(hex2[targets], weights=weights)))
        mapped_side.append(-1 if receptor_rows.iloc[column]["rootSide"] == "L" else 1)

    node_indices = np.asarray(mapped_nodes, dtype=np.int32)
    side = np.asarray(mapped_side, dtype=np.int8)
    local_u = _normalized(np.asarray(mapped_h1, dtype=np.float32))
    # Keep the two anatomical eyes in separate visual hemifields. This is a
    # documented stimulus proxy, not a calibrated fly compound-eye model.
    u = np.where(side < 0, local_u * 0.5, 0.5 + local_u * 0.5).astype(np.float32)
    v = (1.0 - _normalized(np.asarray(mapped_h2, dtype=np.float32))).astype(np.float32)
    result = RetinaMap(node_indices, np.asarray(mapped_ids, dtype=np.int64), u, v, side)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, node_indices=result.node_indices, body_ids=result.body_ids,
        u=result.u, v=result.v, side=result.side,
    )
    return result


def load_retina_map(path: Path) -> RetinaMap:
    payload = np.load(path, allow_pickle=False)
    return RetinaMap(
        payload["node_indices"], payload["body_ids"], payload["u"],
        payload["v"], payload["side"],
    )


def load_or_build_retina_map(
    graph: ConnectomeGraph, annotations_path: Path, path: Path
) -> RetinaMap:
    if path.exists():
        retina = load_retina_map(path)
        if np.all(graph.body_ids[retina.node_indices] == retina.body_ids):
            return retina
    # Raw contact counts preserve the relative evidence used for coordinate inference.
    raw_graph = ConnectomeGraph(
        graph.body_ids,
        sparse.load_npz(path.parent / "adjacency_raw.npz").tocsr(),
        graph.metadata_path,
    )
    return build_retina_map(raw_graph, annotations_path, path)
