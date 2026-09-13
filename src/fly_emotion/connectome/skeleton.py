from __future__ import annotations

import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SKELETON_ROOT = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
    "skeletons-malecns/skeletons-precomputed"
)


@dataclass(frozen=True)
class Skeleton:
    vertices: np.ndarray
    edges: np.ndarray


def parse_precomputed_skeleton(payload: bytes) -> Skeleton:
    if len(payload) < 8:
        raise ValueError("skeleton payload is shorter than its header")
    vertices_count, edges_count = struct.unpack_from("<II", payload)
    expected = 8 + 12 * vertices_count + 8 * edges_count
    if len(payload) != expected:
        raise ValueError(f"invalid skeleton size: {len(payload)} != {expected}")
    vertices = np.frombuffer(payload, dtype="<f4", count=vertices_count * 3, offset=8).reshape(
        -1, 3
    )
    edges = np.frombuffer(
        payload, dtype="<u4", count=edges_count * 2, offset=8 + vertices_count * 12
    ).reshape(-1, 2)
    if edges.size and int(edges.max()) >= vertices_count:
        raise ValueError("skeleton edge references an invalid vertex")
    return Skeleton(vertices.copy(), edges.copy())


def fetch_skeleton(body_id: int, cache_dir: Path, *, source_root: str = SKELETON_ROOT) -> Skeleton:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / str(body_id)
    if not path.exists():
        request = urllib.request.Request(f"{source_root}/{body_id}")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                payload = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(f"MaleCNS skeleton {body_id} does not exist") from error
            raise
        partial = path.with_suffix(".partial")
        partial.write_bytes(payload)
        parse_precomputed_skeleton(payload)
        partial.replace(path)
    return parse_precomputed_skeleton(path.read_bytes())


def skeleton_segments(skeleton: Skeleton, *, max_edges: int | None = None) -> np.ndarray:
    edges = skeleton.edges
    if max_edges and edges.shape[0] > max_edges:
        indexes = np.linspace(0, edges.shape[0] - 1, max_edges, dtype=np.int64)
        edges = edges[indexes]
    # Source coordinates are nm. Sending microns keeps browser values numerically stable.
    return (skeleton.vertices[edges] / 1_000.0).astype(np.float32)
