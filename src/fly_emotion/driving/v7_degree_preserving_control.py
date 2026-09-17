"""Strict degree- and strength-preserving control for v7 visual target inputs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.driving.v7 import V7_CONFIG, V7_IMPLEMENTATION, V7VisualProbe
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-degree-preserving-control.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_degree_preserving_control.py")
RAW_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_raw.npz")
NORMALIZED_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_target_norm.npz")
BODY_IDS = Path("data/processed/malecns-v1.0/body_ids.npy")


def _digest_edges(rows: np.ndarray, cols: np.ndarray, weights: np.ndarray) -> str:
    order = np.lexsort((cols, rows))
    digest = hashlib.sha256()
    digest.update(rows[order].astype(np.int32, copy=False).tobytes())
    digest.update(cols[order].astype(np.int32, copy=False).tobytes())
    digest.update(weights[order].astype(np.float32, copy=False).tobytes())
    return digest.hexdigest()


def _incidence_weight_digest(nodes: np.ndarray, weights: np.ndarray) -> str:
    bits = weights.astype(np.float32, copy=False).view(np.uint32)
    order = np.lexsort((bits, nodes))
    digest = hashlib.sha256()
    digest.update(nodes[order].astype(np.int32, copy=False).tobytes())
    digest.update(bits[order].tobytes())
    return digest.hexdigest()


def degree_preserving_edge_swap(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    attempts: int,
) -> tuple[np.ndarray, dict]:
    rows = np.asarray(rows, dtype=np.int32)
    original_cols = np.asarray(cols, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float32)
    if not (rows.shape == original_cols.shape == weights.shape):
        raise ValueError("edge arrays must have equal shapes")
    shuffled = original_cols.copy()
    occupied = set(zip(rows.tolist(), shuffled.tolist(), strict=True))
    weight_groups: dict[float, list[int]] = defaultdict(list)
    for index, weight in enumerate(weights):
        weight_groups[float(weight)].append(index)
    groups = {
        value: np.asarray(indices, dtype=np.int64) for value, indices in weight_groups.items()
    }
    rng = np.random.default_rng(seed)
    accepted = 0
    for _ in range(attempts):
        first = int(rng.integers(len(rows)))
        group = groups[float(weights[first])]
        if len(group) < 2:
            continue
        second = int(group[int(rng.integers(len(group)))])
        if first == second:
            continue
        r1, r2 = int(rows[first]), int(rows[second])
        c1, c2 = int(shuffled[first]), int(shuffled[second])
        if r1 in (r2, c2) or c1 == c2 or r2 == c1:
            continue
        old1, old2 = (r1, c1), (r2, c2)
        new1, new2 = (r1, c2), (r2, c1)
        if new1 in occupied or new2 in occupied:
            continue
        occupied.remove(old1)
        occupied.remove(old2)
        occupied.add(new1)
        occupied.add(new2)
        shuffled[first], shuffled[second] = c2, c1
        accepted += 1
    return shuffled, {
        "attempted_swaps": attempts,
        "accepted_swaps": accepted,
        "accepted_swap_fraction": float(accepted / attempts) if attempts else 0.0,
        "rewired_edge_count": int(np.count_nonzero(shuffled != original_cols)),
        "rewired_edge_fraction": float(np.mean(shuffled != original_cols)),
    }


def evaluate_v7_degree_preserving_control(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    evidence_path = Path(config["controlled_vision_evidence"])
    evidence = json.loads((root / evidence_path).read_text())
    if evidence["controlled_response_gates_pass"]:
        raise ValueError("structural precheck is reserved for the current failed visual gate")
    probe = V7VisualProbe(root, brain_substeps=1, dynamics_backend="typed_visual_subgraph_v1")
    target_nodes = np.unique(np.concatenate(list(probe.populations.values()))).astype(np.int32)
    scoped = probe.adjacency[target_nodes, :].tocoo()
    rows = target_nodes[scoped.row].astype(np.int32)
    cols = scoped.col.astype(np.int32)
    weights = scoped.data.astype(np.float32)
    attempts = int(config["swap_attempt_multiplier"] * len(rows))
    shuffled_cols, stats = degree_preserving_edge_swap(
        rows, cols, weights, seed=int(config["seed"]), attempts=attempts
    )
    shape = probe.adjacency.shape
    original = sparse.csr_matrix((weights, (rows, cols)), shape=shape)
    shuffled = sparse.csr_matrix((weights, (rows, shuffled_cols)), shape=shape)
    original_in_degree = np.diff(original.indptr)
    shuffled_in_degree = np.diff(shuffled.indptr)
    original_csc = original.tocsc()
    shuffled_csc = shuffled.tocsc()
    original_out_degree = np.diff(original_csc.indptr)
    shuffled_out_degree = np.diff(shuffled_csc.indptr)
    original_target_weight_digest = _incidence_weight_digest(rows, weights)
    shuffled_target_weight_digest = _incidence_weight_digest(rows, weights)
    original_source_weight_digest = _incidence_weight_digest(cols, weights)
    shuffled_source_weight_digest = _incidence_weight_digest(shuffled_cols, weights)
    original_pairs = set(zip(rows.tolist(), cols.tolist(), strict=True))
    shuffled_pairs = set(zip(rows.tolist(), shuffled_cols.tolist(), strict=True))
    invariants = {
        "edge_count_preserved": shuffled.nnz == original.nnz == len(rows),
        "each_target_indegree_preserved": bool(
            np.array_equal(original_in_degree, shuffled_in_degree)
        ),
        "each_source_outdegree_within_scope_preserved": bool(
            np.array_equal(original_out_degree, shuffled_out_degree)
        ),
        "each_target_incoming_weight_multiset_preserved": (
            original_target_weight_digest == shuffled_target_weight_digest
        ),
        "each_source_outgoing_weight_multiset_within_scope_preserved": (
            original_source_weight_digest == shuffled_source_weight_digest
        ),
        "global_weight_multiset_preserved": bool(
            np.array_equal(np.sort(weights), np.sort(shuffled.data))
        ),
        "duplicate_edges_absent": len(shuffled_pairs) == len(rows),
        "new_self_loops_absent": not any(
            row == col and (row, col) not in original_pairs for row, col in shuffled_pairs
        ),
        "deterministic_replay": bool(
            np.array_equal(
                shuffled_cols,
                degree_preserving_edge_swap(
                    rows, cols, weights, seed=int(config["seed"]), attempts=attempts
                )[0],
            )
        ),
        "minimum_rewired_edge_fraction": stats["rewired_edge_fraction"]
        >= float(config["minimum_rewired_edge_fraction"]),
    }
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(V7_CONFIG): _sha256(root / V7_CONFIG),
                str(V7_IMPLEMENTATION): _sha256(root / V7_IMPLEMENTATION),
                str(evidence_path): _sha256(root / evidence_path),
                str(RAW_ADJACENCY): _sha256(root / RAW_ADJACENCY),
                str(NORMALIZED_ADJACENCY): _sha256(root / NORMALIZED_ADJACENCY),
                str(BODY_IDS): _sha256(root / BODY_IDS),
            },
            "seed": int(config["seed"]),
            "scope": config["scope"],
            "node_count": int(shape[0]),
            "target_count": int(len(target_nodes)),
            "edge_count": int(len(rows)),
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "swap_statistics": stats,
        "invariants": invariants,
        "original_edge_sha256": _digest_edges(rows, cols, weights),
        "shuffled_edge_sha256": _digest_edges(rows, shuffled_cols, weights),
        "strict_degree_preserving_control_constructed": bool(all(invariants.values())),
        "visual_response_evaluation_performed": False,
        "real_topology_advantage_established_by_this_control": False,
        "advance_to_model_selection": False,
        "boundary": config["boundary"],
    }
