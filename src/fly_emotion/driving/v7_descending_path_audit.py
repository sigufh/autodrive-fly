from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.engine import INHIBITORY_TRANSMITTERS, MODULATORY_TRANSMITTERS
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-descending-path-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_descending_path_audit.py")
ANNOTATIONS = Path("data/raw/malecns-v1.0/body-annotations.feather")
NEUROTRANSMITTERS = Path("data/raw/malecns-v1.0/body-neurotransmitters.feather")
BODY_IDS = Path("data/processed/malecns-v1.0/body_ids.npy")
RAW_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_raw.npz")
ENGINE_IMPLEMENTATION = Path("src/fly_emotion/driving/engine.py")


def _transmitter_sign(name: object) -> int:
    if name in INHIBITORY_TRANSMITTERS:
        return -1
    if name in MODULATORY_TRANSMITTERS:
        return 0
    if isinstance(name, str) and name:
        return 1
    return 0


def _node_metadata(root: Path, body_ids: np.ndarray) -> dict[str, np.ndarray]:
    annotations = feather.read_table(
        root / ANNOTATIONS,
        columns=["bodyId", "type", "somaSide", "superclass"],
        memory_map=True,
    ).to_pandas()
    ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(body_ids, ids)
    valid = nodes < body_ids.size
    valid[valid] &= body_ids[nodes[valid]] == ids[valid]
    nodes = nodes[valid].astype(np.int32)
    annotations = annotations.iloc[np.flatnonzero(valid)]

    result = {
        "type": np.full(body_ids.size, "", dtype=object),
        "side": np.full(body_ids.size, "", dtype=object),
        "superclass": np.full(body_ids.size, "", dtype=object),
        "transmitter": np.full(body_ids.size, "", dtype=object),
    }
    for output, column in (("type", "type"), ("side", "somaSide"), ("superclass", "superclass")):
        values = annotations[column].to_numpy(dtype=object)
        result[output][nodes] = [value if isinstance(value, str) else "" for value in values]

    transmitters = feather.read_table(
        root / NEUROTRANSMITTERS, columns=["body", "consensus_nt"], memory_map=True
    ).to_pandas()
    ids = transmitters["body"].to_numpy(dtype=np.int64)
    nt_nodes = np.searchsorted(body_ids, ids)
    valid = nt_nodes < body_ids.size
    valid[valid] &= body_ids[nt_nodes[valid]] == ids[valid]
    values = transmitters.loc[valid, "consensus_nt"].to_numpy(dtype=object)
    result["transmitter"][nt_nodes[valid]] = [
        value if isinstance(value, str) else "" for value in values
    ]
    result["sign"] = np.asarray(
        [_transmitter_sign(value) for value in result["transmitter"]], dtype=np.int8
    )
    return result


def _shortest_path_dag(adjacency, sources: np.ndarray, maximum_hops: int):
    """Return distances and all predecessor edges on shortest directed paths."""

    outgoing = adjacency.tocsc()
    distance = np.full(adjacency.shape[0], -1, dtype=np.int16)
    source_set = frozenset(int(node) for node in sources)
    frontier = np.asarray(sorted(source_set), dtype=np.int32)
    distance[frontier] = 0
    predecessors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    frontier_sizes = {"0": int(frontier.size)}
    for depth in range(1, maximum_hops + 1):
        next_nodes: set[int] = set()
        for source in frontier:
            start, end = outgoing.indptr[int(source) : int(source) + 2]
            for target, weight in zip(
                outgoing.indices[start:end], outgoing.data[start:end], strict=True
            ):
                target = int(target)
                if distance[target] == -1:
                    distance[target] = depth
                    next_nodes.add(target)
                if distance[target] == depth:
                    predecessors[target].append((int(source), int(weight)))
        frontier = np.asarray(sorted(next_nodes), dtype=np.int32)
        frontier_sizes[str(depth)] = int(frontier.size)
        if not frontier.size:
            break
    for edges in predecessors.values():
        edges.sort()
    return distance, predecessors, source_set, frontier_sizes


def _enumerate_paths(
    target: int,
    distance: np.ndarray,
    predecessors: dict[int, list[tuple[int, int]]],
    source_set: frozenset[int],
    maximum_paths: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    @cache
    def paths_to(node: int) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        if node in source_set:
            return (((node,), ()),)
        paths = []
        for source, weight in predecessors.get(node, []):
            for nodes, weights in paths_to(source):
                paths.append((nodes + (node,), weights + (weight,)))
                if len(paths) > maximum_paths:
                    raise ValueError(
                        f"shortest-path enumeration for node {target} exceeds frozen limit "
                        f"{maximum_paths}"
                    )
        return tuple(paths)

    if distance[target] < 0:
        return []
    return list(paths_to(target))


def _side_relation(source: str, target: str) -> str:
    if source not in {"L", "R"} or target not in {"L", "R"}:
        return "unknown"
    return "ipsilateral" if source == target else "contralateral"


def _cell(node: int, body_ids: np.ndarray, metadata: dict[str, np.ndarray]) -> dict:
    return {
        "body_id": int(body_ids[node]),
        "type": str(metadata["type"][node]),
        "soma_side": str(metadata["side"][node]),
        "superclass": str(metadata["superclass"][node]),
        "consensus_neurotransmitter": str(metadata["transmitter"][node]),
        "transmitter_sign": int(metadata["sign"][node]),
    }


def _path_record(
    nodes: tuple[int, ...],
    weights: tuple[int, ...],
    body_ids: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> dict:
    cells = [_cell(node, body_ids, metadata) for node in nodes]
    edges = []
    net_sign = 1
    for source, target, weight in zip(nodes[:-1], nodes[1:], weights, strict=True):
        sign = int(metadata["sign"][source])
        net_sign *= sign
        edges.append(
            {
                "source_body_id": int(body_ids[source]),
                "target_body_id": int(body_ids[target]),
                "synapse_weight": int(weight),
                "presynaptic_neurotransmitter": str(metadata["transmitter"][source]),
                "presynaptic_sign": sign,
            }
        )
    return {
        "nodes": cells,
        "edges": edges,
        "source_side": cells[0]["soma_side"],
        "target_side": cells[-1]["soma_side"],
        "side_relation": _side_relation(cells[0]["soma_side"], cells[-1]["soma_side"]),
        "intermediate_types": [cell["type"] for cell in cells[1:-1]],
        "net_presynaptic_sign_product": int(net_sign),
    }


def _target_summary(target: int, paths: list[dict], body_ids, metadata, distance) -> dict:
    unique_intermediates = {
        cell["body_id"]: cell
        for path in paths
        for cell in path["nodes"][1:-1]
    }
    edge_weights = [edge["synapse_weight"] for path in paths for edge in path["edges"]]
    edge_catalog = {}
    for path in paths:
        cells_by_id = {cell["body_id"]: cell for cell in path["nodes"]}
        for edge in path["edges"]:
            key = (edge["source_body_id"], edge["target_body_id"])
            edge_catalog[key] = {
                **edge,
                "source_type": cells_by_id[key[0]]["type"],
                "source_side": cells_by_id[key[0]]["soma_side"],
                "target_type": cells_by_id[key[1]]["type"],
                "target_side": cells_by_id[key[1]]["soma_side"],
            }
    canonical_paths = json.dumps(paths, sort_keys=True, separators=(",", ":")).encode()
    return {
        "target": _cell(target, body_ids, metadata),
        "reachable": bool(paths),
        "shortest_hops": int(distance[target]) if distance[target] >= 0 else None,
        "shortest_path_count": len(paths),
        "all_shortest_paths_enumerated": True,
        "all_shortest_paths_sha256": hashlib.sha256(canonical_paths).hexdigest(),
        "side_relationship_counts": dict(
            sorted(Counter(path["side_relation"] for path in paths).items())
        ),
        "path_sign_counts": {
            str(key): value
            for key, value in sorted(
                Counter(path["net_presynaptic_sign_product"] for path in paths).items()
            )
        },
        "unique_intermediate_count": len(unique_intermediates),
        "unique_intermediate_type_counts": dict(
            sorted(Counter(cell["type"] for cell in unique_intermediates.values()).items())
        ),
        "path_intermediate_type_counts": dict(
            sorted(
                Counter(
                    cell_type
                    for path in paths
                    for cell_type in path["intermediate_types"]
                ).items()
            )
        ),
        "minimum_enumerated_edge_weight": min(edge_weights) if edge_weights else None,
        "maximum_enumerated_edge_weight": max(edge_weights) if edge_weights else None,
        "unique_shortest_path_dag_edges": [
            edge_catalog[key] for key in sorted(edge_catalog)
        ],
        "shortest_path_examples": paths[:8],
        "stored_example_count": min(len(paths), 8),
        "all_paths_stored_verbatim": len(paths) <= 8,
    }


def evaluate_v7_descending_path_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
    metadata = _node_metadata(root, graph.body_ids)
    sources = np.flatnonzero(metadata["type"] == config["source_type"]).astype(np.int32)
    target_nodes = {
        target_type: np.flatnonzero(metadata["type"] == target_type).astype(np.int32)
        for target_type in config["target_types"]
    }
    distance, predecessors, source_set, frontier_sizes = _shortest_path_dag(
        graph.adjacency, sources, int(config["maximum_hops"])
    )
    targets = {}
    for target_type, nodes in target_nodes.items():
        records = []
        for target in sorted(int(node) for node in nodes):
            raw_paths = _enumerate_paths(
                target,
                distance,
                predecessors,
                source_set,
                int(config["maximum_shortest_paths_per_target"]),
            )
            paths = [
                _path_record(path_nodes, weights, graph.body_ids, metadata)
                for path_nodes, weights in raw_paths
            ]
            records.append(_target_summary(target, paths, graph.body_ids, metadata, distance))
        targets[target_type] = records

    population_reachability = {}
    for target_type, records in targets.items():
        side_counts = Counter(
            item["target"]["soma_side"] for item in records if item["reachable"]
        )
        expected_sides = set(config["required_target_sides"])
        population_reachability[target_type] = {
            "target_count": len(records),
            "reachable_target_count": sum(item["reachable"] for item in records),
            "reachable_sides": sorted(side_counts),
            "bilateral_reachability_passed": set(side_counts) == expected_sides,
        }
    structural_gate = bool(
        sources.size
        and all(
            result["reachable_target_count"] == result["target_count"]
            and result["bilateral_reachability_passed"]
            for result in population_reachability.values()
        )
    )

    action_path = Path(config["algebraic_action_evidence"])
    action = json.loads((root / action_path).read_text(encoding="utf-8"))
    action_boundary = {
        "reported_action_equivalence": action["readout_action_equivalence"],
        "arms": config["algebraic_action_arms"],
        "arms_consume_real_DNa01_DNa02_DNp20_states": False,
        "equivalence_is_formula_level_not_neural_state_level": True,
        "real_neural_state_readout_performed": False,
        "neural_readout_equivalence_established": False,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(ANNOTATIONS): _sha256(root / ANNOTATIONS),
                str(NEUROTRANSMITTERS): _sha256(root / NEUROTRANSMITTERS),
                str(BODY_IDS): _sha256(root / BODY_IDS),
                str(RAW_ADJACENCY): _sha256(root / RAW_ADJACENCY),
                str(ENGINE_IMPLEMENTATION): _sha256(root / ENGINE_IMPLEMENTATION),
                str(action_path): _sha256(root / action_path),
            },
            "graph_orientation": "rows=postsynaptic targets, columns=presynaptic sources",
            "search": "sparse_directed_breadth_first_search",
            "maximum_hops": int(config["maximum_hops"]),
            "maximum_shortest_paths_per_target": int(
                config["maximum_shortest_paths_per_target"]
            ),
            "matrix_power_used": False,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "source_population": {
            "type": config["source_type"],
            "count": int(sources.size),
            "side_counts": dict(
                sorted(Counter(str(metadata["side"][node]) for node in sources).items())
            ),
        },
        "bfs_frontier_sizes": frontier_sizes,
        "population_reachability": population_reachability,
        "target_shortest_paths": targets,
        "structural_bilateral_path_gate_passed": structural_gate,
        "authorize_fixed_input_functional_propagation_precheck": structural_gate,
        "algebraic_action_readout_boundary": action_boundary,
        "functional_propagation_evaluated": False,
        "functional_neural_readout_validated": False,
        "advance_to_navigation_release": False,
        "advance_to_mushroom_body": False,
        "boundary": config["boundary"],
    }
