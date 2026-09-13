from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from scipy import sparse

from fly_emotion.connectome.graph import ConnectomeGraph
from fly_emotion.connectome.pathways import build_pathways


def test_pathway_bundles_account_for_every_edge(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.feather"
    feather.write_feather(
        pa.table(
            {
                "bodyId": [1, 2, 3],
                "superclass": ["sensory", "central", "motor"],
                "somaLocation": [[0, 0, 0], [1000, 0, 0], [2000, 0, 0]],
                "tosomaLocation": [None, None, None],
            }
        ),
        annotations,
    )
    adjacency = sparse.csr_matrix(([2, 3], ([1, 2], [0, 1])), shape=(3, 3), dtype=np.int32)
    graph = ConnectomeGraph(np.array([1, 2, 3]), adjacency, tmp_path / "graph.json")
    result = build_pathways(graph, annotations, tmp_path / "pathways.json", strongest_edges=2)
    assert result["all_edges_accounted"] == 2
    assert result["all_synapse_weight_accounted"] == 5
    assert len(result["bundled_paths"]) == 2
    assert len(result["strong_paths"]) == 2


def test_topology_retains_missing_coordinates_and_self_loops(tmp_path):
    annotations = tmp_path / "annotations.feather"
    feather.write_feather(
        pa.table(
            {
                "bodyId": [1, 2],
                "superclass": ["no_soma", "has_soma"],
                "somaLocation": [None, [1000, 0, 0]],
                "tosomaLocation": [None, None],
            }
        ),
        annotations,
    )
    adjacency = sparse.csr_matrix(([2, 3, 4], ([0, 1, 1], [0, 0, 1])), shape=(2, 2))
    graph = ConnectomeGraph(np.array([1, 2]), adjacency, tmp_path / "graph.json")
    result = build_pathways(graph, annotations, tmp_path / "paths.json")
    assert sum(p["edge_count"] for p in result["bundled_paths"]) == 3
    assert sum(p["synapse_weight"] for p in result["bundled_paths"]) == 9
    assert len([p for p in result["bundled_paths"] if p["source"] == p["target"]]) == 2
    assert any(not node["positioned"] for node in result["topology_nodes"])
