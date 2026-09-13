from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from fly_emotion.connectome.dynamics import propagate
from fly_emotion.connectome.graph import build_canonical_graph, load_graph


def test_build_graph_preserves_nodes_direction_and_weights(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.feather"
    edges = tmp_path / "edges.feather"
    output = tmp_path / "graph"
    feather.write_feather(
        pa.table(
            {
                "bodyId": [30, 10, 20, 99],
                "status": ["Traced", "Traced", "Traced", "Orphan"],
                "superclass": ["a", "b", "c", None],
            }
        ),
        annotations,
    )
    feather.write_feather(
        pa.table(
            {
                "body_pre": [10, 10, 20, 99],
                "body_post": [20, 20, 30, 10],
                "weight": [2, 3, 4, 100],
            }
        ),
        edges,
    )
    graph = build_canonical_graph(annotations, edges, output, batch_size=2)
    raw = load_graph(output, normalized=False)
    assert graph.body_ids.tolist() == [10, 20, 30]
    assert raw.adjacency.nnz == 2
    assert raw.adjacency[1, 0] == 5
    assert raw.adjacency[2, 1] == 4
    assert np.isclose(graph.adjacency[1, 0], 1.0)


def test_propagation_reaches_postsynaptic_node() -> None:
    adjacency = pa.table({})  # keep pyarrow import exercised by this test module
    del adjacency
    from scipy import sparse

    matrix = sparse.csr_matrix(([1.0], ([1], [0])), shape=(2, 2), dtype=np.float32)
    history = propagate(matrix, np.array([1.0, 0.0], dtype=np.float32), steps=1, leak=1.0)
    assert history[1, 1] > 0
    assert history[1, 0] == 0


def test_linear_propagation_matches_basis_equation() -> None:
    from scipy import sparse

    matrix = sparse.csr_matrix(([0.5], ([1], [0])), shape=(2, 2), dtype=np.float32)
    initial = np.array([1.0, 0.0], dtype=np.float32)
    history = propagate(matrix, initial, steps=1, leak=0.4)
    expected = 0.6 * initial + 0.4 * (matrix @ initial)
    assert np.allclose(history[1], expected)


def test_iteration_does_not_precompute_future_steps():
    from fly_emotion.connectome.dynamics import iter_propagation

    class Matrix:
        shape = (2, 2)
        calls = 0

        def __matmul__(self, state):
            self.calls += 1
            return state * 0.5

    matrix = Matrix()
    frames = iter_propagation(matrix, np.ones(2), steps=2)
    next(frames)
    assert matrix.calls == 0
    next(frames)
    assert matrix.calls == 1
    next(frames)
    assert matrix.calls == 2
