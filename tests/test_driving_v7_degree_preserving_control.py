import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_degree_preserving_control import degree_preserving_edge_swap

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-degree-preserving-control.json"


def test_equal_weight_edge_swap_is_deterministic_and_degree_preserving() -> None:
    rows = np.repeat(np.arange(20, dtype=np.int32), 4)
    cols = np.concatenate(
        [((np.arange(20, dtype=np.int32) + offset) % 20) + 100 for offset in (0, 2, 5, 9)]
    ).reshape(4, 20).T.reshape(-1)
    weights = np.ones(len(rows), dtype=np.float32)
    first, stats = degree_preserving_edge_swap(rows, cols, weights, seed=7, attempts=5_000)
    second, _ = degree_preserving_edge_swap(rows, cols, weights, seed=7, attempts=5_000)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        np.bincount(rows, minlength=20), np.bincount(rows, minlength=20)
    )
    np.testing.assert_array_equal(np.bincount(cols), np.bincount(first))
    for row in np.unique(rows):
        original = weights[rows == row]
        shuffled = weights[rows == row]
        np.testing.assert_array_equal(np.sort(original), np.sort(shuffled))
    for source in np.unique(cols):
        np.testing.assert_array_equal(
            np.sort(weights[cols == source]), np.sort(weights[first == source])
        )
    assert len(set(zip(rows.tolist(), first.tolist(), strict=True))) == len(rows)
    assert stats["rewired_edge_fraction"] > 0.80


def test_saved_degree_preserving_control_is_hash_bound_and_non_advancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["target_count"] == 14_331
    assert report["protocol"]["edge_count"] == 1_108_726
    assert report["swap_statistics"]["rewired_edge_fraction"] > 0.80
    assert all(report["invariants"].values())
    assert report["original_edge_sha256"] != report["shuffled_edge_sha256"]
    assert report["strict_degree_preserving_control_constructed"] is True
    assert report["visual_response_evaluation_performed"] is False
    assert report["real_topology_advantage_established_by_this_control"] is False
    assert report["advance_to_model_selection"] is False
