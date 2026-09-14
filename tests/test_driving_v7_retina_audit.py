import hashlib
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_retina_audit import (
    RETINA_AUDIT_CONFIG,
    RETINA_AUDIT_IMPLEMENTATION,
    build_balanced_retina_control,
    evaluate_v7_retina_column_audit,
    infer_retinal_columns,
)

ROOT = Path(__file__).parents[1]


def test_v7_retinal_column_assignment_is_deterministic_and_mostly_unambiguous() -> None:
    retina, first = infer_retinal_columns(ROOT)
    _, second = infer_retinal_columns(ROOT)
    assert retina.size == 3344
    assert np.array_equal(first.coordinates, second.coordinates)
    assert np.array_equal(first.weighted_means, second.weighted_means)
    assert np.mean(first.ambiguity > 0) < 0.02
    assert np.quantile(first.weighted_spreads, 0.95) == 0.0


def test_v7_balanced_retina_has_exact_bilateral_pairs_and_mirror_drive() -> None:
    cache_path = ROOT / "data/processed/malecns-v1.0/retina_map.npz"
    cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    control = build_balanced_retina_control(ROOT)
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == cache_hash
    original, assignment = infer_retinal_columns(ROOT)
    assert len(np.unique(control.retina.node_indices)) == control.retina.size
    assert np.array_equal(control.retina.body_ids, original.body_ids[control.source_indices])
    for coordinate in np.unique(control.pair_coordinates, axis=0):
        column = np.all(assignment.coordinates == coordinate, axis=1)
        paired_count = np.count_nonzero(np.all(control.pair_coordinates == coordinate, axis=1))
        for side in (-1, 1):
            candidates = np.sort(original.body_ids[column & (original.side == side)])
            selected = np.sort(
                control.retina.body_ids[
                    (control.retina.side == side)
                    & np.all(assignment.coordinates[control.source_indices] == coordinate, axis=1)
                ]
            )
            assert np.array_equal(selected, candidates[:paired_count])
    assert control.pair_count == 957
    assert control.retina.size == 1914
    assert len(np.unique(control.pair_coordinates, axis=0)) == 270
    assert np.all(control.retina.side[control.left_positions] == -1)
    assert np.all(control.retina.side[control.right_positions] == 1)
    assert np.array_equal(
        control.pair_ids[control.left_positions], control.pair_ids[control.right_positions]
    )
    image = np.arange(24 * 48, dtype=np.float32).reshape(24, 48)
    left = control.retina.encode(image)[control.left_positions]
    right = control.retina.encode(image[:, ::-1])[control.right_positions]
    assert np.array_equal(left, right)


def test_v7_retina_audit_is_non_advancing_and_hash_bound() -> None:
    report = evaluate_v7_retina_column_audit(ROOT)
    assert report["protocol"]["advance_allowed"] is False
    assert (
        report["protocol"]["config_sha256"]
        == hashlib.sha256((ROOT / RETINA_AUDIT_CONFIG).read_bytes()).hexdigest()
    )
    assert (
        report["protocol"]["implementation_sha256"]
        == hashlib.sha256((ROOT / RETINA_AUDIT_IMPLEMENTATION).read_bytes()).hexdigest()
    )
    assert report["legacy_mapping"]["eye_counts"] == {
        "left": 1107,
        "right": 2237,
    }
    assert report["balanced_common_column_control"]["exact_mirror_drive"] is True
    assert report["diagnosis"]["balanced_control_is_biological_reconstruction"] is False
    assert report["advance_to_central_complex"] is False
