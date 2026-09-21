import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_t5_increment_order_control_discovery_audit import (
    _increment_order_control_frames,
)

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-increment-order-control-discovery-audit.json"


def test_increment_order_control_keeps_initialization_and_reorders_motion() -> None:
    frames = np.zeros((6, 1, 1), dtype=np.float32)
    frames[2:, 0, 0] = np.asarray([0.2, 0.4, 0.7, 1.0], dtype=np.float32)
    controlled, order = _increment_order_control_frames(frames, 2, 20260920)
    assert order[0] == 0
    assert sorted(order.tolist()) == list(range(4))
    assert not np.array_equal(order[1:], np.arange(1, 4))
    assert np.all((controlled >= 0.0) & (controlled <= 1.0))
    assert np.isclose(controlled[-1], frames[-1]).all()


def test_increment_order_discovery_is_hash_bound_and_input_matched() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["discovery_condition_id"] == "S1-T01"
    assert report["protocol"]["stimulus_count"] == 120
    assert report["protocol"]["settle_frames"] == 10
    assert report["control_images_within_unit_interval"] is True
    assert report["terminal_frame_preserved_for_every_control"] is True
    assert report["terminal_frame_maximum_absolute_error"] <= 2e-8
    assert report["pixel_increment_multiset_preserved_for_every_control"] is True
    assert report["R1_R6_drive_multiset_preserved_for_every_control"] is True
    assert report["R1_R6_mean_absolute_energy_preserved_within_tolerance"] is True
    assert np.allclose(
        list(report["R1_R6_mean_absolute_energy_ratio_summary"].values()),
        1.0,
        atol=1e-12,
    )
    assert report["increment_order_fixed_point_count"] == 1
    assert report["preserved_forward_adjacency_count"] == 1
    assert report["preserved_reverse_adjacency_count"] == 3


def test_discovery_results_do_not_retroactively_replace_gates() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "1": {
            "lamina": [
                0.9999434351017648,
                0.999972658081661,
                0.9998730547519342,
                0.9990375130061279,
            ],
            "shuffle": [
                1.0087448156400647,
                1.1383877639567048,
                1.3853597275054466,
            ],
        },
        "4": {
            "lamina": [
                0.9999854541519411,
                0.9999951663982193,
                0.9999675661298183,
                0.9999731276000139,
            ],
            "shuffle": [
                1.0162011434360847,
                1.1262430557870489,
                1.3045208104632329,
            ],
        },
    }
    for update, values in expected.items():
        result = report["by_brain_updates_per_frame"][update]
        assert np.allclose(
            list(result["lamina_only_source_energy_ratio"].values()),
            values["lamina"],
        )
        assert np.allclose(
            [
                candidate["shuffle_to_ordered_residual_energy_ratio"]
                for candidate in result["candidate_output_results"].values()
            ],
            values["shuffle"],
        )
        assert all(
            candidate["passed"] is False
            for candidate in result["candidate_output_results"].values()
        )
    assert report["existing_temporal_gate_replaced"] is False
    assert report["independent_condition_evaluation_performed"] is False
    assert report["new_acceptance_threshold_defined"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
