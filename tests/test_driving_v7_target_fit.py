import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_target_fit import (
    CONFIG,
    authorize_optimizer_conditions,
    bootstrap_condition_consistency,
    score_lc4,
    score_lplc1,
    score_lplc2,
    simulate_t5_target_head,
    target_grid_search,
)

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-target-fit.json"


def test_optimizer_accepts_only_exactly_three_tuning_bundles() -> None:
    assert authorize_optimizer_conditions(["tuning", "tuning", "tuning"]) is True
    assert authorize_optimizer_conditions(["tuning", "tuning"]) is False
    assert authorize_optimizer_conditions(["tuning", "tuning", "tuning", "tuning"]) is False
    assert authorize_optimizer_conditions(["tuning", "tuning", "calibration"]) is False
    assert authorize_optimizer_conditions(["tuning", "tuning", "regression_only"]) is False


def test_t5_head_keeps_tm9_and_ct1_components_separate() -> None:
    fast = np.ones((3, 2, 4))
    tm9 = np.full_like(fast, 0.4)
    ct1 = np.full_like(fast, 0.1)
    output, evidence = simulate_t5_target_head(
        fast,
        tm9,
        ct1,
        fast_weight=np.array([1.0, 0.5]),
        delayed_weight=np.array([1.0, 2.0]),
        leak=np.array([0.5, 0.25]),
        tm9_effect=1.0,
        ct1_effect=-1.0,
    )
    assert output.shape == fast.shape
    np.testing.assert_array_equal(evidence["tm9_component"], tm9)
    np.testing.assert_array_equal(evidence["ct1_component"], -ct1)
    assert evidence["components_merged_before_source_specific_transform"] is False


def test_target_grid_search_recovers_distinct_target_parameters() -> None:
    rng = np.random.default_rng(7)
    fast = rng.uniform(0, 1, (3, 2, 8))
    tm9 = rng.uniform(0, 1, fast.shape)
    ct1 = rng.uniform(0, 1, fast.shape)
    desired, _ = simulate_t5_target_head(
        fast,
        tm9,
        ct1,
        fast_weight=np.array([0.5, 1.0]),
        delayed_weight=np.array([1.0, 0.5]),
        leak=np.array([0.5, 1.0]),
        tm9_effect=1.0,
        ct1_effect=-1.0,
    )
    result = target_grid_search(
        fast,
        tm9,
        ct1,
        desired,
        grid={"fast_weight": [0.5, 1.0], "delayed_weight": [0.5, 1.0], "leak": [0.5, 1.0]},
        tm9_effect=1.0,
        ct1_effect=-1.0,
        regularization_strength=0.0,
    )
    np.testing.assert_allclose(result["fast_weight"], [0.5, 1.0])
    np.testing.assert_allclose(result["delayed_weight"], [1.0, 0.5])
    np.testing.assert_allclose(result["leak"], [0.5, 1.0])
    assert result["candidate_count"] == 8


def test_bootstrap_uses_target_clusters_and_requires_three_conditions() -> None:
    positive = np.full((20, 3), 0.2)
    result = bootstrap_condition_consistency(
        positive,
        replicates=1000,
        seed=11,
        minimum_effect=0.1,
        minimum_conditions=3,
        minimum_probability=0.95,
    )
    assert result["independent_cluster_count"] == 20
    assert result["bootstrap_replicates"] == 1000
    assert result["point_passing_condition_count"] == 3
    assert result["passed"] is True
    mixed = positive.copy()
    mixed[:, 2] = -0.2
    assert (
        bootstrap_condition_consistency(
            mixed,
            replicates=1000,
            seed=11,
            minimum_effect=0.1,
            minimum_conditions=3,
            minimum_probability=0.95,
        )["passed"]
        is False
    )


def test_looming_readouts_are_target_specific() -> None:
    lplc1 = score_lplc1(np.array([2.0]), np.array([1.0]), np.array([3.0]), np.array([1.0]))
    assert lplc1["shared_looming_comparator_used"] is False
    lplc2 = score_lplc2(np.array([3.0]), np.array([1.0]), np.array([0.5]), np.array([0.25]))
    assert set(key for key in lplc2 if key.startswith("outward_minus")) == {
        "outward_minus_inward",
        "outward_minus_motion_free_darkening",
        "outward_minus_wide_field_translation",
    }
    lc4 = score_lc4(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]), np.full(3, 20.0))
    assert lc4["through_origin_slope"] == 2.0
    assert lc4["terminal_size_gate_used"] is False


def test_saved_target_fit_contract_is_fresh_and_locked() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    assert report["candidate_count_per_target"] == 729
    assert report["declared_parameter_count"] == 3 * 6719
    assert config["execution"]["optimizer_condition_ids"] == ["S1-T01", "S1-T02", "S1-T03"]
    assert config["execution"]["legacy_regression_weight"] == 0
    assert report["boundary"]["real_fit_performed"] is False
    assert report["calibration_authorized"] is False
    assert report["final_authorized"] is False
    assert report["advance_to_visual_gate"] is False
