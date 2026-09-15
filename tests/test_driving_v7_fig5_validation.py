import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from fly_emotion.driving.v7_fig5_validation import (
    CONFIG,
    _without_notebook_magics,
    curve_comparison,
    direction_selectivity,
)

ROOT = Path(__file__).parents[1]


def test_notebook_magics_are_removed_without_executing_source() -> None:
    source = "%matplotlib inline\nx = 2\n  %autoreload 2\ny = dangerous()"
    cleaned = _without_notebook_magics(source)
    assert "%" not in cleaned
    assert "x = 2" in cleaned
    assert "dangerous()" in cleaned


def test_direction_selectivity_is_rotation_invariant_and_scaled() -> None:
    angles = np.arange(36) * 2 * np.pi / 36
    curves = np.stack((1 + np.cos(angles), 4 + 3 * np.cos(angles - 0.7)))
    shifted = np.roll(curves, 7, axis=1)
    np.testing.assert_allclose(direction_selectivity(curves), direction_selectivity(shifted))
    assert np.all((direction_selectivity(curves) >= 0) & (direction_selectivity(curves) <= 1))
    with pytest.raises(ValueError, match="36 directions"):
        direction_selectivity(np.ones((2, 35)))


def test_curve_comparison_keeps_cell_and_group_statistics_separate() -> None:
    model = np.arange(36, dtype=float)
    cells = np.stack((model, model + 2))
    report = curve_comparison(model, cells)
    np.testing.assert_array_equal(report["observed_mean_curve"], model + 1)
    assert report["rmse_millivolts"] == 1
    assert len(report["observed_cell_direction_selectivity"]) == 2


def test_fig5_manifest_excludes_large_behavior_files() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    assert len(config["files"]) == 10
    assert set(config["excluded_files"]) == {
        "fig5_R39H12_barfixation.csv",
        "fig5_R39H12_edges.csv",
    }
    assert all(item["size"] > 1_000_000_000 for item in config["excluded_files"].values())
    assert config["validation_boundary"]["independent_stimulus_conditions"] is True
    assert config["validation_boundary"]["independent_cell_identities_verified"] is False
    assert config["validation_boundary"]["mi9_block_equals_gluclar_rnai"] is False


def test_saved_fig5_validation_is_hash_bound_and_scope_limited() -> None:
    report = json.loads((ROOT / "artifacts/v7-fig5-validation.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fitting"] is False
    assert report["protocol"]["raw_files_committed"] is False
    assert len(report["verified_files"]) == 15
    assert report["parameter_match_with_fig3_audit"] is True
    assert report["advance_to_time_calibrated_fit"] is False
    assert report["advance_to_central_complex"] is False
    boundary = report["validation_boundary"]
    assert boundary["independent_stimulus_conditions"] is True
    assert boundary["independent_cell_identities_verified"] is False
    protocol = report["notebook_protocol"]
    assert protocol["internal_direction_degrees"] == list(np.arange(36) * 10.0)
    assert protocol["plot_direction_degrees"][-1] == 360
    assert protocol["plot_coordinate_maximum_error_degrees"] == 10
    assert protocol["flat_roll_18_preserves_group_mean"] is True
    assert protocol["flat_roll_18_preserves_group_sem"] is True
    assert protocol["flat_roll_18_individual_max_error_millivolts"] > 10
    measured = report["measured_groups"]
    assert {name: value["cells"] for name, value in measured.items()} == {
        "gfp": 25,
        "gluclarnai": 17,
        "nmdar1rnai": 12,
    }
    assert all(value["directions"] == 36 for value in measured.values())
    wild = report["validation"]["wild_type"]
    assert wild["pearson_correlation"] > 0.97
    assert 0.7 < wild["rmse_millivolts"] < 0.8
    assert wild["model_peak_direction_index"] == wild["observed_peak_direction_index"] == 0
    intervention = report["intervention_effect"]
    assert intervention["same_direction_of_change"] is True
    assert intervention["mechanistic_equivalence_claimed"] is False
    interpretation = report["interpretation"]
    assert interpretation["supports_external_stimulus_condition_generalization"] is True
    assert not any(
        interpretation[key]
        for key in (
            "supports_independent_cell_holdout",
            "supports_independent_input_recordings",
            "supports_T5_validation",
            "supports_MaleCNS_state_calibration",
            "supports_driving_behavior",
        )
    )
