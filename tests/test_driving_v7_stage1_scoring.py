import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_stage1_scoring import (
    CONFIG,
    authorize_split,
    looming_contrast_summary,
    mirror_error_summary,
    strict_contrast_summary,
)

ROOT = Path(__file__).parents[1]


def _thresholds() -> dict:
    return yaml.safe_load((ROOT / CONFIG).read_text())["thresholds"]


def test_strict_contrast_rejects_silence_reversed_labels_and_low_coverage() -> None:
    ids = np.arange(10)
    thresholds = _thresholds()
    assert strict_contrast_summary(ids, np.ones(10), np.full(10, 0.5), thresholds)["passed"]
    assert not strict_contrast_summary(ids, np.full(10, 0.5), np.ones(10), thresholds)["passed"]
    silent = strict_contrast_summary(ids, np.zeros(10), np.zeros(10), thresholds)
    assert silent["valid_cell_count"] == 0
    assert silent["median_signed_contrast"] is None
    assert not silent["passed"]
    partial = strict_contrast_summary(
        ids, np.r_[np.ones(7), np.zeros(3)], np.r_[np.full(7, 0.5), np.zeros(3)], thresholds
    )
    assert partial["valid_cell_fraction"] == 0.7
    assert not partial["passed"]


def test_looming_requires_expansion_above_receding_and_static() -> None:
    ids = np.arange(10)
    thresholds = _thresholds()
    assert looming_contrast_summary(
        ids, np.ones(10), np.full(10, 0.4), np.full(10, 0.6), thresholds
    )["passed"]
    assert not looming_contrast_summary(
        ids, np.ones(10), np.full(10, 0.4), np.ones(10), thresholds
    )["passed"]


def test_mirror_score_rejects_silent_equivariance() -> None:
    thresholds = _thresholds()
    active = np.tile([0.2, 0.4, 0.1], (10, 1))
    assert mirror_error_summary(active, active.copy(), thresholds)["passed"]
    silent = mirror_error_summary(np.zeros((10, 3)), np.zeros((10, 3)), thresholds)
    assert silent["energy_weighted_mirror_error"] is None
    assert not silent["passed"]


def test_final_requires_all_prerequisites_and_external_custody() -> None:
    assert authorize_split("development")
    assert not authorize_split("validation")
    assert authorize_split("validation", development_passed=True)
    assert not authorize_split("ood", development_passed=True)
    assert authorize_split("ood", development_passed=True, validation_passed=True)
    assert not authorize_split("final", stage1_passed=True, topology_passed=True)
    assert authorize_split(
        "final",
        stage1_passed=True,
        topology_passed=True,
        external_final_custody=True,
    )


def test_saved_scoring_contract_is_hash_bound_and_non_advancing() -> None:
    report = json.loads((ROOT / "artifacts/v7-stage1-scoring.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["population_contract"]["required_direction_groups"] == 16
    assert report["population_contract"]["required_polarity_groups"] == 16
    assert report["population_contract"]["required_looming_groups"] == 6
    assert all(report["synthetic_controls"].values())
    assert report["final_lock"]["reserved_final_blinded"] is False
    assert report["final_lock"]["authorization_without_prerequisites"] is False
    assert report["protocol"]["model_evaluated"] is False
    assert report["protocol"]["existing_artifacts_rescored"] is False
    assert report["advance_to_model_fit"] is False
    assert report["advance_to_final_test"] is False
    assert report["advance_to_central_complex"] is False
