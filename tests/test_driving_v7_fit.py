import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_fit import (
    FIT_CONFIG,
    _cell_split,
    _response_energy,
    build_t4_fit_stimuli,
)

ROOT = Path(__file__).parents[1]


def _config() -> dict:
    return yaml.safe_load((ROOT / FIT_CONFIG).read_text(encoding="utf-8"))


def test_v7_fit_stimulus_splits_are_disjoint_and_deterministic() -> None:
    config = _config()
    hashes = {}
    for split, expected_count in (("train", 16), ("validation", 8), ("test", 8)):
        first = build_t4_fit_stimuli(config, split)
        second = build_t4_fit_stimuli(config, split)
        assert len(first) == expected_count
        assert [item.sha256 for item in first] == [item.sha256 for item in second]
        assert all(item.family == "fit_edge" for item in first)
        hashes[split] = {item.sha256 for item in first}
    assert not hashes["train"] & hashes["validation"]
    assert not hashes["train"] & hashes["test"]
    assert not hashes["validation"] & hashes["test"]


def test_v7_fit_cell_split_is_stratified_disjoint_and_deterministic() -> None:
    config = _config()
    body_ids = np.arange(800, dtype=np.int64)
    populations = np.repeat(
        np.asarray([f"T4{subtype}_{side}" for subtype in "abcd" for side in "LR"]),
        100,
    )
    first = _cell_split(body_ids, populations, config)
    second = _cell_split(body_ids, populations, config)
    for name in first:
        assert np.array_equal(first[name], second[name])
    assert np.all(sum(first.values()) == 1)
    assert np.count_nonzero(first["train"]) == 480
    assert np.count_nonzero(first["validation"]) == 160
    assert np.count_nonzero(first["test"]) == 160


def test_v7_fit_response_energy_depends_on_real_source_features() -> None:
    features = {name: np.zeros((4, 3), dtype=float) for name in ("Mi9", "Tm3", "Mi1", "Mi4", "C3")}
    parameters = np.asarray(
        [0.92, 0.35, 0.65, 1.10, 1.49, 0.20, 0.35, 0.88, 0.44, 0.70, -65.0, 0.50]
    )
    baseline = _response_energy(features, parameters)
    features["Tm3"][1:, 0] = 1.0
    excited = _response_energy(features, parameters)
    assert excited[0] > baseline[0]
    assert np.array_equal(excited[1:], baseline[1:])


def test_v7_fit_contract_seals_test_until_validation() -> None:
    config = _config()
    assert config["deployment_enabled"] is False
    assert config["driving_data_allowed"] is False
    assert config["test_policy"]["evaluate_only_if_all_validation_gates_pass"] is True
    assert config["test_policy"]["original_visual_battery_only_after_test_pass"] is True
    assert len(hashlib.sha256((ROOT / FIT_CONFIG).read_bytes()).hexdigest()) == 64


def test_v7_fit_artifact_failed_validation_and_kept_test_sealed() -> None:
    report = json.loads((ROOT / "artifacts/v7-t4-conductance-fit.json").read_text(encoding="utf-8"))
    assert report["protocol"]["driving_data_used"] is False
    assert report["split_body_id_overlap"] == {
        "train_validation": 0,
        "train_test": 0,
    }
    assert report["validation_passed"] is False
    assert report["test"]["evaluated"] is False
    assert report["original_visual_battery_evaluated"] is False
    assert report["validation_metrics"]["on_off_accuracy"] > 0.75
    assert report["validation_metrics"]["direction_accuracy"] < 0.60
