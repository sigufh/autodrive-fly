import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_stage1_development import (
    CONFIG,
    _as_visual_stimulus,
    run_signed_cell_responses,
)
from fly_emotion.driving.v7_stage1_scoring import CONFIG as SCORING_CONFIG
from fly_emotion.driving.v7_stage1_split import CONFIG as SPLIT_CONFIG
from fly_emotion.driving.v7_stage1_split import build_stage1_split

ROOT = Path(__file__).parents[1]


class IdentityProbe:
    def __init__(self) -> None:
        self.graph = type("Graph", (), {"node_count": 3})()
        self.source_delays = np.zeros(3, dtype=np.int8)
        self.correlator = None
        self.retina = type("Retina", (), {"node_indices": np.array([0, 1, 2])})()
        self.retinal_permutation = np.arange(3)
        self.baseline_frames = 1
        self.brain_substeps = 1
        self.populations = {"T4a_L": np.array([0, 1, 2])}

    def _sample_retina(self, image: np.ndarray) -> np.ndarray:
        return image.reshape(-1)[:3]

    def _retinal_code(self, values, baseline, previous):
        return values - previous

    def _advance(self, state, drive, history):
        history[0] = state.copy()
        return drive.copy()


def test_signed_cell_response_retains_negative_values_and_body_order() -> None:
    split = yaml.safe_load((ROOT / SPLIT_CONFIG).read_text())
    item = build_stage1_split(split)["development"][0]
    item = type(item)(
        item.identity,
        item.split,
        item.family,
        item.polarity,
        item.direction,
        item.parameter_name,
        item.parameter_value,
        item.noise_standard_deviation,
        item.seed,
        np.array([[[0.5, 0.5, 0.5]], [[0.2, 0.7, 0.5]]], dtype=np.float32),
        item.mirror_of,
    )
    response = run_signed_cell_responses(IdentityProbe(), _as_visual_stimulus(item))
    np.testing.assert_allclose(response["peaks"]["T4a_L"], [0.0, 0.2, 0.0])
    assert response["population_traces"]["T4a_L"].shape == (2,)


def test_development_contract_uses_only_development_and_no_fitting() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    assert config["split"] == "development"
    assert config["retinal_backends"] == ["linear_luminance", "signed_frame_difference"]
    boundary = config["evaluation_boundary"]
    assert boundary["evaluate_development"] is True
    assert boundary["evaluate_validation"] is False
    assert boundary["evaluate_ood"] is False
    assert boundary["evaluate_final"] is False
    assert boundary["fit_parameters"] is False
    assert config["candidate_semantics"]["T5_direction_labels_externally_verified"] is False
    scoring = yaml.safe_load((ROOT / SCORING_CONFIG).read_text())
    assert scoring["aggregation"]["require_every_population"] is True


def test_saved_development_screen_is_hash_bound_and_non_advancing() -> None:
    report = json.loads((ROOT / "artifacts/v7-stage1-development.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["split"] == "development"
    assert report["protocol"]["parameter_fitting"] is False
    assert report["protocol"]["validation_evaluated"] is False
    assert report["protocol"]["ood_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False
    assert report["stimulus_count"] == 172
    assert report["topology_controls_performed"] is False
    assert report["advance_to_validation"] is False
    assert report["advance_to_final"] is False
    assert report["advance_to_central_complex"] is False
    expected_pass_counts = {
        "linear_luminance": {"direction": 0, "polarity": 9, "looming": 0, "mirror": 24},
        "signed_frame_difference": {
            "direction": 0,
            "polarity": 11,
            "looming": 0,
            "mirror": 4,
        },
    }
    for backend, result in report["retinal_results"].items():
        assert result["T4_conductance_target_count"] == 6749
        assert result["T4_all_target_count"] == 6852
        assert result["T4_conductance_target_fraction"] == 6749 / 6852
        scores = result["strict_scores"]
        assert len(scores["direction"]) == 16
        assert len(scores["polarity"]) == 16
        assert len(scores["looming"]) == 6
        for kind in ("direction", "polarity", "looming", "mirror"):
            assert (
                sum(item["passed"] for item in scores[kind].values())
                == (expected_pass_counts[backend][kind])
            )
        for group in (
            *scores["direction"].values(),
            *scores["polarity"].values(),
            *scores["looming"].values(),
        ):
            assert len(group["cell_ids"]) == group["cell_count"]
            assert len(group["denominator"]) == group["cell_count"]
