import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_closed_loop import (
    CALIBRATION_ARTIFACT,
    CONFIG,
    TUNING_ARTIFACT,
    EventParameters,
    NeuralEventAdapter,
    ReceptiveFieldVisualProbe,
)

ROOT = Path(__file__).parents[1]
CONTROLS = ROOT / "artifacts/v7-closed-loop-controls.json"
MULTI = ROOT / "artifacts/v7-closed-loop-multi.json"


def test_receptive_field_pooling_stays_within_each_eye_and_is_reflection_equivariant() -> None:
    probe = object.__new__(ReceptiveFieldVisualProbe)
    probe.receptive_field_radius = 2
    probe.retinal_u = np.array([0.0, 0.25, 0.49, 0.51, 0.75, 1.0], dtype=np.float32)
    probe.retinal_v = np.full(6, 0.5, dtype=np.float32)
    rng = np.random.default_rng(7)
    image = rng.random((12, 24), dtype=np.float32)
    left_changed = image.copy()
    left_changed[:, :12] = 1.0 - left_changed[:, :12]
    original = probe._sample_retina(image)
    changed = probe._sample_retina(left_changed)
    np.testing.assert_array_equal(original[3:], changed[3:])
    reflected = probe._sample_retina(image[:, ::-1])
    np.testing.assert_allclose(original, reflected[::-1], atol=1e-7)


def test_event_adapter_is_fixed_environment_blind_turn_then_counter_turn() -> None:
    adapter = NeuralEventAdapter(
        EventParameters(
            signal_threshold=2e-6,
            turn_steps=4,
            counter_turn_steps=4,
            steering_amplitude=0.8,
            fixed_throttle=0.35,
        )
    )
    assert adapter.step(0.0) == (0.0, 0.35, 0.0)
    actions = [adapter.step(-3e-6) for _ in range(8)]
    np.testing.assert_allclose([item[0] for item in actions], [0.8] * 4 + [-0.8] * 4)
    assert all(item[1:] == (0.35, 0.0) for item in actions)
    assert adapter.step(1.0) == (0.0, 0.35, 0.0)


def test_saved_tuning_freezes_a_six_of_six_closed_loop_candidate() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    report = json.loads((ROOT / TUNING_ARTIFACT).read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["visual_input_only_via_R1_R6"] is True
    assert report["protocol"]["environment_state_used_by_controller"] is False
    assert report["candidate_count"] == 9
    selected = report["selected_candidate"]
    assert selected["signal_threshold"] == 2e-6
    assert selected["turn_steps"] == selected["counter_turn_steps"] == 4
    winner = next(
        item
        for item in report["candidate_summaries"]
        if item["candidate"]
        == {
            "signal_threshold": 2e-6,
            "turn_steps": 4,
            "counter_turn_steps": 4,
        }
    )
    assert winner["success_count"] == winner["first_obstacle_pass_count"] == 6
    assert winner["terminal_counts"] == {
        "success": 6,
        "obstacle": 0,
        "road_boundary": 0,
        "timeout": 0,
    }
    assert all(
        max(
            item["maximum_signal_odd_error"],
            item["maximum_action_odd_error"],
            item["maximum_trajectory_mirror_error"],
        )
        == 0
        for item in winner["mirror_checks"]
    )
    assert set(config["tuning"]["condition_pairs"]) == {"CL-T01", "CL-T02", "CL-T03"}
    assert report["tuning_passed"] is True
    assert report["final_evaluated"] is False


def test_saved_calibration_is_one_frozen_pair_and_does_not_authorize_release() -> None:
    report = json.loads((ROOT / CALIBRATION_ARTIFACT).read_text())
    tuning = json.loads((ROOT / TUNING_ARTIFACT).read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["seeds"] == [7300, 7301]
    assert report["protocol"]["candidate_sha256"] == tuning["selected_candidate_sha256"]
    assert report["protocol"]["parameters_changed_after_tuning"] is False
    assert [item["terminal_reason"] for item in report["episodes"]] == [
        "success",
        "success",
    ]
    assert all(report["gates"].values())
    assert report["calibration_passed"] is True
    assert report["final_evaluated"] is False
    assert report["advance_to_external_final"] is False
    assert report["advance_to_navigation_release"] is False


def test_causal_controls_require_input_support_and_correct_actions() -> None:
    report = json.loads(CONTROLS.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    arms = report["arms"]
    assert arms["frozen_candidate"]["success_count"] == 6
    assert arms["zero_action"]["success_count"] == 0
    assert arms["wrong_action_sign"]["success_count"] == 0
    assert arms["point_sampled_R1_R6"]["success_count"] == 2
    assert all(report["gates"].values())
    assert report["causal_controls_passed"] is True
    assert report["protocol"]["candidate_parameters_changed"] is False
    assert report["protocol"]["control_results_used_for_selection"] is False
    assert report["advance_to_navigation_release"] is False


def test_frozen_single_event_candidate_retains_multi_obstacle_failure() -> None:
    report = json.loads(MULTI.read_text())
    tuning = json.loads((ROOT / TUNING_ARTIFACT).read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["candidate_sha256"] == tuning["selected_candidate_sha256"]
    assert report["protocol"]["candidate_parameters_changed"] is False
    assert report["protocol"]["parameter_search_performed"] is False
    assert report["summary"]["episode_count"] == 6
    assert report["summary"]["success_count"] == 0
    assert {item["obstacles_passed"] for item in report["episodes"]} == {0, 1, 3}
    assert report["summary"]["terminal_counts"] == {
        "success": 0,
        "obstacle": 6,
        "road_boundary": 0,
        "timeout": 0,
    }
    assert all(
        max(
            item["maximum_signal_odd_error"],
            item["maximum_action_odd_error"],
            item["maximum_trajectory_mirror_error"],
        )
        == 0
        for item in report["mirror_checks"]
    )
    assert report["diagnostic_passed"] is False
    assert report["final_evaluated"] is False
    assert report["advance_to_navigation_release"] is False
