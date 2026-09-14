import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import fly_emotion.driving.evaluate as evaluation_module
from fly_emotion.driving.city import CityDrivingEnvironment
from fly_emotion.driving.engine import (
    DopaminePolicy,
    DrivingEngine,
    NeuralMotorAdapter,
    assisted_steering,
    map_neural_motor_output,
)
from fly_emotion.driving.environment import DrivingEnvironment, Obstacle
from fly_emotion.driving.evaluate import (
    evaluate_city_alpha,
    evaluate_neural_decision_baseline,
    mirror_summary,
    paired_statistics,
    summarize_post_pass_windows,
    usability_gates,
    validate_mirror_protocol,
)
from fly_emotion.driving.retina import RetinaMap
from fly_emotion.driving.sensory import (
    SensoryFrame,
    horizontal_flow_proxy,
)


def motor_fixture():
    body_ids = np.array([10059, 10162, 10527, 10763, 11288, 11332, 12348, 555871])
    adjacency = sparse.eye(8, format="csr", dtype=np.float32)
    return body_ids, adjacency


def test_retina_samples_image_at_mapped_coordinates() -> None:
    retina = RetinaMap(
        np.array([2, 3]),
        np.array([12, 13]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([-1, 1]),
    )
    image = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    assert np.allclose(retina.encode(image), [0.2, 0.8])


def test_retina_mapping_version_is_explicit() -> None:
    retina = RetinaMap(
        np.array([2]),
        np.array([12]),
        np.array([0.5]),
        np.array([0.5]),
        np.array([-1]),
    )
    assert retina.mapping_version == 2


def test_horizontal_flow_proxy_is_signed_and_mirror_equivariant() -> None:
    previous = np.zeros((8, 24), dtype=np.float32)
    previous[:, 6:10] = 1
    shifted = np.roll(previous, 3, axis=1)
    flow = horizontal_flow_proxy(previous, shifted)
    mirrored = horizontal_flow_proxy(previous[:, ::-1], shifted[:, ::-1])
    assert flow != 0
    assert np.isclose(flow, -mirrored)


def test_anatomy_sensory_projection_uses_real_balanced_groups() -> None:
    root = Path(__file__).parents[1]
    engine = DrivingEngine(root, top_k=1, load_checkpoint=False)
    projection = engine.sensory_projection
    summary = projection.summary()
    assert summary["T4_T5_horizontal_flow"] > 6000
    assert summary["haltere_yaw_rate"] == 205
    assert summary["ascending_proprioception"] == 424
    assert all(value > 90 for value in summary["haltere_by_side"])
    drive = np.zeros(engine.graph.node_count, dtype=np.float32)
    projection.add_drive(
        drive,
        SensoryFrame(flow=0.5, yaw_rate=0.6, steering=0.4, speed=3.0),
        optic_flow=True,
        body=True,
    )
    assert np.any(drive[projection.flow_positive] > 0)
    assert np.any(drive[projection.haltere_right] > 0)
    assert np.any(drive[projection.proprio_right] > 0)


def test_environment_reward_and_collision_are_closed_loop() -> None:
    environment = DrivingEnvironment()
    _, reward, done = environment.step(0.0, 1.0)
    assert reward > 0
    assert not done
    environment.x = environment.road_half_width
    _, reward, done = environment.step(0.0, 1.0)
    assert reward < 0
    assert done


def test_adjacent_environment_seeds_are_exact_mirrors() -> None:
    left = DrivingEnvironment()
    right = DrivingEnvironment()
    left.reset(200)
    right.reset(201)
    assert [item.y for item in left.obstacles] == [item.y for item in right.obstacles]
    assert [item.radius for item in left.obstacles] == [item.radius for item in right.obstacles]
    assert np.allclose(
        [item.x for item in left.obstacles],
        [-item.x for item in right.obstacles],
    )
    assert left.obstacles[0].x < 0 < right.obstacles[0].x


def test_steering_actuator_has_deadband_and_rate_limit() -> None:
    environment = DrivingEnvironment()
    environment.obstacles = []
    environment.step(0.01, 0.5)
    assert environment.steering == 0.0
    environment.step(1.0, 0.5)
    assert environment.steering == environment.steering_rate_limit
    environment.step(-1.0, 0.5)
    assert environment.steering == 0.0


def test_empty_road_obstacle_rays_remain_clear() -> None:
    environment = DrivingEnvironment()
    environment.obstacles = []
    environment.observe()
    assert np.all(environment.last_obstacle_rays == environment.max_sensor_distance)
    assert np.any(environment.last_wall_rays < environment.max_sensor_distance)


def test_city_route_has_continuous_multi_road_geometry_and_rule_state() -> None:
    city = CityDrivingEnvironment()
    assert city.road_length > 280
    before = city.route_pose(city.route[0].end)
    after = city.route_pose(city.route[1].end)
    assert np.allclose(before[:2], (0, 62))
    assert np.allclose(after[:2], (18, 80))
    city.y = 42
    guidance = city.traffic_guidance()
    assert guidance["traffic_light"] == "red"
    assert guidance["speed_cap"] < city.cruise_throttle
    assert guidance["next_maneuver"] == "right"


def test_city_rule_layer_requests_lane_change_then_returns_to_route_lane() -> None:
    city = CityDrivingEnvironment()
    city.y = 85
    assert city.traffic_guidance()["lane_action"] == "change_right_for_actor"
    city.y = 112
    assert city.traffic_guidance()["lane_action"] == "return_route_lane"


def test_city_red_signal_requires_stop_before_crossing() -> None:
    city = CityDrivingEnvironment()
    city.y, city.speed = 49.9, 3.0
    city.step(0, 0.62)
    assert city.terminal_reason == "traffic_violation"
    city.reset()
    city.y, city.speed = 49.9, 3.0
    city.stopped_for_signal.add("Harbor / Market")
    city.step(0, 0.62)
    assert "red_light:Harbor / Market" not in city.violations


def test_city_alpha_evaluation_reports_integration_boundary(monkeypatch, tmp_path: Path) -> None:
    class FakeEngine:
        def __init__(self, *_args, **_kwargs):
            self.env = SimpleNamespace(done=False)
            self.policy = SimpleNamespace(checkpoint_kind="learned_v5")
            self.policy_checkpoint = tmp_path / "policy.npz"
            self.policy_checkpoint.write_bytes(b"policy")

        def reset(self, *_args, **_kwargs):
            self.env.done = False

        def step(self, **_kwargs):
            self.env.done = True
            return {
                "environment": {
                    "terminal_reason": "success",
                    "vehicle": {"y": 282.0},
                    "obstacles_passed": 2,
                    "city": {"violations": []},
                },
                "control_statistics": {"mean_abs_steering_change": 0.02},
            }

    monkeypatch.setattr(evaluation_module, "DrivingEngine", FakeEngine)
    report = evaluate_city_alpha(tmp_path, seeds=(0, 1))
    assert report["summary"]["success_rate"] == 1
    assert report["summary"]["traffic_violation_rate"] == 0
    assert "not a city generalization" in report["protocol"]["claim_boundary"]


def test_neural_decision_evaluation_requires_zero_action_override(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "artifacts/checkpoints/driving-policy.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"policy")

    class FakeEngine:
        def __init__(self, *_args, **_kwargs):
            self.policy_checkpoint = checkpoint

    def fake_episode(_engine, value, *, control_mode, **_kwargs):
        return {
            "seed": value,
            "pair_seed": value // 2,
            "mirror": 1,
            "first_obstacle_side": "left",
            "steps": 1,
            "distance": 10.0,
            "return": 1.0,
            "success": control_mode == "assisted",
            "terminal_reason": "success" if control_mode == "assisted" else "obstacle",
            "obstacles_passed": 1,
            "first_obstacle_passed": True,
            "mean_abs_steering": 0.0,
            "mean_abs_steering_change": 0.0,
            "far_mean_abs_steering": 0.0,
            "steering_sign_changes": 0,
            "max_abs_lateral": 0.0,
            "constraint_rate": 0.0,
            "mean_abs_constraint": 0.0,
            "mean_signed_raw_steering": 0.0,
            "mean_signed_steering": 0.0,
            "right_turn_fraction": 0.0,
            "left_turn_fraction": 0.0,
            "pre_first_mean_signed_steering": 0.0,
            "mean_drive": 0.5,
            "mean_reverse": 0.0,
            "reverse_fraction": 0.0,
            "negative_speed_fraction": 0.0,
            "reverse_gate_fraction": 0.0,
            "post_pass_mean_abs_raw_steering": 0.0,
            "post_pass_mean_abs_steering": 0.0,
            "post_pass_mean_lateral_drift": 0.0,
            "collision_before_first_pass": False,
            "control_trace": [[0, 0, 0, 10, 0.5, 0]],
            "control_mode": control_mode,
        }

    monkeypatch.setattr(evaluation_module, "DrivingEngine", FakeEngine)
    monkeypatch.setattr(evaluation_module, "run_episode", fake_episode)
    report = evaluate_neural_decision_baseline(tmp_path, start=400, count=2)
    assert report["neural_decision"]["constraint_rate"] == 0
    assert report["delta_neural_minus_assisted"]["success_rate"] == -1


def test_lane_constraint_only_intervenes_near_boundary_or_outward_heading() -> None:
    engine = object.__new__(DrivingEngine)
    engine.env = DrivingEnvironment()
    steering, status = engine.apply_lane_constraint(0.4, enabled=True)
    assert steering == 0.4 and not status["active"]
    engine.env.x = 4.5
    engine.env.heading = 0.4
    steering, status = engine.apply_lane_constraint(0.4, enabled=True)
    assert status["active"] and steering < 0.4
    raw, status = engine.apply_lane_constraint(0.4, enabled=False)
    assert raw == 0.4 and not status["active"]


def test_dopamine_policy_only_changes_existing_motor_inputs() -> None:
    body_ids, adjacency = motor_fixture()
    policy = DopaminePolicy(adjacency, body_ids, seed=1)
    activity = np.ones(8, dtype=np.float32)
    policy.action(
        activity,
        np.ones(8, dtype=np.float32),
        explore=True,
        adapt=True,
        mirrored_activity=np.zeros(8, dtype=np.float32),
    )
    _, _, _, features = policy.action(
        activity * np.array([1.00001, 0.99999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.ones(8, dtype=np.float32),
        explore=True,
        adapt=True,
        mirrored_activity=np.zeros(8, dtype=np.float32),
    )
    policy.learn(
        1.0,
        features,
        avoidance_target=0.5,
        speed_target=0.25,
        reverse_target=0.8,
        danger=1.0,
        enabled=True,
    )
    assert policy.plastic_synapses == 8
    assert policy.summary()["changed_synapses"] > 0


def test_real_driving_engine_preserves_full_graph_and_maps_retina() -> None:
    engine = DrivingEngine(Path(__file__).parents[1], seed=4, top_k=12, load_checkpoint=False)
    state = engine.step(learning=True)
    assert engine.graph.node_count == 166_700
    assert engine.graph.edge_count == 25_582_938
    assert state["retina"]["mapped_receptors"] == 3_344
    assert len(state["motor"]["body_ids"]) == 8
    assert state["motor"]["body_ids"][-4:] == [10763, 11288, 11332, 12348]
    assert state["motor"]["brain_substeps_per_action"] == 4
    assert state["dopamine_neurons"]["body_ids"] == [11327, 11900]
    assert len(state["environment"]["trajectory"]) == 2
    assert len(state["environment"]["projected_trajectory"]) == 12
    assert state["activity"]["units"] == "simulated_activation_not_millivolts"
    assert state["environment"]["step"] == 1


def test_paired_statistics_measure_learning_gain() -> None:
    frozen = [{"distance": float(value), "return": float(value)} for value in range(1, 9)]
    learned = [{"distance": float(value + 2), "return": float(value + 1)} for value in range(1, 9)]
    result = paired_statistics(frozen, learned, seed=7)
    assert result["distance"]["delta_mean"] == 2.0
    assert result["return"]["delta_mean"] == 1.0
    assert result["distance"]["bootstrap_95_ci"] == [2.0, 2.0]


def test_policy_checkpoint_validates_synapse_identity(tmp_path: Path) -> None:
    body_ids, adjacency = motor_fixture()
    original = DopaminePolicy(adjacency, body_ids, seed=1)
    original.gains[0][0] = 1.02
    original.running_mean = [np.zeros_like(gain) for gain in original.gains]
    original.steering_bias = 0.0
    original.speed_bias = 0.0
    original.reverse_mean = 0.0
    path = tmp_path / "policy.npz"
    original.save(path, body_ids)
    restored = DopaminePolicy(adjacency, body_ids, seed=2)
    restored.load(path, body_ids)
    assert np.isclose(restored.gains[0][0], 1.02)
    assert restored.running_mean[0] is not None
    assert np.array_equal(restored.running_mean[0], original.running_mean[0])
    assert np.array_equal(restored.running_variance[0], original.running_variance[0])
    assert restored.steering_bias == 0.0
    assert restored.checkpoint_kind == "learned"


def test_policy_checkpoint_reproduces_frozen_actions(tmp_path: Path) -> None:
    body_ids, adjacency = motor_fixture()
    original = DopaminePolicy(adjacency, body_ids, seed=1)
    signs = np.ones(8, dtype=np.float32)
    original.action(
        np.array([0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.4], dtype=np.float32),
        signs,
        explore=False,
        adapt=True,
        mirrored_activity=np.zeros(8, dtype=np.float32),
    )
    original.action(
        np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.4], dtype=np.float32),
        signs,
        explore=False,
        adapt=True,
        mirrored_activity=np.zeros(8, dtype=np.float32),
    )
    path = tmp_path / "complete-policy.npz"
    original.save(path, body_ids, checkpoint_kind="frozen_calibrated")
    restored = DopaminePolicy(adjacency, body_ids, seed=99)
    restored.load(path, body_ids)
    original.reset_traces(episode_seed=7)
    restored.reset_traces(episode_seed=7)
    stimulus = np.array([0.35, 0.16, 0.31, 0.0, 0.0, 0.0, 0.0, 0.39], dtype=np.float32)
    original_action = original.action(
        stimulus, signs, explore=False, adapt=False, mirrored_activity=stimulus[::-1]
    )[:3]
    restored_action = restored.action(
        stimulus, signs, explore=False, adapt=False, mirrored_activity=stimulus[::-1]
    )[:3]
    assert np.allclose(original_action, restored_action, atol=0, rtol=0)
    assert restored.checkpoint_kind == "frozen_calibrated"


def test_legacy_checkpoint_is_rejected(tmp_path: Path) -> None:
    body_ids, adjacency = motor_fixture()
    path = tmp_path / "legacy.npz"
    np.savez(path, format_version=np.array([1]), motor_body_ids=body_ids)
    policy = DopaminePolicy(adjacency, body_ids)
    with np.testing.assert_raises_regex(ValueError, "unsupported"):
        policy.load(path, body_ids)


@pytest.mark.parametrize("width", [47, 48])
def test_mirrored_camera_and_dynamics(width: int) -> None:
    left, right = DrivingEnvironment(width=width), DrivingEnvironment(width=width)
    left.reset(200)
    right.reset(201)
    for steering in [0.3] * 15 + [-0.4] * 15:
        a, reward_a, done_a = left.step(steering, 0.6)
        b, reward_b, done_b = right.step(-steering, 0.6)
        np.testing.assert_array_equal(a, b[:, ::-1])
        assert left.x == -right.x
        assert left.y == right.y
        assert reward_a == reward_b and done_a == done_b


def test_obstacle_pass_is_full_clearance_unique_and_not_collision() -> None:
    env = DrivingEnvironment()
    env.obstacles = [Obstacle(4.0, 4.0, 1.0)]
    env.y = 5.5
    env.step(0, 0)
    assert env.obstacles_passed == 0
    env.step(0, 0)
    assert env.obstacles_passed == 1
    env.y = 3
    env.speed = 0
    env.step(0, 0)
    assert env.obstacles_passed == 1
    env.y = 5.6
    _, reward, _ = env.step(0, 0)
    assert env.obstacles_passed == 1
    assert reward < 0.35
    env.reset()
    env.obstacles = [Obstacle(4.0, 4.0, 1.0)]
    env.y, env.x = 6, 6
    env.step(0, 0)
    assert env.obstacles_passed == 0
    assert not env.snapshot()["success"]


@pytest.mark.parametrize("speed", [-2.0, 0.0, 6.0])
def test_lane_constraint_is_odd_under_mirroring(speed: float) -> None:
    engine = object.__new__(DrivingEngine)
    engine.env = DrivingEnvironment()
    for x, heading in [(0, 0), (0, 0.8), (2, 0.5), (4.5, -0.3)]:
        engine.env.x, engine.env.heading, engine.env.speed = x, heading, speed
        a, status_a = engine.apply_lane_constraint(0.4, enabled=True)
        engine.env.x, engine.env.heading = -x, -heading
        b, status_b = engine.apply_lane_constraint(-0.4, enabled=True)
        assert a == -b
        assert status_a["blend"] == status_b["blend"]


def test_policy_is_odd_for_steering_even_for_longitudinal_actions() -> None:
    ids, graph = motor_fixture()
    policy = DopaminePolicy(graph, ids)
    signs = np.ones(8, dtype=np.float32)
    a = np.asarray([0.4, 0.1, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1], dtype=np.float32)
    b = a[::-1].copy()
    policy.action(a, signs, mirrored_activity=b, explore=False, adapt=True)
    for _ in range(5):
        left = policy.action(a, signs, mirrored_activity=b, explore=False, adapt=False)
        right = policy.action(b, signs, mirrored_activity=a, explore=False, adapt=False)
        assert left[0] != 0
        assert left[0] == -right[0]
        assert left[1:3] == right[1:3]
    neutral = policy.action(a, signs, mirrored_activity=a, explore=False, adapt=False)
    assert neutral[0] == 0


def test_assisted_baseline_steering_chooses_the_free_side_and_recovers() -> None:
    # Positive steering moves right.  Thus the left obstacle has a positive
    # asymmetry and must produce a positive (rightward) response.
    left_obstacle = assisted_steering(
        0,
        obstacle_danger=0.8,
        obstacle_asymmetry=0.5,
        road_target=0,
    )
    right_obstacle = assisted_steering(
        0,
        obstacle_danger=0.8,
        obstacle_asymmetry=-0.5,
        road_target=0,
    )
    assert left_obstacle > 0 > right_obstacle
    assert np.isclose(left_obstacle, -right_obstacle)
    assert assisted_steering(0, obstacle_danger=0, obstacle_asymmetry=0, road_target=-0.5) < 0


def test_neural_motor_adapter_is_fixed_and_environment_independent() -> None:
    first = map_neural_motor_output(1.4, -0.2, 1.5)
    assert np.isclose(first[0], np.tanh(5.6))
    assert first[1:] == (0.0, 1.0)
    second = map_neural_motor_output(-0.35, 0.62, 0.2)
    assert np.isclose(second[0], np.tanh(-1.4))
    assert second[1:] == (0.31, 0.2)


def test_neural_motor_adapter_attenuates_sustained_turn_without_environment_input() -> None:
    adapter = NeuralMotorAdapter(steering_gain=4.0, adaptation_rate=0.08)
    outputs = [adapter.step(0.2, 0.5, 0.0)[0] for _ in range(40)]
    assert outputs[0] > 0.6
    assert outputs[-1] < 0.05
    adapter.reset()
    assert adapter.step(0.2, 0.5, 0.0)[0] == outputs[0]


def test_neural_mode_bypasses_environment_action_overrides() -> None:
    engine = DrivingEngine(Path(__file__).parents[1], top_k=1, load_checkpoint=False)
    engine.reset(400, control_mode="neural")
    features = [np.zeros_like(values) for values in engine.policy.gains]
    engine.policy.action = lambda *_args, **_kwargs: (0.4, 0.5, 0.2, features)
    engine.apply_lane_constraint = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("neural mode must not invoke the road safety controller")
    )
    state = engine.step(learning=False, safety_constraints=True, include_activity=False)
    assert state["control_mode"] == "neural"
    assert np.isclose(state["action"]["steering"], np.tanh(1.6))
    assert state["action"]["throttle"] == 0.25
    assert state["action"]["reverse"] == 0.2
    assert state["action"]["drive"] == 0.0
    assert state["lane_constraint"]["blend"] == 0
    assert state["lane_constraint"]["visual_avoidance"] == 0
    assert state["lane_constraint"]["road_recovery"] == 0


@pytest.mark.parametrize(
    "profile", ["front", "panorama", "panorama_flow", "panorama_flow_body"]
)
def test_sensory_profiles_cannot_bypass_fixed_neural_action(profile: str) -> None:
    engine = DrivingEngine(
        Path(__file__).parents[1], top_k=1, load_checkpoint=False,
        control_mode="neural", sensory_profile=profile,
    )
    engine.reset(400, control_mode="neural", sensory_profile=profile)
    features = [np.zeros_like(values) for values in engine.policy.gains]
    engine.policy.action = lambda *_args, **_kwargs: (0.2, 0.5, 0.0, features)
    state = engine.step(learning=False, safety_constraints=True, include_activity=False)
    assert np.isclose(state["action"]["steering"], np.tanh(0.8))
    assert state["lane_constraint"]["blend"] == 0
    assert state["lane_constraint"]["visual_avoidance"] == 0
    assert state["lane_constraint"]["road_recovery"] == 0


def test_single_obstacle_curriculum_is_mirrored_and_resettable() -> None:
    left, right = DrivingEnvironment(), DrivingEnvironment()
    left.reset(600)
    right.reset(601)
    left.configure_curriculum("single")
    right.configure_curriculum("single")
    assert left.obstacles[0].x == -right.obstacles[0].x
    assert left.obstacles[0].y == right.obstacles[0].y
    assert left.obstacles[0].radius == right.obstacles[0].radius
    assert left.pass_reward == 1.5 and left.collision_penalty == 8.0
    assert left.progress_reward_scale == 0.02
    left.reset(600)
    assert left.curriculum_stage == "full"
    assert left.road_length == 120.0 and len(left.obstacles) == 9


def test_neural_v6_checkpoint_keeps_independent_learning_contract(tmp_path: Path) -> None:
    ids, graph = motor_fixture()
    policy = DopaminePolicy(graph, ids)
    policy.configure_neural_curriculum()
    policy.action(np.zeros(8), np.ones(8), mirrored_activity=np.zeros(8), explore=False, adapt=True)
    path = tmp_path / "neural-v6.npz"
    policy.save(path, ids, checkpoint_kind="neural_curriculum_v6", format_version=6)
    restored = DopaminePolicy(graph, ids)
    restored.load(path, ids)
    assert restored.checkpoint_kind == "neural_curriculum_v6"
    assert restored.eligibility_decay == 0.995
    assert restored.min_gain == 0.65 and restored.max_gain == 1.35
    assert restored.exploration_sigma == 0.16


def test_mirror_protocol_rejects_incomplete_or_leaking_pairs() -> None:
    for count, start, exposure in [(3, 200, 4), (4, 201, 4), (4, 10002, 4)]:
        with pytest.raises(ValueError):
            validate_mirror_protocol(count, start, train_episodes=exposure)
    validate_mirror_protocol(32, 200, train_episodes=48)
    with pytest.raises(ValueError):
        mirror_summary([{"seed": 200}])


def test_early_collision_cannot_pass_usability_gate() -> None:
    gates = usability_gates(
        {
            "success_rate": 0,
            "first_obstacle_pass_rate": 0,
            "mean_obstacles_passed": 0,
            "collision_before_first_pass_rate": 1,
            "timeout_rate": 0,
            "road_exit_rate": 0,
            "mean_drive": 0.62,
            "reverse_fraction": 0,
            "negative_speed_fraction": 0,
        }
    )
    assert not all(gates.values())
    assert gates["road_exit_at_most_10pct"]
    assert not gates["early_collision_at_most_25pct"]


def test_longitudinal_and_reverse_actions_have_nonzero_eligibility() -> None:
    ids, graph = motor_fixture()
    policy = DopaminePolicy(graph, ids, seed=3)
    signs = np.ones(8, dtype=np.float32)
    baseline = np.ones(8, dtype=np.float32)
    policy.action(baseline, signs, mirrored_activity=baseline, explore=True, adapt=True)
    changed = baseline.copy()
    changed[[2, 3, 4, 5, 6, 7]] += 1e-5
    _, _, _, features = policy.action(
        changed, signs, mirrored_activity=changed, explore=True, adapt=True
    )
    assert np.any(features[2] != 0) and np.any(features[3] != 0)
    assert all(np.any(features[index] != 0) for index in range(4, 8))


def test_mdn_baseline_is_sparse_without_visual_danger() -> None:
    ids, graph = motor_fixture()
    policy = DopaminePolicy(graph, ids, seed=3)
    signs = np.ones(8, dtype=np.float32)
    activity = np.ones(8, dtype=np.float32)
    policy.action(activity, signs, mirrored_activity=activity, explore=False, adapt=True)
    outputs = [
        policy.action(activity, signs, mirrored_activity=activity, explore=False, adapt=False)[2]
        for _ in range(20)
    ]
    assert outputs == [0.0] * 20


def test_reverse_requires_persistent_close_hazard() -> None:
    engine = DrivingEngine(Path(__file__).parents[1], top_k=1, load_checkpoint=False)
    engine.env.obstacles = [Obstacle(0.0, 3.8, 1.0)]
    engine.env.observe()
    for _ in range(3):
        state = engine.step(include_activity=False)
        assert state["action"]["reverse"] == 0.0


def test_obstacle_pass_reward_modulates_policy() -> None:
    env = DrivingEnvironment()
    env.obstacles = [Obstacle(4.0, 4.0, 0.5)]
    env.y = 5.2
    _, reward, _ = env.step(0, 0)
    assert env.obstacles_passed == 1
    assert reward > 0.35


def test_episode_reports_post_pass_steering_and_lateral_drift() -> None:
    engine = DrivingEngine(
        Path(__file__).parents[1], top_k=1, load_checkpoint=True, control_mode="neural"
    )
    episode = evaluation_module.run_episode(
        engine,
        700,
        learning=False,
        explore=False,
        safety_constraints=False,
        control_mode="neural",
        curriculum_stage="triple",
    )
    assert episode["obstacles_passed"] == 3
    assert episode["post_pass_mean_abs_steering"] >= 0
    assert episode["post_pass_mean_abs_raw_steering"] >= 0
    assert episode["post_pass_mean_lateral_drift"] >= 0
    assert episode["post_pass_window_count"] == episode["obstacles_passed"]
    assert (
        episode["post_pass_complete_window_count"]
        + episode["post_pass_failure_truncated_window_count"]
        + episode["post_pass_censored_window_count"]
        == episode["post_pass_window_count"]
    )
    assert 0 <= episode["post_pass_complete_window_rate"] <= 1
    assert 0 <= episode["post_pass_30_step_early_failure_rate"] <= 1


def test_post_pass_metrics_do_not_treat_early_collision_as_stability() -> None:
    trace = np.zeros((40, 6), dtype=float)
    trace[:, 0] = 0.4
    trace[:, 1] = 0.2
    trace[:, 2] = np.arange(40) * 0.1
    trace[:, 3] = np.arange(40) * 0.2
    metrics = summarize_post_pass_windows(trace, [1, 31], "obstacle")
    assert metrics["post_pass_window_count"] == 2
    assert metrics["post_pass_complete_window_count"] == 1
    assert metrics["post_pass_failure_truncated_window_count"] == 1
    assert metrics["post_pass_censored_window_count"] == 0
    assert metrics["post_pass_complete_window_rate"] == 0.5
    assert metrics["post_pass_30_step_early_failure_rate"] == 0.5
    assert np.isclose(metrics["post_pass_complete_mean_abs_steering"], 0.2)
    assert np.isclose(metrics["post_pass_complete_mean_lateral_drift_per_metre"], 0.5)


def test_post_pass_short_success_window_is_censored_not_failed() -> None:
    trace = np.zeros((12, 6), dtype=float)
    metrics = summarize_post_pass_windows(trace, [5], "success")
    assert metrics["post_pass_complete_window_count"] == 0
    assert metrics["post_pass_failure_truncated_window_count"] == 0
    assert metrics["post_pass_censored_window_count"] == 1


def test_bootstrap_clusters_mirror_pairs_and_rejects_misalignment() -> None:
    baseline = [
        {"seed": seed, "pair_seed": seed // 2, "distance": 1.0, "return": 1.0}
        for seed in range(200, 204)
    ]
    result = paired_statistics(baseline, baseline, seed=7)
    assert result["independent_units"] == 2
    with pytest.raises(ValueError):
        paired_statistics(baseline, baseline[::-1], seed=7)


def test_real_engine_mirror_replay_and_checkpoint_reset(tmp_path: Path) -> None:
    engine = DrivingEngine(Path(__file__).parents[1], top_k=1, load_checkpoint=False)
    engine.step(explore=True, include_activity=False)
    engine.root = tmp_path
    engine.policy_checkpoint = tmp_path / "policy.npz"
    engine.policy.save(
        engine.policy_checkpoint, engine.graph.body_ids, checkpoint_kind="frozen_calibrated"
    )
    runs = []
    for seed in (200, 201, 200):
        engine.policy.gains[0].fill(1.03)
        engine.reset(seed, keep_learning=False)
        assert engine.checkpoint_loaded
        assert np.all(engine.policy.gains[0] == 1)
        trace = []
        for _ in range(12):
            state = engine.step(include_activity=False)
            trace.append(
                [
                    state["raw_action"]["steering"],
                    engine.env.x,
                    engine.env.y,
                    state["action"]["throttle"],
                    state["action"]["reverse"],
                ]
            )
        runs.append(np.asarray(trace))
    np.testing.assert_allclose(runs[0], runs[1] * [-1, -1, 1, 1, 1], atol=1e-7)
    np.testing.assert_array_equal(runs[0], runs[2])


def test_checkpoint_rejects_nonfinite_scalar_state(tmp_path: Path) -> None:
    ids, graph = motor_fixture()
    policy = DopaminePolicy(graph, ids)
    policy.action(
        np.zeros(8),
        np.ones(8),
        mirrored_activity=np.zeros(8),
        explore=False,
        adapt=True,
    )
    path = tmp_path / "policy.npz"
    policy.save(path, ids)
    with np.load(path) as source:
        payload = dict(source)
    payload["reverse_mean"] = np.asarray([np.nan])
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="scalar"):
        policy.load(path, ids)


def test_constraint_report_reuse_is_strict_and_skips_only_enabled_runs(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.npz"
    checkpoint.write_bytes(b"fixture checkpoint")
    reference_path = tmp_path / "reference.json"
    contract = {"policy_version": 5, "implementation_sha256": {"engine.py": "one"}}
    monkeypatch.setattr(evaluation_module, "evaluation_contract", lambda _root: contract)
    episodes = [
        {
            "seed": seed,
            "pair_seed": seed // 2,
            "mirror": 1 if seed % 2 == 0 else -1,
            "first_obstacle_side": "left" if seed % 2 == 0 else "right",
            "steps": 1,
            "distance": 12.0,
            "return": 1.0,
            "control_trace": [[0.0, 0.0, 0.0, 12.0, 0.62, 0.0]],
        }
        for seed in (400, 401)
    ]
    reference = {
        "protocol_version": 5,
        "candidate_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluation_contract": contract,
        "test_seed_range": [400, 401],
        "evaluation_learning": False,
        "evaluation_explore": False,
        "evaluation_safety_constraints": True,
        "deployed": {"details": episodes},
    }
    calls = []
    monkeypatch.setattr(
        evaluation_module,
        "DrivingEngine",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy=SimpleNamespace(checkpoint_kind="frozen_calibrated", load=lambda *_: None),
            graph=SimpleNamespace(body_ids=np.array([])),
            policy_checkpoint=checkpoint,
        ),
    )

    def run(_engine, seed, **options):
        calls.append((seed, options))
        return copy.deepcopy(episodes[seed - 400])

    monkeypatch.setattr(evaluation_module, "run_episode", run)
    monkeypatch.setattr(evaluation_module, "summarize", lambda rows: {"details": rows})
    monkeypatch.setattr(evaluation_module, "mirror_summary", lambda _rows: {})
    reference_path.write_text(json.dumps(reference))
    result = evaluation_module.evaluate_constraints(
        tmp_path,
        evaluation_start=400,
        evaluation_seeds=2,
        checkpoint=checkpoint,
        reference_report=reference_path,
    )
    assert [seed for seed, _ in calls] == [400, 401]
    assert all(
        options
        == {
            "learning": False,
            "explore": False,
            "safety_constraints": False,
        }
        for _, options in calls
    )
    assert result["constraint_on"]["details"] == episodes
    assert (
        result["protocol"]["reference_report_sha256"]
        == hashlib.sha256(reference_path.read_bytes()).hexdigest()
    )

    mismatches = [
        {"candidate_sha256": "wrong"},
        {"protocol_version": 4},
        {"evaluation_contract": {"policy_version": 5}},
        {"evaluation_learning": True},
        {"evaluation_explore": True},
        {"evaluation_safety_constraints": False},
        {"test_seed_range": [402, 403]},
        {"deployed": {"details": list(reversed(episodes))}},
    ]
    invalid_trace = copy.deepcopy(episodes)
    invalid_trace[0]["control_trace"] = [[0, 0, 0, 13, 0.62, 0]]
    mismatches.append({"deployed": {"details": invalid_trace}})
    for change in mismatches:
        calls.clear()
        reference_path.write_text(json.dumps({**reference, **change}))
        with pytest.raises(ValueError, match="reference report"):
            evaluation_module.evaluate_constraints(
                tmp_path,
                evaluation_start=400,
                evaluation_seeds=2,
                checkpoint=checkpoint,
                reference_report=reference_path,
            )
        assert calls == []
