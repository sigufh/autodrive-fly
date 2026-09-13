from pathlib import Path

import numpy as np
from scipy import sparse

from fly_emotion.driving.engine import DopaminePolicy, DrivingEngine
from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.evaluate import paired_statistics
from fly_emotion.driving.retina import RetinaMap


def test_retina_samples_image_at_mapped_coordinates() -> None:
    retina = RetinaMap(
        np.array([2, 3]), np.array([12, 13]),
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([-1, 1]),
    )
    image = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    assert np.allclose(retina.encode(image), [0.2, 0.8])


def test_environment_reward_and_collision_are_closed_loop() -> None:
    environment = DrivingEnvironment()
    _, reward, done = environment.step(0.0, 1.0)
    assert reward > 0
    assert not done
    environment.x = environment.road_half_width
    _, reward, done = environment.step(0.0, 1.0)
    assert reward < 0
    assert done


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
    body_ids = np.array([10059, 10162, 10527, 555871])
    adjacency = sparse.eye(4, format="csr", dtype=np.float32)
    policy = DopaminePolicy(adjacency, body_ids, seed=1)
    activity = np.ones(4, dtype=np.float32)
    policy.action(activity, np.ones(4, dtype=np.float32), explore=True, adapt=True)
    _, _, features = policy.action(
        activity * np.array([1.00001, 0.99999, 1.0, 1.0]),
        np.ones(4, dtype=np.float32),
        explore=True,
        adapt=True,
    )
    policy.learn(
        1.0, features, avoidance_target=0.5, danger=1.0, enabled=True
    )
    assert policy.plastic_synapses == 4
    assert policy.summary()["changed_synapses"] > 0


def test_real_driving_engine_preserves_full_graph_and_maps_retina() -> None:
    engine = DrivingEngine(
        Path(__file__).parents[1], seed=4, top_k=12, load_checkpoint=False
    )
    state = engine.step(learning=True)
    assert engine.graph.node_count == 166_700
    assert engine.graph.edge_count == 25_582_938
    assert state["retina"]["mapped_receptors"] == 3_344
    assert len(state["motor"]["body_ids"]) == 4
    assert state["motor"]["brain_substeps_per_action"] == 4
    assert state["dopamine_neurons"]["body_ids"] == [11327, 11900]
    assert len(state["environment"]["trajectory"]) == 2
    assert len(state["environment"]["projected_trajectory"]) == 12
    assert state["activity"]["units"] == "simulated_activation_not_millivolts"
    assert state["environment"]["step"] == 1


def test_reset_without_keep_learning_restores_published_policy() -> None:
    root = Path(__file__).parents[1]
    engine = DrivingEngine(root, seed=5, top_k=1, load_checkpoint=True)
    assert engine.policy.checkpoint_kind == "frozen_calibrated"
    published = engine.policy.gains[0].copy()
    engine.policy.gains[0].fill(1.03)
    engine.reset(6, keep_learning=False)
    assert engine.checkpoint_loaded
    assert engine.policy.checkpoint_kind == "frozen_calibrated"
    assert np.array_equal(engine.policy.gains[0], published)


def test_paired_statistics_measure_learning_gain() -> None:
    frozen = [
        {"distance": float(value), "return": float(value)}
        for value in range(1, 9)
    ]
    learned = [
        {"distance": float(value + 2), "return": float(value + 1)}
        for value in range(1, 9)
    ]
    result = paired_statistics(frozen, learned, seed=7)
    assert result["distance"]["delta_mean"] == 2.0
    assert result["return"]["delta_mean"] == 1.0
    assert result["distance"]["bootstrap_95_ci"] == [2.0, 2.0]


def test_policy_checkpoint_validates_synapse_identity(tmp_path: Path) -> None:
    body_ids = np.array([10059, 10162, 10527, 555871])
    adjacency = sparse.eye(4, format="csr", dtype=np.float32)
    original = DopaminePolicy(adjacency, body_ids, seed=1)
    original.gains[0][0] = 1.02
    original.running_mean = [np.zeros_like(gain) for gain in original.gains]
    original.steering_bias = 0.25
    path = tmp_path / "policy.npz"
    original.save(path, body_ids)
    restored = DopaminePolicy(adjacency, body_ids, seed=2)
    restored.load(path, body_ids)
    assert np.isclose(restored.gains[0][0], 1.02)
    assert restored.running_mean[0] is not None
    assert np.array_equal(restored.running_mean[0], original.running_mean[0])
    assert np.array_equal(restored.running_variance[0], original.running_variance[0])
    assert restored.steering_bias == 0.25
    assert restored.checkpoint_kind == "learned"


def test_policy_checkpoint_reproduces_frozen_actions(tmp_path: Path) -> None:
    body_ids = np.array([10059, 10162, 10527, 555871])
    adjacency = sparse.eye(4, format="csr", dtype=np.float32)
    original = DopaminePolicy(adjacency, body_ids, seed=1)
    signs = np.ones(4, dtype=np.float32)
    original.action(
        np.array([0.2, 0.1, 0.3, 0.4], dtype=np.float32),
        signs, explore=False, adapt=True,
    )
    original.action(
        np.array([0.4, 0.2, 0.3, 0.4], dtype=np.float32),
        signs, explore=False, adapt=True,
    )
    path = tmp_path / "complete-policy.npz"
    original.save(path, body_ids, checkpoint_kind="frozen_calibrated")
    restored = DopaminePolicy(adjacency, body_ids, seed=99)
    restored.load(path, body_ids)
    original.reset_traces(episode_seed=7)
    restored.reset_traces(episode_seed=7)
    stimulus = np.array([0.35, 0.16, 0.31, 0.39], dtype=np.float32)
    original_action = original.action(stimulus, signs, explore=False, adapt=False)[:2]
    restored_action = restored.action(stimulus, signs, explore=False, adapt=False)[:2]
    assert np.allclose(original_action, restored_action, atol=0, rtol=0)
    assert restored.checkpoint_kind == "frozen_calibrated"


def test_legacy_checkpoint_is_rejected(tmp_path: Path) -> None:
    body_ids = np.array([10059, 10162, 10527, 555871])
    adjacency = sparse.eye(4, format="csr", dtype=np.float32)
    path = tmp_path / "legacy.npz"
    np.savez(path, format_version=np.array([1]), motor_body_ids=body_ids)
    policy = DopaminePolicy(adjacency, body_ids)
    with np.testing.assert_raises_regex(ValueError, "unsupported"):
        policy.load(path, body_ids)
