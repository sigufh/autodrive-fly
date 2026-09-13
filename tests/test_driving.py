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


def test_dopamine_policy_only_changes_existing_motor_inputs() -> None:
    body_ids = np.array([10059, 10162, 10527, 555871])
    adjacency = sparse.eye(4, format="csr", dtype=np.float32)
    policy = DopaminePolicy(adjacency, body_ids, seed=1)
    activity = np.ones(4, dtype=np.float32)
    policy.action(activity, np.ones(4, dtype=np.float32), explore=True)
    _, _, features = policy.action(
        activity * np.array([1.00001, 0.99999, 1.0, 1.0]),
        np.ones(4, dtype=np.float32),
        explore=True,
    )
    policy.learn(
        1.0, features, avoidance_target=0.5, danger=1.0, enabled=True
    )
    assert policy.plastic_synapses == 4
    assert policy.summary()["changed_synapses"] > 0


def test_real_driving_engine_preserves_full_graph_and_maps_retina() -> None:
    engine = DrivingEngine(Path(__file__).parents[1], seed=4, top_k=12)
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
    original.gains[0][0] = 1.5
    path = tmp_path / "policy.npz"
    original.save(path, body_ids)
    restored = DopaminePolicy(adjacency, body_ids, seed=2)
    restored.load(path, body_ids)
    assert restored.gains[0][0] == 1.5
