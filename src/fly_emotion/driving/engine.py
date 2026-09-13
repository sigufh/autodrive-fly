from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.retina import load_or_build_retina_map

MOTOR_BODY_IDS = np.asarray([10059, 10162, 10527, 555871], dtype=np.int64)
MOTOR_NAMES = ("turn_right_DNp20", "turn_left_DNp20", "drive_left_DNpe017", "drive_right_DNpe017")
DOPAMINE_BODY_IDS = np.asarray([11327, 11900], dtype=np.int64)
INHIBITORY_TRANSMITTERS = {"gaba", "glutamate", "histamine"}
MODULATORY_TRANSMITTERS = {"dopamine", "octopamine", "serotonin"}


class DopaminePolicy:
    """Plastic gain on real synapses entering four documented DN cells."""

    def __init__(
        self,
        adjacency,
        body_ids: np.ndarray,
        *,
        seed: int = 0,
        learning_rate: float = 0.04,
        exploration_sigma: float = 0.22,
    ):
        self.rng = np.random.default_rng(seed)
        self.motor_nodes = np.searchsorted(body_ids, MOTOR_BODY_IDS).astype(np.int32)
        if np.any(self.motor_nodes >= len(body_ids)) or not np.array_equal(
            body_ids[self.motor_nodes], MOTOR_BODY_IDS
        ):
            raise ValueError("configured motor neurons are absent from the MaleCNS graph")
        self.sources: list[np.ndarray] = []
        self.base: list[np.ndarray] = []
        self.gains: list[np.ndarray] = []
        self.eligibility: list[np.ndarray] = []
        self.running_mean: list[np.ndarray | None] = []
        self.running_variance: list[np.ndarray] = []
        for target in self.motor_nodes:
            start, end = adjacency.indptr[target : target + 2]
            self.sources.append(adjacency.indices[start:end].copy())
            base = adjacency.data[start:end].astype(np.float32, copy=True)
            base /= max(float(base.sum()), 1e-8)
            self.base.append(base)
            self.gains.append(np.ones_like(base))
            self.eligibility.append(np.zeros_like(base))
            self.running_mean.append(None)
            self.running_variance.append(np.full_like(base, 1e-5))
        self.reward_baseline = 0.0
        self.dopamine = 0.0
        self.lateral_dopamine = [0.0, 0.0]
        self.updates = 0
        self.learning_rate = learning_rate
        self.exploration_sigma = exploration_sigma

    @property
    def plastic_synapses(self) -> int:
        return sum(len(values) for values in self.gains)

    def reset_traces(self, *, episode_seed: int | None = None) -> None:
        for trace in self.eligibility:
            trace.fill(0)
        if episode_seed is not None:
            self.rng = np.random.default_rng(episode_seed)
        self.dopamine = 0.0
        self.lateral_dopamine = [0.0, 0.0]

    def action(
        self, activity: np.ndarray, source_sign: np.ndarray, *, explore: bool
    ) -> tuple[float, float, list[np.ndarray]]:
        inputs = []
        scores = []
        centered = []
        for index, (sources, base, gains) in enumerate(
            zip(self.sources, self.base, self.gains, strict=True)
        ):
            values = activity[sources] * source_sign[sources]
            if self.running_mean[index] is None:
                self.running_mean[index] = values.copy()
            delta = values - self.running_mean[index]
            self.running_mean[index] += 0.03 * delta
            self.running_variance[index] *= 0.995
            self.running_variance[index] += 0.005 * delta * delta
            normalized = np.clip(
                delta / np.sqrt(self.running_variance[index] + 1e-6), -5.0, 5.0
            )
            inputs.append(values)
            centered.append(normalized)
            scores.append(float(np.dot(base * gains, normalized)))
        steering_mean = float(np.tanh((scores[0] - scores[1]) * 3.0))
        throttle_mean = float(1 / (1 + np.exp(-((scores[2] + scores[3]) * 2.0 + 0.2))))
        steering_noise = self.rng.normal(0, self.exploration_sigma) if explore else 0.0
        steering = float(
            np.clip(steering_mean + steering_noise, -1, 1)
        )
        throttle = float(
            np.clip(throttle_mean + (self.rng.normal(0, 0.08) if explore else 0), 0, 1)
        )
        steering_gradient = (
            steering_noise
            / (self.exploration_sigma**2)
            * (1.0 - steering_mean**2)
            * 3.0
            if explore
            else 0.0
        )
        features = [
            self.base[0] * centered[0] * steering_gradient,
            -self.base[1] * centered[1] * steering_gradient,
            np.zeros_like(centered[2]),
            np.zeros_like(centered[3]),
        ]
        self._last_centered = centered
        self._last_steering_mean = steering_mean
        return steering, throttle, features

    def learn(
        self,
        modulation: float,
        features: list[np.ndarray],
        *,
        avoidance_target: float,
        danger: float,
        enabled: bool,
    ) -> float:
        self.reward_baseline = 0.995 * self.reward_baseline + 0.005 * modulation
        self.dopamine = float(np.clip(modulation - self.reward_baseline, -2.0, 2.0))
        for trace, feature in zip(self.eligibility, features, strict=True):
            trace *= 0.96
            trace += feature
        if enabled:
            # Compartment-like opponent teaching: both updates remain confined to
            # existing presynaptic partners of the left/right DNp20 cells. The
            # ray-derived target is a training-only aversive signal and never
            # overrides the neural action at inference.
            error = avoidance_target - self._last_steering_mean
            gradient = danger * error * (1.0 - self._last_steering_mean**2) * 3.0
            self.lateral_dopamine = [float(gradient), float(-gradient)]
            self.gains[0] += self.learning_rate * gradient * self.base[0] * self._last_centered[0]
            self.gains[1] -= self.learning_rate * gradient * self.base[1] * self._last_centered[1]
            for gains, trace in zip(self.gains, self.eligibility, strict=True):
                gains += 0.15 * self.learning_rate * self.dopamine * trace
                np.clip(gains, 0.2, 3.0, out=gains)
            self.updates += 1
        return self.dopamine

    def summary(self) -> dict:
        changed = np.concatenate([np.abs(gain - 1) > 1e-6 for gain in self.gains])
        gains = np.concatenate(self.gains)
        return {
            "rule": "reward_prediction_error_x_eligibility",
            "dopamine": self.dopamine,
            "lateral_dopamine": self.lateral_dopamine,
            "plastic_synapses": self.plastic_synapses,
            "changed_synapses": int(changed.sum()),
            "mean_gain": float(gains.mean()),
            "min_gain": float(gains.min()),
            "max_gain": float(gains.max()),
            "updates": self.updates,
            "learning_rate": self.learning_rate,
            "exploration_sigma": self.exploration_sigma,
            "centering": "running_mean_and_variance",
        }

    def save(self, path: Path, body_ids: np.ndarray) -> None:
        payload = {
            "format_version": np.asarray([1], dtype=np.int32),
            "motor_body_ids": MOTOR_BODY_IDS,
        }
        for index, (sources, gains) in enumerate(
            zip(self.sources, self.gains, strict=True)
        ):
            payload[f"source_body_ids_{index}"] = body_ids[sources]
            payload[f"gains_{index}"] = gains
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    def load(self, path: Path, body_ids: np.ndarray) -> None:
        payload = np.load(path, allow_pickle=False)
        if payload["format_version"].tolist() != [1]:
            raise ValueError("unsupported driving policy checkpoint")
        if not np.array_equal(payload["motor_body_ids"], MOTOR_BODY_IDS):
            raise ValueError("driving policy motor-neuron contract mismatch")
        loaded = []
        for index, sources in enumerate(self.sources):
            if not np.array_equal(payload[f"source_body_ids_{index}"], body_ids[sources]):
                raise ValueError("driving policy synapse contract mismatch")
            gains = payload[f"gains_{index}"].astype(np.float32)
            if gains.shape != self.gains[index].shape or not np.all(np.isfinite(gains)):
                raise ValueError("invalid driving policy gains")
            if np.any((gains < 0.2) | (gains > 3.0)):
                raise ValueError("driving policy gains exceed configured bounds")
            loaded.append(gains)
        self.gains = loaded


class DrivingEngine:
    def __init__(
        self,
        root: Path,
        *,
        seed: int = 0,
        top_k: int = 220,
        brain_substeps: int = 4,
        load_checkpoint: bool = True,
    ):
        self.root = root
        processed = root / "data/processed/malecns-v1.0"
        raw = root / "data/raw/malecns-v1.0"
        self.graph = load_graph(processed)
        self.retina = load_or_build_retina_map(
            self.graph, raw / "body-annotations.feather", processed / "retina_map.npz"
        )
        self.source_sign = self._source_sign(raw / "body-neurotransmitters.feather")
        self.policy = DopaminePolicy(self.graph.adjacency, self.graph.body_ids, seed=seed)
        self.policy_checkpoint = root / "artifacts/checkpoints/driving-policy.npz"
        self.checkpoint_loaded = False
        if load_checkpoint and self.policy_checkpoint.exists():
            self.policy.load(self.policy_checkpoint, self.graph.body_ids)
            self.checkpoint_loaded = True
        self.dopamine_nodes = np.searchsorted(self.graph.body_ids, DOPAMINE_BODY_IDS)
        if not np.array_equal(self.graph.body_ids[self.dopamine_nodes], DOPAMINE_BODY_IDS):
            raise ValueError("configured PPL101 dopamine cells are absent from MaleCNS")
        self.env = DrivingEnvironment()
        self.activity = np.zeros(self.graph.node_count, dtype=np.float32)
        self.visual_drive = np.zeros_like(self.activity)
        self.top_k = top_k
        if brain_substeps < 1:
            raise ValueError("brain_substeps must be positive")
        self.brain_substeps = brain_substeps
        overview = json.loads((processed / "overview.json").read_text())
        self.visible_mask = np.isin(self.graph.body_ids, overview["body_ids"])
        self.visible = np.flatnonzero(self.visible_mask)
        self.last_action = {"steering": 0.0, "throttle": 0.0}
        self.last_reward = 0.0
        self.last_safety_signal = 0.0
        self.learning = True
        self.reset(seed)

    def _source_sign(self, path: Path) -> np.ndarray:
        table = feather.read_table(path, columns=["body", "consensus_nt"], memory_map=True)
        ids = table["body"].to_numpy(zero_copy_only=False).astype(np.int64)
        names = table["consensus_nt"].to_pylist()
        nodes = np.searchsorted(self.graph.body_ids, ids)
        valid = nodes < self.graph.node_count
        valid[valid] &= self.graph.body_ids[nodes[valid]] == ids[valid]
        signs = np.ones(self.graph.node_count, dtype=np.float32)
        for node, name in zip(nodes[valid], np.asarray(names, dtype=object)[valid], strict=True):
            if name in INHIBITORY_TRANSMITTERS:
                signs[node] = -1.0
            elif name in MODULATORY_TRANSMITTERS:
                signs[node] = 0.0
        return signs

    def reset(self, seed: int = 0, *, keep_learning: bool = True) -> dict:
        self.env.reset(seed)
        self.activity.fill(0)
        self.visual_drive.fill(0)
        self.policy.reset_traces(episode_seed=seed)
        if not keep_learning:
            self.policy = DopaminePolicy(self.graph.adjacency, self.graph.body_ids, seed=seed)
            self.checkpoint_loaded = False
        self.last_action = {"steering": 0.0, "throttle": 0.0}
        self.last_reward = 0.0
        self.last_safety_signal = 0.0
        for _ in range(self.brain_substeps):
            self._advance_brain(self.env.observe(), dopamine=0.0)
        return self.state(include_activity=True)

    def _advance_brain(self, image: np.ndarray, *, dopamine: float) -> None:
        self.visual_drive.fill(0)
        # Photoreceptor activity follows local contrast plus luminance.
        receptor_values = self.retina.encode(image)
        receptor_values = 0.25 * receptor_values + 0.75 * np.abs(
            receptor_values - float(receptor_values.mean())
        )
        self.visual_drive[self.retina.node_indices] = receptor_values
        recurrent = self.graph.adjacency @ (self.activity * self.source_sign)
        self.activity = (
            0.72 * self.activity + 0.28 * np.tanh(1.8 * recurrent + self.visual_drive)
        ).astype(np.float32)
        # The current image is clamped at the sensory boundary for this time step.
        self.activity[self.retina.node_indices] = receptor_values
        # Reward prediction error is represented on the two annotated PPL101
        # dopamine cells as a signed model signal. It gates plasticity below but
        # is excluded from the fast transmitter-weighted recurrent current.
        self.activity[self.dopamine_nodes] = np.tanh(dopamine)

    def step(
        self,
        *,
        learning: bool = True,
        explore: bool | None = None,
        include_activity: bool = True,
    ) -> dict:
        if self.env.done:
            return self.state(include_activity=include_activity)
        if explore is None:
            explore = learning
        started = time.perf_counter()
        rays = self.env.last_rays
        quarter = len(rays) // 4
        left_clearance = float(np.mean(rays[quarter : 2 * quarter]))
        right_clearance = float(np.mean(rays[2 * quarter : 3 * quarter]))
        asymmetry = (right_clearance - left_clearance) / self.env.max_sensor_distance
        forward_clearance = float(np.min(rays[quarter : 3 * quarter]))
        danger = np.clip(1.0 - forward_clearance / self.env.max_sensor_distance, 0.0, 1.0)
        steering, throttle, features = self.policy.action(
            self.activity, self.source_sign, explore=explore
        )
        image, reward, done = self.env.step(steering, throttle)
        avoidance_target = float(np.tanh(4.0 * danger * asymmetry))
        safety_signal = float(danger * steering * asymmetry)
        terminal_signal = 3.0 if self.env.y >= self.env.road_length else (-1.5 if done else 0.0)
        dopamine = self.policy.learn(
            terminal_signal + safety_signal,
            features,
            avoidance_target=avoidance_target,
            danger=float(danger),
            enabled=learning,
        )
        for _ in range(self.brain_substeps):
            self._advance_brain(image, dopamine=dopamine)
        self.last_action = {"steering": steering, "throttle": throttle}
        self.last_reward = reward
        self.last_safety_signal = safety_signal
        self.learning = learning
        result = self.state(include_activity=include_activity)
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        result["dopamine"]["dopamine"] = dopamine
        return result

    def _activity_frame(self) -> dict:
        count = min(self.top_k, len(self.visible))
        local = np.argpartition(np.abs(self.activity[self.visible]), -count)[-count:]
        selected = self.visible[local]
        selected = selected[np.argsort(-np.abs(self.activity[selected]), kind="stable")]
        neurons = [
            {"body_id": int(self.graph.body_ids[node]), "value": float(self.activity[node])}
            for node in selected if self.activity[node] != 0
        ]
        pathways = []
        selected_set = set(map(int, selected))
        for target in selected[:80]:
            start, end = self.graph.adjacency.indptr[target : target + 2]
            sources = self.graph.adjacency.indices[start:end]
            values = (
                self.graph.adjacency.data[start:end]
                * self.activity[sources]
                * self.source_sign[sources]
            )
            for index in np.argsort(-np.abs(values), kind="stable")[:4]:
                source = int(sources[index])
                if source in selected_set and values[index] != 0:
                    pathways.append({
                        "pre": int(self.graph.body_ids[source]),
                        "post": int(self.graph.body_ids[target]),
                        "value": float(values[index]),
                    })
        pathways.sort(key=lambda edge: abs(edge["value"]), reverse=True)
        return {
            "type": "activity", "step": self.env.steps, "total_steps": 500,
            "elapsed_ms": 0.0, "units": "simulated_activation_not_millivolts",
            "neurons": neurons, "pathways": pathways[:320],
            "statistics": {
                "all_nodes": self.graph.node_count,
                "active_nodes": int(np.count_nonzero(self.activity)),
                "positioned_nodes": len(self.visible),
                "unpositioned_active_nodes": int(
                    np.count_nonzero(self.activity[~self.visible_mask])
                ),
                "displayed_nodes": len(neurons), "displayed_edges": min(len(pathways), 320),
                "max_abs_state": float(np.abs(self.activity).max()),
                "display_max_abs_state": max((abs(n["value"]) for n in neurons), default=0.0),
            },
        }

    def state(self, *, include_activity: bool = False) -> dict:
        state = {
            "environment": self.env.snapshot(),
            "action": self.last_action,
            "reward": self.last_reward,
            "safety_signal": self.last_safety_signal,
            "learning": self.learning,
            "policy_checkpoint": {
                "loaded": self.checkpoint_loaded,
                "path": str(self.policy_checkpoint.relative_to(self.root)),
            },
            "dopamine": self.policy.summary(),
            "retina": {
                "mapped_receptors": self.retina.size, "source_type": "R1-R6",
                "mapping": "contact_weighted_postsynaptic_optic_hex_proxy",
                "width": self.env.image_width, "height": self.env.image_height,
                "stimulus": self.env.observe().tolist(),
            },
            "motor": {
                "body_ids": MOTOR_BODY_IDS.tolist(), "names": list(MOTOR_NAMES),
                "mapping": "engineered_bilateral_DN_readout",
                "brain_substeps_per_action": self.brain_substeps,
            },
            "dopamine_neurons": {
                "body_ids": DOPAMINE_BODY_IDS.tolist(),
                "type": "PPL101",
                "signal": "signed_reward_prediction_error_model_state",
            },
        }
        if include_activity:
            state["activity"] = self._activity_frame()
        return state
