from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.city import CityDrivingEnvironment
from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.retina import load_or_build_retina_map

MOTOR_BODY_IDS = np.asarray([10059, 10162, 10527, 555871], dtype=np.int64)
MOTOR_NAMES = ("turn_right_DNp20", "turn_left_DNp20", "drive_left_DNpe017", "drive_right_DNpe017")
REVERSE_BODY_IDS = np.asarray([10763, 11288, 11332, 12348], dtype=np.int64)
REVERSE_NAMES = ("reverse_MDN_R1", "reverse_MDN_L1", "reverse_MDN_R2", "reverse_MDN_L2")
DOPAMINE_BODY_IDS = np.asarray([11327, 11900], dtype=np.int64)
INHIBITORY_TRANSMITTERS = {"gaba", "glutamate", "histamine"}
MODULATORY_TRANSMITTERS = {"dopamine", "octopamine", "serotonin"}
MIN_PLASTIC_GAIN = 0.97
MAX_PLASTIC_GAIN = 1.03
POLICY_VERSION = 5
NEURAL_POLICY_VERSION = 6


def assisted_steering(
    neural_residual: float,
    *,
    obstacle_danger: float,
    obstacle_asymmetry: float,
    road_target: float,
) -> float:
    """Legacy engineering baseline, not a neural-driver action map.

    Positive steering is rightward in the simulator.  An obstacle on the left
    makes ``obstacle_asymmetry`` positive, so the visual term deliberately
    chooses the free (right) side.  The road term is odd and pulls the vehicle
    back towards the centre after it has passed an obstacle.
    """
    visual_avoidance = 2.0 * obstacle_danger * obstacle_asymmetry
    road_recovery = 2.0 * road_target
    return float(np.tanh(visual_avoidance + road_recovery + 0.20 * neural_residual))


def map_neural_motor_output(
    steering: float,
    throttle: float,
    reverse: float,
    *,
    steering_gain: float = 4.0,
    throttle_scale: float = 0.5,
) -> tuple[float, float, float]:
    """Map DNp20/DNpe017/MDN motor primitives to vehicle actuators only.

    This adapter enforces the actuator domain but never reads obstacle rays,
    lane position, route geometry, traffic light state or reward.
    """
    return (
        float(np.tanh(steering_gain * steering)),
        float(np.clip(throttle_scale * throttle, 0.0, 1.0)),
        float(np.clip(reverse, 0.0, 1.0)),
    )


class DopaminePolicy:
    """Plastic gain on real synapses entering documented motor cells."""

    def __init__(
        self,
        adjacency,
        body_ids: np.ndarray,
        *,
        seed: int = 0,
        learning_rate: float = 0.003,
        exploration_sigma: float = 0.22,
    ):
        self.rng = np.random.default_rng(seed)
        self.motor_nodes = np.searchsorted(body_ids, MOTOR_BODY_IDS).astype(np.int32)
        if np.any(self.motor_nodes >= len(body_ids)) or not np.array_equal(
            body_ids[self.motor_nodes], MOTOR_BODY_IDS
        ):
            raise ValueError("configured motor neurons are absent from the MaleCNS graph")
        self.reverse_nodes = np.searchsorted(body_ids, REVERSE_BODY_IDS).astype(np.int32)
        if np.any(self.reverse_nodes >= len(body_ids)) or not np.array_equal(
            body_ids[self.reverse_nodes], REVERSE_BODY_IDS
        ):
            raise ValueError("configured MDN reverse neurons are absent from MaleCNS")
        self.target_nodes = np.concatenate([self.motor_nodes, self.reverse_nodes])
        self.sources: list[np.ndarray] = []
        self.base: list[np.ndarray] = []
        self.gains: list[np.ndarray] = []
        self.eligibility: list[np.ndarray] = []
        self.running_mean: list[np.ndarray | None] = []
        self.running_variance: list[np.ndarray] = []
        for target in self.target_nodes:
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
        self.steering_bias: float | None = None
        self.speed_bias: float | None = None
        self.reverse_mean: float | None = None
        # A finite prior prevents the first small MDN fluctuation from becoming
        # an artificial many-sigma escape response.
        self.reverse_variance = 0.25
        self.dopamine = 0.0
        self.lateral_dopamine = [0.0, 0.0]
        self.updates = 0
        self.learning_rate = learning_rate
        self.exploration_sigma = exploration_sigma
        self.exploration_sign = 1.0
        self.global_eligibility_scale = 0.02
        self.gain_decay = 0.0002
        self.eligibility_decay = 0.96
        self.min_gain = MIN_PLASTIC_GAIN
        self.max_gain = MAX_PLASTIC_GAIN
        self.checkpoint_kind = "untrained"

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

    def configure_neural_curriculum(self) -> None:
        """Use longer reward credit and wider, still bounded real-synapse gains."""
        self.learning_rate = 0.01
        self.global_eligibility_scale = 0.35
        self.eligibility_decay = 0.995
        self.gain_decay = 0.00005
        self.exploration_sigma = 0.16
        self.min_gain, self.max_gain = 0.65, 1.35

    def action(
        self,
        activity: np.ndarray,
        source_sign: np.ndarray,
        *,
        explore: bool,
        adapt: bool,
        mirrored_activity: np.ndarray,
    ) -> tuple[float, float, float, list[np.ndarray]]:
        scores = []
        centered = []
        for index, (sources, base, gains) in enumerate(
            zip(self.sources, self.base, self.gains, strict=True)
        ):
            # Odd visual response controls yaw; even response controls speed.
            # Both states traverse the same full graph with shared synaptic gains.
            values = (
                0.5
                * (activity[sources] + (-1 if index < 2 else 1) * mirrored_activity[sources])
                * source_sign[sources]
            )
            if self.running_mean[index] is None:
                self.running_mean[index] = np.zeros_like(values) if index < 2 else values.copy()
            delta = values - self.running_mean[index]
            if adapt:
                if index >= 2:
                    self.running_mean[index] += 0.03 * delta
                self.running_variance[index] *= 0.995
                self.running_variance[index] += 0.005 * delta * delta
            normalized = np.clip(delta / np.sqrt(self.running_variance[index] + 1e-6), -5.0, 5.0)
            centered.append(normalized)
            scores.append(float(np.dot(base * gains, normalized)))
        steering_drive = scores[1] - scores[0]
        self.steering_bias = 0.0
        steering_activation = float(np.tanh(steering_drive * 3.0))
        steering_threshold = 0.08
        steering_mean = float(
            np.sign(steering_activation)
            * max(0.0, abs(steering_activation) - steering_threshold)
            / (1.0 - steering_threshold)
        )
        speed_drive = scores[2] + scores[3]
        if self.speed_bias is None:
            self.speed_bias = speed_drive
        centered_speed = speed_drive - self.speed_bias
        if adapt:
            self.speed_bias += 0.01 * centered_speed
        # Forward is the default motor primitive. DNpe017 only adjusts its speed.
        throttle_mean = float(np.clip(0.62 + 0.18 * np.tanh(centered_speed * 2.0), 0.25, 0.85))
        reverse_drive = float(-np.mean(scores[4:]))
        if self.reverse_mean is None:
            self.reverse_mean = reverse_drive
        reverse_delta = reverse_drive - self.reverse_mean
        if adapt:
            self.reverse_mean += 0.01 * reverse_delta
            self.reverse_variance = 0.995 * self.reverse_variance + 0.005 * reverse_delta**2
        reverse_z = reverse_delta / np.sqrt(self.reverse_variance + 1e-10)
        reverse_logit = float(np.clip(2.0 * reverse_z - 6.0, -40.0, 40.0))
        reverse_logit_noise = self.rng.normal(0, 0.35) if explore else 0.0
        reverse_activation = float(1.0 / (1.0 + np.exp(-(reverse_logit + reverse_logit_noise))))
        reverse_mean = float(max(0.0, reverse_activation - 0.08) / 0.92)
        steering_noise = (
            self.exploration_sign * self.rng.normal(0, self.exploration_sigma) if explore else 0.0
        )
        steering = float(np.clip(steering_mean + steering_noise, -1, 1))
        throttle_noise = self.rng.normal(0, 0.08) if explore else 0.0
        throttle = float(np.clip(throttle_mean + throttle_noise, 0, 1))
        reverse = float(np.clip(reverse_mean, 0, 1))
        steering_gradient = (
            steering_noise / (self.exploration_sigma**2) * (1.0 - steering_mean**2) * 3.0
            if explore
            else 0.0
        )
        speed_gradient = (
            throttle_noise / 0.08**2 * 0.36 * (1.0 - np.tanh(centered_speed * 2.0) ** 2)
            if explore
            else 0.0
        )
        reverse_gradient = (
            reverse_logit_noise / 0.35**2 * 2.0 * reverse_activation * (1.0 - reverse_activation)
            if explore
            else 0.0
        )
        features = [
            -self.base[0] * centered[0] * steering_gradient,
            self.base[1] * centered[1] * steering_gradient,
            self.base[2] * centered[2] * speed_gradient,
            self.base[3] * centered[3] * speed_gradient,
            *[
                -self.base[index] * centered[index] * reverse_gradient / 4.0
                for index in range(4, 8)
            ],
        ]
        self._last_centered = centered
        self._last_steering_drive = steering_drive
        self._last_steering_mean = steering_mean
        self._last_throttle_mean = throttle_mean
        self._last_reverse_mean = reverse_mean
        return steering, throttle, reverse, features

    def learn(
        self,
        modulation: float,
        features: list[np.ndarray],
        *,
        avoidance_target: float,
        speed_target: float,
        reverse_target: float,
        danger: float,
        enabled: bool,
        teacher_enabled: bool = True,
    ) -> float:
        self.reward_baseline = 0.995 * self.reward_baseline + 0.005 * modulation
        self.dopamine = float(np.clip(modulation - self.reward_baseline, -2.0, 2.0))
        for trace, feature in zip(self.eligibility, features, strict=True):
            trace *= self.eligibility_decay
            trace += feature
        if enabled:
            # Compartment-like opponent teaching: both updates remain confined to
            # existing presynaptic partners of the left/right DNp20 cells. The
            # ray-derived target is a training-only aversive signal and never
            # overrides the neural action at inference.
            if teacher_enabled:
                error = avoidance_target - self._last_steering_mean
                gradient = danger * error * (1.0 - self._last_steering_mean**2) * 3.0
                self.lateral_dopamine = [float(gradient), float(-gradient)]
                self.gains[0] -= (
                    self.learning_rate * gradient * self.base[0] * self._last_centered[0]
                )
                self.gains[1] += (
                    self.learning_rate * gradient * self.base[1] * self._last_centered[1]
                )
                speed_error = speed_target - self._last_throttle_mean
                speed_gradient = danger * speed_error * 0.36
                for index in (2, 3):
                    self.gains[index] += (
                        self.learning_rate
                        * speed_gradient
                        * self.base[index]
                        * self._last_centered[index]
                    )
                reverse_error = reverse_target - self._last_reverse_mean
                reverse_gradient = danger * reverse_error
                for index in range(4, 8):
                    self.gains[index] -= (
                        self.learning_rate
                        * reverse_gradient
                        * self.base[index]
                        * self._last_centered[index]
                        / 4.0
                    )
            else:
                self.lateral_dopamine = [0.0, 0.0]
            for gains, trace in zip(self.gains, self.eligibility, strict=True):
                gains += self.global_eligibility_scale * self.learning_rate * self.dopamine * trace
                gains += self.gain_decay * (1.0 - gains)
                np.clip(gains, self.min_gain, self.max_gain, out=gains)
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
            "global_eligibility_scale": self.global_eligibility_scale,
            "gain_decay": self.gain_decay,
            "eligibility_decay": self.eligibility_decay,
            "gain_bounds": [self.min_gain, self.max_gain],
            "centering": "odd_steering_even_speed_shared_graph",
        }

    def save(
        self,
        path: Path,
        body_ids: np.ndarray,
        *,
        checkpoint_kind: str = "learned",
        format_version: int = POLICY_VERSION,
    ) -> None:
        scalars = [
            self.reward_baseline,
            self.steering_bias,
            self.speed_bias,
            self.reverse_mean,
            self.reverse_variance,
        ]
        if any(value is None or not np.isfinite(value) for value in scalars):
            raise ValueError("cannot save policy without finite calibration state")
        payload = {
            "format_version": np.asarray([format_version], dtype=np.int32),
            "motor_body_ids": MOTOR_BODY_IDS,
            "reverse_body_ids": REVERSE_BODY_IDS,
            "reward_baseline": np.asarray([self.reward_baseline], dtype=np.float64),
            "steering_bias": np.asarray([self.steering_bias], dtype=np.float64),
            "speed_bias": np.asarray([self.speed_bias], dtype=np.float64),
            "reverse_mean": np.asarray([self.reverse_mean], dtype=np.float64),
            "reverse_variance": np.asarray([self.reverse_variance], dtype=np.float64),
            "checkpoint_kind": np.asarray([checkpoint_kind]),
            "learning_config": np.asarray(
                [
                    self.learning_rate,
                    self.global_eligibility_scale,
                    self.eligibility_decay,
                    self.gain_decay,
                    self.min_gain,
                    self.max_gain,
                    self.exploration_sigma,
                ],
                dtype=np.float64,
            ),
        }
        for index, (sources, gains) in enumerate(zip(self.sources, self.gains, strict=True)):
            payload[f"source_body_ids_{index}"] = body_ids[sources]
            payload[f"gains_{index}"] = gains
            if self.running_mean[index] is None:
                raise ValueError("cannot save policy before adaptation statistics exist")
            payload[f"running_mean_{index}"] = self.running_mean[index]
            payload[f"running_variance_{index}"] = self.running_variance[index]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    def load(self, path: Path, body_ids: np.ndarray) -> None:
        payload = np.load(path, allow_pickle=False)
        format_version = int(payload["format_version"][0])
        if format_version not in {POLICY_VERSION, NEURAL_POLICY_VERSION}:
            raise ValueError("unsupported driving policy checkpoint")
        if format_version == NEURAL_POLICY_VERSION:
            if "learning_config" not in payload.files or payload["learning_config"].shape != (7,):
                raise ValueError("invalid neural curriculum configuration")
            config = payload["learning_config"].astype(float)
            self.learning_rate, self.global_eligibility_scale, self.eligibility_decay = config[:3]
            self.gain_decay, self.min_gain, self.max_gain, self.exploration_sigma = config[3:]
        if not np.array_equal(payload["motor_body_ids"], MOTOR_BODY_IDS):
            raise ValueError("driving policy motor-neuron contract mismatch")
        if not np.array_equal(payload["reverse_body_ids"], REVERSE_BODY_IDS):
            raise ValueError("driving policy reverse-neuron contract mismatch")
        loaded = []
        loaded_means = []
        loaded_variances = []
        for index, sources in enumerate(self.sources):
            if not np.array_equal(payload[f"source_body_ids_{index}"], body_ids[sources]):
                raise ValueError("driving policy synapse contract mismatch")
            gains = payload[f"gains_{index}"].astype(np.float32)
            if gains.shape != self.gains[index].shape or not np.all(np.isfinite(gains)):
                raise ValueError("invalid driving policy gains")
            if np.any((gains < self.min_gain) | (gains > self.max_gain)):
                raise ValueError("driving policy gains exceed configured bounds")
            loaded.append(gains)
            mean = payload[f"running_mean_{index}"].astype(np.float32)
            variance = payload[f"running_variance_{index}"].astype(np.float32)
            if mean.shape != gains.shape or variance.shape != gains.shape:
                raise ValueError("driving policy adaptation-state shape mismatch")
            if not np.all(np.isfinite(mean)) or np.any(~np.isfinite(variance)):
                raise ValueError("invalid driving policy adaptation state")
            if np.any(variance < 0):
                raise ValueError("negative driving policy variance")
            if index < 2 and np.any(mean != 0):
                raise ValueError("odd steering normalization must have zero mean")
            loaded_means.append(mean)
            loaded_variances.append(variance)
        scalar_names = (
            "reward_baseline",
            "steering_bias",
            "speed_bias",
            "reverse_mean",
            "reverse_variance",
        )
        if any(
            payload[name].shape != (1,) or not np.isfinite(payload[name][0])
            for name in scalar_names
        ):
            raise ValueError("invalid driving policy scalar state")
        if payload["steering_bias"][0] != 0 or payload["reverse_variance"][0] < 0:
            raise ValueError("invalid symmetric driving policy calibration")
        self.gains = loaded
        self.running_mean = loaded_means
        self.running_variance = loaded_variances
        self.reward_baseline = float(payload["reward_baseline"][0])
        self.steering_bias = float(payload["steering_bias"][0])
        self.speed_bias = float(payload["speed_bias"][0])
        self.reverse_mean = float(payload["reverse_mean"][0])
        self.reverse_variance = float(payload["reverse_variance"][0])
        self.checkpoint_kind = (
            str(payload["checkpoint_kind"][0])
            if "checkpoint_kind" in payload.files
            else "legacy_v2"
        )


class DrivingEngine:
    def __init__(
        self,
        root: Path,
        *,
        seed: int = 0,
        top_k: int = 220,
        brain_substeps: int = 4,
        load_checkpoint: bool = True,
        scenario: str = "highway",
        control_mode: str = "assisted",
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
        self.assisted_policy_checkpoint = root / "artifacts/checkpoints/driving-policy.npz"
        self.neural_policy_checkpoint = root / "artifacts/checkpoints/driving-policy.neural-v6.npz"
        self.policy_checkpoint = self.assisted_policy_checkpoint
        self.checkpoint_loaded = False
        self.checkpoint_rejection: str | None = None
        self.dopamine_nodes = np.searchsorted(self.graph.body_ids, DOPAMINE_BODY_IDS)
        if not np.array_equal(self.graph.body_ids[self.dopamine_nodes], DOPAMINE_BODY_IDS):
            raise ValueError("configured PPL101 dopamine cells are absent from MaleCNS")
        self.scenario = ""
        self.control_mode = ""
        self.env: DrivingEnvironment | CityDrivingEnvironment
        self.set_scenario(scenario)
        self.set_control_mode(control_mode)
        if load_checkpoint and self.policy_checkpoint.exists():
            self._load_published_policy()
        self.activity = np.zeros(self.graph.node_count, dtype=np.float32)
        self.mirrored_activity = np.zeros_like(self.activity)
        self.visual_drive = np.zeros_like(self.activity)
        self.top_k = top_k
        if brain_substeps < 1:
            raise ValueError("brain_substeps must be positive")
        self.brain_substeps = brain_substeps
        overview = json.loads((processed / "overview.json").read_text())
        self.visible_mask = np.isin(self.graph.body_ids, overview["body_ids"])
        self.visible = np.flatnonzero(self.visible_mask)
        self.last_action = {"steering": 0.0, "throttle": 0.0, "reverse": 0.0, "drive": 0.0}
        self.last_raw_action = {"steering": 0.0, "throttle": 0.0, "reverse": 0.0, "drive": 0.0}
        self.last_constraint = {"active": False, "blend": 0.0, "correction": 0.0}
        self.last_reward = 0.0
        self.last_safety_signal = 0.0
        self.control_statistics = {
            "steering_sum_abs": 0.0,
            "steering_change_sum_abs": 0.0,
            "far_steering_sum_abs": 0.0,
            "far_steps": 0,
            "sign_changes": 0,
            "previous_sign": 0,
            "max_abs_lateral": 0.0,
            "constraint_steps": 0,
            "constraint_sum_abs": 0.0,
            "reverse_gate_steps": 0,
        }
        self.close_hazard_streak = 0
        self.learning = False
        self.reset(seed)

    def set_scenario(self, scenario: str) -> None:
        if scenario not in {"highway", "city"}:
            raise ValueError(f"unknown driving scenario: {scenario}")
        if scenario == self.scenario:
            return
        self.scenario = scenario
        self.env = CityDrivingEnvironment() if scenario == "city" else DrivingEnvironment()

    def set_control_mode(self, control_mode: str) -> None:
        if control_mode not in {"assisted", "neural"}:
            raise ValueError(f"unknown driving control mode: {control_mode}")
        self.control_mode = control_mode
        self.policy_checkpoint = (
            self.neural_policy_checkpoint
            if control_mode == "neural"
            else self.assisted_policy_checkpoint
        )

    def _load_published_policy(self) -> None:
        with np.load(self.policy_checkpoint, allow_pickle=False) as payload:
            expected = (
                NEURAL_POLICY_VERSION
                if self.policy_checkpoint == self.neural_policy_checkpoint
                else POLICY_VERSION
            )
            if payload["format_version"].tolist() != [expected]:
                self.checkpoint_loaded = False
                self.checkpoint_rejection = "obsolete_format_requires_recalibration"
                return
        self.policy.load(self.policy_checkpoint, self.graph.body_ids)
        self.checkpoint_loaded = True
        self.checkpoint_rejection = None

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

    def reset(
        self,
        seed: int = 0,
        *,
        keep_learning: bool = True,
        scenario: str | None = None,
        control_mode: str | None = None,
        curriculum_stage: str = "full",
    ) -> dict:
        if scenario is not None:
            self.set_scenario(scenario)
        if control_mode is not None:
            self.set_control_mode(control_mode)
        self.env.reset(seed)
        if curriculum_stage != "full":
            if not isinstance(self.env, DrivingEnvironment):
                raise ValueError("neural curriculum requires the random-obstacle environment")
            self.env.configure_curriculum(curriculum_stage)
        self.activity.fill(0)
        self.mirrored_activity.fill(0)
        self.visual_drive.fill(0)
        if not keep_learning:
            self.policy = DopaminePolicy(self.graph.adjacency, self.graph.body_ids, seed=seed)
            if self.policy_checkpoint.exists():
                self._load_published_policy()
            else:
                self.checkpoint_loaded = False
                self.checkpoint_rejection = None
        self.policy.reset_traces(
            episode_seed=self.env.pair_seed if self.control_mode == "neural" else seed
        )
        self.policy.exploration_sign = (
            float(self.env.mirror) if self.control_mode == "neural" else 1.0
        )
        self.last_action = {"steering": 0.0, "throttle": 0.0, "reverse": 0.0, "drive": 0.0}
        self.last_raw_action = {"steering": 0.0, "throttle": 0.0, "reverse": 0.0, "drive": 0.0}
        self.last_constraint = {"active": False, "blend": 0.0, "correction": 0.0}
        self.last_reward = 0.0
        self.last_safety_signal = 0.0
        self.learning = False
        self.control_statistics = {
            "steering_sum_abs": 0.0,
            "steering_change_sum_abs": 0.0,
            "far_steering_sum_abs": 0.0,
            "far_steps": 0,
            "sign_changes": 0,
            "previous_sign": 0,
            "max_abs_lateral": 0.0,
            "constraint_steps": 0,
            "constraint_sum_abs": 0.0,
            "reverse_gate_steps": 0,
        }
        self.close_hazard_streak = 0
        for _ in range(self.brain_substeps):
            self._advance_brain(self.env.observe(), dopamine=0.0)
        return self.state(include_activity=True)

    def _advance_brain(self, image: np.ndarray, *, dopamine: float) -> None:
        self.activity = self._advance_state(self.activity, image, dopamine=dopamine)
        self.mirrored_activity = self._advance_state(
            self.mirrored_activity, image[:, ::-1], dopamine=dopamine
        )

    def _advance_state(
        self, activity: np.ndarray, image: np.ndarray, *, dopamine: float
    ) -> np.ndarray:
        self.visual_drive.fill(0)
        # Photoreceptor activity follows local contrast plus luminance.
        receptor_values = self.retina.encode(image)
        receptor_values = 0.25 * receptor_values + 0.75 * np.abs(
            receptor_values - float(receptor_values.mean())
        )
        self.visual_drive[self.retina.node_indices] = receptor_values
        recurrent = self.graph.adjacency @ (activity * self.source_sign)
        activity = (0.72 * activity + 0.28 * np.tanh(1.8 * recurrent + self.visual_drive)).astype(
            np.float32
        )
        # The current image is clamped at the sensory boundary for this time step.
        activity[self.retina.node_indices] = receptor_values
        # Reward prediction error is represented on the two annotated PPL101
        # dopamine cells as a signed model signal. It gates plasticity below but
        # is excluded from the fast transmitter-weighted recurrent current.
        activity[self.dopamine_nodes] = np.tanh(dopamine)
        return activity

    def step(
        self,
        *,
        learning: bool = False,
        explore: bool = False,
        safety_constraints: bool = True,
        include_activity: bool = True,
    ) -> dict:
        if self.env.done:
            return self.state(include_activity=include_activity)
        started = time.perf_counter()
        quarter = len(self.env.last_rays) // 4
        obstacle_rays = self.env.last_obstacle_rays.copy()
        obstacle_left = float(np.mean(obstacle_rays[quarter : 2 * quarter]))
        obstacle_right = float(np.mean(obstacle_rays[2 * quarter : 3 * quarter]))
        obstacle_asymmetry = (obstacle_right - obstacle_left) / self.env.max_sensor_distance
        obstacle_danger = np.clip(
            1.0
            - float(np.min(obstacle_rays[quarter : 3 * quarter])) / self.env.max_sensor_distance,
            0.0,
            1.0,
        )
        forward_clearance = float(np.min(obstacle_rays[quarter : 3 * quarter]))
        guidance = self.env.traffic_guidance() if self.scenario == "city" else None
        target_lateral = (
            float(guidance["target_lateral_offset"])
            if guidance and self.control_mode == "assisted"
            else 0.0
        )
        road_pressure = np.clip(
            abs(self.env.x - target_lateral) / (self.env.road_half_width - self.env.vehicle_radius),
            0.0,
            1.0,
        )
        heading_pressure = np.clip(abs(self.env.heading) / 0.6, 0.0, 1.0)
        lane_danger = float(
            np.clip(0.18 + 0.62 * road_pressure + 0.20 * heading_pressure, 0.0, 1.0)
        )
        road_target = np.tanh(
            -1.5 * (self.env.x - target_lateral) / self.env.road_half_width - self.env.heading
        )
        raw_steering, raw_throttle, raw_reverse, features = self.policy.action(
            self.activity,
            self.source_sign,
            explore=explore,
            adapt=learning or explore,
            mirrored_activity=self.mirrored_activity,
        )
        # Assisted mode is the preserved v5 engineering baseline. Neural mode
        # maps the three documented motor outputs directly to vehicle actuators.
        visual_avoidance = 2.0 * obstacle_danger * obstacle_asymmetry
        road_recovery = 2.0 * road_target
        if self.control_mode == "assisted":
            behavioral_steering = assisted_steering(
                raw_steering,
                obstacle_danger=obstacle_danger,
                obstacle_asymmetry=obstacle_asymmetry,
                road_target=road_target,
            )
            if guidance:
                behavioral_steering = float(
                    np.tanh(
                        np.arctanh(np.clip(behavioral_steering, -0.999, 0.999))
                        + 1.35 * float(guidance["desired_steering"])
                    )
                )
            steering, constraint = self.apply_lane_constraint(
                behavioral_steering, enabled=safety_constraints
            )
            throttle = raw_throttle
            speed_target = float(0.25 + 0.37 * np.clip((forward_clearance - 2.0) / 6.0, 0.0, 1.0))
            if guidance:
                speed_target = min(speed_target, float(guidance["speed_cap"]))
            throttle = float(min(raw_throttle, speed_target))
        else:
            steering, throttle, reverse = map_neural_motor_output(
                raw_steering, raw_throttle, raw_reverse
            )
            constraint = {"active": False, "blend": 0.0, "correction": 0.0}
            speed_target = raw_throttle
        constraint.update(
            {
                "neural_steering": raw_steering,
                "visual_avoidance": float(visual_avoidance)
                if self.control_mode == "assisted"
                else 0.0,
                "road_recovery": float(road_recovery) if self.control_mode == "assisted" else 0.0,
                "route_steering": (
                    float(guidance["desired_steering"])
                    if guidance and self.control_mode == "assisted"
                    else 0.0
                ),
            }
        )
        if self.control_mode == "assisted" and forward_clearance < 1.5:
            self.close_hazard_streak += 1
        elif self.control_mode == "assisted":
            self.close_hazard_streak = 0
        reverse_gate = self.control_mode == "assisted" and self.close_hazard_streak >= 4
        if self.control_mode == "assisted":
            reverse = float(raw_reverse if reverse_gate else 0.0)
        if reverse_gate:
            stats = self.control_statistics
            stats["reverse_gate_steps"] += 1
        image, reward, done = self.env.step(steering, throttle, reverse)
        stats = self.control_statistics
        current_sign = int(np.sign(self.env.steering))
        if current_sign and stats["previous_sign"] and current_sign != stats["previous_sign"]:
            stats["sign_changes"] += 1
        if current_sign:
            stats["previous_sign"] = current_sign
        stats["steering_sum_abs"] += abs(self.env.steering)
        stats["steering_change_sum_abs"] += abs(self.env.steering - self.env.previous_steering)
        if float(np.min(obstacle_rays)) > 10.0:
            stats["far_steps"] += 1
            stats["far_steering_sum_abs"] += abs(self.env.steering)
        stats["max_abs_lateral"] = max(stats["max_abs_lateral"], abs(self.env.x))
        if constraint["active"]:
            stats["constraint_steps"] += 1
            stats["constraint_sum_abs"] += abs(steering - raw_steering)
        avoidance_target = float(
            np.clip(
                np.tanh(4.0 * obstacle_danger * obstacle_asymmetry)
                + road_pressure**2 * road_target,
                -1.0,
                1.0,
            )
        )
        steering_change = self.env.steering - self.env.previous_steering
        behavior_cost = (
            0.025 * road_pressure**2
            + 0.015 * self.env.heading**2
            + 0.006 * self.env.steering**2
            + 0.08 * steering_change**2
        )
        safety_signal = float(obstacle_danger * steering * obstacle_asymmetry - behavior_cost)
        reverse_target = float(
            0.8 * np.clip((1.5 - forward_clearance) / 0.75, 0.0, 1.0) if reverse_gate else 0.0
        )
        dopamine = self.policy.learn(
            reward + safety_signal if self.control_mode == "assisted" else reward,
            features,
            avoidance_target=avoidance_target,
            speed_target=speed_target,
            reverse_target=reverse_target,
            danger=float(max(obstacle_danger, lane_danger)),
            enabled=learning,
            teacher_enabled=self.control_mode == "assisted",
        )
        for _ in range(self.brain_substeps):
            self._advance_brain(image, dopamine=dopamine)
        drive = float((1.0 - reverse) * throttle - reverse)
        self.last_raw_action = {
            "steering": raw_steering,
            "throttle": raw_throttle,
            "reverse": raw_reverse,
            "drive": float((1.0 - raw_reverse) * raw_throttle - raw_reverse),
        }
        self.last_action = {
            "steering": steering,
            "throttle": throttle,
            "reverse": reverse,
            "drive": drive,
        }
        self.last_constraint = constraint
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
            for node in selected
            if self.activity[node] != 0
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
                    pathways.append(
                        {
                            "pre": int(self.graph.body_ids[source]),
                            "post": int(self.graph.body_ids[target]),
                            "value": float(values[index]),
                        }
                    )
        pathways.sort(key=lambda edge: abs(edge["value"]), reverse=True)
        return {
            "type": "activity",
            "step": self.env.steps,
            "total_steps": 500,
            "elapsed_ms": 0.0,
            "units": "simulated_activation_not_millivolts",
            "neurons": neurons,
            "pathways": pathways[:320],
            "statistics": {
                "all_nodes": self.graph.node_count,
                "active_nodes": int(np.count_nonzero(self.activity)),
                "positioned_nodes": len(self.visible),
                "unpositioned_active_nodes": int(
                    np.count_nonzero(self.activity[~self.visible_mask])
                ),
                "displayed_nodes": len(neurons),
                "displayed_edges": min(len(pathways), 320),
                "max_abs_state": float(np.abs(self.activity).max()),
                "display_max_abs_state": max((abs(n["value"]) for n in neurons), default=0.0),
            },
        }

    def state(self, *, include_activity: bool = False) -> dict:
        state = {
            "environment": self.env.snapshot(),
            "scenario": self.scenario,
            "control_mode": self.control_mode,
            "action": self.last_action,
            "raw_action": self.last_raw_action,
            "lane_constraint": self.last_constraint,
            "reward": self.last_reward,
            "safety_signal": self.last_safety_signal,
            "control_statistics": self.control_summary(),
            "learning": self.learning,
            "policy_checkpoint": {
                "loaded": self.checkpoint_loaded,
                "path": str(self.policy_checkpoint.relative_to(self.root)),
                "kind": self.policy.checkpoint_kind,
                "rejection": self.checkpoint_rejection,
            },
            "dopamine": self.policy.summary(),
            "retina": {
                "mapped_receptors": self.retina.size,
                "source_type": "R1-R6",
                "mapping": "contact_weighted_postsynaptic_optic_hex_proxy",
                "width": self.env.image_width,
                "height": self.env.image_height,
                "stimulus": self.env.observe().tolist(),
            },
            "motor": {
                "body_ids": np.concatenate([MOTOR_BODY_IDS, REVERSE_BODY_IDS]).tolist(),
                "names": list(MOTOR_NAMES + REVERSE_NAMES),
                "mapping": "mirror_odd_DNp20_even_DNpe017_MDN",
                "symmetry": "shared_full_graph_original_and_mirrored_visual_states",
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

    def control_summary(self) -> dict:
        steps = max(self.env.steps, 1)
        far_steps = self.control_statistics["far_steps"]
        return {
            "mean_abs_steering": self.control_statistics["steering_sum_abs"] / steps,
            "mean_abs_steering_change": (
                self.control_statistics["steering_change_sum_abs"] / steps
            ),
            "far_mean_abs_steering": (
                self.control_statistics["far_steering_sum_abs"] / far_steps if far_steps else 0.0
            ),
            "far_steps": far_steps,
            "steering_sign_changes": self.control_statistics["sign_changes"],
            "max_abs_lateral": self.control_statistics["max_abs_lateral"],
            "constraint_rate": self.control_statistics["constraint_steps"] / steps,
            "mean_abs_constraint": (self.control_statistics["constraint_sum_abs"] / steps),
            "reverse_gate_fraction": (self.control_statistics["reverse_gate_steps"] / steps),
        }

    def apply_lane_constraint(self, steering: float, *, enabled: bool) -> tuple[float, dict]:
        if not enabled:
            return steering, {"active": False, "blend": 0.0, "correction": 0.0}
        usable_half_width = self.env.road_half_width - self.env.vehicle_radius
        lateral = self.env.x / usable_half_width
        projected_lateral = (
            self.env.x + np.sin(self.env.heading) * self.env.speed * 1.2
        ) / usable_half_width
        outward_heading = np.sign(lateral) * self.env.heading * np.sign(self.env.speed)
        lateral_pressure = np.clip((abs(lateral) - 0.45) / 0.40, 0.0, 1.0)
        projected_pressure = np.clip((abs(projected_lateral) - 0.45) / 0.40, 0.0, 1.0)
        heading_pressure = (
            np.clip((outward_heading - 0.12) / 0.45, 0.0, 1.0) if abs(lateral) > 0.20 else 0.0
        )
        blend = float(max(lateral_pressure, projected_pressure, heading_pressure))
        if blend <= 0:
            return steering, {"active": False, "blend": 0.0, "correction": 0.0}
        direction = -1.0 if self.env.speed < 0 else 1.0
        correction = float(
            np.clip(-0.85 * lateral * direction - 0.75 * self.env.heading, -1.0, 1.0)
        )
        constrained = float(np.clip((1.0 - blend) * steering + blend * correction, -1.0, 1.0))
        return constrained, {"active": True, "blend": blend, "correction": correction}
