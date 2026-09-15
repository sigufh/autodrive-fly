from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import uniform_filter

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.sensory import AnatomySensoryProjection
from fly_emotion.driving.v7 import V7VisualProbe
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-closed-loop.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_closed_loop.py")
TUNING_ARTIFACT = Path("artifacts/v7-closed-loop-tuning.json")
CALIBRATION_ARTIFACT = Path("artifacts/v7-closed-loop-calibration.json")


class ReceptiveFieldVisualProbe(V7VisualProbe):
    """Experimental same-eye receptive-field sampling; never used by default services."""

    def __init__(self, *args, receptive_field_radius: int, **kwargs):
        if receptive_field_radius < 0:
            raise ValueError("receptive-field radius must be nonnegative")
        self.receptive_field_radius = receptive_field_radius
        super().__init__(*args, **kwargs)

    def _sample_retina(self, image: np.ndarray) -> np.ndarray:
        if self.receptive_field_radius == 0:
            return super()._sample_retina(image)
        height, width = image.shape
        if width % 2:
            raise ValueError("same-eye receptive fields require an even image width")
        half = width // 2
        filtered = np.empty_like(image, dtype=np.float32)
        size = 2 * self.receptive_field_radius + 1
        filtered[:, :half] = uniform_filter(image[:, :half], size=size, mode="nearest")
        filtered[:, half:] = uniform_filter(image[:, half:], size=size, mode="nearest")
        return super()._sample_retina(filtered)


@dataclass(frozen=True)
class EventParameters:
    signal_threshold: float
    turn_steps: int
    counter_turn_steps: int
    steering_amplitude: float
    fixed_throttle: float


class NeuralEventAdapter:
    """Fixed environment-blind conversion from a neural spatial moment to actuators."""

    def __init__(self, parameters: EventParameters):
        self.parameters = parameters
        self.phase = "waiting"
        self.remaining = 0
        self.side = 0.0

    def step(self, neural_signal: float) -> tuple[float, float, float]:
        if self.phase == "waiting" and abs(neural_signal) >= self.parameters.signal_threshold:
            self.phase = "turn"
            self.remaining = self.parameters.turn_steps
            self.side = -float(np.sign(neural_signal))
        if self.phase == "turn":
            steering = self.parameters.steering_amplitude * self.side
            self.remaining -= 1
            if self.remaining == 0:
                self.phase = "counter_turn"
                self.remaining = self.parameters.counter_turn_steps
        elif self.phase == "counter_turn":
            steering = -self.parameters.steering_amplitude * self.side
            self.remaining -= 1
            if self.remaining == 0:
                self.phase = "complete"
        else:
            steering = 0.0
        return steering, self.parameters.fixed_throttle, 0.0


class ClosedLoopVisualCore:
    """Stateful original/mirror visual graph yielding an exactly odd spatial moment."""

    def __init__(self, root: Path, config: dict):
        visual = config["visual_core"]
        radius = int(visual["same_eye_receptive_field"]["radius_pixels"])
        self.probe = ReceptiveFieldVisualProbe(
            root,
            receptive_field_radius=radius,
            brain_substeps=int(visual["brain_substeps_per_vehicle_step"]),
            baseline_frames=1,
            retinal_backend=visual["retinal_backend"],
            retinal_geometry=visual["retinal_geometry"],
            dynamics_backend=visual["dynamics_backend"],
        )
        raw = root / "data/raw/malecns-v1.0/body-annotations.feather"
        projection = AnatomySensoryProjection.from_annotations(
            self.probe.graph.body_ids, raw, adjacency=self.probe.graph.adjacency
        )
        self.bins = tuple(
            np.unique(np.concatenate((positive, negative))).astype(np.int32)
            for positive, negative in zip(
                projection.flow_positive_bins, projection.flow_negative_bins, strict=True
            )
        )
        self.spatial_weights = np.asarray(
            visual["steering_signal"]["spatial_weights"], dtype=np.float64
        )
        if len(self.bins) != len(self.spatial_weights) or any(
            len(nodes) == 0 for nodes in self.bins
        ):
            raise ValueError("spatial T4/T5 bins are incomplete")
        self.states: list[np.ndarray] = []
        self.histories: list[list[np.ndarray]] = []
        self.previous: list[np.ndarray] = []

    def reset(self, image: np.ndarray) -> None:
        probe = self.probe
        self.states = [np.zeros(probe.graph.node_count, dtype=np.float32) for _ in range(2)]
        history_length = max(
            1,
            int(probe.source_delays.max()),
            int(probe.correlator["history_substeps"]) if probe.correlator else 0,
        )
        self.histories = [[state.copy() for _ in range(history_length)] for state in self.states]
        self.previous = [
            probe._sample_retina(frame)[probe.retinal_permutation]
            for frame in (image, image[:, ::-1])
        ]

    def step(self, image: np.ndarray) -> dict:
        probe = self.probe
        retinal_hash = hashlib.sha256()
        bin_differences = np.empty(len(self.bins), dtype=np.float64)
        for index, frame in enumerate((image, image[:, ::-1])):
            sampled = probe._sample_retina(frame)[probe.retinal_permutation]
            receptor = probe._retinal_code(sampled, self.previous[index], self.previous[index])
            self.previous[index] = sampled
            retinal_hash.update(receptor.astype(np.float32, copy=False).tobytes())
            drive = np.zeros_like(self.states[index])
            drive[probe.retina.node_indices] = receptor
            for _ in range(probe.brain_substeps):
                self.states[index] = probe._advance(
                    self.states[index], drive, self.histories[index]
                )
                self.states[index][probe.retina.node_indices] = receptor
        for bin_index, nodes in enumerate(self.bins):
            bin_differences[bin_index] = float(
                np.mean(self.states[0][nodes]) - np.mean(self.states[1][nodes])
            )
        return {
            "neural_spatial_moment": float(self.spatial_weights @ bin_differences),
            "bin_odd_responses": bin_differences.tolist(),
            "retinal_pair_sha256": retinal_hash.hexdigest(),
        }


def _episode(
    root: Path, config: dict, core: ClosedLoopVisualCore, seed: int, candidate: dict
) -> dict:
    environment = DrivingEnvironment(
        width=int(config["visual_core"]["width"]),
        height=int(config["visual_core"]["height"]),
    )
    image = environment.reset(seed)
    environment.configure_curriculum(config["tuning"]["curriculum_stage"])
    image = environment.observe()
    core.reset(image)
    parameters = EventParameters(
        signal_threshold=float(candidate["signal_threshold"]),
        turn_steps=int(candidate["turn_steps"]),
        counter_turn_steps=int(candidate["counter_turn_steps"]),
        steering_amplitude=float(config["vehicle_adapter"]["steering_amplitude"]),
        fixed_throttle=float(config["vehicle_adapter"]["fixed_throttle"]),
    )
    adapter = NeuralEventAdapter(parameters)
    trace = []
    while not environment.done:
        neural = core.step(image)
        steering, throttle, reverse = adapter.step(neural["neural_spatial_moment"])
        image, _, _ = environment.step(steering, throttle, reverse)
        trace.append(
            {
                "step": environment.steps,
                "neural_spatial_moment": neural["neural_spatial_moment"],
                "steering_command": steering,
                "vehicle_steering": environment.steering,
                "vehicle_x": environment.x,
                "vehicle_y": environment.y,
                "adapter_phase": adapter.phase,
            }
        )
    values = np.asarray(
        [
            [
                item["neural_spatial_moment"],
                item["steering_command"],
                item["vehicle_x"],
                item["vehicle_y"],
            ]
            for item in trace
        ],
        dtype=np.float64,
    )
    return {
        "seed": seed,
        "pair_seed": environment.pair_seed,
        "mirror": environment.mirror,
        "first_obstacle_x": environment.obstacles[0].x,
        "terminal_reason": environment.terminal_reason,
        "success": environment.terminal_reason == "success",
        "first_obstacle_passed": 0 in environment.passed_obstacle_indices,
        "obstacles_passed": environment.obstacles_passed,
        "distance": environment.y,
        "steps": environment.steps,
        "maximum_absolute_neural_signal": float(np.max(np.abs(values[:, 0]))),
        "maximum_absolute_lateral_position": float(np.max(np.abs(values[:, 2]))),
        "trace": trace,
    }


def _mirror_errors(episodes: list[dict]) -> list[dict]:
    result = []
    for first, second in zip(episodes[::2], episodes[1::2], strict=True):
        count = min(len(first["trace"]), len(second["trace"]))
        a = first["trace"][:count]
        b = second["trace"][:count]
        result.append(
            {
                "seeds": [first["seed"], second["seed"]],
                "paired_steps": count,
                "maximum_signal_odd_error": float(
                    max(
                        abs(x["neural_spatial_moment"] + y["neural_spatial_moment"])
                        for x, y in zip(a, b, strict=True)
                    )
                ),
                "maximum_action_odd_error": float(
                    max(
                        abs(x["steering_command"] + y["steering_command"])
                        for x, y in zip(a, b, strict=True)
                    )
                ),
                "maximum_trajectory_mirror_error": float(
                    max(abs(x["vehicle_x"] + y["vehicle_x"]) for x, y in zip(a, b, strict=True))
                ),
            }
        )
    return result


def _candidate_grid(config: dict) -> list[dict]:
    values = config["tuning"]["candidates"]
    candidates = []
    for threshold in values["signal_threshold"]:
        for turn_steps in values["turn_steps"]:
            for counter_steps in values["counter_turn_steps_by_turn_steps"][str(turn_steps)]:
                candidates.append(
                    {
                        "signal_threshold": float(threshold),
                        "turn_steps": int(turn_steps),
                        "counter_turn_steps": int(counter_steps),
                    }
                )
    return candidates


def _candidate_summary(candidate: dict, episodes: list[dict]) -> dict:
    mirrors = _mirror_errors(episodes)
    terminal_counts = {
        name: sum(item["terminal_reason"] == name for item in episodes)
        for name in ("success", "obstacle", "road_boundary", "timeout")
    }
    return {
        "candidate": candidate,
        "success_count": sum(item["success"] for item in episodes),
        "first_obstacle_pass_count": sum(item["first_obstacle_passed"] for item in episodes),
        "terminal_counts": terminal_counts,
        "mean_distance": float(np.mean([item["distance"] for item in episodes])),
        "mean_maximum_absolute_lateral_position": float(
            np.mean([item["maximum_absolute_lateral_position"] for item in episodes])
        ),
        "mirror_checks": mirrors,
        "episodes": episodes,
    }


def _selection_key(summary: dict) -> tuple:
    candidate = summary["candidate"]
    return (
        -summary["success_count"],
        -summary["first_obstacle_pass_count"],
        summary["terminal_counts"]["road_boundary"],
        summary["mean_maximum_absolute_lateral_position"],
        abs(candidate["signal_threshold"] - 0.000002),
        candidate["turn_steps"] + candidate["counter_turn_steps"],
        candidate["signal_threshold"],
    )


def evaluate_v7_closed_loop_tuning(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_path = Path(config["stage1_nested_evidence"])
    nested = json.loads((root / nested_path).read_text())
    if not nested["tuning_authorized"] or nested["calibration_authorized"]:
        raise ValueError("nested protocol does not authorize tuning-only evaluation")
    core = ClosedLoopVisualCore(root, config)
    seeds = [seed for pair in config["tuning"]["condition_pairs"].values() for seed in pair]
    summaries = []
    for candidate in _candidate_grid(config):
        episodes = [_episode(root, config, core, int(seed), candidate) for seed in seeds]
        summaries.append(_candidate_summary(candidate, episodes))
    winner = min(summaries, key=_selection_key)
    frozen_candidate = {
        **winner["candidate"],
        "steering_amplitude": config["vehicle_adapter"]["steering_amplitude"],
        "fixed_throttle": config["vehicle_adapter"]["fixed_throttle"],
        "visual_core_sha256": _sha256(root / IMPLEMENTATION),
        "config_sha256": _sha256(root / CONFIG),
    }
    candidate_sha256 = hashlib.sha256(
        json.dumps(frozen_candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "protocol": {
            "name": config["name"],
            "phase": "tuning",
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(nested_path): _sha256(root / nested_path),
            },
            "visual_input_only_via_R1_R6": True,
            "environment_state_used_by_controller": False,
            "reward_used_by_controller": False,
            "runtime_modified": False,
        },
        "condition_pairs": config["tuning"]["condition_pairs"],
        "candidate_count": len(summaries),
        "candidate_summaries": summaries,
        "selected_candidate": frozen_candidate,
        "selected_candidate_sha256": candidate_sha256,
        "tuning_passed": winner["success_count"] == len(seeds),
        "calibration_evaluated": False,
        "final_evaluated": False,
        "advance_to_calibration": winner["success_count"] == len(seeds),
        "advance_to_final": False,
    }


def evaluate_v7_closed_loop_calibration(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    tuning = json.loads((root / TUNING_ARTIFACT).read_text())
    dependencies = tuning["protocol"]["dependencies_sha256"]
    for path, expected in dependencies.items():
        if _sha256(root / path) != expected:
            raise ValueError(f"tuning evidence is stale: {path}")
    if not tuning["advance_to_calibration"]:
        raise ValueError("tuning did not authorize calibration")
    frozen = tuning["selected_candidate"]
    candidate_hash = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if candidate_hash != tuning["selected_candidate_sha256"]:
        raise ValueError("frozen candidate digest mismatch")
    core = ClosedLoopVisualCore(root, config)
    seeds = [int(value) for value in config["calibration"]["mirror_pair_seeds"]]
    episodes = [_episode(root, config, core, seed, frozen) for seed in seeds]
    mirrors = _mirror_errors(episodes)
    limits = config["calibration"]["pass_requirements"]
    gates = {
        "both_episodes_success": all(item["success"] for item in episodes),
        "both_first_obstacles_passed": all(item["first_obstacle_passed"] for item in episodes),
        "zero_terminal_collision_or_road_boundary": all(
            item["terminal_reason"] not in {"obstacle", "road_boundary"} for item in episodes
        ),
        "signal_mirror_error": max(item["maximum_signal_odd_error"] for item in mirrors)
        <= float(limits["maximum_signal_mirror_error"]),
        "action_mirror_error": max(item["maximum_action_odd_error"] for item in mirrors)
        <= float(limits["maximum_action_mirror_error"]),
    }
    return {
        "protocol": {
            "name": config["name"],
            "phase": "calibration_once",
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(TUNING_ARTIFACT): _sha256(root / TUNING_ARTIFACT),
            },
            "candidate_sha256": candidate_hash,
            "parameters_changed_after_tuning": False,
            "visual_input_only_via_R1_R6": True,
            "environment_state_used_by_controller": False,
            "reward_used_by_controller": False,
            "runtime_modified": False,
        },
        "condition_id": config["calibration"]["condition_id"],
        "seeds": seeds,
        "episodes": episodes,
        "mirror_checks": mirrors,
        "gates": gates,
        "calibration_passed": all(gates.values()),
        "final_evaluated": False,
        "advance_to_external_final": False,
        "advance_to_navigation_release": False,
    }
