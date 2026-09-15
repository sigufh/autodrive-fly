from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.sensory import AnatomySensoryProjection
from fly_emotion.driving.v7 import V7VisualProbe
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_r1r6_local import (
    local_visual_channels,
    reconstruct_image,
)
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina

CONFIG = Path("configs/driving-v7-neural-channels.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_channels.py")
TUNING_ARTIFACT = Path("artifacts/v7-neural-channels-tuning.json")


class StructuredNeuralFeatures:
    def __init__(
        self,
        root: Path,
        config: dict,
        *,
        topology_control: str = "real_malecns",
        control_seed: int = 20260915,
    ):
        self.root = root
        self.config = config
        self.retina = build_mass_balanced_retina(root)
        self.probe = V7VisualProbe(
            root,
            brain_substeps=1,
            baseline_frames=1,
            retinal_backend="linear_luminance",
            retinal_geometry="legacy_proxy_v2",
            dynamics_backend=config["input"]["visual_graph"],
            control=topology_control,
            control_seed=control_seed,
        )
        nodes = np.searchsorted(self.probe.graph.body_ids, self.retina.body_ids).astype(np.int32)
        if not np.array_equal(self.probe.graph.body_ids[nodes], self.retina.body_ids):
            raise ValueError("balanced R1-R6 body IDs do not align with the graph")
        self.probe.retina = type(self.probe.retina)(
            nodes,
            self.retina.body_ids,
            self.retina.x / 47.0,
            self.retina.y / 23.0,
            self.retina.side,
            4,
        )
        self.probe.retinal_u = self.probe.retina.u
        self.probe.retinal_v = self.probe.retina.v
        self.probe.retinal_permutation = np.arange(len(nodes), dtype=np.int32)
        if topology_control == "shuffled_retina_coordinates":
            self.probe.retinal_permutation = (
                np.random.default_rng(control_seed).permutation(len(nodes)).astype(np.int32)
            )
        self.retinal_gain = self.retina.mass * (len(nodes) / (2 * 24 * 24))
        raw = root / "data/raw/malecns-v1.0/body-annotations.feather"
        projection = AnatomySensoryProjection.from_annotations(
            self.probe.graph.body_ids, raw, adjacency=self.probe.graph.adjacency
        )
        self.groups: list[np.ndarray] = []
        self.group_names: list[str] = []
        for region, (positive, negative) in enumerate(
            zip(projection.flow_positive_bins, projection.flow_negative_bins, strict=True)
        ):
            self.groups.extend((positive, negative))
            self.group_names.extend((f"front_to_back_{region}", f"back_to_front_{region}"))
        for name in config["features"]["looming_populations"]:
            self.groups.append(self.probe.populations[name])
            self.group_names.append(name)
        self.group_mirror = self._group_mirror()
        self.stat_count = len(config["features"]["per_group_statistics"])
        self.stat_mirror = np.concatenate(
            [np.arange(self.stat_count) + self.stat_count * index for index in self.group_mirror]
        )
        self.feature_group_names = [
            name
            for temporal in config["features"]["temporal_terms"]
            for name in self.group_names
            for _ in config["features"]["per_group_statistics"]
        ]
        self.states: list[np.ndarray] = []
        self.histories: list[list[np.ndarray]] = []
        self.previous_even = np.empty(0)
        self.previous_odd = np.empty(0)
        self.ema_even = np.empty(0)
        self.ema_odd = np.empty(0)

    def _group_mirror(self) -> np.ndarray:
        index = {name: position for position, name in enumerate(self.group_names)}
        mirror = []
        for name in self.group_names:
            if name.startswith("front_to_back_"):
                region = int(name.rsplit("_", 1)[1])
                counterpart = f"back_to_front_{5 - region}"
            elif name.startswith("back_to_front_"):
                region = int(name.rsplit("_", 1)[1])
                counterpart = f"front_to_back_{5 - region}"
            else:
                counterpart = name[:-1] + ("R" if name.endswith("L") else "L")
            mirror.append(index[counterpart])
        return np.asarray(mirror, dtype=np.int32)

    def reset(self) -> None:
        count = self.probe.graph.node_count
        self.states = [np.zeros(count, dtype=np.float32) for _ in range(2)]
        self.histories = [[state.copy()] for state in self.states]
        base_size = len(self.groups) * self.stat_count
        self.previous_even = np.zeros(base_size, dtype=np.float64)
        self.previous_odd = np.zeros(base_size, dtype=np.float64)
        self.ema_even = np.zeros(base_size, dtype=np.float64)
        self.ema_odd = np.zeros(base_size, dtype=np.float64)

    def _advance(self, index: int, image: np.ndarray) -> None:
        values = (self.probe._sample_retina(image) * self.retinal_gain)[
            self.probe.retinal_permutation
        ]
        drive = np.zeros_like(self.states[index])
        drive[self.probe.retina.node_indices] = values
        self.states[index] = self.probe._advance(self.states[index], drive, self.histories[index])
        self.states[index][self.probe.retina.node_indices] = values

    def _statistics(self, state: np.ndarray) -> np.ndarray:
        output = []
        for nodes in self.groups:
            values = state[nodes].astype(np.float64)
            output.extend(
                (
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.quantile(values, 0.10)),
                    float(np.quantile(values, 0.90)),
                    float(np.max(values)),
                )
            )
        return np.asarray(output, dtype=np.float64)

    @staticmethod
    def _expand(current: np.ndarray, previous: np.ndarray, ema: np.ndarray) -> np.ndarray:
        return np.concatenate((current, current - previous, np.abs(current), current**2, ema))

    def step(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._advance(0, image)
        self._advance(1, image[:, ::-1])
        original = self._statistics(self.states[0])
        mirrored = self._statistics(self.states[1])[self.stat_mirror]
        even = (original + mirrored) / 2.0
        odd = (original - mirrored) / 2.0
        rate = float(self.config["features"]["exponential_mean_update"])
        self.ema_even = (1.0 - rate) * self.ema_even + rate * even
        self.ema_odd = (1.0 - rate) * self.ema_odd + rate * odd
        even_features = self._expand(even, self.previous_even, self.ema_even)
        odd_features = self._expand(odd, self.previous_odd, self.ema_odd)
        self.previous_even, self.previous_odd = even, odd
        return even_features, odd_features


def _teacher_step(environment, image, command, retina, teacher_config, teacher):
    channels = local_visual_channels(
        reconstruct_image(retina, retina.encode(image)), teacher_config
    )
    target = np.tanh(
        float(teacher["obstacle_gain"]) * channels["danger"] * channels["obstacle_asymmetry"]
        + float(teacher["road_gain"]) * channels["road_centroid"]
        - float(teacher["heading_gain"]) * environment.heading
    )
    command = 0.4 * command + 0.6 * target
    image, _, _ = environment.step(command, float(teacher["throttle"]), 0.0)
    return image, command, channels


def _fit_model(root: Path, config: dict) -> tuple[StructuredNeuralFeatures, dict, list[dict]]:
    teacher_config_path = Path(config["teacher_tuning_evidence"])
    teacher_report = json.loads((root / teacher_config_path).read_text())
    teacher = teacher_report["selected_candidate"]
    teacher_yaml = yaml.safe_load((root / "configs/driving-v7-r1r6-local.yaml").read_text())
    retina = build_mass_balanced_retina(root)
    features = StructuredNeuralFeatures(root, config)
    even_rows, odd_rows, targets, episode_summaries = [], [], [], []
    for seed in config["tuning"]["mirror_pair_seeds"]:
        environment = DrivingEnvironment()
        image = environment.reset(int(seed))
        features.reset()
        command = 0.0
        count = 0
        while not environment.done:
            even, odd = features.step(image)
            image, command, channels = _teacher_step(
                environment, image, command, retina, teacher_yaml, teacher
            )
            even_rows.append(even)
            odd_rows.append(odd)
            targets.append(
                [channels["danger"], channels["obstacle_asymmetry"], channels["road_centroid"]]
            )
            count += 1
        episode_summaries.append(
            {
                "seed": int(seed),
                "teacher_terminal_reason": environment.terminal_reason,
                "teacher_obstacles_passed": environment.obstacles_passed,
                "samples": count,
            }
        )
    even_matrix = np.asarray(even_rows)
    odd_matrix = np.asarray(odd_rows)
    target_matrix = np.asarray(targets)
    alpha = float(config["readout"]["alpha"])
    coefficients, scales, fit_r2 = [], [], []
    for output_index, matrix in enumerate((even_matrix, odd_matrix, odd_matrix)):
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale[scale < 1e-7] = 1.0
        normalized = (matrix - mean) / scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        penalty = np.diag([0.0] + [alpha] * normalized.shape[1])
        coefficient = np.linalg.solve(
            design.T @ design + penalty, design.T @ target_matrix[:, output_index]
        )
        prediction = design @ coefficient
        denominator = np.sum(
            (target_matrix[:, output_index] - np.mean(target_matrix[:, output_index])) ** 2
        )
        fit_r2.append(
            float(
                1.0
                - np.sum((target_matrix[:, output_index] - prediction) ** 2)
                / max(float(denominator), 1e-12)
            )
        )
        coefficients.append(coefficient.tolist())
        scales.append({"mean": mean.tolist(), "scale": scale.tolist()})
    model = {
        "output_names": config["readout"]["outputs"],
        "alpha": alpha,
        "coefficients": coefficients,
        "scales": scales,
        "fit_r2": fit_r2,
        "group_names": features.group_names,
        "group_mirror": features.group_mirror.tolist(),
        "feature_dimension_per_parity": int(even_matrix.shape[1]),
        "training_samples": int(len(target_matrix)),
    }
    return features, model, episode_summaries


def _predict(
    model: dict,
    even: np.ndarray,
    odd: np.ndarray,
    *,
    ablated_groups: set[str] | None = None,
) -> tuple[float, float, float]:
    if ablated_groups:
        names = model["group_names"]
        stat_count = 5
        temporal_count = 5
        mask = np.ones_like(even, dtype=bool)
        group_count = len(names)
        for group_index, name in enumerate(names):
            if name in ablated_groups:
                for temporal_index in range(temporal_count):
                    start = temporal_index * group_count * stat_count + group_index * stat_count
                    mask[start : start + stat_count] = False
        even = even.copy()
        odd = odd.copy()
        for index, scale in enumerate(model["scales"]):
            baseline = np.asarray(scale["mean"])
            if index == 0:
                even[~mask] = baseline[~mask]
            else:
                odd[~mask] = baseline[~mask]
    output = []
    for index, feature in enumerate((even, odd, odd)):
        scale = model["scales"][index]
        normalized = (feature - np.asarray(scale["mean"])) / np.asarray(scale["scale"])
        value = float(np.r_[1.0, normalized] @ np.asarray(model["coefficients"][index]))
        output.append(value)
    return (
        float(np.clip(output[0], 0.0, 1.0)),
        float(np.clip(output[1], -1.0, 1.0)),
        float(np.clip(output[2], -1.0, 1.0)),
    )


def _neural_episode(
    features,
    model,
    teacher,
    seed: int,
    *,
    ablated_groups: set[str] | None = None,
    heading_feedback: bool = True,
) -> dict:
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    features.reset()
    command = 0.0
    trace = []
    while not environment.done:
        even, odd = features.step(image)
        danger, asymmetry, road = _predict(model, even, odd, ablated_groups=ablated_groups)
        target = np.tanh(
            float(teacher["obstacle_gain"]) * danger * asymmetry
            + float(teacher["road_gain"]) * road
            - (float(teacher["heading_gain"]) * environment.heading if heading_feedback else 0.0)
        )
        command = 0.4 * command + 0.6 * target
        image, _, _ = environment.step(command, float(teacher["throttle"]), 0.0)
        trace.append(
            {
                "danger": danger,
                "obstacle_asymmetry": asymmetry,
                "road_center": road,
                "steering_command": command,
                "vehicle_x": environment.x,
                "vehicle_y": environment.y,
            }
        )
    return {
        "seed": seed,
        "terminal_reason": environment.terminal_reason,
        "success": environment.terminal_reason == "success",
        "obstacles_passed": environment.obstacles_passed,
        "distance": environment.y,
        "steps": environment.steps,
        "maximum_absolute_lateral_position": float(max(abs(item["vehicle_x"]) for item in trace)),
        "trace": trace,
    }


def _mirror_checks(episodes: list[dict]) -> list[dict]:
    checks = []
    for first, second in zip(episodes[::2], episodes[1::2], strict=True):
        count = min(first["steps"], second["steps"])
        a, b = first["trace"][:count], second["trace"][:count]
        checks.append(
            {
                "seeds": [first["seed"], second["seed"]],
                "maximum_danger_even_error": float(
                    max(abs(x["danger"] - y["danger"]) for x, y in zip(a, b, strict=True))
                ),
                "maximum_asymmetry_odd_error": float(
                    max(
                        abs(x["obstacle_asymmetry"] + y["obstacle_asymmetry"])
                        for x, y in zip(a, b, strict=True)
                    )
                ),
                "maximum_road_odd_error": float(
                    max(abs(x["road_center"] + y["road_center"]) for x, y in zip(a, b, strict=True))
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
    return checks


def _model_digest(model: dict) -> str:
    return hashlib.sha256(
        json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluate_v7_neural_channels_tuning(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    features, model, teacher_episodes = _fit_model(root, config)
    teacher_report = json.loads((root / config["adapter_source"]).read_text())
    teacher = teacher_report["selected_candidate"]
    seeds = [int(seed) for seed in config["tuning"]["mirror_pair_seeds"]]
    episodes = [_neural_episode(features, model, teacher, seed) for seed in seeds]
    mirrors = _mirror_checks(episodes)
    return {
        "protocol": {
            "name": config["name"],
            "phase": "tuning",
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                config["teacher_tuning_evidence"]: _sha256(
                    root / config["teacher_tuning_evidence"]
                ),
            },
            "visual_input_only_via_R1_R6": True,
            "teacher_values_injected_into_neural_state": False,
            "environment_geometry_used_by_neural_controller": False,
            "runtime_modified": False,
        },
        "teacher_episodes": teacher_episodes,
        "model": model,
        "model_sha256": _model_digest(model),
        "episodes": [
            {key: value for key, value in item.items() if key != "trace"} for item in episodes
        ],
        "mirror_checks": mirrors,
        "tuning_passed": all(
            item["success"] and item["obstacles_passed"] == 9 for item in episodes
        ),
        "calibration_evaluated": False,
        "final_evaluated": False,
        "advance_to_calibration": all(item["success"] for item in episodes),
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }


def evaluate_v7_neural_channels_calibration(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    tuning = json.loads((root / TUNING_ARTIFACT).read_text())
    for path, digest in tuning["protocol"]["dependencies_sha256"].items():
        if _sha256(root / path) != digest:
            raise ValueError(f"neural channel tuning evidence is stale: {path}")
    if not tuning["advance_to_calibration"]:
        raise ValueError("neural channel tuning did not pass")
    if _model_digest(tuning["model"]) != tuning["model_sha256"]:
        raise ValueError("frozen neural channel model changed")
    features = StructuredNeuralFeatures(root, config)
    teacher = json.loads((root / config["adapter_source"]).read_text())["selected_candidate"]
    seeds = [int(seed) for seed in config["calibration"]["mirror_pair_seeds"]]
    episodes = [_neural_episode(features, tuning["model"], teacher, seed) for seed in seeds]
    mirrors = _mirror_checks(episodes)
    limit = float(config["calibration"]["pass_requirements"]["maximum_channel_mirror_error"])
    gates = {
        "both_success": all(item["success"] for item in episodes),
        "both_pass_all_nine_obstacles": all(item["obstacles_passed"] == 9 for item in episodes),
        "no_collision_or_road_boundary": all(
            item["terminal_reason"] == "success" for item in episodes
        ),
        "channel_mirror_error": max(
            max(
                item["maximum_danger_even_error"],
                item["maximum_asymmetry_odd_error"],
                item["maximum_road_odd_error"],
            )
            for item in mirrors
        )
        <= limit,
        "action_mirror_error": max(item["maximum_action_odd_error"] for item in mirrors)
        <= float(config["calibration"]["pass_requirements"]["maximum_action_mirror_error"]),
        "trajectory_mirror_error": max(item["maximum_trajectory_mirror_error"] for item in mirrors)
        <= float(config["calibration"]["pass_requirements"]["maximum_trajectory_mirror_error"]),
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
            "model_sha256": tuning["model_sha256"],
            "model_changed_after_tuning": False,
            "runtime_modified": False,
        },
        "seeds": seeds,
        "episodes": episodes,
        "mirror_checks": mirrors,
        "gates": gates,
        "calibration_passed": all(gates.values()),
        "final_evaluated": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
