from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_neural_channels import (
    StructuredNeuralFeatures,
    _mirror_checks,
    _model_digest,
    _neural_episode,
    _predict,
)
from fly_emotion.driving.v7_r1r6_local import _episode as _r1r6_local_episode
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina

CONFIG = Path("configs/driving-v7-heading-ring.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_heading_ring.py")


class EPGPENPEGHeadingRing:
    def __init__(self, config: dict):
        ring = config["ring"]
        self.bin_count = int(ring["bins"])
        self.phases = np.linspace(-np.pi, np.pi, self.bin_count, endpoint=False)
        self.concentration = float(ring["epg_concentration"])
        self.peg_memory = float(ring["peg_memory"])
        self.pen_update_gain = float(ring["pen_update_gain"])
        self.decode_limit = float(ring["decode_limit"])
        self.reset(float(ring["initial_heading"]))

    def _encode(self, heading: float) -> np.ndarray:
        angle = float(heading) / self.decode_limit * np.pi
        activity = 1.0 + 0.9 * np.cos(self.phases - angle)
        return activity / np.sum(activity)

    def decode(self) -> float:
        vector = np.sum(self.epg * np.exp(1j * self.phases))
        return float(np.angle(vector) / np.pi * self.decode_limit)

    def reset(self, heading: float = 0.0) -> None:
        self.integrated_heading = float(heading)
        self.epg = self._encode(heading)
        self.peg = self.epg.copy()
        self.pen_left = np.zeros(self.bin_count, dtype=np.float64)
        self.pen_right = np.zeros(self.bin_count, dtype=np.float64)

    def step(self, yaw_rate: float, dt: float, *, cue_visible: bool = True) -> float:
        delta = float(yaw_rate) * float(dt) * self.pen_update_gain
        self.integrated_heading = float(
            np.clip(
                self.integrated_heading + delta, -self.decode_limit, self.decode_limit
            )
        )
        if delta > 0.0:
            self.pen_right = self._encode(self.integrated_heading)
            self.pen_left.fill(0.0)
        elif delta < 0.0:
            self.pen_left = self._encode(self.integrated_heading)
            self.pen_right.fill(0.0)
        else:
            self.pen_left.fill(0.0)
            self.pen_right.fill(0.0)
        # PEN path integration advances EPG from proprioceptive yaw.  PEG stores
        # and stabilizes the current bump but never pulls EPG toward an old phase.
        self.epg = self._encode(self.integrated_heading)
        self.peg = self.peg_memory * self.peg + (1.0 - self.peg_memory) * self.epg
        self.peg /= np.sum(self.peg)
        return self.decode()


def _population_nodes(root: Path, graph, cell_types: tuple[str, ...]) -> np.ndarray:
    table = feather.read_table(
        root / "data/raw/malecns-v1.0/body-annotations.feather",
        columns=["bodyId", "type"],
        memory_map=True,
    ).to_pandas()
    body_ids = table.loc[table["type"].isin(cell_types), "bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(graph.body_ids, body_ids)
    valid = nodes < graph.node_count
    valid[valid] &= graph.body_ids[nodes[valid]] == body_ids[valid]
    return nodes[valid].astype(np.int32)


def _central_complex_structure(root: Path) -> dict:
    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
    populations = {
        "EPG": ("EPG",),
        "PEN": ("PEN_a(PEN1)", "PEN_b(PEN2)"),
        "PEG": ("PEG",),
        "FC2": ("FC2A", "FC2B", "FC2C"),
        "PFL3": ("PFL3",),
        "PFL2": ("PFL2",),
        "DNa": ("DNa01", "DNa02"),
        "DNp20": ("DNp20",),
    }
    nodes = {name: _population_nodes(root, graph, types) for name, types in populations.items()}
    edges = {}
    for source in populations:
        for target in populations:
            matrix = graph.adjacency[nodes[target], :][:, nodes[source]]
            if matrix.nnz:
                edges[f"{source}->{target}"] = {
                    "edge_count": int(matrix.nnz),
                    "synapse_weight": float(np.sum(matrix.data)),
                }
    body_ids = {name: graph.body_ids[value].tolist() for name, value in nodes.items()}
    return {
        "population_counts": {name: len(value) for name, value in nodes.items()},
        "body_ids_sha256": {
            name: hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
            for name, ids in body_ids.items()
        },
        "selected_edges": edges,
        "structure_is_not_functional_validation": True,
    }


def _heading_assays(config: dict) -> dict:
    dt = float(config["assays"]["dt"])
    sequences = [
        np.asarray(values, dtype=np.float64)
        for values in config["assays"]["rotation_sequences"]
    ]
    tracking = []
    mirror_errors = []
    for yaw_rates in sequences:
        ring = EPGPENPEGHeadingRing(config)
        expected = 0.0
        errors = []
        mirrored = EPGPENPEGHeadingRing(config)
        for yaw_rate in yaw_rates:
            expected = float(
                np.clip(expected + yaw_rate * dt, -ring.decode_limit, ring.decode_limit)
            )
            decoded = ring.step(float(yaw_rate), dt)
            reflected = mirrored.step(float(-yaw_rate), dt)
            errors.append(abs(decoded - expected))
            mirror_errors.append(abs(decoded + reflected))
        tracking.append(float(max(errors)))
    occlusion = EPGPENPEGHeadingRing(config)
    occlusion.step(0.24, dt)
    before = occlusion.decode()
    for _ in range(int(config["assays"]["occlusion_steps"])):
        occlusion.step(0.0, dt, cue_visible=False)
    drift = abs(occlusion.decode() - before)
    gates = {
        "rotation_tracking": max(tracking)
        <= float(config["assays"]["maximum_tracking_error"]),
        "occlusion_memory": drift
        <= float(config["assays"]["maximum_occlusion_drift"]),
        "mirror_equivariance": max(mirror_errors)
        <= float(config["assays"]["maximum_mirror_error"]),
    }
    return {
        "maximum_tracking_error": max(tracking),
        "occlusion_drift": float(drift),
        "maximum_mirror_error": max(mirror_errors),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _heading_episode(
    features,
    model,
    teacher: dict,
    config: dict,
    seed: int,
    arm: str,
    *,
    ablated_groups: set[str] | None = None,
) -> dict:
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    features.reset()
    ring = EPGPENPEGHeadingRing(config)
    command = 0.0
    decoded_heading = ring.decode()
    maximum_heading_error = 0.0
    trace = []
    while not environment.done:
        even, odd = features.step(image)
        danger, asymmetry, road = _predict(
            model, even, odd, ablated_groups=ablated_groups
        )
        target = np.tanh(
            float(teacher["obstacle_gain"]) * danger * asymmetry
            + float(teacher["road_gain"]) * road
            - float(teacher["heading_gain"]) * decoded_heading
        )
        command = 0.4 * command + 0.6 * target
        image, _, _ = environment.step(command, float(teacher["throttle"]), 0.0)
        yaw_rate = environment.last_yaw_rate
        if arm == "frozen_heading":
            yaw_rate = 0.0
        elif arm == "reversed_yaw_update":
            yaw_rate = -yaw_rate
        elif arm != "neural_heading":
            raise ValueError(f"unknown heading arm: {arm}")
        decoded_heading = ring.step(yaw_rate, environment.dt, cue_visible=False)
        maximum_heading_error = max(
            maximum_heading_error, abs(decoded_heading - environment.heading)
        )
        trace.append(
            {
                "danger": danger,
                "obstacle_asymmetry": asymmetry,
                "road_center": road,
                "decoded_heading": decoded_heading,
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
        "maximum_heading_decode_error": float(maximum_heading_error),
        "maximum_absolute_lateral_position": float(
            max(abs(item["vehicle_x"]) for item in trace)
        ),
        "trace": trace,
    }


def evaluate_v7_heading_ring(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    neural_config_path = Path(config["neural_channel_config"])
    neural_config = yaml.safe_load((root / neural_config_path).read_text())
    tuning_path = Path(config["neural_tuning_evidence"])
    calibration_path = Path(config["neural_calibration_evidence"])
    tuning = json.loads((root / tuning_path).read_text())
    calibration = json.loads((root / calibration_path).read_text())
    if not tuning["tuning_passed"] or not calibration["calibration_passed"]:
        raise ValueError("heading ring requires passed frozen neural navigation evidence")
    if _model_digest(tuning["model"]) != tuning["model_sha256"]:
        raise ValueError("frozen neural visual model changed")
    teacher = json.loads((root / neural_config["adapter_source"]).read_text())[
        "selected_candidate"
    ]
    features = StructuredNeuralFeatures(root, neural_config)
    arms = {}
    for arm in config["navigation"]["controls"]:
        tuning_episodes = [
            _heading_episode(features, tuning["model"], teacher, config, int(seed), arm)
            for seed in config["navigation"]["tuning_seeds"]
        ]
        regression_episodes = [
            _heading_episode(features, tuning["model"], teacher, config, int(seed), arm)
            for seed in config["navigation"]["development_regression_seeds"]
        ]
        calibration_episodes = (
            [
                _heading_episode(features, tuning["model"], teacher, config, int(seed), arm)
                for seed in config["navigation"]["calibration_seeds"]
            ]
            if arm == "neural_heading"
            else []
        )
        episodes = tuning_episodes + regression_episodes + calibration_episodes
        arms[arm] = {
            "tuning_success_count": sum(item["success"] for item in tuning_episodes),
            "tuning_obstacles_passed": sum(
                item["obstacles_passed"] for item in tuning_episodes
            ),
            "calibration_success_count": sum(
                item["success"] for item in calibration_episodes
            ),
            "calibration_obstacles_passed": sum(
                item["obstacles_passed"] for item in calibration_episodes
            ),
            "maximum_heading_decode_error": max(
                item["maximum_heading_decode_error"] for item in episodes
            ),
            "development_regression": [
                {key: value for key, value in item.items() if key != "trace"}
                for item in regression_episodes
            ],
            "calibration": [
                {key: value for key, value in item.items() if key != "trace"}
                for item in calibration_episodes
            ],
            "episodes": [
                {key: value for key, value in item.items() if key != "trace"}
                for item in episodes
            ],
            "mirror_checks": _mirror_checks(episodes),
        }
    neural = arms["neural_heading"]
    navigation_passed = (
        neural["tuning_success_count"] == 6
        and neural["tuning_obstacles_passed"] == 54
        and neural["calibration_success_count"] == 2
        and neural["calibration_obstacles_passed"] == 18
    )
    controls_passed = all(
        arms[name]["tuning_success_count"] < neural["tuning_success_count"]
        or arms[name]["tuning_obstacles_passed"] < neural["tuning_obstacles_passed"]
        for name in ("frozen_heading", "reversed_yaw_update")
    )
    attribution_seeds = [int(seed) for seed in config["navigation"]["calibration_seeds"]]
    raw_heading_reference = [
        _neural_episode(features, tuning["model"], teacher, seed)
        for seed in attribution_seeds
    ]
    r1r6_config = yaml.safe_load((root / "configs/driving-v7-r1r6-local.yaml").read_text())
    r1r6_candidate = json.loads(
        (root / "artifacts/v7-r1r6-local-tuning.json").read_text()
    )["selected_candidate"]
    retina = build_mass_balanced_retina(root)
    r1r6_upper_bound = [
        _r1r6_local_episode(retina, r1r6_config, r1r6_candidate, seed)
        for seed in attribution_seeds
    ]
    raw_matches_ring = all(
        raw["terminal_reason"] == ring["terminal_reason"]
        and raw["obstacles_passed"] == ring["obstacles_passed"]
        and raw["steps"] == ring["steps"]
        for raw, ring in zip(raw_heading_reference, neural["calibration"], strict=True)
    )
    attribution = {
        "role": "post_failure_attribution_only_not_parameter_selection",
        "raw_heading_reference": [
            {key: value for key, value in item.items() if key != "trace"}
            for item in raw_heading_reference
        ],
        "r1r6_local_visual_upper_bound": [
            {key: value for key, value in item.items() if key != "trace"}
            for item in r1r6_upper_bound
        ],
        "raw_heading_matches_neural_heading_failure": bool(raw_matches_ring),
        "r1r6_local_upper_bound_passes_both": all(
            item["success"] and item["obstacles_passed"] == 9
            for item in r1r6_upper_bound
        ),
        "diagnosis": (
            "frozen_neural_visual_readout_generalization_gap"
            if raw_matches_ring
            and all(item["success"] for item in r1r6_upper_bound)
            else "attribution_inconclusive"
        ),
    }
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(neural_config_path): _sha256(root / neural_config_path),
                str(tuning_path): _sha256(root / tuning_path),
                str(calibration_path): _sha256(root / calibration_path),
            },
            "neural_visual_model_sha256": tuning["model_sha256"],
            "visual_model_changed": False,
            "action_adapter_changed": False,
            "raw_heading_read_by_controller": False,
            "yaw_rate_is_proprioceptive_input": True,
            "external_final_evaluated": False,
            "development_regression_seeds_not_counted_as_calibration": True,
            "calibration_run_once_after_candidate_freeze": True,
            "runtime_modified": False,
        },
        "malecns_structure": _central_complex_structure(root),
        "heading_assays": _heading_assays(config),
        "navigation_arms": arms,
        "calibration_failure_attribution": attribution,
        "navigation_passed": bool(navigation_passed),
        "causal_controls_passed": bool(controls_passed),
        "advance_to_fc2_pfl_comparison": bool(navigation_passed and controls_passed),
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
        "limitations": [
            "MaleCNS population membership and edges establish structural support only.",
            "Yaw rate is a proprioceptive vehicle signal, not inferred from the visual graph.",
            "The fixed ring equation is an engineering abstraction, not a fitted cell model.",
            "FC2/PFL/DNa goal comparison and external final remain unevaluated.",
        ],
    }
