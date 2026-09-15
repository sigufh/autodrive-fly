from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_retina_audit import build_balanced_retina_control

CONFIG = Path("configs/driving-v7-r1r6-multi.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_r1r6_multi.py")
TUNING_ARTIFACT = Path("artifacts/v7-r1r6-multi-tuning.json")


@dataclass(frozen=True)
class MassBalancedRetina:
    x: np.ndarray
    y: np.ndarray
    side: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    pixel_id: np.ndarray
    mass: np.ndarray
    body_ids: np.ndarray

    def encode(self, image: np.ndarray) -> np.ndarray:
        if image.shape != (24, 48):
            raise ValueError("mass-balanced retina requires 24x48 images")
        return np.asarray(image[self.y, self.x], dtype=np.float64)


def build_mass_balanced_retina(root: Path) -> MassBalancedRetina:
    balanced = build_balanced_retina_control(root)
    coordinates = balanced.pair_coordinates.astype(np.float64)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    normalized = (coordinates - low) / np.maximum(high - low, 1.0)
    centres = np.stack((normalized[:, 0] * 23.0, normalized[:, 1] * 23.0), axis=1)
    yy, xx = np.mgrid[:24, :24]
    pixels = np.stack((xx.ravel(), yy.ravel()), axis=1)
    cost = np.sum((centres[:, None, :] - pixels[None, :, :]) ** 2, axis=2)
    rows, columns = linear_sum_assignment(cost)
    pair_pixel = np.full(len(coordinates), -1, dtype=np.int32)
    pair_pixel[rows] = columns
    remaining = np.flatnonzero(pair_pixel < 0)
    pair_pixel[remaining] = np.argmin(cost[remaining], axis=1).astype(np.int32)
    pair_xy = pixels[pair_pixel].astype(np.int32)
    count = np.bincount(pair_pixel, minlength=24 * 24)
    pair_mass = 1.0 / count[pair_pixel]
    size = balanced.retina.size
    x = np.empty(size, dtype=np.int32)
    y = np.empty(size, dtype=np.int32)
    pixel_id = np.empty(size, dtype=np.int32)
    mass = np.empty(size, dtype=np.float64)
    x[balanced.left_positions] = 23 - pair_xy[:, 0]
    x[balanced.right_positions] = 24 + pair_xy[:, 0]
    y[balanced.left_positions] = pair_xy[:, 1]
    y[balanced.right_positions] = pair_xy[:, 1]
    pixel_id[balanced.left_positions] = pair_pixel
    pixel_id[balanced.right_positions] = pair_pixel
    mass[balanced.left_positions] = pair_mass
    mass[balanced.right_positions] = pair_mass
    return MassBalancedRetina(
        x=x,
        y=y,
        side=balanced.retina.side.copy(),
        horizontal=np.linspace(-1.0, 1.0, 48)[x],
        vertical=np.linspace(0.0, 1.0, 24)[y],
        pixel_id=pixel_id,
        mass=mass,
        body_ids=balanced.retina.body_ids.copy(),
    )


def retinal_channels(retina: MassBalancedRetina, image: np.ndarray, config: dict) -> dict:
    values = retina.encode(image)
    obstacle_config = config["channels"]["obstacle"]
    obstacle = (values > float(obstacle_config["threshold"])).astype(np.float64)
    obstacle *= retina.vertical ** int(obstacle_config["vertical_weight_power"])
    obstacle *= retina.mass
    left = float(np.sum(obstacle[retina.side < 0]))
    right = float(np.sum(obstacle[retina.side > 0]))
    total = left + right
    danger = total / (24 * 24)
    asymmetry = (left - right) / (total + 1e-12)
    road_config = config["channels"]["road_center"]
    marker = np.isclose(
        values,
        float(road_config["target_luminance"]),
        atol=float(road_config["absolute_tolerance"]),
    ).astype(np.float64)
    marker *= retina.mass
    road_centroid = float(np.sum(retina.horizontal * marker) / (np.sum(marker) + 1e-12))
    return {
        "danger": danger,
        "obstacle_asymmetry": asymmetry,
        "road_centroid": road_centroid,
        "obstacle_receptor_count": int(np.count_nonzero(obstacle)),
        "road_marker_receptor_count": int(np.count_nonzero(marker)),
    }


def _episode(retina: MassBalancedRetina, config: dict, candidate: dict, seed: int) -> dict:
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    command = 0.0
    trace = []
    while not environment.done:
        channels = retinal_channels(retina, image, config)
        target = np.tanh(
            float(candidate["obstacle_gain"]) * channels["danger"] * channels["obstacle_asymmetry"]
            + float(candidate["road_gain"]) * channels["road_centroid"]
            - float(candidate["heading_gain"]) * environment.heading
        )
        adapter = config["adapter"]
        command = float(
            float(adapter["smoothing_previous"]) * command
            + float(adapter["smoothing_current"]) * target
        )
        image, _, _ = environment.step(command, float(candidate["throttle"]), 0.0)
        trace.append(
            {
                "danger": channels["danger"],
                "obstacle_asymmetry": channels["obstacle_asymmetry"],
                "road_centroid": channels["road_centroid"],
                "steering_command": command,
                "vehicle_x": environment.x,
                "vehicle_y": environment.y,
            }
        )
    return {
        "seed": seed,
        "pair_seed": environment.pair_seed,
        "mirror": environment.mirror,
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
                "maximum_obstacle_asymmetry_odd_error": float(
                    max(
                        abs(x["obstacle_asymmetry"] + y["obstacle_asymmetry"])
                        for x, y in zip(a, b, strict=True)
                    )
                ),
                "maximum_road_centroid_odd_error": float(
                    max(
                        abs(x["road_centroid"] + y["road_centroid"])
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
    return checks


def _summary(candidate: dict, episodes: list[dict]) -> dict:
    return {
        "candidate": candidate,
        "success_count": sum(item["success"] for item in episodes),
        "total_obstacles_passed": sum(item["obstacles_passed"] for item in episodes),
        "road_boundary_count": sum(item["terminal_reason"] == "road_boundary" for item in episodes),
        "mean_maximum_absolute_lateral_position": float(
            np.mean([item["maximum_absolute_lateral_position"] for item in episodes])
        ),
        "episodes": [
            {key: value for key, value in item.items() if key != "trace"} for item in episodes
        ],
        "mirror_checks": _mirror_checks(episodes),
    }


def _key(item: dict) -> tuple:
    candidate = item["candidate"]
    return (
        -item["success_count"],
        -item["total_obstacles_passed"],
        item["road_boundary_count"],
        item["mean_maximum_absolute_lateral_position"],
        candidate["throttle"],
        candidate["obstacle_gain"],
        candidate["road_gain"],
        candidate["heading_gain"],
    )


def _grid(config: dict) -> list[dict]:
    values = config["tuning"]["candidate_grid"]
    return [
        {
            "throttle": float(throttle),
            "obstacle_gain": float(obstacle),
            "road_gain": float(road),
            "heading_gain": float(heading),
        }
        for throttle in values["throttle"]
        for obstacle in values["obstacle_gain"]
        for road in values["road_gain"]
        for heading in values["heading_gain"]
    ]


def evaluate_v7_r1r6_multi_tuning(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    retina = build_mass_balanced_retina(root)
    per_pixel_mass = np.bincount(retina.pixel_id, weights=retina.mass, minlength=24 * 24)
    if not np.allclose(per_pixel_mass, 2.0):
        raise ValueError("left-plus-right receptor mass is not exactly two per paired pixel")
    seeds = [int(seed) for seed in config["tuning"]["mirror_pair_seeds"]]
    summaries = []
    for candidate in _grid(config):
        summaries.append(
            _summary(candidate, [_episode(retina, config, candidate, seed) for seed in seeds])
        )
    selected = min(summaries, key=_key)
    frozen = {
        **selected["candidate"],
        "config_sha256": _sha256(root / CONFIG),
        "implementation_sha256": _sha256(root / IMPLEMENTATION),
        "retina_body_ids_sha256": hashlib.sha256(retina.body_ids.tobytes()).hexdigest(),
        "retina_assignment_sha256": hashlib.sha256(
            retina.x.tobytes() + retina.y.tobytes() + retina.mass.astype("<f8").tobytes()
        ).hexdigest(),
    }
    digest = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "protocol": {
            "name": config["name"],
            "phase": "tuning",
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "visual_input_only_via_R1_R6": True,
            "target_neuron_activity_used": False,
            "environment_geometry_used": False,
            "runtime_modified": False,
        },
        "retina": {
            "receptor_count": len(retina.x),
            "receptors_per_eye": int(np.count_nonzero(retina.side < 0)),
            "pixels_per_eye": 24 * 24,
            "covered_pixels_per_eye": int(len(np.unique(retina.pixel_id))),
            "maximum_pairs_per_pixel": int(np.max(np.bincount(retina.pixel_id))),
            "maximum_pixel_mass_error": float(np.max(np.abs(per_pixel_mass - 2.0))),
        },
        "candidate_count": len(summaries),
        "candidate_summaries": summaries,
        "selected_candidate": frozen,
        "selected_candidate_sha256": digest,
        "tuning_passed": selected["success_count"] == len(seeds)
        and selected["total_obstacles_passed"] == 9 * len(seeds),
        "calibration_evaluated": False,
        "final_evaluated": False,
        "advance_to_calibration": selected["success_count"] == len(seeds),
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }


def evaluate_v7_r1r6_multi_calibration(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    tuning = json.loads((root / TUNING_ARTIFACT).read_text())
    for path, digest in tuning["protocol"]["dependencies_sha256"].items():
        if _sha256(root / path) != digest:
            raise ValueError(f"R1-R6 multi-obstacle tuning evidence is stale: {path}")
    if not tuning["advance_to_calibration"]:
        raise ValueError("R1-R6 multi-obstacle tuning did not pass")
    frozen = tuning["selected_candidate"]
    digest = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != tuning["selected_candidate_sha256"]:
        raise ValueError("R1-R6 multi-obstacle frozen candidate changed")
    retina = build_mass_balanced_retina(root)
    seeds = [int(seed) for seed in config["calibration"]["mirror_pair_seeds"]]
    episodes = [_episode(retina, config, frozen, seed) for seed in seeds]
    mirrors = _mirror_checks(episodes)
    limits = config["calibration"]["pass_requirements"]
    gates = {
        "both_success": all(item["success"] for item in episodes),
        "both_pass_all_nine_obstacles": all(item["obstacles_passed"] == 9 for item in episodes),
        "no_collision_or_road_boundary": all(
            item["terminal_reason"] == "success" for item in episodes
        ),
        "action_mirror_error": max(item["maximum_action_odd_error"] for item in mirrors)
        <= float(limits["maximum_action_mirror_error"]),
        "trajectory_mirror_error": max(item["maximum_trajectory_mirror_error"] for item in mirrors)
        <= float(limits["maximum_trajectory_mirror_error"]),
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
            "candidate_sha256": digest,
            "parameters_changed_after_tuning": False,
            "runtime_modified": False,
        },
        "seeds": seeds,
        "episodes": episodes,
        "mirror_checks": mirrors,
        "gates": gates,
        "calibration_passed": all(gates.values()),
        "engineering_upper_bound_only": True,
        "final_evaluated": False,
        "advance_to_navigation_release": False,
    }
