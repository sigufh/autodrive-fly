from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_r1r6_multi import MassBalancedRetina, build_mass_balanced_retina

CONFIG = Path("configs/driving-v7-r1r6-local.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_r1r6_local.py")
TUNING_ARTIFACT = Path("artifacts/v7-r1r6-local-tuning.json")


def reconstruct_image(retina: MassBalancedRetina, receptor_values: np.ndarray) -> np.ndarray:
    values = np.asarray(receptor_values, dtype=np.float64)
    if values.shape != retina.x.shape:
        raise ValueError("receptor values do not match the mass-balanced retina")
    image = np.zeros((24, 48), dtype=np.float64)
    mass = np.zeros((24, 48), dtype=np.float64)
    np.add.at(image, (retina.y, retina.x), values * retina.mass)
    np.add.at(mass, (retina.y, retina.x), retina.mass)
    if np.any(mass <= 0):
        raise ValueError("mass-balanced retina does not cover every image pixel")
    return image / mass


def local_visual_channels(image: np.ndarray, config: dict) -> dict:
    if image.shape != (24, 48):
        raise ValueError("local visual channels require a 24x48 reconstruction")
    obstacle_config = config["channels"]["obstacle"]
    obstacle = image > float(obstacle_config["luminance_threshold"])
    height, width = image.shape
    horizon = height // 3
    obstacle_height = np.zeros(width, dtype=np.float64)
    for column in range(width):
        positions = np.flatnonzero(obstacle[:, column])
        if positions.size:
            obstacle_height[column] = height - int(positions.min())
    distance = 1.0 - obstacle_height / (height - horizon)
    quarter = width // 4
    left = float(np.mean(distance[quarter : 2 * quarter]))
    right = float(np.mean(distance[2 * quarter : 3 * quarter]))
    danger = float(np.clip(1.0 - np.min(distance[quarter : 3 * quarter]), 0.0, 1.0))
    asymmetry = right - left
    road_config = config["channels"]["road_center"]
    marker = np.isclose(
        image,
        float(road_config["target_luminance"]),
        atol=float(road_config["absolute_tolerance"]),
    )
    horizontal = np.linspace(-1.0, 1.0, width)[None, :]
    road_centroid = float(np.sum(horizontal * marker) / (np.sum(marker) + 1e-12))
    return {
        "danger": danger,
        "obstacle_asymmetry": asymmetry,
        "road_centroid": road_centroid,
        "active_obstacle_columns": int(np.count_nonzero(obstacle_height)),
        "road_marker_pixels": int(np.count_nonzero(marker)),
    }


def _episode(retina: MassBalancedRetina, config: dict, candidate: dict, seed: int) -> dict:
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    command = 0.0
    trace = []
    maximum_reconstruction_error = 0.0
    while not environment.done:
        receptor_values = retina.encode(image)
        reconstructed = reconstruct_image(retina, receptor_values)
        maximum_reconstruction_error = max(
            maximum_reconstruction_error, float(np.max(np.abs(reconstructed - image)))
        )
        channels = local_visual_channels(reconstructed, config)
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
        "maximum_reconstruction_error": maximum_reconstruction_error,
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


def _key(summary: dict) -> tuple:
    candidate = summary["candidate"]
    return (
        -summary["success_count"],
        -summary["total_obstacles_passed"],
        summary["road_boundary_count"],
        summary["mean_maximum_absolute_lateral_position"],
        candidate["throttle"],
        candidate["obstacle_gain"],
        candidate["road_gain"],
        candidate["heading_gain"],
    )


def evaluate_v7_r1r6_local_tuning(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    retina = build_mass_balanced_retina(root)
    test_image = np.arange(24 * 48, dtype=np.float64).reshape(24, 48) / (24 * 48)
    reconstruction_error = float(
        np.max(np.abs(reconstruct_image(retina, retina.encode(test_image)) - test_image))
    )
    if reconstruction_error > 1e-12:
        raise ValueError("R1-R6 mass-balanced reconstruction is not exact")
    seeds = [int(seed) for seed in config["tuning"]["mirror_pair_seeds"]]
    summaries = [
        _summary(candidate, [_episode(retina, config, candidate, seed) for seed in seeds])
        for candidate in _grid(config)
    ]
    selected = min(summaries, key=_key)
    frozen = {
        **selected["candidate"],
        "config_sha256": _sha256(root / CONFIG),
        "implementation_sha256": _sha256(root / IMPLEMENTATION),
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
        "maximum_synthetic_reconstruction_error": reconstruction_error,
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


def evaluate_v7_r1r6_local_calibration(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    tuning = json.loads((root / TUNING_ARTIFACT).read_text())
    for path, digest in tuning["protocol"]["dependencies_sha256"].items():
        if _sha256(root / path) != digest:
            raise ValueError(f"R1-R6 local tuning evidence is stale: {path}")
    if not tuning["advance_to_calibration"]:
        raise ValueError("R1-R6 local tuning did not pass")
    frozen = tuning["selected_candidate"]
    digest = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != tuning["selected_candidate_sha256"]:
        raise ValueError("R1-R6 local frozen candidate changed")
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
        "reconstruction_error": max(item["maximum_reconstruction_error"] for item in episodes)
        <= float(limits["maximum_reconstruction_error"]),
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
        "engineering_input_upper_bound_only": True,
        "final_evaluated": False,
        "advance_to_navigation_release": False,
    }
