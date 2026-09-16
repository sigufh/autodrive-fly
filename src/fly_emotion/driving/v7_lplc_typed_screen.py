from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_stage1_development import run_signed_cell_responses
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary

CONFIG = Path("configs/driving-v7-lplc-typed-screen.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_lplc_typed_screen.py")
WIDTH = 48
HEIGHT = 24


def _noise(frames: np.ndarray, standard_deviation: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perturbation = rng.normal(0.0, standard_deviation, frames.shape)
    perturbation = (perturbation + perturbation[:, :, ::-1]) / 2.0
    return np.clip(frames + perturbation, 0.0, 1.0).astype(np.float32)


def _moving_object(
    duration: int, width: int, height: int, vertical_fraction: float, direction: str
) -> np.ndarray:
    frames = np.full((duration, HEIGHT, WIDTH), 0.92, dtype=np.float32)
    positions = np.linspace(-width, WIDTH, duration)
    if direction == "front_to_back":
        positions = positions[::-1]
    center_y = int(round(vertical_fraction * (HEIGHT - 1)))
    top = max(0, center_y - height // 2)
    bottom = min(HEIGHT, top + height)
    for index, position in enumerate(positions):
        left = max(0, int(round(position)))
        right = min(WIDTH, left + width)
        if right > left:
            frames[index, top:bottom, left:right] = 0.08
    return frames


def _radial_ring(duration: int, start: float, terminal: float, outward: bool) -> np.ndarray:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    distance = np.sqrt((xx - (WIDTH - 1) / 2) ** 2 + (yy - (HEIGHT - 1) / 2) ** 2)
    radii = np.linspace(start, terminal, duration)
    if not outward:
        radii = radii[::-1]
    frames = np.full((duration, HEIGHT, WIDTH), 0.92, dtype=np.float32)
    for index, radius in enumerate(radii):
        frames[index, np.abs(distance - radius) <= 1.25] = 0.08
    return frames


def _motion_free_darkening(duration: int, terminal: float) -> np.ndarray:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    mask = (xx - (WIDTH - 1) / 2) ** 2 + (yy - (HEIGHT - 1) / 2) ** 2 <= terminal**2
    frames = np.full((duration, HEIGHT, WIDTH), 0.92, dtype=np.float32)
    levels = np.linspace(0.92, 0.08, duration)
    for index, level in enumerate(levels):
        frames[index, mask] = level
    return frames


def _translation(duration: int, period: float) -> np.ndarray:
    _, xx = np.mgrid[:HEIGHT, :WIDTH]
    return np.stack(
        [
            np.broadcast_to(
                0.5 - 0.4 * np.sin(2 * np.pi * xx / period - phase), (HEIGHT, WIDTH)
            ).astype(np.float32)
            for phase in np.linspace(0, 4 * np.pi, duration, endpoint=False)
        ]
    )


def _filled_loom(duration: int, start: float, terminal: float) -> np.ndarray:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    distance = np.sqrt((xx - (WIDTH - 1) / 2) ** 2 + (yy - (HEIGHT - 1) / 2) ** 2)
    frames = np.full((duration, HEIGHT, WIDTH), 0.92, dtype=np.float32)
    for index, radius in enumerate(np.linspace(start, terminal, duration)):
        frames[index, distance <= radius] = 0.08
    return frames


def _condition_stimuli(condition: dict, config: dict) -> dict[str, VisualStimulus]:
    duration = int(condition["duration_frames"])
    terminal = float(condition["terminal_radius_pixels"])
    start = float(condition["start_radius_pixels"])
    noise = float(condition["noise_standard_deviation"])
    seed = int(condition["seed"])
    stimulus = config["stimulus"]
    raw = {
        "lplc1_near_back_to_front": _moving_object(
            duration,
            int(stimulus["object_width_pixels"]),
            int(stimulus["object_height_pixels"]),
            float(stimulus["near_collision_vertical_fraction"]),
            "back_to_front",
        ),
        "lplc1_near_front_to_back": _moving_object(
            duration,
            int(stimulus["object_width_pixels"]),
            int(stimulus["object_height_pixels"]),
            float(stimulus["near_collision_vertical_fraction"]),
            "front_to_back",
        ),
        "lplc1_matched_miss": _moving_object(
            duration,
            int(stimulus["object_width_pixels"]),
            int(stimulus["object_height_pixels"]),
            float(stimulus["matched_miss_vertical_fraction"]),
            "back_to_front",
        ),
        "lplc1_rotating_background": _translation(
            duration, float(stimulus["translation_period_pixels"])
        ),
        "lplc2_outward": _radial_ring(duration, start, terminal, True),
        "lplc2_inward": _radial_ring(duration, start, terminal, False),
        "lplc2_motion_free": _motion_free_darkening(duration, terminal),
        "lplc2_translation": _translation(
            duration, float(stimulus["translation_period_pixels"])
        ),
        "lc4_slow": _filled_loom(duration, terminal * 0.55, terminal),
        "lc4_medium": _filled_loom(max(8, duration // 2), terminal * 0.55, terminal),
        "lc4_fast": _filled_loom(max(4, duration // 4), terminal * 0.55, terminal),
    }
    return {
        name: VisualStimulus(
            f"{condition['condition_id']}:{name}",
            name,
            "off",
            name.rsplit("_", 1)[-1],
            _noise(frames, noise, seed + index),
        )
        for index, (name, frames) in enumerate(raw.items())
    }


def _compact(score: dict) -> dict:
    return {
        key: score[key]
        for key in (
            "cell_count",
            "valid_cell_count",
            "valid_cell_fraction",
            "median_signed_contrast",
            "positive_cell_fraction",
            "passed",
        )
    }


def _pair_score(
    probe, responses: dict, population: str, preferred: str, comparator: str, thresholds
):
    ids = probe.graph.body_ids[probe.populations[population]]
    return strict_contrast_summary(
        ids,
        responses[preferred]["peaks"][population],
        responses[comparator]["peaks"][population],
        thresholds,
    )


def _lc4_score(probe, responses: dict, population: str, thresholds: dict) -> dict:
    values = np.stack(
        [responses[name]["peaks"][population] for name in ("lc4_slow", "lc4_medium", "lc4_fast")]
    )
    velocities = np.asarray((1.0, 2.0, 4.0), dtype=np.float64)
    centered_velocity = velocities - np.mean(velocities)
    centered_values = values - np.mean(values, axis=0)
    slopes = centered_velocity @ centered_values / float(centered_velocity @ centered_velocity)
    fitted = np.mean(values, axis=0) + centered_velocity[:, None] * slopes[None, :]
    residual = np.sum((values - fitted) ** 2, axis=0)
    total = np.sum(centered_values**2, axis=0)
    valid = np.isfinite(slopes) & (total >= float(thresholds["minimum_valid_denominator"]))
    r2 = np.full(len(slopes), np.nan)
    r2[valid] = 1.0 - residual[valid] / total[valid]
    valid_fraction = float(np.mean(valid))
    positive_fraction = float(np.mean(slopes[valid] > 0)) if np.any(valid) else 0.0
    median_r2 = float(np.median(r2[valid])) if np.any(valid) else None
    passed = (
        valid_fraction >= float(thresholds["minimum_valid_cell_fraction"])
        and positive_fraction >= float(thresholds["minimum_LC4_positive_slope_fraction"])
        and median_r2 is not None
        and median_r2 >= float(thresholds["minimum_LC4_median_r_squared"])
    )
    return {
        "cell_count": len(slopes),
        "valid_cell_count": int(np.count_nonzero(valid)),
        "valid_cell_fraction": valid_fraction,
        "positive_slope_fraction": positive_fraction,
        "median_r_squared": median_r2,
        "passed": bool(passed),
    }


def evaluate_v7_lplc_typed_screen(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    if any(item["role"] != "tuning" for item in config["conditions"]):
        raise ValueError("typed LPLC screen may consume tuning conditions only")
    probe = MassBalancedVisualProbe(root, config)
    condition_scores = {}
    manifests = []
    for condition in config["conditions"]:
        stimuli = _condition_stimuli(condition, config)
        responses = {
            name: run_signed_cell_responses(probe, stimulus)
            for name, stimulus in stimuli.items()
        }
        manifests.extend(
            {
                "identity": item.name,
                "frame_sha256": hashlib.sha256(item.frames.tobytes()).hexdigest(),
                "retinal_drive_sha256": responses[name]["retinal_drive_sha256"],
            }
            for name, item in stimuli.items()
        )
        per_population = {}
        for side in ("L", "R"):
            lplc1 = f"LPLC1_{side}"
            lplc2 = f"LPLC2_{side}"
            lc4 = f"LC4_{side}"
            per_population[lplc1] = {
                "near_vs_miss": _compact(
                    _pair_score(
                        probe,
                        responses,
                        lplc1,
                        "lplc1_near_back_to_front",
                        "lplc1_matched_miss",
                        scoring["thresholds"],
                    )
                ),
                "back_to_front_vs_front_to_back": _compact(
                    _pair_score(
                        probe,
                        responses,
                        lplc1,
                        "lplc1_near_back_to_front",
                        "lplc1_near_front_to_back",
                        scoring["thresholds"],
                    )
                ),
                "object_vs_rotating_background": _compact(
                    _pair_score(
                        probe,
                        responses,
                        lplc1,
                        "lplc1_near_back_to_front",
                        "lplc1_rotating_background",
                        scoring["thresholds"],
                    )
                ),
            }
            per_population[lplc2] = {
                name: _compact(
                    _pair_score(
                        probe,
                        responses,
                        lplc2,
                        "lplc2_outward",
                        comparator,
                        scoring["thresholds"],
                    )
                )
                for name, comparator in (
                    ("outward_vs_inward", "lplc2_inward"),
                    ("outward_vs_motion_free", "lplc2_motion_free"),
                    ("outward_vs_translation", "lplc2_translation"),
                )
            }
            per_population[lc4] = {
                "angular_velocity": _lc4_score(
                    probe, responses, lc4, config["thresholds"]
                )
            }
        condition_scores[condition["condition_id"]] = per_population
    population_consistency = {}
    for population in ("LPLC1_L", "LPLC1_R", "LPLC2_L", "LPLC2_R", "LC4_L", "LC4_R"):
        mechanisms = condition_scores[config["condition_ids"][0]][population]
        population_consistency[population] = {
            mechanism: {
                "passing_condition_count": sum(
                    condition_scores[condition][population][mechanism]["passed"]
                    for condition in config["condition_ids"]
                ),
                "required_condition_count": len(config["condition_ids"]),
                "passed": all(
                    condition_scores[condition][population][mechanism]["passed"]
                    for condition in config["condition_ids"]
                ),
            }
            for mechanism in mechanisms
        }
    all_passed = all(
        result["passed"]
        for population in population_consistency.values()
        for result in population.values()
    )
    mirror = {}
    for condition in config["condition_ids"]:
        left = condition_scores[condition]
        pairs = []
        for family in ("LPLC1", "LPLC2", "LC4"):
            left_score = next(iter(left[f"{family}_L"].values()))
            right_score = next(iter(left[f"{family}_R"].values()))
            pairs.append(
                [
                    left_score.get(
                        "median_signed_contrast", left_score.get("median_r_squared", 0.0)
                    )
                    or 0.0,
                    right_score.get(
                        "median_signed_contrast", right_score.get("median_r_squared", 0.0)
                    )
                    or 0.0,
                ]
            )
        values = np.asarray(pairs, dtype=np.float64)
        maximum_error = float(np.max(np.abs(values[:, 0] - values[:, 1])))
        mirror[condition] = {
            "maximum_population_metric_error": maximum_error,
            "passed": bool(
                maximum_error <= config["thresholds"]["maximum_mirror_error"]
            ),
        }
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_ids": config["condition_ids"],
            "stimulus_count": len(manifests),
            "shared_generic_looming_rule": False,
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "stimulus_manifest": manifests,
        "per_condition_scores": condition_scores,
        "population_consistency": population_consistency,
        "mirror_summary": mirror,
        "typed_lplc_lc4_gates_passed": bool(
            all_passed and all(item["passed"] for item in mirror.values())
        ),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
