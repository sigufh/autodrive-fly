"""Fixed-grid, matched-noise LPLC2 filled-loom position diagnostic."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import (
    _build_radial_projection,
    _compact_scalar_score,
    _coverage,
    _layer_nodes,
    _layer_traces,
    _radial_traces,
    _score,
)
from fly_emotion.driving.v7_lplc_typed_screen import HEIGHT, WIDTH, _translation
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe

CONFIG = Path("configs/driving-v7-lplc2-position-coverage.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_lplc2_position_coverage.py")
SIDES = ("L", "R")


def _filled_disc(
    duration: int, start: float, terminal: float, center_x: float, center_y: float, outward: bool
) -> np.ndarray:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    radii = np.linspace(start, terminal, duration)
    if not outward:
        radii = radii[::-1]
    frames = np.full((duration, HEIGHT, WIDTH), 0.92, dtype=np.float32)
    for index, radius in enumerate(radii):
        frames[index, distance <= radius] = 0.08
    return frames


def _motion_free(duration: int, terminal: float, center_x: float, center_y: float) -> np.ndarray:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= terminal**2
    frames = np.full((duration, HEIGHT, WIDTH), 0.92, dtype=np.float32)
    for index, level in enumerate(np.linspace(0.92, 0.08, duration)):
        frames[index, mask] = level
    return frames


def _noise(shape: tuple[int, ...], standard_deviation: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, standard_deviation, shape)
    return (values + values[:, :, ::-1]) / 2.0


def _bundle(condition: dict, config: dict) -> dict[tuple[int, int], dict[str, VisualStimulus]]:
    duration = int(condition["duration_frames"])
    start = float(condition["start_radius_pixels"])
    terminal = float(condition["terminal_radius_pixels"])
    standard_deviation = float(condition["noise_standard_deviation"])
    baseline = np.full(
        (int(config["common_background_frames"]), HEIGHT, WIDTH), 0.92, dtype=np.float32
    )
    x_values = list(map(float, config["centers"]["x"]))
    y_values = list(map(float, config["centers"]["y"]))
    output = {}
    for y_index, center_y in enumerate(y_values):
        for x_index, center_x in enumerate(x_values):
            mirror_index = len(x_values) - 1 - x_index
            pair_index = min(x_index, mirror_index)
            perturbation = _noise(
                (duration, HEIGHT, WIDTH),
                standard_deviation,
                int(condition["seed"]) + y_index * 100 + pair_index,
            )
            raw = {
                "filled_expansion": _filled_disc(
                    duration, start, terminal, center_x, center_y, True
                ),
                "filled_contraction": _filled_disc(
                    duration, start, terminal, center_x, center_y, False
                ),
                "motion_free_darkening": _motion_free(duration, terminal, center_x, center_y),
            }
            position = (x_index, y_index)
            output[position] = {
                name: VisualStimulus(
                    f"{condition['condition_id']}:{name}:x{x_index}:y{y_index}",
                    name,
                    "off",
                    name,
                    np.concatenate(
                        (baseline, np.clip(frames + perturbation, 0.0, 1.0).astype(np.float32))
                    ),
                )
                for name, frames in raw.items()
            }
    return output


def _translation_pair(condition: dict, config: dict) -> dict[str, VisualStimulus]:
    duration = int(condition["duration_frames"])
    baseline = np.full(
        (int(config["common_background_frames"]), HEIGHT, WIDTH), 0.92, dtype=np.float32
    )
    forward = _translation(duration, float(config["stimuli"]["translation_period_pixels"]))
    perturbation = _noise(
        (duration, HEIGHT, WIDTH),
        float(condition["noise_standard_deviation"]),
        int(condition["seed"]) + 10_000,
    )
    first = np.clip(forward + perturbation, 0.0, 1.0).astype(np.float32)
    second = first[:, :, ::-1].copy()
    return {
        "forward": VisualStimulus(
            f"{condition['condition_id']}:wide_field_translation:forward",
            "wide_field_translation",
            "mixed",
            "forward",
            np.concatenate((baseline, first)),
        ),
        "mirrored": VisualStimulus(
            f"{condition['condition_id']}:wide_field_translation:mirrored",
            "wide_field_translation",
            "mixed",
            "mirrored",
            np.concatenate((baseline, second)),
        ),
    }


def _observability(
    body_ids: np.ndarray, outward: list[np.ndarray], inward: list[np.ndarray], thresholds: dict
) -> tuple[dict, np.ndarray]:
    per_position = []
    for preferred, comparator in zip(outward, inward, strict=True):
        scale = np.mean(np.abs(preferred), axis=0) + np.mean(np.abs(comparator), axis=0)
        values = np.full(len(body_ids), np.nan, dtype=np.float64)
        valid = np.isfinite(scale) & (scale >= float(thresholds["minimum_valid_scale"]))
        values[valid] = (
            np.mean(np.abs(preferred[:, valid] - comparator[:, valid]), axis=0) / scale[valid]
        )
        per_position.append(values)
    matrix = np.stack(per_position)
    best = np.full(len(body_ids), np.nan, dtype=np.float64)
    any_valid = np.any(np.isfinite(matrix), axis=0)
    best[any_valid] = np.nanmax(matrix[:, any_valid], axis=0)
    separated = np.isfinite(best) & (best >= float(thresholds["minimum_median_trace_separation"]))
    valid_fraction = float(np.mean(np.isfinite(best)))
    separated_fraction = float(np.mean(separated))
    median = float(np.nanmedian(best)) if np.any(np.isfinite(best)) else None
    gates = {
        "valid_cell_fraction": valid_fraction >= float(thresholds["minimum_joint_valid_fraction"]),
        "median_trace_separation": median is not None
        and median >= float(thresholds["minimum_median_trace_separation"]),
        "separated_cell_fraction": separated_fraction
        >= float(thresholds["minimum_separated_cell_fraction"]),
    }
    return (
        {
            "cell_count": int(len(body_ids)),
            "valid_cell_count": int(np.count_nonzero(np.isfinite(best))),
            "valid_cell_fraction": valid_fraction,
            "median_trace_separation": median,
            "separated_cell_count": int(np.count_nonzero(separated)),
            "separated_cell_fraction": separated_fraction,
            "gates": gates,
            "passed": all(gates.values()),
        },
        best,
    )


def evaluate_v7_lplc2_position_coverage(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    source_path = Path(config["source_protocol"])
    source_config = yaml.safe_load((root / source_path).read_text())
    typed_path = Path(config["typed_stimulus_protocol"])
    typed = yaml.safe_load((root / typed_path).read_text())
    scoring_path = Path(config["scoring_config"])
    conditions = {row["condition_id"]: row for row in typed["conditions"]}
    condition_ids = list(config["condition_ids"])
    if any(conditions[name]["role"] != "tuning" for name in condition_ids):
        raise ValueError("position coverage may consume tuning conditions only")
    probe = MassBalancedVisualProbe(root, config)
    matrices, anatomy = _build_radial_projection(root, probe, source_config)
    receptor_nodes = _layer_nodes(probe, {"populations": ["mapped_R1-R6"]})["mapped_R1-R6"]
    observations = []
    radial_scores = {
        side: {name: [] for name in config["stimuli"]["comparators"]} for side in SIDES
    }
    per_condition = {}
    mirror_summary = {}
    manifest = []
    for condition_id in condition_ids:
        bundle = _bundle(conditions[condition_id], config)
        translations = _translation_pair(conditions[condition_id], config)
        preferred_layer = []
        comparator_layer = []
        radial_by_name = {
            name: {side: [] for side in SIDES}
            for name in (config["stimuli"]["preferred"], *config["stimuli"]["comparators"])
        }
        for position in sorted(bundle):
            stimuli = bundle[position]
            layer_preferred = _layer_traces(
                probe, stimuli[config["stimuli"]["preferred"]], {"R": receptor_nodes}
            )["R"]
            layer_comparator = _layer_traces(
                probe, stimuli["filled_contraction"], {"R": receptor_nodes}
            )["R"]
            preferred_layer.append(layer_preferred)
            comparator_layer.append(layer_comparator)
            for name, stimulus in stimuli.items():
                traces, retinal_hash = _radial_traces(probe, stimulus, matrices)
                for side in SIDES:
                    radial_by_name[name][side].append(np.max(traces[side], axis=0))
                manifest.append(
                    {
                        "identity": stimulus.name,
                        "frame_sha256": hashlib.sha256(stimulus.frames.tobytes()).hexdigest(),
                        "retinal_drive_sha256": retinal_hash,
                    }
                )
        translation_responses = {}
        for direction, stimulus in translations.items():
            traces, retinal_hash = _radial_traces(probe, stimulus, matrices)
            translation_responses[direction] = {
                side: np.max(values, axis=0) for side, values in traces.items()
            }
            manifest.append(
                {
                    "identity": stimulus.name,
                    "frame_sha256": hashlib.sha256(stimulus.frames.tobytes()).hexdigest(),
                    "retinal_drive_sha256": retinal_hash,
                }
            )
        observation, raw_observation = _observability(
            probe.graph.body_ids[receptor_nodes],
            preferred_layer,
            comparator_layer,
            config["observability"],
        )
        observations.append(raw_observation)
        condition_scores = {"observability": observation, "LPLC2": {}}
        for side in SIDES:
            ids = probe.graph.body_ids[probe.populations[f"LPLC2_{side}"]]
            condition_scores["LPLC2"][side] = {}
            preferred = np.max(np.stack(radial_by_name["filled_expansion"][side]), axis=0)
            for comparator_name in config["stimuli"]["comparators"]:
                comparator = (
                    np.maximum(
                        translation_responses["forward"][side],
                        translation_responses["mirrored"][side],
                    )
                    if comparator_name == "wide_field_translation"
                    else np.max(np.stack(radial_by_name[comparator_name][side]), axis=0)
                )
                score = _score(ids, preferred, comparator, config["thresholds"])
                radial_scores[side][comparator_name].append(score)
                condition_scores["LPLC2"][side][comparator_name] = _compact_scalar_score(score)
        per_condition[condition_id] = condition_scores
        x_count = len(config["centers"]["x"])
        frame_errors = []
        for y_index in range(len(config["centers"]["y"])):
            for x_index in range((x_count + 1) // 2):
                counterpart = x_count - 1 - x_index
                first = bundle[(x_index, y_index)]["filled_expansion"]
                second = bundle[(counterpart, y_index)]["filled_expansion"]
                frame_errors.append(float(np.max(np.abs(first.frames[:, :, ::-1] - second.frames))))
        mirror_summary[condition_id] = {
            "maximum_position_pair_frame_error": float(max(frame_errors)),
            "translation_pair_frame_error": float(
                np.max(
                    np.abs(
                        translations["forward"].frames[:, :, ::-1] - translations["mirrored"].frames
                    )
                )
            ),
        }
        mirror_summary[condition_id]["passed"] = bool(
            mirror_summary[condition_id]["maximum_position_pair_frame_error"] <= 1e-7
            and mirror_summary[condition_id]["translation_pair_frame_error"] <= 1e-7
        )
        metric_errors = {}
        for comparator_name in config["stimuli"]["comparators"]:
            left = condition_scores["LPLC2"]["L"][comparator_name]["median_signed_contrast"]
            right = condition_scores["LPLC2"]["R"][comparator_name]["median_signed_contrast"]
            metric_errors[comparator_name] = (
                abs(left - right) if left is not None and right is not None else None
            )
        finite_metric_errors = [value for value in metric_errors.values() if value is not None]
        maximum_metric_error = (
            max(finite_metric_errors) if len(finite_metric_errors) == len(metric_errors) else None
        )
        mirror_summary[condition_id]["per_comparison_population_metric_error"] = metric_errors
        mirror_summary[condition_id]["maximum_population_metric_error"] = maximum_metric_error
        mirror_summary[condition_id]["passed"] = bool(
            mirror_summary[condition_id]["passed"]
            and maximum_metric_error is not None
            and maximum_metric_error <= float(config["thresholds"]["maximum_mirror_error"])
        )
    observation_matrix = np.stack(observations)
    observation_joint_valid = np.all(np.isfinite(observation_matrix), axis=0)
    observation_all_success = np.all(
        np.isfinite(observation_matrix)
        & (observation_matrix >= float(config["observability"]["minimum_median_trace_separation"])),
        axis=0,
    )
    observation_consistency = {
        "cell_count": int(len(receptor_nodes)),
        "passing_condition_count": int(
            sum(per_condition[name]["observability"]["passed"] for name in condition_ids)
        ),
        "joint_valid_fraction": float(np.mean(observation_joint_valid)),
        "all_condition_separated_count": int(np.count_nonzero(observation_all_success)),
        "all_condition_separated_fraction": float(np.mean(observation_all_success)),
    }
    observation_consistency["passed"] = bool(
        observation_consistency["passing_condition_count"] == len(condition_ids)
        and observation_consistency["joint_valid_fraction"]
        >= float(config["observability"]["minimum_joint_valid_fraction"])
        and observation_consistency["all_condition_separated_fraction"]
        >= float(config["observability"]["minimum_all_condition_separated_fraction"])
    )
    radial_consistency = {
        side: {
            name: {
                "passing_condition_count": int(sum(score["passed"] for score in values)),
                "required_condition_count": len(values),
                "coverage": _coverage(values, config["thresholds"]),
                "passed": bool(
                    all(score["passed"] for score in values)
                    and _coverage(values, config["thresholds"])["passed"]
                ),
            }
            for name, values in mechanisms.items()
        }
        for side, mechanisms in radial_scores.items()
    }
    radial_passed = all(
        result["passed"]
        for mechanisms in radial_consistency.values()
        for result in mechanisms.values()
    )
    mirror_passed = all(item["passed"] for item in mirror_summary.values())
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(source_path): _sha256(root / source_path),
                str(typed_path): _sha256(root / typed_path),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_ids": condition_ids,
            "position_count": len(config["centers"]["x"]) * len(config["centers"]["y"]),
            "stimulus_count": len(manifest),
            "matched_noise_within_position": True,
            "common_background_frames": int(config["common_background_frames"]),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "candidate": {
            "centers": config["centers"],
            "preferred": config["stimuli"]["preferred"],
            "comparators": config["stimuli"]["comparators"],
            "position_grid_selection_uses_target_or_response": False,
            "per_target_pooling": "maximum over every fixed grid position",
            "observability_is_functional_preference": False,
        },
        "anatomy_target_counts": {
            side: anatomy["populations"][side]["target_count"] for side in SIDES
        },
        "stimulus_manifest": manifest,
        "per_condition": per_condition,
        "observability_consistency": observation_consistency,
        "radial_population_consistency": radial_consistency,
        "mirror_summary": mirror_summary,
        "observability_gate_passed": observation_consistency["passed"],
        "LPLC2_position_coverage_gates_passed": bool(
            observation_consistency["passed"] and radial_passed and mirror_passed
        ),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
