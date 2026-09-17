"""Three-hop structural ON/OFF moment-Reichardt diagnostic for T4/T5."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.driving.v7 import (
    HORIZONTAL_PREFERENCE,
    VERTICAL_PREFERENCE,
    VisualStimulus,
)
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    CONFIG as LOCAL_EDGE_CONFIG,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    IMPLEMENTATION as LOCAL_EDGE_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    OPPOSITE,
)

CONFIG = Path("configs/driving-v7-three-hop-moment.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_three_hop_moment.py")
WIDTH = 48
HEIGHT = 24


def _local_step_edge(
    center_x: float,
    center_y: float,
    radius: float,
    speed: float,
    polarity: str,
    direction: str,
    baseline_frames: int,
) -> VisualStimulus:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    aperture = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    background, foreground = (0.08, 0.92) if polarity == "on" else (0.92, 0.08)
    coordinate = xx if direction in {"left", "right"} else yy
    center = center_x if direction in {"left", "right"} else center_y
    steps = int(np.ceil(2.0 * radius / speed)) + 1
    positions = np.minimum(center - radius + np.arange(steps) * speed, center + radius)
    if direction in {"left", "up"}:
        positions = positions[::-1]
    frames = np.full((steps, HEIGHT, WIDTH), 0.5, dtype=np.float32)
    for index, position in enumerate(positions):
        frames[index, aperture] = background
        behind = (
            coordinate <= position if direction in {"right", "down"} else coordinate >= position
        )
        frames[index, aperture & behind] = foreground
    common = np.full((baseline_frames, HEIGHT, WIDTH), 0.5, dtype=np.float32)
    return VisualStimulus(
        f"local-step:{polarity}:{direction}:x={center_x:g}:y={center_y:g}:speed={speed:g}",
        "local_moving_step_edge",
        polarity,
        direction,
        np.concatenate((common, frames)),
    )


def _path_matrix(
    probe, targets: np.ndarray, sources: tuple[str, ...]
) -> tuple[sparse.csr_matrix, dict]:
    source_nodes = np.flatnonzero(np.isin(probe.node_types, sources))
    receptors = probe.retina.node_indices
    matrix = (
        probe.adjacency[targets, :][:, source_nodes]
        @ probe.adjacency[source_nodes, :]
        @ probe.adjacency[:, receptors]
    ).tocsr()
    matrix.eliminate_zeros()
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.zeros_like(totals, dtype=np.float64)
    np.divide(1.0, totals, out=scale, where=totals > 0.0)
    normalized = (sparse.diags(scale) @ matrix).tocsr()
    return normalized, {
        "known_source_cell_count": int(len(source_nodes)),
        "path_nonzero_count": int(matrix.nnz),
        "reachable_target_count": int(np.count_nonzero(totals > 0.0)),
        "reachable_target_fraction": float(np.mean(totals > 0.0)),
    }


def _energy(
    probe, stimulus, matrix: sparse.csr_matrix, x: np.ndarray, y: np.ndarray, channel: str
) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack(
        [probe._sample_retina(image)[probe.retinal_permutation] for image in stimulus.frames]
    )
    delta = np.diff(values, axis=0, prepend=values[:1])
    signal = np.maximum(delta if channel == "on" else -delta, 0.0)
    signal *= probe.mass_gain[None]
    mass = (matrix @ signal.T).T
    moment_x = (matrix @ (signal * x).T).T
    moment_y = (matrix @ (signal * y).T).T
    previous_mass = np.concatenate((np.zeros_like(mass[:1]), mass[:-1]))
    previous_x = np.concatenate((np.zeros_like(moment_x[:1]), moment_x[:-1]))
    previous_y = np.concatenate((np.zeros_like(moment_y[:1]), moment_y[:-1]))
    forward_x = moment_x * previous_mass
    reverse_x = mass * previous_x
    forward_y = moment_y * previous_mass
    reverse_y = mass * previous_y
    denominator = (
        np.abs(forward_x) + np.abs(reverse_x) + np.abs(forward_y) + np.abs(reverse_y) + 1e-9
    )
    return (forward_x - reverse_x) / denominator, (forward_y - reverse_y) / denominator


def _peak(
    values: list[tuple[np.ndarray, np.ndarray]],
    vector: tuple[float, float],
    temporal_reduction: str = "maximum",
) -> np.ndarray:
    by_position = []
    for x, y in values:
        projected = vector[0] * x + vector[1] * y
        if temporal_reduction == "maximum":
            by_position.append(np.max(projected, axis=0))
        elif temporal_reduction == "signed_mean":
            by_position.append(np.mean(projected, axis=0))
        else:
            raise ValueError(f"unknown temporal reduction: {temporal_reduction}")
    stacked = np.stack(by_position)
    return np.max(stacked, axis=0) if temporal_reduction == "maximum" else np.mean(stacked, axis=0)


def _compact(score: dict) -> dict:
    return {
        key: score[key]
        for key in (
            "cell_count",
            "valid_cell_count",
            "valid_cell_fraction",
            "median_signed_contrast",
            "positive_cell_fraction",
            "invalid_cell_ids",
            "cell_ids",
            "contrast",
            "gates",
            "passed",
        )
    }


def _coverage(scores: list[dict], minimum: float) -> dict:
    ids = np.asarray(scores[0]["cell_ids"], dtype=np.int64)
    values = np.stack(
        [
            np.asarray([x if x is not None else np.nan for x in score["contrast"]])
            for score in scores
        ]
    )
    joint = np.all(np.isfinite(values), axis=0)
    success = np.all(np.isfinite(values) & (values >= minimum), axis=0)
    return {
        "cell_count": int(len(ids)),
        "joint_valid_fraction": float(np.mean(joint)),
        "all_condition_success_count": int(np.count_nonzero(success)),
        "all_condition_success_fraction": float(np.mean(success)),
    }


def _score_energies(
    probe,
    populations: dict,
    family_data: dict,
    energies: dict,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    scoring: dict,
    temporal_reduction: str = "maximum",
) -> tuple[dict, dict]:
    condition_scores = {}
    raw = {}
    for name, nodes in populations.items():
        family, subtype, side = name[:2], name[2], name[-1]
        data = family_data[family]
        group = np.flatnonzero(data["population"] == name)
        expected_direction = (
            HORIZONTAL_PREFERENCE[(subtype, side)]
            if subtype in {"a", "b"}
            else VERTICAL_PREFERENCE[subtype]
        )
        expected_polarity = "on" if family == "T4" else "off"
        other_polarity = "off" if expected_polarity == "on" else "on"
        vector = {
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
            "up": (0.0, -1.0),
            "down": (0.0, 1.0),
        }[expected_direction]

        def response(
            stimulus_polarity: str,
            direction: str,
            *,
            selected_group: np.ndarray = group,
            selected_family: str = family,
            selected_vector: tuple[float, float] = vector,
        ) -> np.ndarray:
            values = [
                tuple(
                    component[:, selected_group]
                    for component in energies[selected_family][
                        (xi, yi, stimulus_polarity, direction)
                    ]
                )
                for yi in range(len(y_centers))
                for xi in range(len(x_centers))
            ]
            return _peak(values, selected_vector, temporal_reduction)

        preferred = response(expected_polarity, expected_direction)
        direction_score = strict_contrast_summary(
            probe.graph.body_ids[nodes],
            preferred,
            response(expected_polarity, OPPOSITE[expected_direction]),
            scoring["thresholds"],
        )
        polarity_score = strict_contrast_summary(
            probe.graph.body_ids[nodes],
            preferred,
            response(other_polarity, expected_direction),
            scoring["thresholds"],
        )
        raw[name] = {"direction": direction_score, "polarity": polarity_score}
        condition_scores[name] = {
            "direction": _compact(direction_score),
            "polarity": _compact(polarity_score),
        }
    return condition_scores, raw


def evaluate_v7_three_hop_moment(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    local = yaml.safe_load((root / LOCAL_EDGE_CONFIG).read_text())
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    conditions = {row["condition_id"]: row for row in stage1["conditions"]}
    if any(conditions[name]["role"] != "tuning" for name in config["condition_ids"]):
        raise ValueError("three-hop moment diagnostic may consume tuning only")
    probe = MassBalancedVisualProbe(root, local)
    temporal_reduction = (
        "signed_mean"
        if config["motion_energy"]["temporal_reduction"]
        == "signed_mean_over_time_and_fixed_position_grid"
        else "maximum"
    )
    populations = {
        f"{family}{subtype}_{side}": probe.populations[f"{family}{subtype}_{side}"]
        for family in ("T4", "T5")
        for subtype in "abcd"
        for side in "LR"
    }
    x = probe.retinal_u.astype(np.float64) * 47.0
    y = probe.retinal_v.astype(np.float64) * 23.0
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    family_data = {}
    for family in ("T4", "T5"):
        family_names = [name for name in populations if name.startswith(family)]
        targets = np.concatenate([populations[name] for name in family_names]).astype(np.int32)
        target_population = np.concatenate(
            [np.full(len(populations[name]), name) for name in family_names]
        )
        matrix, path = _path_matrix(probe, targets, tuple(config["known_final_sources"][family]))
        family_data[family] = {
            "names": family_names,
            "targets": targets,
            "population": target_population,
            "matrix": matrix,
            "path": path,
        }
    per_condition = {}
    controls = {}
    raw_scores = {name: {"direction": [], "polarity": []} for name in populations}
    stimulus_hashes = []
    for condition_id in config["condition_ids"]:
        parameters = conditions[condition_id]["generator_parameters"]
        speed = float(parameters["edge_speed_pixels_per_frame"])
        energies = {family: {} for family in ("T4", "T5")}
        for xi, cx in enumerate(x_centers):
            for yi, cy in enumerate(y_centers):
                for polarity in ("on", "off"):
                    for direction in ("left", "right", "up", "down"):
                        stimulus = _local_step_edge(
                            float(cx),
                            float(cy),
                            float(config["stimulus"]["aperture_radius_pixels"]),
                            speed,
                            polarity,
                            direction,
                            int(config["stimulus"]["common_background_frames"]),
                        )
                        stimulus_hashes.append(
                            hashlib.sha256(stimulus.frames.tobytes()).hexdigest()
                        )
                        for family, channel in (("T4", "on"), ("T5", "off")):
                            energies[family][(xi, yi, polarity, direction)] = _energy(
                                probe, stimulus, family_data[family]["matrix"], x, y, channel
                            )
        condition_scores, condition_raw = _score_energies(
            probe,
            populations,
            family_data,
            energies,
            x_centers,
            y_centers,
            scoring,
            temporal_reduction,
        )
        for name in populations:
            raw_scores[name]["direction"].append(condition_raw[name]["direction"])
            raw_scores[name]["polarity"].append(condition_raw[name]["polarity"])
        per_condition[condition_id] = condition_scores
        if condition_id == config["controls"]["temporal_and_static_sham_condition"]:
            control_energies = {
                name: {family: {} for family in ("T4", "T5")}
                for name in ("temporal_shuffle", "static_sham")
            }
            for xi, cx in enumerate(x_centers):
                for yi, cy in enumerate(y_centers):
                    for polarity in ("on", "off"):
                        for direction in ("left", "right", "up", "down"):
                            stimulus = _local_step_edge(
                                float(cx),
                                float(cy),
                                float(config["stimulus"]["aperture_radius_pixels"]),
                                speed,
                                polarity,
                                direction,
                                int(config["stimulus"]["common_background_frames"]),
                            )
                            baseline_count = int(config["stimulus"]["common_background_frames"])
                            prefix = stimulus.frames[:baseline_count]
                            moving = stimulus.frames[baseline_count:]
                            order = np.random.default_rng(
                                int(config["controls"]["temporal_shuffle_seed"])
                                + xi * 1000
                                + yi * 100
                            ).permutation(len(moving))
                            variants = {
                                "temporal_shuffle": np.concatenate((prefix, moving[order])),
                                "static_sham": np.concatenate(
                                    (prefix, np.repeat(moving[:1], len(moving), axis=0))
                                ),
                            }
                            for control_name, frames in variants.items():
                                sham = VisualStimulus(
                                    stimulus.name + f":{control_name}",
                                    stimulus.family,
                                    polarity,
                                    direction,
                                    frames,
                                )
                                for family, channel in (("T4", "on"), ("T5", "off")):
                                    control_energies[control_name][family][
                                        (xi, yi, polarity, direction)
                                    ] = _energy(
                                        probe,
                                        sham,
                                        family_data[family]["matrix"],
                                        x,
                                        y,
                                        channel,
                                    )
            for control_name, energy in control_energies.items():
                scores, _ = _score_energies(
                    probe,
                    populations,
                    family_data,
                    energy,
                    x_centers,
                    y_centers,
                    scoring,
                    temporal_reduction,
                )
                controls[control_name] = {
                    "condition_id": condition_id,
                    "direction_pass_count": int(
                        sum(value["direction"]["passed"] for value in scores.values())
                    ),
                    "polarity_pass_count": int(
                        sum(value["polarity"]["passed"] for value in scores.values())
                    ),
                    "passed_as_failure_control": not any(
                        value["direction"]["passed"] for value in scores.values()
                    ),
                }
    consistency = {}
    for name, heads in raw_scores.items():
        consistency[name] = {}
        for head, scores in heads.items():
            coverage = _coverage(
                scores, float(config["thresholds"]["minimum_median_signed_contrast"])
            )
            consistency[name][head] = {
                "passing_condition_count": int(sum(score["passed"] for score in scores)),
                "required_condition_count": len(scores),
                "coverage": coverage,
                "passed": bool(
                    all(score["passed"] for score in scores)
                    and coverage["joint_valid_fraction"]
                    >= float(config["thresholds"]["minimum_joint_valid_fraction"])
                    and coverage["all_condition_success_fraction"]
                    >= float(config["thresholds"]["minimum_all_condition_success_fraction"])
                ),
            }
    all_passed = all(
        result["passed"] for heads in consistency.values() for result in heads.values()
    )
    temporal_shuffle_failed = controls["temporal_shuffle"]["passed_as_failure_control"]
    static_failed = controls["static_sham"]["passed_as_failure_control"]
    controls["coordinate_shuffle"] = {
        "evaluated": False,
        "reason": "not run because temporal-shuffle prerequisite failed",
    }
    controls["direction_reversal"] = {
        "evaluated": False,
        "reason": "not run because temporal-shuffle prerequisite failed",
    }
    bilateral_direction = [
        f"{family}{subtype}"
        for family in ("T4", "T5")
        for subtype in "abcd"
        if consistency[f"{family}{subtype}_L"]["direction"]["passed"]
        and consistency[f"{family}{subtype}_R"]["direction"]["passed"]
    ]
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(LOCAL_EDGE_CONFIG): _sha256(root / LOCAL_EDGE_CONFIG),
                str(LOCAL_EDGE_IMPLEMENTATION): _sha256(root / LOCAL_EDGE_IMPLEMENTATION),
                str(stage1_path): _sha256(root / stage1_path),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_ids": config["condition_ids"],
            "stimulus_count": len(set(stimulus_hashes)),
            "temporal_reduction": temporal_reduction,
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "path_summary": {family: data["path"] for family, data in family_data.items()},
        "per_condition": per_condition,
        "population_consistency": consistency,
        "controls": controls,
        "temporal_shuffle_control_passed": temporal_shuffle_failed,
        "static_sham_control_passed": static_failed,
        "bilateral_direction_populations": bilateral_direction,
        "interpretation": {
            "diagnostic_output": "target-specific two-dimensional path-weighted optical flow",
            "subtype_label_used_for_final_projection": True,
            "subtype_label_used_for_activity_generation": False,
            "may_validate_target_cell_direction_selectivity": False,
            "main_score_is_not_a_T4_T5_scalar_activity_response": True,
        },
        "strict_three_hop_gates_passed": bool(
            all_passed and temporal_shuffle_failed and static_failed
        ),
        "advance_to_target_dynamics": bool(
            all_passed and temporal_shuffle_failed and static_failed
        ),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
