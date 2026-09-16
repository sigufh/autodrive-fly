"""Single-condition stop/go precheck for anatomy-local T4/T5 edge stimuli."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import (
    HORIZONTAL_PREFERENCE,
    VERTICAL_PREFERENCE,
    VisualStimulus,
)
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import (
    _infer_t4_t5_positions,
    _layer_traces,
)
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary

CONFIG = Path("configs/driving-v7-t4t5-local-edge-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4t5_local_edge_precheck.py")
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
WIDTH = 48
HEIGHT = 24


def _local_edge(
    duration: int,
    center_x: float,
    center_y: float,
    radius: float,
    half_width: float,
    polarity: str,
    direction: str,
    baseline_frames: int,
) -> VisualStimulus:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    aperture = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    background, foreground = (0.08, 0.92) if polarity == "on" else (0.92, 0.08)
    coordinate = xx if direction in {"left", "right"} else yy
    center = center_x if direction in {"left", "right"} else center_y
    positions = np.linspace(center - radius, center + radius, duration)
    if direction in {"left", "up"}:
        positions = positions[::-1]
    frames = np.full((duration, HEIGHT, WIDTH), 0.5, dtype=np.float32)
    for index, position in enumerate(positions):
        frames[index, aperture] = background
        frames[index, aperture & (np.abs(coordinate - position) <= half_width)] = foreground
    common = np.full((baseline_frames, HEIGHT, WIDTH), 0.5, dtype=np.float32)
    return VisualStimulus(
        f"local-edge:{polarity}:{direction}:x={center_x:g}:y={center_y:g}",
        "local_moving_edge",
        polarity,
        direction,
        np.concatenate((common, frames)),
    )


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
            "gates",
            "passed",
        )
    }


def evaluate_v7_t4t5_local_edge_precheck(
    root: Path, *, retinal_backend: str | None = None, dynamics_backend: str | None = None
) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    if retinal_backend is not None:
        config["retinal_backend"] = retinal_backend
    if dynamics_backend is not None:
        config["dynamics_backend"] = dynamics_backend
    source_path = Path(config["source_protocol"])
    source = yaml.safe_load((root / source_path).read_text())
    typed_path = Path(config["typed_stimulus_protocol"])
    typed = yaml.safe_load((root / typed_path).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    condition = next(
        row for row in typed["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("local edge precheck may consume tuning only")
    if float(config["stimulus"]["noise_standard_deviation"]) != 0.0:
        raise ValueError("local edge precheck is frozen to zero noise")
    probe = MassBalancedVisualProbe(root, config)
    positions, position_metadata = _infer_t4_t5_positions(root, probe, source)
    finite_positions = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = finite_positions.min(axis=0), finite_positions.max(axis=0)
    x_centers = np.asarray(config["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(config["centers"]["y"], dtype=np.float64)
    populations = {
        f"{family}{subtype}_{side}": probe.populations[f"{family}{subtype}_{side}"]
        for family in ("T4", "T5")
        for subtype in "abcd"
        for side in "LR"
    }
    assignments = {}
    assignment_summary = {}
    for name, nodes in populations.items():
        mapped = np.all(np.isfinite(positions[nodes]), axis=1)
        camera = np.full((len(nodes), 2), np.nan, dtype=np.float64)
        camera[mapped] = (
            (positions[nodes[mapped]] - low)
            / np.maximum(high - low, 1e-12)
            * (WIDTH - 1, HEIGHT - 1)
        )
        indices = np.full((len(nodes), 2), -1, dtype=np.int32)
        indices[mapped, 0] = np.argmin(np.abs(x_centers[:, None] - camera[mapped, 0]), axis=0)
        indices[mapped, 1] = np.argmin(np.abs(y_centers[:, None] - camera[mapped, 1]), axis=0)
        assignments[name] = indices
        assignment_summary[name] = {
            "cell_count": int(len(nodes)),
            "mapped_count": int(np.count_nonzero(mapped)),
            "unmapped_count": int(np.count_nonzero(~mapped)),
        }
    responses = {}
    duration = int(condition["duration_frames"])
    for x_index, center_x in enumerate(x_centers):
        for y_index, center_y in enumerate(y_centers):
            for polarity in ("on", "off"):
                for direction in ("left", "right", "up", "down"):
                    stimulus = _local_edge(
                        duration,
                        float(center_x),
                        float(center_y),
                        float(config["stimulus"]["aperture_radius_pixels"]),
                        float(config["stimulus"]["bar_half_width_pixels"]),
                        polarity,
                        direction,
                        int(config["common_background_frames"]),
                    )
                    traces = _layer_traces(probe, stimulus, populations)
                    responses[(x_index, y_index, polarity, direction)] = {
                        name: np.max(values, axis=0) for name, values in traces.items()
                    }
    results = {}
    for name, nodes in populations.items():
        family, subtype, side = name[:2], name[2], name[-1]
        preferred_direction = (
            HORIZONTAL_PREFERENCE[(subtype, side)]
            if subtype in {"a", "b"}
            else VERTICAL_PREFERENCE[subtype]
        )
        expected_polarity = "on" if family == "T4" else "off"
        other_polarity = "off" if expected_polarity == "on" else "on"
        indices = assignments[name]
        columns = np.arange(len(nodes))

        def gather(
            polarity: str,
            direction: str,
            *,
            population_nodes: np.ndarray = nodes,
            population_indices: np.ndarray = indices,
            population_name: str = name,
            population_columns: np.ndarray = columns,
        ) -> np.ndarray:
            output = np.full(len(population_nodes), np.nan, dtype=np.float64)
            mapped = np.all(population_indices >= 0, axis=1)
            grid = np.stack(
                [
                    responses[(x, y, polarity, direction)][population_name]
                    for y in range(len(y_centers))
                    for x in range(len(x_centers))
                ]
            )
            rows = population_indices[mapped, 1] * len(x_centers) + population_indices[mapped, 0]
            output[mapped] = grid[rows, population_columns[mapped]]
            return output

        preferred = gather(expected_polarity, preferred_direction)
        direction_score = strict_contrast_summary(
            probe.graph.body_ids[nodes],
            preferred,
            gather(expected_polarity, OPPOSITE[preferred_direction]),
            scoring["thresholds"],
        )
        polarity_score = strict_contrast_summary(
            probe.graph.body_ids[nodes],
            preferred,
            gather(other_polarity, preferred_direction),
            scoring["thresholds"],
        )
        results[name] = {
            "expected_direction": preferred_direction,
            "expected_polarity": expected_polarity,
            "direction": _compact(direction_score),
            "polarity": _compact(polarity_score),
        }
    direction_pass_count = sum(item["direction"]["passed"] for item in results.values())
    polarity_pass_count = sum(item["polarity"]["passed"] for item in results.values())
    expand = direction_pass_count >= int(
        config["precheck_gate"]["minimum_direction_populations_to_expand"]
    )
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
            "condition_id": config["condition_id"],
            "position_count": len(x_centers) * len(y_centers),
            "stimulus_count": len(responses),
            "noise_standard_deviation": 0.0,
            "retinal_backend": config["retinal_backend"],
            "dynamics_backend": config["dynamics_backend"],
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "assignment": {
            "source_position_metadata": position_metadata,
            "global_source_bounds": {"low": low.tolist(), "high": high.tolist()},
            "by_population": assignment_summary,
            "target_response_used_for_assignment": False,
            "unmapped_cells_retained_as_invalid": True,
        },
        "population_scores": results,
        "summary": {
            "direction_pass_count": int(direction_pass_count),
            "direction_population_count": len(results),
            "polarity_pass_count": int(polarity_pass_count),
            "polarity_population_count": len(results),
            "minimum_direction_populations_to_expand": int(
                config["precheck_gate"]["minimum_direction_populations_to_expand"]
            ),
            "expand_to_three_conditions": bool(expand),
        },
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
