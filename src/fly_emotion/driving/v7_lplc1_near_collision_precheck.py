"""Matched-control LPLC1 near-collision stimulus and response precheck."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import (
    _infer_t4_t5_positions,
    _layer_traces,
)
from fly_emotion.driving.v7_lplc_typed_screen import HEIGHT, WIDTH
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t5_lamina_split import (
    IMPLEMENTATION as LAMINA_SPLIT_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe

CONFIG = Path("configs/driving-v7-lplc1-near-collision-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_lplc1_near_collision_precheck.py")
RAW_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_raw.npz")
BODY_IDS = Path("data/processed/malecns-v1.0/body_ids.npy")
ANNOTATIONS = Path("data/raw/malecns-v1.0/body-annotations.feather")


def _object_frames(
    duration: int,
    stimulus: dict,
    side: str,
    vertical_fraction: float,
    background: np.ndarray | None = None,
) -> np.ndarray:
    if background is None:
        frames = np.full((duration, HEIGHT, WIDTH), stimulus["bright_level"], dtype=np.float32)
    else:
        frames = background.copy()
    half = (WIDTH - 1) / 2.0
    start = float(stimulus["start_margin_pixels"])
    terminal = half - float(stimulus["terminal_center_offset_pixels"])
    x_positions = np.linspace(start, terminal, duration)
    if side == "R":
        x_positions = (WIDTH - 1) - x_positions
    center_y = float(round(float(vertical_fraction) * (HEIGHT - 1)))
    widths = np.linspace(
        stimulus["start_width_pixels"], stimulus["terminal_width_pixels"], duration
    )
    heights = np.linspace(
        stimulus["start_height_pixels"], stimulus["terminal_height_pixels"], duration
    )
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    for index, (center_x, width, height) in enumerate(
        zip(x_positions, widths, heights, strict=True)
    ):
        mask = (np.abs(xx - center_x) <= width / 2.0) & (
            np.abs(yy - center_y) <= height / 2.0
        )
        frames[index, mask] = float(stimulus["dark_level"])
    return frames


def _moving_background(duration: int, stimulus: dict, side: str) -> np.ndarray:
    _, xx = np.mgrid[:HEIGHT, :WIDTH]
    sign = 1.0 if side == "L" else -1.0
    phases = np.linspace(
        0.0, 2.0 * np.pi * float(stimulus["background_cycles"]), duration, endpoint=False
    )
    return np.stack(
        [
            np.broadcast_to(
                0.5
                + float(stimulus["background_contrast"])
                * np.sin(
                    2.0 * np.pi * xx / float(stimulus["background_period_pixels"])
                    - sign * phase
                ),
                (HEIGHT, WIDTH),
            ).astype(np.float32)
            for phase in phases
        ]
    )


def _stimuli(condition: dict, config: dict) -> tuple[dict[str, VisualStimulus], dict]:
    duration = int(condition["duration_frames"])
    stimulus = config["stimulus"]
    rng = np.random.default_rng(int(condition["seed"]))
    noise = rng.normal(
        0.0, float(condition["noise_standard_deviation"]), (duration, HEIGHT, WIDTH)
    )
    noise = (noise + noise[:, :, ::-1]) / 2.0
    output = {}
    geometry = {}
    left_background = _moving_background(duration, stimulus, "L")
    for side in "LR":
        near = _object_frames(
            duration, stimulus, side, float(stimulus["near_collision_vertical_fraction"])
        )
        miss = _object_frames(
            duration, stimulus, side, float(stimulus["matched_miss_vertical_fraction"])
        )
        background = left_background if side == "L" else left_background[:, :, ::-1]
        stationary_background = np.repeat(background[:1], duration, axis=0)
        with_stationary_background = _object_frames(
            duration,
            stimulus,
            side,
            float(stimulus["near_collision_vertical_fraction"]),
            stationary_background,
        )
        with_background = _object_frames(
            duration,
            stimulus,
            side,
            float(stimulus["near_collision_vertical_fraction"]),
            background,
        )
        bases = {
            "near_approach": near,
            "matched_miss": miss,
            "near_recede": near[::-1],
            "near_stationary_background": with_stationary_background,
            "near_rotating_background": with_background,
        }
        side_noise = noise if side == "L" else noise[:, :, ::-1]
        baseline = np.full(
            (int(config["baseline_frames"]), HEIGHT, WIDTH), 0.5, dtype=np.float32
        )
        for name, frames in bases.items():
            final = np.clip(frames + side_noise, 0.0, 1.0).astype(np.float32)
            output[f"{side}:{name}"] = VisualStimulus(
                f"{condition['condition_id']}:{side}:{name}",
                "lplc1_near_collision",
                "off",
                name,
                np.concatenate((baseline, final)),
            )
        geometry[side] = {
            "approach_recede_time_reverse_error": float(
                np.max(np.abs(near - bases["near_recede"][::-1]))
            ),
            "near_miss_equal_dark_pixel_count_per_frame": bool(
                np.array_equal(
                    np.sum(near < 0.5, axis=(1, 2)), np.sum(miss < 0.5, axis=(1, 2))
                )
            ),
        }
    for name in (
        "near_approach",
        "matched_miss",
        "near_recede",
        "near_stationary_background",
        "near_rotating_background",
    ):
        geometry[name] = {
            "left_right_frame_mirror_error": float(
                np.max(
                    np.abs(
                        output[f"L:{name}"].frames[:, :, ::-1]
                        - output[f"R:{name}"].frames
                    )
                )
            )
        }
    return output, geometry


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


def evaluate_v7_lplc1_near_collision_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    typed_path = Path(config["typed_protocol"])
    typed = yaml.safe_load((root / typed_path).read_text())
    lamina_path = Path(config["lamina_split_protocol"])
    lamina = yaml.safe_load((root / lamina_path).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    condition = next(
        item for item in typed["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("LPLC1 precheck may consume tuning only")
    probe = LaminaSplitProbe(root, lamina)
    populations = {side: probe.populations[f"LPLC1_{side}"] for side in "LR"}
    source_path = Path(lamina["source_protocol"])
    source_config = yaml.safe_load((root / source_path).read_text())
    source_positions, _ = _infer_t4_t5_positions(root, probe, source_config)
    raw = load_graph(root / "data/processed/malecns-v1.0", normalized=False).adjacency
    t4_t5_types = [f"{family}{subtype}" for family in ("T4", "T5") for subtype in "abcd"]
    source_coverage = {}
    for side, nodes in populations.items():
        target_records = []
        for target in nodes:
            row = raw.getrow(int(target))
            sources = row.indices
            weights = np.abs(row.data).astype(np.float64)
            direct = np.isin(probe.node_types[sources], t4_t5_types)
            located = direct & np.all(np.isfinite(source_positions[sources]), axis=1)
            target_records.append(
                {
                    "body_id": int(probe.graph.body_ids[target]),
                    "direct_T4_T5_source_count": int(np.count_nonzero(direct)),
                    "located_direct_T4_T5_source_count": int(np.count_nonzero(located)),
                    "direct_T4_T5_weight": float(np.sum(weights[direct])),
                    "located_direct_T4_T5_weight": float(np.sum(weights[located])),
                }
            )
        target_count = len(target_records)
        with_direct = sum(item["direct_T4_T5_source_count"] > 0 for item in target_records)
        all_located = sum(
            item["direct_T4_T5_source_count"] > 0
            and item["direct_T4_T5_source_count"]
            == item["located_direct_T4_T5_source_count"]
            for item in target_records
        )
        total_weight = sum(item["direct_T4_T5_weight"] for item in target_records)
        located_weight = sum(item["located_direct_T4_T5_weight"] for item in target_records)
        source_coverage[side] = {
            "target_count": target_count,
            "targets_with_direct_T4_T5_count": int(with_direct),
            "targets_with_direct_T4_T5_fraction": float(with_direct / target_count),
            "targets_with_all_direct_T4_T5_located_count": int(all_located),
            "targets_with_all_direct_T4_T5_located_fraction": float(all_located / target_count),
            "direct_T4_T5_source_count_median": float(
                np.median([item["direct_T4_T5_source_count"] for item in target_records])
            ),
            "located_direct_T4_T5_weight_fraction": (
                float(located_weight / total_weight) if total_weight > 0 else 0.0
            ),
            "target_records": target_records,
        }
    stimuli, geometry = _stimuli(condition, config)
    responses = {}
    manifest = []
    for name, stimulus in stimuli.items():
        traces = _layer_traces(
            probe, stimulus, {f"LPLC1_{side}": nodes for side, nodes in populations.items()}
        )
        responses[name] = {
            population: np.max(trace, axis=0) for population, trace in traces.items()
        }
        manifest.append(
            {
                "identity": stimulus.name,
                "frame_sha256": hashlib.sha256(stimulus.frames.tobytes()).hexdigest(),
            }
        )
    comparisons = {
        "near_vs_miss": ("near_approach", "matched_miss"),
        "approach_vs_recede": ("near_approach", "near_recede"),
        "stationary_vs_rotating_background": (
            "near_stationary_background",
            "near_rotating_background",
        ),
    }
    per_side = {}
    for side, nodes in populations.items():
        population = f"LPLC1_{side}"
        per_side[side] = {
            name: _compact(
                strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    responses[f"{side}:{pair[0]}"][population],
                    responses[f"{side}:{pair[1]}"][population],
                    scoring["thresholds"],
                )
            )
            for name, pair in comparisons.items()
        }
    mirror = {}
    for name in comparisons:
        left = per_side["L"][name]["median_signed_contrast"]
        right = per_side["R"][name]["median_signed_contrast"]
        error = abs(left - right) if left is not None and right is not None else None
        mirror[name] = {
            "absolute_median_contrast_error": error,
            "passed": bool(
                error is not None
                and error <= float(config["stop_gate"]["maximum_mirror_error"])
            ),
        }
    response_gate = all(result["passed"] for side in per_side.values() for result in side.values())
    internal_geometry_gate = all(
        result.get("approach_recede_time_reverse_error", 0.0) == 0.0
        and result.get("near_miss_equal_dark_pixel_count_per_frame", True)
        for result in geometry.values()
    )
    pair_geometry_gate = all(
        result["left_right_frame_mirror_error"] == 0.0
        for result in geometry.values()
        if "left_right_frame_mirror_error" in result
    )
    geometry_gate = internal_geometry_gate and pair_geometry_gate
    mirror_gate = all(result["passed"] for result in mirror.values())
    passed = bool(response_gate and geometry_gate and mirror_gate)
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(typed_path): _sha256(root / typed_path),
                str(lamina_path): _sha256(root / lamina_path),
                str(LAMINA_SPLIT_IMPLEMENTATION): _sha256(root / LAMINA_SPLIT_IMPLEMENTATION),
                str(scoring_path): _sha256(root / scoring_path),
                str(source_path): _sha256(root / source_path),
                str(RAW_ADJACENCY): _sha256(root / RAW_ADJACENCY),
                str(BODY_IDS): _sha256(root / BODY_IDS),
                str(ANNOTATIONS): _sha256(root / ANNOTATIONS),
            },
            "condition_id": config["condition_id"],
            "stimulus_count": len(stimuli),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "stimulus_geometry": geometry,
        "direct_source_coverage": source_coverage,
        "stimulus_manifest": manifest,
        "per_side": per_side,
        "mirror": mirror,
        "stimulus_geometry_gate_passed": bool(geometry_gate),
        "response_gate_passed": bool(response_gate),
        "mirror_gate_passed": bool(mirror_gate),
        "LPLC1_near_collision_precheck_passed": passed,
        "expand_to_three_conditions": passed,
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
