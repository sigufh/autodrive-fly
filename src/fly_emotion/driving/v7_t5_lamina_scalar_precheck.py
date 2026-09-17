"""Stop/go T5 source-pool timing probe after the frozen lamina OFF split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_four_hop_scalar_precheck import (
    IMPLEMENTATION as FOUR_HOP_IMPLEMENTATION,
)
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    IMPLEMENTATION as LOCAL_EDGE_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE
from fly_emotion.driving.v7_t5_lamina_split import (
    IMPLEMENTATION as LAMINA_SPLIT_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-lamina-scalar-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_lamina_scalar_precheck.py")


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


def _source_pool_trace(probe, stimulus, fast, delayed, mode: str, seed: int) -> dict:
    motion_frames = stimulus.frames[probe.baseline_frames :]
    if mode == "temporal_shuffle":
        motion_frames = motion_frames[np.random.default_rng(seed).permutation(len(motion_frames))]
    elif mode == "static_sham":
        motion_frames = np.repeat(motion_frames[:1], len(motion_frames), axis=0)
    elif mode != "ordered":
        raise ValueError(f"unknown T5 lamina scalar mode: {mode}")

    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    baseline_image = stimulus.frames[0]
    baseline_values = probe._sample_retina(baseline_image)[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(baseline_values, baseline_values, baseline_values)
    for _ in range(probe.baseline_frames):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = baseline_drive

    previous_retina = baseline_values.copy()
    previous_fast = np.zeros(fast.shape[0], dtype=np.float64)
    previous_delayed = np.zeros(delayed.shape[0], dtype=np.float64)
    fast_values = []
    delayed_values = []
    correlation_values = []
    for image in motion_frames:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous_retina)
        previous_retina = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        positive = np.maximum(state.astype(np.float64), 0.0)
        fast_now = fast @ positive
        delayed_now = delayed @ positive
        forward = delayed_now * previous_fast
        reverse = fast_now * previous_delayed
        correlation = (forward - reverse) / (np.abs(forward) + np.abs(reverse) + 1e-9)
        fast_values.append(fast_now)
        delayed_values.append(delayed_now)
        correlation_values.append(correlation)
        previous_fast, previous_delayed = fast_now, delayed_now
    return {
        "fast_state_peak": np.max(np.stack(fast_values), axis=0),
        "delayed_state_peak": np.max(np.stack(delayed_values), axis=0),
        "fast_delayed_correlation": np.mean(np.stack(correlation_values), axis=0),
    }


def evaluate_v7_t5_lamina_scalar_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    lamina_path = Path(config["lamina_split_protocol"])
    lamina = yaml.safe_load((root / lamina_path).read_text())
    local_path = Path(lamina["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text())
    four_hop_path = Path(config["four_hop_protocol"])
    four_hop = yaml.safe_load((root / four_hop_path).read_text())
    if config["formula"] != four_hop["formula"]:
        raise ValueError("T5 lamina scalar precheck must reuse the frozen scalar formula")
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    condition = next(
        row for row in stage1["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("T5 lamina scalar precheck may consume tuning only")

    probe = LaminaSplitProbe(root, lamina)
    populations = {
        f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    names = np.concatenate([np.full(len(nodes), name) for name, nodes in populations.items()])
    fast = probe._normalized_target_inputs(targets, tuple(config["source_groups"]["fast"]))
    delayed = probe._normalized_target_inputs(
        targets, tuple(config["source_groups"]["delayed"])
    )
    fast_present = np.diff(fast.indptr) > 0
    delayed_present = np.diff(delayed.indptr) > 0
    structurally_valid = fast_present & delayed_present
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    modes = {mode: {} for mode in ("ordered", "temporal_shuffle", "static_sham")}
    for xi, center_x in enumerate(x_centers):
        for yi, center_y in enumerate(y_centers):
            for polarity in ("on", "off"):
                for direction in ("left", "right", "up", "down"):
                    stimulus = _local_step_edge(
                        float(center_x),
                        float(center_y),
                        float(local["stimulus"]["aperture_radius_pixels"]),
                        speed,
                        polarity,
                        direction,
                        int(local["common_background_frames"]),
                    )
                    for mode in modes:
                        modes[mode][(xi, yi, polarity, direction)] = _source_pool_trace(
                            probe,
                            stimulus,
                            fast,
                            delayed,
                            mode,
                            int(config["controls"]["temporal_shuffle_seed"]),
                        )

    readouts = ("fast_state_peak", "delayed_state_peak", "fast_delayed_correlation")
    mode_results = {}
    for mode, responses in modes.items():
        readout_results = {}
        for readout in readouts:
            population_scores = {}
            for population, nodes in populations.items():
                group = np.flatnonzero(names == population)
                subtype, side = population[2], population[-1]
                preferred_direction = (
                    HORIZONTAL_PREFERENCE[(subtype, side)]
                    if subtype in {"a", "b"}
                    else VERTICAL_PREFERENCE[subtype]
                )

                def value(
                    polarity: str,
                    direction: str,
                    *,
                    selected_responses: dict = responses,
                    selected_readout: str = readout,
                    selected_group: np.ndarray = group,
                ) -> np.ndarray:
                    result = np.max(
                        np.stack(
                            [
                                selected_responses[(xi, yi, polarity, direction)][
                                    selected_readout
                                ][selected_group]
                                for yi in range(len(y_centers))
                                for xi in range(len(x_centers))
                            ]
                        ),
                        axis=0,
                    )
                    result[~structurally_valid[selected_group]] = np.nan
                    return result

                preferred = value("off", preferred_direction)
                direction_score = strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred,
                    value("off", OPPOSITE[preferred_direction]),
                    scoring["thresholds"],
                )
                polarity_score = strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred,
                    value("on", preferred_direction),
                    scoring["thresholds"],
                )
                population_scores[population] = {
                    "direction": _compact(direction_score),
                    "polarity": _compact(polarity_score),
                }
            bilateral = [
                subtype
                for subtype in "abcd"
                if population_scores[f"T5{subtype}_L"]["direction"]["passed"]
                and population_scores[f"T5{subtype}_R"]["direction"]["passed"]
            ]
            readout_results[readout] = {
                "direction_pass_count": int(
                    sum(item["direction"]["passed"] for item in population_scores.values())
                ),
                "polarity_pass_count": int(
                    sum(item["polarity"]["passed"] for item in population_scores.values())
                ),
                "bilateral_direction_subtypes": bilateral,
                "population_scores": population_scores,
            }
        mode_results[mode] = readout_results

    correlation = mode_results["ordered"]["fast_delayed_correlation"]
    mirror_errors = {}
    for subtype in "abcd":
        mirror_errors[subtype] = {}
        for head in ("direction", "polarity"):
            left = correlation["population_scores"][f"T5{subtype}_L"][head][
                "median_signed_contrast"
            ]
            right = correlation["population_scores"][f"T5{subtype}_R"][head][
                "median_signed_contrast"
            ]
            mirror_errors[subtype][head] = abs(left - right) if (
                left is not None and right is not None
            ) else None
    mirror_gate = all(
        error is not None
        and error
        <= float(config["stop_gate"]["maximum_mirror_error"])
        for heads in mirror_errors.values()
        for error in heads.values()
    )
    minimum = int(config["stop_gate"]["minimum_bilateral_direction_subtypes_to_expand"])
    candidate_passed = bool(
        len(correlation["bilateral_direction_subtypes"]) >= minimum
        and correlation["polarity_pass_count"] == len(populations)
        and not mode_results["temporal_shuffle"]["fast_delayed_correlation"][
            "bilateral_direction_subtypes"
        ]
        and not mode_results["static_sham"]["fast_delayed_correlation"][
            "bilateral_direction_subtypes"
        ]
        and mirror_gate
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(lamina_path): _sha256(root / lamina_path),
                str(LAMINA_SPLIT_IMPLEMENTATION): _sha256(root / LAMINA_SPLIT_IMPLEMENTATION),
                str(local_path): _sha256(root / local_path),
                str(LOCAL_EDGE_IMPLEMENTATION): _sha256(root / LOCAL_EDGE_IMPLEMENTATION),
                str(four_hop_path): _sha256(root / four_hop_path),
                str(FOUR_HOP_IMPLEMENTATION): _sha256(root / FOUR_HOP_IMPLEMENTATION),
                str(stage1_path): _sha256(root / stage1_path),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_id": config["condition_id"],
            "position_count": int(len(x_centers) * len(y_centers)),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "source_coverage": {
            "target_count": int(len(targets)),
            "fast_present_count": int(np.count_nonzero(fast_present)),
            "delayed_present_count": int(np.count_nonzero(delayed_present)),
            "joint_present_count": int(np.count_nonzero(structurally_valid)),
            "joint_present_fraction": float(np.mean(structurally_valid)),
        },
        "modes": mode_results,
        "ordered_correlation_mirror_absolute_median_contrast_errors": mirror_errors,
        "ordered_correlation_mirror_gate_passed": bool(mirror_gate),
        "candidate_passed": candidate_passed,
        "expand_to_three_conditions": candidate_passed,
        "three_condition_evaluation_performed": False,
        "stop_reason": (
            None
            if candidate_passed
            else "ordered correlation produced no bilateral direction-selective T5 subtype"
        ),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
