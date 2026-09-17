"""Stop/go audit for scalar fast/delayed signals on shortest four-hop paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
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
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-four-hop-scalar-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_four_hop_scalar_precheck.py")


def _shortest_source_paths(probe, sources: np.ndarray) -> tuple[sparse.csr_matrix, dict]:
    adjacency = probe.adjacency
    receptors = probe.retina.node_indices
    exact = [
        adjacency[sources, :][:, receptors].tocsr(),
        (adjacency[sources, :] @ adjacency[:, receptors]).tocsr(),
        (adjacency[sources, :] @ adjacency @ adjacency[:, receptors]).tocsr(),
    ]
    found = np.zeros(len(sources), dtype=bool)
    parts = []
    depth_counts = {}
    for depth, matrix in enumerate(exact, start=1):
        has_path = np.diff(matrix.indptr) > 0
        selected = has_path & ~found
        parts.append(sparse.diags(selected.astype(np.float32)) @ matrix)
        depth_counts[str(depth)] = int(np.count_nonzero(selected))
        found |= has_path
    return sum(parts[1:], parts[0]).tocsr(), {
        "source_count": int(len(sources)),
        "source_first_depth_counts": depth_counts,
        "unreachable_source_count": int(np.count_nonzero(~found)),
    }


def _target_path(
    probe, targets: np.ndarray, types: tuple[str, ...]
) -> tuple[sparse.csr_matrix, dict]:
    sources = np.flatnonzero(np.isin(probe.node_types, types))
    upstream, source_summary = _shortest_source_paths(probe, sources)
    matrix = (probe.adjacency[targets, :][:, sources] @ upstream).tocsr()
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.zeros_like(totals, dtype=np.float64)
    np.divide(1.0, totals, out=scale, where=totals > 0.0)
    return (sparse.diags(scale) @ matrix).tocsr(), {
        **source_summary,
        "target_count": int(len(targets)),
        "reachable_target_count": int(np.count_nonzero(totals > 0.0)),
        "reachable_target_fraction": float(np.mean(totals > 0.0)),
    }


def _scalar_response(
    probe, stimulus, fast, delayed, channel: str, mode: str, seed: int
) -> np.ndarray:
    frames = stimulus.frames[2:]
    if mode == "temporal_shuffle":
        frames = frames[np.random.default_rng(seed).permutation(len(frames))]
    elif mode == "static_sham":
        frames = np.repeat(frames[:1], len(frames), axis=0)
    values = np.stack([probe._sample_retina(frame)[probe.retinal_permutation] for frame in frames])
    delta = np.diff(values, axis=0, prepend=values[:1])
    signal = np.maximum(delta if channel == "on" else -delta, 0.0) * probe.mass_gain[None]
    fast_trace = (fast @ signal.T).T
    delayed_trace = (delayed @ signal.T).T
    fast_previous = np.concatenate((np.zeros_like(fast_trace[:1]), fast_trace[:-1]))
    delayed_previous = np.concatenate((np.zeros_like(delayed_trace[:1]), delayed_trace[:-1]))
    forward = delayed_trace * fast_previous
    reverse = fast_trace * delayed_previous
    correlation = (forward - reverse) / (np.abs(forward) + np.abs(reverse) + 1e-9)
    return np.mean(correlation, axis=0)


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


def evaluate_v7_four_hop_scalar_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    local = yaml.safe_load((root / LOCAL_EDGE_CONFIG).read_text())
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text())
    coverage_path = Path(config["source_coverage_evidence"])
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    coverage = __import__("json").loads((root / coverage_path).read_text())
    if coverage["minimum_sufficient_total_path_length_balanced"] != 4:
        raise ValueError("four-hop precheck requires frozen structural depth four")
    condition = next(
        row for row in stage1["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("four-hop scalar precheck may consume tuning only")
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    probe = MassBalancedVisualProbe(root, local)
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    results = {}
    for candidate_name, candidate in config["candidates"].items():
        family = str(candidate["family"])
        populations = {
            f"{family}{subtype}_{side}": probe.populations[f"{family}{subtype}_{side}"]
            for subtype in "abcd"
            for side in "LR"
        }
        targets = np.concatenate(list(populations.values())).astype(np.int32)
        names = np.concatenate([np.full(len(nodes), name) for name, nodes in populations.items()])
        fast, fast_summary = _target_path(probe, targets, tuple(candidate["fast"]))
        delayed, delayed_summary = _target_path(probe, targets, tuple(candidate["delayed"]))
        modes = {mode: {} for mode in ("ordered", "temporal_shuffle", "static_sham")}
        channel = "on" if family == "T4" else "off"
        for xi, cx in enumerate(x_centers):
            for yi, cy in enumerate(y_centers):
                for polarity in ("on", "off"):
                    for direction in ("left", "right", "up", "down"):
                        stimulus = _local_step_edge(
                            float(cx),
                            float(cy),
                            float(local["stimulus"]["aperture_radius_pixels"]),
                            speed,
                            polarity,
                            direction,
                            int(local["common_background_frames"]),
                        )
                        for mode in modes:
                            modes[mode][(xi, yi, polarity, direction)] = _scalar_response(
                                probe,
                                stimulus,
                                fast,
                                delayed,
                                channel,
                                mode,
                                int(config["controls"]["temporal_shuffle_seed"]),
                            )
        mode_results = {}
        for mode, responses in modes.items():
            population_scores = {}
            for population, nodes in populations.items():
                group = np.flatnonzero(names == population)
                subtype, side = population[2], population[-1]
                preferred = (
                    HORIZONTAL_PREFERENCE[(subtype, side)]
                    if subtype in {"a", "b"}
                    else VERTICAL_PREFERENCE[subtype]
                )
                expected_polarity = channel
                other_polarity = "off" if channel == "on" else "on"

                def value(
                    polarity: str,
                    direction: str,
                    *,
                    selected_responses: dict = responses,
                    selected_group: np.ndarray = group,
                ) -> np.ndarray:
                    return np.max(
                        np.stack(
                            [
                                selected_responses[(xi, yi, polarity, direction)][selected_group]
                                for yi in range(len(y_centers))
                                for xi in range(len(x_centers))
                            ]
                        ),
                        axis=0,
                    )

                preferred_value = value(expected_polarity, preferred)
                direction_score = strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_value,
                    value(expected_polarity, OPPOSITE[preferred]),
                    scoring["thresholds"],
                )
                polarity_score = strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_value,
                    value(other_polarity, preferred),
                    scoring["thresholds"],
                )
                population_scores[population] = {
                    "direction": _compact(direction_score),
                    "polarity": _compact(polarity_score),
                }
            bilateral = [
                subtype
                for subtype in "abcd"
                if population_scores[f"{family}{subtype}_L"]["direction"]["passed"]
                and population_scores[f"{family}{subtype}_R"]["direction"]["passed"]
            ]
            mode_results[mode] = {
                "direction_pass_count": int(
                    sum(x["direction"]["passed"] for x in population_scores.values())
                ),
                "polarity_pass_count": int(
                    sum(x["polarity"]["passed"] for x in population_scores.values())
                ),
                "bilateral_direction_subtypes": bilateral,
                "population_scores": population_scores,
            }
        minimum = int(config["stop_gate"]["minimum_bilateral_direction_subtypes_to_expand"])
        candidate_passed = (
            len(mode_results["ordered"]["bilateral_direction_subtypes"]) >= minimum
            and not mode_results["temporal_shuffle"]["bilateral_direction_subtypes"]
            and not mode_results["static_sham"]["bilateral_direction_subtypes"]
        )
        results[candidate_name] = {
            "family": family,
            "fast_sources": candidate["fast"],
            "delayed_sources": candidate["delayed"],
            "fast_path_summary": fast_summary,
            "delayed_path_summary": delayed_summary,
            "modes": mode_results,
            "passed": bool(candidate_passed),
        }
    advancing = [name for name, result in results.items() if result["passed"]]
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(LOCAL_EDGE_CONFIG): _sha256(root / LOCAL_EDGE_CONFIG),
                str(LOCAL_EDGE_IMPLEMENTATION): _sha256(root / LOCAL_EDGE_IMPLEMENTATION),
                str(stage1_path): _sha256(root / stage1_path),
                str(coverage_path): _sha256(root / coverage_path),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_id": config["condition_id"],
            "candidate_count": len(results),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "candidates": results,
        "advancing_candidates": advancing,
        "four_hop_scalar_gate_passed": bool(advancing),
        "expand_to_three_conditions": bool(advancing),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
