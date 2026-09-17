"""Locate LC4 angular-speed information at its direct presynaptic input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lc4_position_speed_precheck import (
    IMPLEMENTATION as LC4_SPEED_IMPLEMENTATION,
)
from fly_emotion.driving.v7_lplc2_position_coverage import _filled_disc
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe

CONFIG = Path("configs/driving-v7-lc4-input-speed-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_lc4_input_speed_precheck.py")
ANNOTATIONS = Path("data/raw/malecns-v1.0/body-annotations.feather")
NEUROTRANSMITTERS = Path("data/raw/malecns-v1.0/body-neurotransmitters.feather")
RAW_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_raw.npz")
BODY_IDS = Path("data/processed/malecns-v1.0/body_ids.npy")


def _direct_input_trace(probe, stimulus, targets: np.ndarray) -> np.ndarray:
    source_mask = (probe.node_types != "LC4").astype(np.float32)
    matrix = probe.adjacency[targets, :].multiply(source_mask).tocsr()
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
    baseline_input = matrix @ (state.astype(np.float64) * probe.source_sign)
    traces = []
    previous = baseline_values.copy()
    for image in stimulus.frames:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous)
        previous = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        value = matrix @ (state.astype(np.float64) * probe.source_sign)
        traces.append(value - baseline_input)
    return np.stack(traces)


def _score_speed(
    body_ids: np.ndarray, matrix: np.ndarray, velocities: np.ndarray, thresholds: dict
) -> dict:
    centered_velocity = velocities - velocities.mean()
    centered = matrix - matrix.mean(axis=0)
    slopes = centered_velocity @ centered / float(centered_velocity @ centered_velocity)
    fitted = matrix.mean(axis=0) + centered_velocity[:, None] * slopes[None, :]
    residual = np.sum((matrix - fitted) ** 2, axis=0)
    total = np.sum(centered**2, axis=0)
    valid = np.isfinite(slopes) & (total >= float(thresholds["minimum_valid_denominator"]))
    r_squared = np.full(len(slopes), np.nan, dtype=np.float64)
    r_squared[valid] = 1.0 - residual[valid] / total[valid]
    positive = valid & (slopes > 0.0)
    monotonic = (matrix[2] > matrix[1]) & (matrix[1] > matrix[0])
    median_r_squared = float(np.median(r_squared[valid])) if np.any(valid) else None
    gates = {
        "valid_cell_fraction": float(np.mean(valid))
        >= float(thresholds["minimum_valid_cell_fraction"]),
        "positive_slope_fraction": float(np.mean(positive))
        >= float(thresholds["minimum_positive_slope_fraction"]),
        "median_r_squared": median_r_squared is not None
        and median_r_squared >= float(thresholds["minimum_median_r_squared"]),
        "monotonic_cell_fraction": float(np.mean(monotonic))
        >= float(thresholds["minimum_monotonic_cell_fraction"]),
    }
    return {
        "cell_count": int(len(slopes)),
        "body_ids": body_ids.tolist(),
        "responses_by_speed": matrix.T.tolist(),
        "slopes": slopes.tolist(),
        "r_squared": [float(x) if np.isfinite(x) else None for x in r_squared],
        "valid_cell_count": int(np.count_nonzero(valid)),
        "valid_cell_fraction": float(np.mean(valid)),
        "positive_slope_count": int(np.count_nonzero(positive)),
        "positive_slope_fraction_all_cells": float(np.mean(positive)),
        "median_r_squared": median_r_squared,
        "monotonic_cell_count": int(np.count_nonzero(monotonic)),
        "monotonic_cell_fraction": float(np.mean(monotonic)),
        "median_response_by_speed": np.median(matrix, axis=1).tolist(),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def _direct_input_structure(root: Path, targets: dict[str, np.ndarray]) -> dict:
    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
    annotations = feather.read_table(
        root / ANNOTATIONS, columns=["bodyId", "type"], memory_map=True
    ).to_pandas()
    transmitters = feather.read_table(
        root / NEUROTRANSMITTERS, columns=["body", "consensus_nt"], memory_map=True
    ).to_pandas()
    annotations = annotations.merge(
        transmitters.rename(columns={"body": "bodyId"}), on="bodyId", how="left"
    )
    ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(graph.body_ids, ids)
    valid = nodes < graph.node_count
    valid[valid] &= graph.body_ids[nodes[valid]] == ids[valid]
    rows = annotations.iloc[np.flatnonzero(valid)]
    nodes = nodes[valid]
    node_types = np.full(graph.node_count, "", dtype=object)
    neurotransmitters = np.full(graph.node_count, "missing", dtype=object)
    node_types[nodes] = rows["type"].fillna("").to_numpy(dtype=object)
    neurotransmitters[nodes] = rows["consensus_nt"].fillna("missing").to_numpy(
        dtype=object
    )
    output = {}
    for side, target_nodes in targets.items():
        type_totals = {}
        transmitter_totals = {}
        for target in target_nodes:
            row = graph.adjacency.getrow(int(target))
            sources = row.indices
            weights = np.abs(row.data).astype(np.float64)
            nonrecurrent = node_types[sources] != "LC4"
            for source_type in np.unique(node_types[sources][nonrecurrent]):
                if not source_type:
                    continue
                selected = nonrecurrent & (node_types[sources] == source_type)
                entry = type_totals.setdefault(
                    str(source_type), {"target_count": 0, "source_cell_edges": 0, "weight": 0}
                )
                entry["target_count"] += 1
                entry["source_cell_edges"] += int(np.count_nonzero(selected))
                entry["weight"] += int(np.sum(weights[selected]))
            for transmitter in np.unique(neurotransmitters[sources][nonrecurrent]):
                selected = nonrecurrent & (neurotransmitters[sources] == transmitter)
                transmitter_totals[str(transmitter)] = int(
                    transmitter_totals.get(str(transmitter), 0) + np.sum(weights[selected])
                )
        output[side] = {
            "target_count": int(len(target_nodes)),
            "source_types_by_synapse_weight": [
                {"type": name, **item}
                for name, item in sorted(
                    type_totals.items(), key=lambda pair: pair[1]["weight"], reverse=True
                )
            ],
            "neurotransmitter_synapse_weights": dict(
                sorted(transmitter_totals.items(), key=lambda pair: pair[1], reverse=True)
            ),
        }
    return output


def evaluate_v7_lc4_input_speed_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    speed_path = Path(config["lc4_speed_protocol"])
    speed = yaml.safe_load((root / speed_path).read_text())
    speed_evidence_path = Path(config["lc4_speed_evidence"])
    position_path = Path(config["position_protocol"])
    position = yaml.safe_load((root / position_path).read_text())
    typed_path = Path(config["typed_stimulus_protocol"])
    typed = yaml.safe_load((root / typed_path).read_text())
    condition = next(
        item for item in typed["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("LC4 input speed precheck may consume tuning only")
    if config["duration_divisors"] != speed["duration_divisors"]:
        raise ValueError("LC4 input precheck must reuse frozen speed durations")
    probe = MassBalancedVisualProbe(root, config)
    populations = {side: probe.populations[f"LC4_{side}"] for side in "LR"}
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    names = np.concatenate(
        [np.full(len(nodes), side, dtype=object) for side, nodes in populations.items()]
    )
    centers = [
        (float(x), float(y)) for y in position["centers"]["y"] for x in position["centers"]["x"]
    ]
    baseline = np.full(
        (int(config["common_background_frames"]), 24, 48), 0.92, dtype=np.float32
    )
    base_duration = int(condition["duration_frames"])
    responses = {readout: {side: [] for side in populations} for readout in config["readouts"]}
    durations = []
    for divisor in config["duration_divisors"]:
        duration = max(4, base_duration // int(divisor))
        durations.append(duration)
        per_position = {
            readout: {side: [] for side in populations} for readout in config["readouts"]
        }
        for center_x, center_y in centers:
            frames = _filled_disc(
                duration,
                float(condition["start_radius_pixels"]),
                float(condition["terminal_radius_pixels"]),
                center_x,
                center_y,
                True,
            )
            stimulus = VisualStimulus(
                f"LC4-input:x={center_x:g}:y={center_y:g}:duration={duration}",
                "filled_looming",
                "off",
                "expansion",
                np.concatenate((baseline, frames)),
            )
            traces = _direct_input_trace(probe, stimulus, targets)
            for side in populations:
                group = np.flatnonzero(names == side)
                per_position["input_peak"][side].append(np.max(traces[:, group], axis=0))
                derivative = np.diff(traces[:, group], axis=0)
                per_position["peak_positive_input_derivative"][side].append(
                    np.max(np.maximum(derivative, 0.0), axis=0)
                )
        for readout in config["readouts"]:
            for side in populations:
                responses[readout][side].append(
                    np.max(np.stack(per_position[readout][side]), axis=0)
                )
    velocities = np.asarray(config["relative_angular_velocities"], dtype=np.float64)
    results = {
        readout: {
            side: _score_speed(
                probe.graph.body_ids[populations[side]],
                np.stack(values),
                velocities,
                config["thresholds"],
            )
            for side, values in by_side.items()
        }
        for readout, by_side in responses.items()
    }
    primary = config["primary_precheck_readout"]
    mirror = {}
    for readout, by_side in results.items():
        left = np.asarray(by_side["L"]["median_response_by_speed"], dtype=np.float64)
        right = np.asarray(by_side["R"]["median_response_by_speed"], dtype=np.float64)
        left /= max(float(np.max(np.abs(left))), 1e-12)
        right /= max(float(np.max(np.abs(right))), 1e-12)
        error = float(np.max(np.abs(left - right)))
        mirror[readout] = {
            "left_normalized_population_curve": left.tolist(),
            "right_normalized_population_curve": right.tolist(),
            "maximum_absolute_error": error,
            "passed": bool(
                error
                <= float(config["thresholds"]["maximum_normalized_population_curve_error"])
            ),
        }
    passed = bool(
        all(result["passed"] for result in results[primary].values())
        and mirror[primary]["passed"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(speed_path): _sha256(root / speed_path),
                str(LC4_SPEED_IMPLEMENTATION): _sha256(root / LC4_SPEED_IMPLEMENTATION),
                str(speed_evidence_path): _sha256(root / speed_evidence_path),
                str(position_path): _sha256(root / position_path),
                str(typed_path): _sha256(root / typed_path),
                str(ANNOTATIONS): _sha256(root / ANNOTATIONS),
                str(NEUROTRANSMITTERS): _sha256(root / NEUROTRANSMITTERS),
                str(RAW_ADJACENCY): _sha256(root / RAW_ADJACENCY),
                str(BODY_IDS): _sha256(root / BODY_IDS),
            },
            "condition_id": config["condition_id"],
            "position_count": len(centers),
            "duration_frames": durations,
            "source_pool": config["source_pool"],
            "LC4_recurrent_sources_excluded": True,
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "readouts": results,
        "direct_input_structure": _direct_input_structure(root, populations),
        "mirror": mirror,
        "primary_precheck_readout": primary,
        "per_side": results[primary],
        "LC4_input_speed_precheck_passed": bool(passed),
        "expand_to_three_conditions": bool(passed),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            None
            if passed
            else "direct non-LC4 input lacks full-denominator positive angular-speed coding"
        ),
        "boundary": config["boundary"],
    }
