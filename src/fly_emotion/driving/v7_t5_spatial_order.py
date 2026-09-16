from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml
from scipy import sparse

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_stage1_development import OPPOSITE_DIRECTION, _as_visual_stimulus
from fly_emotion.driving.v7_stage1_nested import CONFIG as NESTED_CONFIG
from fly_emotion.driving.v7_stage1_nested import build_condition_bundle

CONFIG = Path("configs/driving-v7-t5-spatial-order.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_spatial_order.py")
POPULATIONS = tuple(f"T5{subtype}_{side}" for subtype in "abcd" for side in "LR")
INCREASING_DIRECTIONS = {"right", "down"}


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_weights = weights[order]
    cutoff = 0.5 * float(np.sum(ordered_weights))
    return float(values[order[np.searchsorted(np.cumsum(ordered_weights), cutoff)]])


def _fixed_sample(body_ids: np.ndarray, count: int, seed: int) -> np.ndarray:
    keys = np.asarray(
        [hashlib.sha256(f"{seed}:{int(body_id)}".encode()).digest() for body_id in body_ids],
        dtype="S32",
    )
    return np.argsort(keys, kind="stable")[:count]


def _optic_coordinates(root: Path, probe: MassBalancedVisualProbe) -> np.ndarray:
    table = feather.read_table(
        root / "data/raw/malecns-v1.0/body-annotations.feather",
        columns=["bodyId", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    body_ids = table["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(probe.graph.body_ids, body_ids)
    valid = nodes < probe.graph.node_count
    valid[valid] &= probe.graph.body_ids[nodes[valid]] == body_ids[valid]
    hex1 = table["assignedOlHex1"].to_numpy(dtype=np.float64)
    hex2 = table["assignedOlHex2"].to_numpy(dtype=np.float64)
    coordinates = np.full((probe.graph.node_count, 2), np.nan, dtype=np.float64)
    # Same axial-to-image orientation used by V7VisualProbe._retinal_coordinates.
    coordinates[nodes[valid], 0] = 1.5 * hex2[valid]
    coordinates[nodes[valid], 1] = np.sqrt(3.0) * (hex1[valid] + hex2[valid] / 2.0)
    return coordinates


def _row_matrix(rows: list[tuple[np.ndarray, np.ndarray]], node_count: int) -> sparse.csr_matrix:
    row_indices: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for row_index, (nodes, weights) in enumerate(rows):
        row_indices.extend([row_index] * len(nodes))
        columns.extend(nodes.tolist())
        data.extend(weights.tolist())
    return sparse.csr_matrix(
        (np.asarray(data), (np.asarray(row_indices), np.asarray(columns))),
        shape=(len(rows), node_count),
        dtype=np.float64,
    )


def _build_spatial_projection(
    root: Path, probe: MassBalancedVisualProbe, config: dict
) -> tuple[dict[str, sparse.csr_matrix], dict]:
    coordinates = _optic_coordinates(root, probe)
    target_nodes = []
    target_body_ids = []
    target_populations = []
    for population in POPULATIONS:
        nodes = probe.populations[population]
        body_ids = probe.graph.body_ids[nodes]
        selected = _fixed_sample(
            body_ids, int(config["targets_per_population"]), int(config["sample_seed"])
        )
        target_nodes.extend(nodes[selected].tolist())
        target_body_ids.extend(body_ids[selected].tolist())
        target_populations.extend([population] * len(selected))
    target_nodes_array = np.asarray(target_nodes, dtype=np.int32)
    fast_types = tuple(config["source_groups"]["fast"])
    tm9_types = tuple(config["source_groups"]["spatial_delayed"])
    ct1_types = tuple(config["source_groups"]["global_modulator"])
    pools: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name in ("fast_low", "fast_high", "tm9_low", "tm9_high", "ct1")
    }
    structural_valid = []
    source_counts = []
    split_coordinates = []
    for target, population in zip(target_nodes_array, target_populations, strict=True):
        row = probe.adjacency.getrow(int(target))
        sources = row.indices
        weights = np.abs(row.data).astype(np.float64)
        subtype, side = population[2], population[-1]
        axis = 0 if subtype in {"a", "b"} else 1
        projected = coordinates[sources, axis].copy()
        if axis == 0 and side == "L":
            projected *= -1.0
        finite = np.isfinite(projected)
        fast = np.isin(probe.node_types[sources], fast_types) & finite
        tm9 = np.isin(probe.node_types[sources], tm9_types) & finite
        ct1 = np.isin(probe.node_types[sources], ct1_types)
        spatial = fast | tm9
        split = (
            _weighted_median(projected[spatial], weights[spatial])
            if np.any(spatial)
            else np.nan
        )
        split_coordinates.append(split if np.isfinite(split) else None)

        def append_pool(
            name: str,
            mask: np.ndarray,
            denominator_mask: np.ndarray,
            source_nodes: np.ndarray = sources,
            source_weights: np.ndarray = weights,
        ) -> None:
            denominator = float(np.sum(source_weights[denominator_mask]))
            if denominator <= 0.0:
                pools[name].append((np.empty(0, dtype=np.int32), np.empty(0)))
            else:
                pools[name].append((source_nodes[mask], source_weights[mask] / denominator))

        low = projected <= split
        high = projected > split
        append_pool("fast_low", fast & low, fast)
        append_pool("fast_high", fast & high, fast)
        append_pool("tm9_low", tm9 & low, tm9)
        append_pool("tm9_high", tm9 & high, tm9)
        append_pool("ct1", ct1, ct1)
        counts = {
            "fast_low": int(np.count_nonzero(fast & low)),
            "fast_high": int(np.count_nonzero(fast & high)),
            "tm9_low": int(np.count_nonzero(tm9 & low)),
            "tm9_high": int(np.count_nonzero(tm9 & high)),
            "ct1": int(np.count_nonzero(ct1)),
        }
        source_counts.append(counts)
        structural_valid.append(all(counts[name] > 0 for name in pools if name != "ct1"))
    matrices = {name: _row_matrix(rows, probe.graph.node_count) for name, rows in pools.items()}
    metadata = {
        "target_nodes": target_nodes_array,
        "target_body_ids": np.asarray(target_body_ids, dtype=np.int64),
        "target_populations": np.asarray(target_populations),
        "structural_valid": np.asarray(structural_valid, dtype=bool),
        "source_counts": source_counts,
        "split_coordinates": split_coordinates,
    }
    return matrices, metadata


def _source_traces(
    probe: MassBalancedVisualProbe, stimulus, matrices: dict[str, sparse.csr_matrix]
) -> dict[str, np.ndarray]:
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
    baselines = {name: matrix @ state.astype(np.float64) for name, matrix in matrices.items()}
    traces = {name: [] for name in matrices}
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
        state64 = state.astype(np.float64)
        for name, matrix in matrices.items():
            traces[name].append(matrix @ state64 - baselines[name])
    return {name: np.stack(values) for name, values in traces.items()}


def _spatial_response(
    traces: dict[str, np.ndarray], preferred_direction: str, lag: int, ct1_mode: str
) -> np.ndarray:
    fast_low = traces["fast_low"]
    fast_high = traces["fast_high"]
    tm9_low = traces["tm9_low"]
    tm9_high = traces["tm9_high"]
    if lag:
        zeros = np.zeros_like(tm9_low[:lag])
        delayed_low = np.concatenate((zeros, tm9_low[:-lag]), axis=0)
        delayed_high = np.concatenate((zeros, tm9_high[:-lag]), axis=0)
    else:
        delayed_low, delayed_high = tm9_low, tm9_high
    low_to_high = fast_high * delayed_low - fast_low * delayed_high
    orientation = 1.0 if preferred_direction in INCREASING_DIRECTIONS else -1.0
    drive = orientation * low_to_high
    if ct1_mode == "inhibitory_gain":
        # CT1 is an unlocalized, GABAergic wide-field cell in each eye.  Keep it
        # outside the spatial pools and apply only its signed global gain.
        drive *= np.clip(1.0 - traces["ct1"], 0.0, 2.0)
    elif ct1_mode != "separate_only":
        raise ValueError(f"unknown CT1 mode: {ct1_mode}")
    return np.max(drive, axis=0)


def _contrast(preferred: np.ndarray, comparator: np.ndarray, minimum: float) -> np.ndarray:
    denominator = np.abs(preferred) + np.abs(comparator)
    result = np.full(preferred.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator >= minimum)
    result[valid] = (preferred[valid] - comparator[valid]) / denominator[valid]
    return result


def evaluate_v7_t5_spatial_order(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested = yaml.safe_load((root / NESTED_CONFIG).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    condition_ids = list(config["condition_ids"])
    conditions = {item["condition_id"]: item for item in nested["conditions"]}
    if any(conditions[name]["role"] != "tuning" for name in condition_ids):
        raise ValueError("spatial-order diagnostic may consume tuning conditions only")
    probe = MassBalancedVisualProbe(root, config)
    matrices, metadata = _build_spatial_projection(root, probe, config)
    condition_stimuli = {
        condition_id: [
            item
            for item in build_condition_bundle(conditions[condition_id])
            if item.family == "moving_edge"
        ]
        for condition_id in condition_ids
    }
    stimuli = {
        item.identity: item
        for items in condition_stimuli.values()
        for item in items
    }
    traces = {
        identity: _source_traces(probe, _as_visual_stimulus(item), matrices)
        for identity, item in stimuli.items()
    }
    lookup = {
        (condition_id, item.polarity, item.direction): item.identity
        for condition_id, items in condition_stimuli.items()
        for item in items
    }
    minimum_denominator = float(scoring["thresholds"]["minimum_valid_denominator"])
    minimum_contrast = float(config["reachability_gate"]["minimum_signed_contrast"])
    candidates = [
        (int(lag), mode) for lag in config["lags_frames"] for mode in config["ct1_modes"]
    ]
    target_results = []
    reachable_by_population = {}
    for population in POPULATIONS:
        group = np.flatnonzero(metadata["target_populations"] == population)
        preferred_direction = scoring["direction_populations"][population]
        opposite_direction = OPPOSITE_DIRECTION[preferred_direction]
        candidate_contrasts = []
        for lag, ct1_mode in candidates:
            comparisons = []
            for condition_id in condition_ids:
                preferred_off = _spatial_response(
                    traces[lookup[(condition_id, "off", preferred_direction)]],
                    preferred_direction,
                    lag,
                    ct1_mode,
                )[group]
                opposite_off = _spatial_response(
                    traces[lookup[(condition_id, "off", opposite_direction)]],
                    preferred_direction,
                    lag,
                    ct1_mode,
                )[group]
                preferred_on = _spatial_response(
                    traces[lookup[(condition_id, "on", preferred_direction)]],
                    preferred_direction,
                    lag,
                    ct1_mode,
                )[group]
                comparisons.extend(
                    (
                        _contrast(preferred_off, opposite_off, minimum_denominator),
                        _contrast(preferred_off, preferred_on, minimum_denominator),
                    )
                )
            candidate_contrasts.append(np.stack(comparisons, axis=1))
        contrast_cube = np.stack(candidate_contrasts)
        success_cube = np.isfinite(contrast_cube) & (contrast_cube >= minimum_contrast)
        candidate_success = np.all(success_cube, axis=2)
        reachable = np.any(candidate_success, axis=0) & metadata["structural_valid"][group]
        reachable_by_population[population] = {
            "target_count": int(len(group)),
            "structurally_valid_count": int(np.count_nonzero(metadata["structural_valid"][group])),
            "all_six_comparisons_reachable_count": int(np.count_nonzero(reachable)),
            "all_six_comparisons_reachable_fraction": float(np.mean(reachable)),
        }
        for local_index, global_index in enumerate(group):
            successes = np.flatnonzero(candidate_success[:, local_index])
            best = int(successes[0]) if len(successes) else int(
                np.argmax(np.sum(success_cube[:, local_index], axis=1))
            )
            lag, ct1_mode = candidates[best]
            target_results.append(
                {
                    "body_id": int(metadata["target_body_ids"][global_index]),
                    "population": population,
                    "structurally_valid": bool(metadata["structural_valid"][global_index]),
                    "source_counts": metadata["source_counts"][global_index],
                    "best_candidate": {"lag_frames": lag, "ct1_mode": ct1_mode},
                    "successful_comparison_count": int(
                        np.sum(success_cube[best, local_index])
                    ),
                    "all_six_comparisons_reachable": bool(reachable[local_index]),
                    "contrasts": [
                        None if not np.isfinite(value) else float(value)
                        for value in contrast_cube[best, local_index]
                    ],
                }
            )
    bd = [item for item in target_results if item["population"][2] in {"b", "d"}]
    bd_count = sum(item["all_six_comparisons_reachable"] for item in bd)
    bd_fraction = bd_count / len(bd)
    gate = bd_fraction >= float(
        config["reachability_gate"]["minimum_b_d_reachable_fraction"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(NESTED_CONFIG): _sha256(root / NESTED_CONFIG),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_ids": condition_ids,
            "moving_edge_stimulus_count": len(stimuli),
            "target_sample_count": len(target_results),
            "exploratory_candidate_count": len(candidates),
            "parameter_fit": False,
            "runtime_modified": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
        },
        "mechanism": {
            "fast_sources": list(config["source_groups"]["fast"]),
            "spatial_delayed_sources": list(
                config["source_groups"]["spatial_delayed"]
            ),
            "global_modulator_sources": list(
                config["source_groups"]["global_modulator"]
            ),
            "ct1_has_optic_hex_coordinates": False,
            "ct1_used_in_spatial_pool": False,
            "labels_frozen": True,
            "thresholds_frozen": True,
            "response_formula": (
                "preferred_sign*"
                "(fast_high*lagged_tm9_low-fast_low*lagged_tm9_high)"
            ),
        },
        "reachable_by_population": reachable_by_population,
        "b_d_reachability": {
            "reachable_count": int(bd_count),
            "target_count": len(bd),
            "reachable_fraction": float(bd_fraction),
            "minimum_required_fraction": float(
                config["reachability_gate"]["minimum_b_d_reachable_fraction"]
            ),
            "passed": bool(gate),
        },
        "target_results": target_results,
        "advance_to_full_population": bool(gate),
        "advance_to_runtime_integration": False,
        "advance_to_navigation": False,
        "boundary": config["boundary"],
    }
