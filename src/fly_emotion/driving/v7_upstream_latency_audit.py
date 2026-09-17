"""Target-specific upstream fast/delayed latency audit for T4 and T5."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import _infer_t4_t5_positions
from fly_emotion.driving.v7_t5_lamina_split import (
    IMPLEMENTATION as LAMINA_SPLIT_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_t5_spatial_order import _source_traces
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-upstream-latency-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_upstream_latency_audit.py")


def _stimulus_modes(stimulus, baseline_frames: int, seed: int) -> dict[str, VisualStimulus]:
    prefix = stimulus.frames[:baseline_frames]
    moving = stimulus.frames[baseline_frames:]
    order = np.random.default_rng(seed).permutation(len(moving))
    variants = {
        "ordered": stimulus.frames,
        "temporal_shuffle": np.concatenate((prefix, moving[order])),
        "static_sham": np.concatenate((prefix, np.repeat(moving[:1], len(moving), axis=0))),
    }
    return {
        name: VisualStimulus(
            f"{stimulus.name}:{name}",
            stimulus.family,
            stimulus.polarity,
            stimulus.direction,
            frames,
        )
        for name, frames in variants.items()
    }


def _population_summary(
    body_ids: np.ndarray,
    fast_amplitude: np.ndarray,
    delayed_amplitude: np.ndarray,
    fast_latency: np.ndarray,
    delayed_latency: np.ndarray,
    thresholds: dict,
) -> dict:
    if not (
        fast_amplitude.shape
        == delayed_amplitude.shape
        == fast_latency.shape
        == delayed_latency.shape
    ):
        raise ValueError("source latency arrays must have equal shape")
    if fast_amplitude.shape[1] != len(body_ids):
        raise ValueError("source latency arrays must retain every target cell")
    active = (
        np.isfinite(fast_amplitude)
        & np.isfinite(delayed_amplitude)
        & (fast_amplitude >= float(thresholds["minimum_activity"]))
        & (delayed_amplitude >= float(thresholds["minimum_activity"]))
    )
    delay = delayed_latency - fast_latency
    valid_fraction = float(np.mean(active))
    median_delay = float(np.median(delay[active])) if np.any(active) else None
    delayed_later = active & (delay >= 1)
    delayed_later_fraction = float(np.mean(delayed_later))
    gates = {
        "joint_active_comparison_fraction": valid_fraction
        >= float(thresholds["minimum_joint_active_comparison_fraction"]),
        "median_delayed_minus_fast_peak_frames": median_delay is not None
        and median_delay
        >= float(thresholds["minimum_median_delayed_minus_fast_peak_frames"]),
        "delayed_later_fraction_all_comparisons": delayed_later_fraction
        >= float(thresholds["minimum_delayed_later_fraction_all_comparisons"]),
    }
    cell_all_active = np.all(active, axis=0)
    cell_all_delayed = np.all(delayed_later, axis=0)
    return {
        "target_count": int(len(body_ids)),
        "comparison_count": int(active.size),
        "joint_active_comparison_count": int(np.count_nonzero(active)),
        "joint_active_comparison_fraction": valid_fraction,
        "all_direction_active_target_count": int(np.count_nonzero(cell_all_active)),
        "all_direction_active_target_fraction": float(np.mean(cell_all_active)),
        "median_fast_peak_latency_frames": (
            float(np.median(fast_latency[active])) if np.any(active) else None
        ),
        "median_delayed_peak_latency_frames": (
            float(np.median(delayed_latency[active])) if np.any(active) else None
        ),
        "median_delayed_minus_fast_peak_frames": median_delay,
        "delayed_later_comparison_count": int(np.count_nonzero(delayed_later)),
        "delayed_later_fraction_all_comparisons": delayed_later_fraction,
        "delayed_later_in_all_directions_target_count": int(np.count_nonzero(cell_all_delayed)),
        "delayed_later_in_all_directions_target_fraction": float(np.mean(cell_all_delayed)),
        "target_body_ids_sha256": hashlib.sha256(
            np.asarray(body_ids, dtype="<i8").tobytes()
        ).hexdigest(),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def evaluate_v7_upstream_latency_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    lamina_path = Path(config["lamina_split_protocol"])
    lamina = yaml.safe_load((root / lamina_path).read_text(encoding="utf-8"))
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text(encoding="utf-8"))
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    source_path = Path(config["source_position_protocol"])
    source_config = yaml.safe_load((root / source_path).read_text(encoding="utf-8"))
    temporal_control_path = Path(config["temporal_control_protocol"])
    temporal_control = yaml.safe_load(
        (root / temporal_control_path).read_text(encoding="utf-8")
    )
    thresholds = config["thresholds"]
    if float(thresholds["minimum_activity"]) != float(
        scoring["thresholds"]["minimum_valid_denominator"]
    ):
        raise ValueError("upstream activity threshold differs from frozen scoring")
    if float(thresholds["minimum_joint_active_comparison_fraction"]) != float(
        scoring["thresholds"]["minimum_valid_cell_fraction"]
    ):
        raise ValueError("upstream active-fraction threshold differs from frozen scoring")
    if float(thresholds["minimum_delayed_later_fraction_all_comparisons"]) != float(
        scoring["thresholds"]["minimum_positive_cell_fraction"]
    ):
        raise ValueError("upstream delay-fraction threshold differs from frozen scoring")
    if float(thresholds["maximum_shuffle_to_ordered_residual_energy_ratio"]) != float(
        temporal_control["controls"][
            "maximum_shuffle_to_ordered_energy_ratio_for_diagnostic_attenuation"
        ]
    ):
        raise ValueError("upstream residual attenuation threshold differs from frozen control")
    condition = next(
        row for row in stage1["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("upstream latency audit may consume tuning only")
    if {family: settings["polarity"] for family, settings in config["families"].items()} != {
        "T4": "on",
        "T5": "off",
    }:
        raise ValueError("upstream source polarities must be explicit quoted strings")

    probe = LaminaSplitProbe(root, lamina)
    positions, position_summary = _infer_t4_t5_positions(root, probe, source_config)
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = finite.min(axis=0), finite.max(axis=0)
    local_path = Path(lamina["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text(encoding="utf-8"))
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])

    family_data = {}
    matrices = {}
    for family, settings in config["families"].items():
        populations = {
            f"{family}{subtype}_{side}": probe.populations[f"{family}{subtype}_{side}"]
            for subtype in "abcd"
            for side in "LR"
        }
        targets = np.concatenate(list(populations.values())).astype(np.int32)
        names = np.concatenate(
            [np.full(len(nodes), name) for name, nodes in populations.items()]
        )
        camera = (positions[targets] - low) / np.maximum(high - low, 1e-12) * (47, 23)
        assignments = np.stack(
            (
                np.argmin(np.abs(x_centers[:, None] - camera[:, 0]), axis=0),
                np.argmin(np.abs(y_centers[:, None] - camera[:, 1]), axis=0),
            ),
            axis=1,
        )
        fast = probe._normalized_target_inputs(targets, tuple(settings["fast_sources"]))
        delayed = probe._normalized_target_inputs(targets, tuple(settings["delayed_sources"]))
        family_data[family] = {
            "populations": populations,
            "targets": targets,
            "names": names,
            "assignments": assignments,
            "structurally_valid": (np.diff(fast.indptr) > 0) & (np.diff(delayed.indptr) > 0),
        }
        matrices[f"{family}_fast"] = fast
        matrices[f"{family}_delayed"] = delayed

    directions = list(config["directions"])
    mode_arrays = {
        mode: {
            family: {
                channel: {
                    "amplitude": np.full((len(directions), len(data["targets"])), np.nan),
                    "latency": np.full((len(directions), len(data["targets"])), -1, dtype=np.int16),
                    "trace": None,
                }
                for channel in ("fast", "delayed")
            }
            for family, data in family_data.items()
        }
        for mode in config["modes"]
    }
    stimulus_hashes = {mode: [] for mode in config["modes"]}
    for xi, center_x in enumerate(x_centers):
        for yi, center_y in enumerate(y_centers):
            for direction_index, direction in enumerate(directions):
                for family, settings in config["families"].items():
                    stimulus = _local_step_edge(
                        float(center_x),
                        float(center_y),
                        float(local["stimulus"]["aperture_radius_pixels"]),
                        speed,
                        settings["polarity"],
                        direction,
                        int(local["common_background_frames"]),
                    )
                    variants = _stimulus_modes(
                        stimulus,
                        int(local["common_background_frames"]),
                        int(config["temporal_shuffle_seed"]) + xi * 1000 + yi * 100,
                    )
                    data = family_data[family]
                    assigned = (data["assignments"][:, 0] == xi) & (
                        data["assignments"][:, 1] == yi
                    )
                    target_indices = np.flatnonzero(assigned)
                    for mode, variant in variants.items():
                        if family == next(iter(config["families"])):
                            stimulus_hashes[mode].append(variant.sha256)
                        traces = _source_traces(
                            probe,
                            variant,
                            {
                                "fast": matrices[f"{family}_fast"],
                                "delayed": matrices[f"{family}_delayed"],
                            },
                        )
                        for channel in ("fast", "delayed"):
                            trace = traces[channel][:, target_indices]
                            if mode_arrays[mode][family][channel]["trace"] is None:
                                mode_arrays[mode][family][channel]["trace"] = np.full(
                                    (
                                        len(directions),
                                        trace.shape[0],
                                        len(data["targets"]),
                                    ),
                                    np.nan,
                                    dtype=np.float32,
                                )
                            mode_arrays[mode][family][channel]["trace"][
                                direction_index, :, target_indices
                            ] = trace.T
                            selected = np.abs(trace)
                            mode_arrays[mode][family][channel]["amplitude"][
                                direction_index, target_indices
                            ] = np.max(selected, axis=0)
                            mode_arrays[mode][family][channel]["latency"][
                                direction_index, target_indices
                            ] = np.argmax(selected, axis=0)

    results = {}
    for mode in config["modes"]:
        results[mode] = {}
        for family, data in family_data.items():
            population_results = {}
            for population, nodes in data["populations"].items():
                group = np.flatnonzero(data["names"] == population)
                fast_amplitude = mode_arrays[mode][family]["fast"]["amplitude"][:, group]
                delayed_amplitude = mode_arrays[mode][family]["delayed"]["amplitude"][:, group]
                fast_latency = mode_arrays[mode][family]["fast"]["latency"][:, group]
                delayed_latency = mode_arrays[mode][family]["delayed"]["latency"][:, group]
                structurally_valid = data["structurally_valid"][group]
                fast_amplitude[:, ~structurally_valid] = np.nan
                delayed_amplitude[:, ~structurally_valid] = np.nan
                population_results[population] = _population_summary(
                    probe.graph.body_ids[nodes],
                    fast_amplitude,
                    delayed_amplitude,
                    fast_latency,
                    delayed_latency,
                    thresholds,
                )
            results[mode][family] = {
                "populations": population_results,
                "passing_population_count": int(
                    sum(item["passed"] for item in population_results.values())
                ),
                "all_population_latency_gate_passed": all(
                    item["passed"] for item in population_results.values()
                ),
            }
    ordered_passed = all(
        result["all_population_latency_gate_passed"]
        for result in results["ordered"].values()
    )
    failure_controls_rejected = all(
        not result["all_population_latency_gate_passed"]
        for mode in ("temporal_shuffle", "static_sham")
        for result in results[mode].values()
    )
    identifiable = bool(ordered_passed and failure_controls_rejected)
    residual_results = {}
    residual_attenuation_passed = True
    for family, data in family_data.items():
        population_results = {}
        for population, nodes in data["populations"].items():
            group = np.flatnonzero(data["names"] == population)
            residual = {}
            for mode in ("ordered", "temporal_shuffle"):
                residual[mode] = {
                    channel: (
                        mode_arrays[mode][family][channel]["trace"][:, :, group]
                        - mode_arrays["static_sham"][family][channel]["trace"][:, :, group]
                    )
                    for channel in ("fast", "delayed")
                }
            ordered_summary = _population_summary(
                probe.graph.body_ids[nodes],
                np.max(np.abs(residual["ordered"]["fast"]), axis=1),
                np.max(np.abs(residual["ordered"]["delayed"]), axis=1),
                np.argmax(np.abs(residual["ordered"]["fast"]), axis=1),
                np.argmax(np.abs(residual["ordered"]["delayed"]), axis=1),
                thresholds,
            )
            channel_ratios = {}
            for channel in ("fast", "delayed"):
                ordered_energy = float(np.nanmean(np.abs(residual["ordered"][channel])))
                shuffled_energy = float(
                    np.nanmean(np.abs(residual["temporal_shuffle"][channel]))
                )
                ratio = shuffled_energy / max(ordered_energy, 1e-12)
                channel_ratios[channel] = {
                    "ordered_residual_mean_absolute_energy": ordered_energy,
                    "shuffle_residual_mean_absolute_energy": shuffled_energy,
                    "shuffle_to_ordered_residual_energy_ratio": ratio,
                    "attenuation_gate_passed": ratio
                    <= float(thresholds["maximum_shuffle_to_ordered_residual_energy_ratio"]),
                }
            attenuation = all(
                item["attenuation_gate_passed"] for item in channel_ratios.values()
            )
            residual_attenuation_passed &= attenuation
            population_results[population] = {
                "ordered_minus_static_latency": ordered_summary,
                "channel_residual_energy": channel_ratios,
                "shuffle_attenuation_gate_passed": attenuation,
                "passed": bool(ordered_summary["passed"] and attenuation),
            }
        residual_results[family] = {
            "populations": population_results,
            "passing_population_count": int(
                sum(item["passed"] for item in population_results.values())
            ),
            "all_population_residual_gate_passed": all(
                item["passed"] for item in population_results.values()
            ),
        }
    residual_identifiable = bool(
        residual_attenuation_passed
        and all(
            result["all_population_residual_gate_passed"]
            for result in residual_results.values()
        )
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(lamina_path): _sha256(root / lamina_path),
                str(LAMINA_SPLIT_IMPLEMENTATION): _sha256(root / LAMINA_SPLIT_IMPLEMENTATION),
                str(stage1_path): _sha256(root / stage1_path),
                str(scoring_path): _sha256(root / scoring_path),
                str(source_path): _sha256(root / source_path),
                str(local_path): _sha256(root / local_path),
                str(temporal_control_path): _sha256(root / temporal_control_path),
            },
            "condition_id": config["condition_id"],
            "position_count": int(len(x_centers) * len(y_centers)),
            "direction_count": len(directions),
            "mode_count": len(config["modes"]),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "source_groups": config["families"],
        "position_inference": position_summary,
        "stimulus_hashes": stimulus_hashes,
        "modes": results,
        "ordered_source_latency_gate_passed": bool(ordered_passed),
        "failure_controls_rejected": bool(failure_controls_rejected),
        "source_latency_temporal_identifiability_passed": identifiable,
        "paired_static_residual": residual_results,
        "paired_static_residual_shuffle_attenuation_passed": bool(
            residual_attenuation_passed
        ),
        "paired_static_residual_temporal_identifiability_passed": residual_identifiable,
        "authorize_new_target_dynamics_candidate": residual_identifiable,
        "advance_to_three_tuning_conditions": False,
        "advance_to_LPLC_mechanism_repair": False,
        "stop_reason": (
            None
            if residual_identifiable
            else "paired_motion_residual_failed_latency_or_shuffle_attenuation"
        ),
        "boundary": config["boundary"],
    }
