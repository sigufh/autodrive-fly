"""Resolve motion-specific temporal evidence for each direct T4/T5 source type."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import _infer_t4_t5_positions
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_t5_spatial_order import _source_traces
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge
from fly_emotion.driving.v7_upstream_latency_audit import _stimulus_modes

CONFIG = Path("configs/driving-v7-source-type-temporal-identifiability.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_source_type_temporal_identifiability.py"
)


def _summary(
    ordered: np.ndarray,
    shuffled: np.ndarray,
    static: np.ndarray,
    active_threshold: float,
    ratio_threshold: float,
) -> dict:
    ordered_residual = ordered - static
    shuffled_residual = shuffled - static
    ordered_amplitude = np.max(np.abs(ordered_residual), axis=1)
    active = np.isfinite(ordered_amplitude) & (ordered_amplitude >= active_threshold)
    ordered_energy = float(np.nanmean(np.abs(ordered_residual)))
    shuffled_energy = float(np.nanmean(np.abs(shuffled_residual)))
    ratio = shuffled_energy / max(ordered_energy, np.finfo(np.float64).tiny)
    return {
        "comparison_count": int(ordered_amplitude.size),
        "active_comparison_count": int(np.count_nonzero(active)),
        "active_comparison_fraction": float(np.mean(active)),
        "ordered_residual_mean_absolute_energy": ordered_energy,
        "shuffle_residual_mean_absolute_energy": shuffled_energy,
        "shuffle_to_ordered_residual_energy_ratio": ratio,
        "ordered_median_peak_frame": (
            float(np.median(np.argmax(np.abs(ordered_residual), axis=1)[active]))
            if np.any(active)
            else None
        ),
        "gates": {
            "activity_coverage": float(np.mean(active)) >= 0.80,
            "shuffle_attenuation": ratio <= ratio_threshold,
        },
        "passed": bool(float(np.mean(active)) >= 0.80 and ratio <= ratio_threshold),
    }


def evaluate_v7_source_type_temporal_identifiability(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        key: Path(config[key])
        for key in (
            "lamina_split_protocol",
            "stage1_protocol",
            "scoring_config",
            "source_position_protocol",
            "prior_grouped_latency_evidence",
        )
    }
    lamina = yaml.safe_load((root / paths["lamina_split_protocol"]).read_text())
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    scoring = yaml.safe_load((root / paths["scoring_config"]).read_text())
    prior = json.loads((root / paths["prior_grouped_latency_evidence"]).read_text())
    if prior["authorize_new_target_dynamics_candidate"]:
        raise ValueError("source-type audit requires preserved grouped-source failure")
    thresholds = config["thresholds"]
    if float(thresholds["minimum_activity"]) != float(
        scoring["thresholds"]["minimum_valid_denominator"]
    ):
        raise ValueError("source-type activity threshold differs from frozen scoring")
    condition = next(
        row for row in stage1["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("source-type temporal audit may consume tuning only")
    probe = LaminaSplitProbe(root, lamina)
    source_position = yaml.safe_load(
        (root / paths["source_position_protocol"]).read_text()
    )
    positions, position_summary = _infer_t4_t5_positions(root, probe, source_position)
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = finite.min(axis=0), finite.max(axis=0)
    local = yaml.safe_load((root / lamina["local_edge_config"]).read_text())
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    family_data, matrices = {}, {}
    for family, settings in config["families"].items():
        populations = {
            f"{family}{subtype}_{side}": probe.populations[f"{family}{subtype}_{side}"]
            for subtype in "abcd"
            for side in "LR"
        }
        targets = np.concatenate(list(populations.values())).astype(np.int32)
        names = np.concatenate([np.full(len(nodes), name) for name, nodes in populations.items()])
        camera = (positions[targets] - low) / np.maximum(high - low, 1e-12) * (47, 23)
        assignments = np.stack(
            (
                np.argmin(np.abs(x_centers[:, None] - camera[:, 0]), axis=0),
                np.argmin(np.abs(y_centers[:, None] - camera[:, 1]), axis=0),
            ),
            axis=1,
        )
        family_data[family] = {
            "populations": populations,
            "targets": targets,
            "names": names,
            "assignments": assignments,
        }
        for source_type in settings["source_types"]:
            matrices[f"{family}:{source_type}"] = probe._normalized_target_inputs(
                targets, (source_type,)
            )
    directions = list(config["directions"])
    modes = list(config["modes"])
    arrays = {
        mode: {
            family: {
                source_type: None for source_type in settings["source_types"]
            }
            for family, settings in config["families"].items()
        }
        for mode in modes
    }
    for xi, center_x in enumerate(x_centers):
        for yi, center_y in enumerate(y_centers):
            for direction_index, direction in enumerate(directions):
                for family, settings in config["families"].items():
                    stimulus = _local_step_edge(
                        float(center_x),
                        float(center_y),
                        float(local["stimulus"]["aperture_radius_pixels"]),
                        speed,
                        str(settings["polarity"]),
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
                    selected_matrices = {
                        source_type: matrices[f"{family}:{source_type}"]
                        for source_type in settings["source_types"]
                    }
                    for mode, variant in variants.items():
                        traces = _source_traces(probe, variant, selected_matrices)
                        for source_type, trace in traces.items():
                            if arrays[mode][family][source_type] is None:
                                arrays[mode][family][source_type] = np.full(
                                    (len(directions), trace.shape[0], len(data["targets"])),
                                    np.nan,
                                    dtype=np.float32,
                                )
                            arrays[mode][family][source_type][
                                direction_index, :, target_indices
                            ] = trace[:, target_indices].T
    results = {}
    for family, data in family_data.items():
        population_results = {}
        for population, nodes in data["populations"].items():
            group = np.flatnonzero(data["names"] == population)
            source_results = {}
            for source_type in config["families"][family]["source_types"]:
                source_results[source_type] = _summary(
                    arrays["ordered"][family][source_type][:, :, group],
                    arrays["temporal_shuffle"][family][source_type][:, :, group],
                    arrays["static_sham"][family][source_type][:, :, group],
                    float(thresholds["minimum_activity"]),
                    float(thresholds["maximum_shuffle_to_ordered_residual_energy_ratio"]),
                )
            population_results[population] = {
                "target_count": len(nodes),
                "source_types": source_results,
                "all_source_types_passed": all(
                    item["passed"] for item in source_results.values()
                ),
            }
        results[family] = {
            "populations": population_results,
            "passing_source_population_count": int(
                sum(
                    item["passed"]
                    for population in population_results.values()
                    for item in population["source_types"].values()
                )
            ),
            "source_population_denominator": int(
                len(population_results) * len(config["families"][family]["source_types"])
            ),
            "all_source_types_and_populations_passed": all(
                item["all_source_types_passed"] for item in population_results.values()
            ),
        }
    passed = all(item["all_source_types_and_populations_passed"] for item in results.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in paths.values()},
            },
            "condition_id": config["condition_id"],
            "position_count": int(len(x_centers) * len(y_centers)),
            "direction_count": len(directions),
            "mode_count": len(modes),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "source_groups": config["families"],
        "position_inference": position_summary,
        "families": results,
        "source_type_temporal_identifiability_passed": passed,
        "authorize_source_time_constant_candidate": passed,
        "advance_to_three_tuning_conditions": False,
        "advance_to_LPLC_mechanism_repair": False,
        "stop_reason": (
            None
            if passed
            else "source_type_motion_residual_not_attenuated_by_temporal_shuffle"
        ),
        "boundary": config["boundary"],
    }
