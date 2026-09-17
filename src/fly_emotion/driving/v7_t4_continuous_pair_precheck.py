"""Tuning-only continuous source-pair diagnostic for T4 direction dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_source_audit import CARDINAL_VECTORS
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4_source_resolved import _frozen_axis_coordinates
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE
from fly_emotion.driving.v7_t5_continuous_moment_precheck import (
    _compact,
    _reduce,
    _row_matrix,
)
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t4-continuous-pair-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_continuous_pair_precheck.py")
SOURCE_AUDIT_IMPLEMENTATION = Path("src/fly_emotion/driving/v7_source_audit.py")
RAW_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_raw.npz")
ANNOTATIONS = Path("data/raw/malecns-v1.0/body-annotations.feather")


def _centroid(mask, weights, positions) -> np.ndarray:
    return (
        np.average(positions[mask], axis=0, weights=weights[mask])
        if np.any(mask)
        else np.full(2, np.nan)
    )


def _build_projection(root: Path, probe, config: dict, audit: dict, targets: np.ndarray) -> dict:
    raw = load_graph(root / "data/processed/malecns-v1.0", normalized=False).adjacency
    coordinates = _frozen_axis_coordinates(
        root, probe, audit["frozen_axis_calibration"]["transforms_by_eye"]
    )
    rows: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        f"{group}_{moment}": []
        for group in ("direct", "offset")
        for moment in ("mass", "x_moment", "y_moment")
    }
    axes = []
    valid = []
    direct_types = tuple(config["source_groups"]["direct"])
    offset_types = tuple(config["source_groups"]["offset"])
    for target in targets:
        row = raw.getrow(int(target))
        sources = row.indices
        weights = np.abs(row.data).astype(np.float64)
        positions = coordinates[sources]
        direct = np.isin(probe.node_types[sources], direct_types) & np.all(
            np.isfinite(positions), axis=1
        )
        offset = np.isin(probe.node_types[sources], offset_types) & np.all(
            np.isfinite(positions), axis=1
        )
        for group_name, source_mask in (("direct", direct), ("offset", offset)):
            denominator = float(np.sum(weights[source_mask]))
            selected_nodes = sources[source_mask]
            normalized = (
                weights[source_mask] / denominator
                if denominator > 0.0
                else weights[source_mask]
            )
            rows[f"{group_name}_mass"].append((selected_nodes, normalized))
            rows[f"{group_name}_x_moment"].append(
                (selected_nodes, normalized * positions[source_mask, 0])
            )
            rows[f"{group_name}_y_moment"].append(
                (selected_nodes, normalized * positions[source_mask, 1])
            )
        axis = _centroid(direct, weights, positions) - _centroid(offset, weights, positions)
        norm = float(np.linalg.norm(axis))
        axis_valid = bool(np.isfinite(norm) and norm > 1e-12)
        axes.append(axis / norm if axis_valid else np.full(2, np.nan))
        valid.append(np.any(direct) and np.any(offset) and axis_valid)
    return {
        "matrices": {
            name: _row_matrix(entries, probe.graph.node_count)
            for name, entries in rows.items()
        },
        "axes": np.asarray(axes),
        "valid": np.asarray(valid, dtype=bool),
    }


def _pair_trace(probe, stimulus, projection: dict) -> np.ndarray:
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
    baseline_state = state.astype(np.float64)

    previous_retina = baseline_values.copy()
    previous = None
    projected = []
    matrices = projection["matrices"]
    axes = projection["axes"]
    for image in stimulus.frames[probe.baseline_frames :]:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous_retina)
        previous_retina = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        positive = np.maximum(state.astype(np.float64) - baseline_state, 0.0)
        current = {name: matrix @ positive for name, matrix in matrices.items()}
        if previous is not None:
            # This is the factored sum over every direct-offset source pair.
            # The structural axis is direct-minus-offset, hence the leading
            # signs below are the inverse of offset-minus-direct displacement.
            cross_x = -(
                current["offset_x_moment"] * previous["direct_mass"]
                - current["offset_mass"] * previous["direct_x_moment"]
                - current["direct_x_moment"] * previous["offset_mass"]
                + current["direct_mass"] * previous["offset_x_moment"]
            )
            cross_y = -(
                current["offset_y_moment"] * previous["direct_mass"]
                - current["offset_mass"] * previous["direct_y_moment"]
                - current["direct_y_moment"] * previous["offset_mass"]
                + current["direct_mass"] * previous["offset_y_moment"]
            )
            terms = (
                current["offset_x_moment"] * previous["direct_mass"],
                current["offset_mass"] * previous["direct_x_moment"],
                current["direct_x_moment"] * previous["offset_mass"],
                current["direct_mass"] * previous["offset_x_moment"],
                current["offset_y_moment"] * previous["direct_mass"],
                current["offset_mass"] * previous["direct_y_moment"],
                current["direct_y_moment"] * previous["offset_mass"],
                current["direct_mass"] * previous["offset_y_moment"],
            )
            denominator = sum(np.abs(term) for term in terms) + 1e-9
            projected.append((axes[:, 0] * cross_x + axes[:, 1] * cross_y) / denominator)
        previous = current
    return np.stack(projected)


def evaluate_v7_t4_continuous_pair_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    audit_path = Path(config["source_audit_evidence"])
    audit = json.loads((root / audit_path).read_text(encoding="utf-8"))
    if not audit["nested_branch_validation"]["nested_validation_pass"]:
        raise ValueError("T4 continuous pair requires passed nested structural axis audit")
    if audit["frozen_axis_calibration"]["direct_sources"] != config["source_groups"]["direct"]:
        raise ValueError("T4 direct sources differ from frozen structural axis")
    if audit["frozen_axis_calibration"]["offset_sources"] != config["source_groups"]["offset"]:
        raise ValueError("T4 offset sources differ from frozen structural axis")
    nested_path = Path(config["stage1_protocol"])
    nested = yaml.safe_load((root / nested_path).read_text(encoding="utf-8"))
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    local_path = Path(config["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text(encoding="utf-8"))
    condition = next(
        row for row in nested["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("T4 continuous pair precheck may consume tuning only")

    probe = MassBalancedVisualProbe(root, local)
    populations = {
        f"T4{subtype}_{side}": probe.populations[f"T4{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    names = np.concatenate([np.full(len(nodes), name) for name, nodes in populations.items()])
    projection = _build_projection(root, probe, config, audit, targets)

    structural = {}
    median_axes = {}
    for population, nodes in populations.items():
        group = np.flatnonzero(names == population)
        axes = projection["axes"][group]
        valid = projection["valid"][group]
        expected = CARDINAL_VECTORS[scoring["direction_populations"][population]]
        cosine = axes @ expected
        center = np.median(axes[valid], axis=0)
        center /= np.linalg.norm(center)
        gates = {
            "valid_cell_fraction": float(np.mean(valid))
            >= float(scoring["thresholds"]["minimum_valid_cell_fraction"]),
            "median_expected_cosine": float(np.median(cosine[valid]))
            >= float(scoring["thresholds"]["minimum_median_signed_contrast"]),
            "positive_alignment_fraction": float(np.mean(valid & (cosine > 0.0)))
            >= float(scoring["thresholds"]["minimum_positive_cell_fraction"]),
        }
        median_axes[population] = center
        structural[population] = {
            "cell_count": int(len(nodes)),
            "valid_axis_count": int(np.count_nonzero(valid)),
            "valid_axis_fraction": float(np.mean(valid)),
            "median_expected_cosine": float(np.median(cosine[valid])),
            "positive_alignment_fraction_all_cells": float(np.mean(valid & (cosine > 0.0))),
            "median_unit_axis": center.tolist(),
            "gates": gates,
            "passed": bool(all(gates.values())),
        }
    structural_mirror = {}
    for subtype in "abcd":
        left = median_axes[f"T4{subtype}_L"]
        right = median_axes[f"T4{subtype}_R"]
        reflected = np.asarray((-right[0], right[1]))
        error = float(np.linalg.norm(left - reflected))
        structural_mirror[subtype] = {
            "error": error,
            "passed": error
            <= float(scoring["thresholds"]["maximum_energy_weighted_mirror_error"]),
        }
    structural_gate = all(item["passed"] for item in structural.values()) and all(
        item["passed"] for item in structural_mirror.values()
    )

    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    responses = {}
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
                    responses[(xi, yi, polarity, direction)] = _pair_trace(
                        probe, stimulus, projection
                    )
    reductions = {}
    for reduction in config["activity_pair"]["temporal_reductions"]:
        scores = {}
        for population, nodes in populations.items():
            group = np.flatnonzero(names == population)
            subtype, side = population[2], population[-1]
            preferred = (
                HORIZONTAL_PREFERENCE[(subtype, side)]
                if subtype in {"a", "b"}
                else VERTICAL_PREFERENCE[subtype]
            )

            def value(
                polarity: str,
                direction: str,
                *,
                selected_group: np.ndarray = group,
                selected_reduction: str = reduction,
            ) -> np.ndarray:
                grid = [
                    _reduce(responses[(xi, yi, polarity, direction)], selected_reduction)
                    for yi in range(len(y_centers))
                    for xi in range(len(x_centers))
                ]
                output = np.max(np.stack(grid), axis=0)[selected_group]
                output[~projection["valid"][selected_group]] = np.nan
                return output

            preferred_response = value("on", preferred)
            direction = strict_contrast_summary(
                probe.graph.body_ids[nodes],
                preferred_response,
                value("on", OPPOSITE[preferred]),
                scoring["thresholds"],
            )
            polarity = strict_contrast_summary(
                probe.graph.body_ids[nodes],
                preferred_response,
                value("off", preferred),
                scoring["thresholds"],
            )
            scores[population] = {
                "direction": _compact(direction),
                "polarity": _compact(polarity),
            }
        bilateral = [
            subtype
            for subtype in "abcd"
            if scores[f"T4{subtype}_L"]["direction"]["passed"]
            and scores[f"T4{subtype}_R"]["direction"]["passed"]
        ]
        reductions[reduction] = {
            "direction_pass_count": int(
                sum(item["direction"]["passed"] for item in scores.values())
            ),
            "polarity_pass_count": int(
                sum(item["polarity"]["passed"] for item in scores.values())
            ),
            "bilateral_direction_subtypes": bilateral,
            "population_scores": scores,
        }
    minimum_bilateral = int(
        config["stop_gate"]["minimum_bilateral_direction_subtypes_to_run_controls"]
    )
    control_eligible = any(
        len(result["bilateral_direction_subtypes"]) >= minimum_bilateral
        for result in reductions.values()
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(audit_path): _sha256(root / audit_path),
                str(SOURCE_AUDIT_IMPLEMENTATION): _sha256(root / SOURCE_AUDIT_IMPLEMENTATION),
                str(nested_path): _sha256(root / nested_path),
                str(scoring_path): _sha256(root / scoring_path),
                str(local_path): _sha256(root / local_path),
                str(RAW_ADJACENCY): _sha256(root / RAW_ADJACENCY),
                str(ANNOTATIONS): _sha256(root / ANNOTATIONS),
            },
            "condition_id": config["condition_id"],
            "position_count": int(len(x_centers) * len(y_centers)),
            "stimulus_count": len(responses),
            "parameter_fit_in_this_precheck": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "structural_axis": {
            "definition": config["structural_axis"],
            "population_results": structural,
            "mirror": structural_mirror,
            "all_population_and_mirror_gates_passed": bool(structural_gate),
        },
        "ordered_activity_pair": reductions,
        "control_eligibility_gate_passed": bool(control_eligible),
        "controls_requested": config["controls_after_ordered_gate"],
        "controls_evaluated": False,
        "three_condition_evaluation_performed": False,
        "advance_to_three_tuning_conditions": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            None
            if control_eligible
            else "continuous_pair_produced_no_bilateral_direction_selective_T4_subtype"
        ),
        "boundary": config["boundary"],
    }
