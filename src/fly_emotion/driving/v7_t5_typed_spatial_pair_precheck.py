"""T5 stop/go precheck with separate typed source-pair spatial moments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import (
    _infer_t4_t5_positions,
    _node_annotations,
)
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-typed-spatial-pair-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_typed_spatial_pair_precheck.py")
RAW_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_raw.npz")
ANNOTATIONS = Path("data/raw/malecns-v1.0/body-annotations.feather")


def _row_matrix(rows: list[tuple[np.ndarray, np.ndarray]], node_count: int):
    row_indices, columns, values = [], [], []
    for row_index, (nodes, weights) in enumerate(rows):
        row_indices.extend([row_index] * len(nodes))
        columns.extend(nodes.tolist())
        values.extend(weights.tolist())
    return sparse.csr_matrix(
        (np.asarray(values), (np.asarray(row_indices), np.asarray(columns))),
        shape=(len(rows), node_count),
        dtype=np.float64,
    )


def _build_typed_moments(root: Path, probe, targets: np.ndarray) -> dict:
    raw = load_graph(root / RAW_ADJACENCY.parent, normalized=False).adjacency
    sides, coordinates = _node_annotations(root, probe)
    source_types = ("Tm1", "Tm2", "Tm4", "Tm9")
    rows = {
        f"{source_type}_{component}": []
        for source_type in source_types
        for component in ("mass", "x", "y")
    }
    present = {source_type: [] for source_type in source_types}
    coordinate_present = {source_type: [] for source_type in source_types}
    axes = []
    for target in targets:
        edge = raw.getrow(int(target))
        sources = edge.indices
        weights = np.abs(edge.data).astype(np.float64)
        positions = coordinates[sources].copy()
        if sides[target] == "L":
            positions[:, 0] *= -1.0
        for source_type in source_types:
            typed = probe.node_types[sources] == source_type
            located = typed & np.all(np.isfinite(positions), axis=1)
            total = float(np.sum(weights[located]))
            selected = sources[located]
            normalized = weights[located] / total if total > 0 else weights[located]
            rows[f"{source_type}_mass"].append((selected, normalized))
            rows[f"{source_type}_x"].append((selected, normalized * positions[located, 0]))
            rows[f"{source_type}_y"].append((selected, normalized * positions[located, 1]))
            present[source_type].append(bool(np.any(typed)))
            coordinate_present[source_type].append(bool(np.any(located)))
        tm1 = np.asarray(
            (
                rows["Tm1_x"][-1][1].sum(),
                rows["Tm1_y"][-1][1].sum(),
            )
        )
        tm2 = np.asarray(
            (
                rows["Tm2_x"][-1][1].sum(),
                rows["Tm2_y"][-1][1].sum(),
            )
        )
        tm9 = np.asarray(
            (
                rows["Tm9_x"][-1][1].sum(),
                rows["Tm9_y"][-1][1].sum(),
            )
        )
        axis = np.asarray((tm2[0] - tm9[0], tm9[1] - tm1[1]))
        norm = float(np.linalg.norm(axis))
        axes.append(axis / norm if np.isfinite(norm) and norm > 1e-12 else np.full(2, np.nan))
    matrices = {
        name: _row_matrix(entries, probe.graph.node_count) for name, entries in rows.items()
    }
    valid = np.logical_and.reduce(
        [np.asarray(coordinate_present[name]) for name in ("Tm1", "Tm2", "Tm9")]
    ) & np.asarray(present["Tm4"])
    return {
        "matrices": matrices,
        "axes": np.asarray(axes),
        "valid": valid,
        "present": {name: np.asarray(values) for name, values in present.items()},
        "coordinate_present": {
            name: np.asarray(values) for name, values in coordinate_present.items()
        },
    }


def _pair_term(
    current: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    first: str,
    second: str,
    component: str,
) -> np.ndarray:
    first_now = current[f"{first}_{component}"]
    second_now = current[f"{second}_{component}"]
    first_mass = current[f"{first}_mass"]
    second_mass = current[f"{second}_mass"]
    first_previous = previous[f"{first}_{component}"]
    second_previous = previous[f"{second}_{component}"]
    first_mass_previous = previous[f"{first}_mass"]
    second_mass_previous = previous[f"{second}_mass"]
    terms = (
        first_now * second_mass_previous,
        first_mass * second_previous,
        second_now * first_mass_previous,
        second_mass * first_previous,
    )
    numerator = terms[0] - terms[1] - terms[2] + terms[3]
    return numerator / (sum(np.abs(term) for term in terms) + 1e-9)


def _controlled_frames(stimulus, mode: str, seed: int) -> np.ndarray:
    frames = stimulus.frames
    prefix = frames[:2]
    moving = frames[2:]
    if mode == "temporal_shuffle":
        moving = moving[np.random.default_rng(seed).permutation(len(moving))]
    elif mode == "static_sham":
        moving = np.repeat(moving[:1], len(moving), axis=0)
    elif mode != "ordered":
        raise ValueError(f"unknown temporal mode: {mode}")
    return np.concatenate((prefix, moving))


def _response(probe, stimulus, moments: dict, mode: str, seed: int) -> dict:
    frames = _controlled_frames(stimulus, mode, seed)
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    baseline_image = frames[0]
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
    baseline_values_out, pair_values = [], []
    for image in frames[probe.baseline_frames :]:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous_retina)
        previous_retina = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        positive = np.maximum(state.astype(np.float64) - baseline_state, 0.0)
        current = {name: matrix @ positive for name, matrix in moments["matrices"].items()}
        baseline_values_out.append(current["Tm9_mass"])
        if previous is not None:
            horizontal = _pair_term(current, previous, "Tm2", "Tm9", "x")
            vertical = _pair_term(current, previous, "Tm9", "Tm1", "y")
            pair_values.append(
                moments["axes"][:, 0] * horizontal
                + moments["axes"][:, 1] * vertical
            )
        previous = current
    return {
        "base_peak": np.max(np.stack(baseline_values_out), axis=0),
        "positive_pair_peak": np.max(np.maximum(np.stack(pair_values), 0.0), axis=0),
    }


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


def _score(
    probe,
    responses: dict,
    populations: dict[str, np.ndarray],
    names: np.ndarray,
    valid: np.ndarray,
    gain: float,
    x_count: int,
    y_count: int,
    assignments: dict[str, np.ndarray],
    scoring: dict,
) -> dict:
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
            selected_population: str = population,
        ) -> np.ndarray:
            grid = [
                responses[(xi, yi, polarity, direction)]["base_peak"]
                + gain * responses[(xi, yi, polarity, direction)]["positive_pair_peak"]
                for yi in range(y_count)
                for xi in range(x_count)
            ]
            grid = np.stack(grid)[:, selected_group]
            assignment = assignments[selected_population]
            rows = assignment[:, 1] * x_count + assignment[:, 0]
            output = grid[rows, np.arange(len(selected_group))]
            output[~valid[selected_group]] = np.nan
            return output

        preferred_response = value("off", preferred)
        scores[population] = {
            "direction": _compact(
                strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("off", OPPOSITE[preferred]),
                    scoring["thresholds"],
                )
            ),
            "polarity": _compact(
                strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("on", preferred),
                    scoring["thresholds"],
                )
            ),
        }
    mirror = {
        subtype: {
            head: abs(
                scores[f"T5{subtype}_L"][head]["median_signed_contrast"]
                - scores[f"T5{subtype}_R"][head]["median_signed_contrast"]
            )
            for head in ("direction", "polarity")
        }
        for subtype in "abcd"
    }
    return {
        "gain": gain,
        "direction_pass_count": int(sum(item["direction"]["passed"] for item in scores.values())),
        "polarity_pass_count": int(sum(item["polarity"]["passed"] for item in scores.values())),
        "bilateral_direction_subtypes": [
            subtype
            for subtype in "abcd"
            if scores[f"T5{subtype}_L"]["direction"]["passed"]
            and scores[f"T5{subtype}_R"]["direction"]["passed"]
        ],
        "mirror_absolute_median_contrast_errors": mirror,
        "mirror_gate_passed": all(
            error <= float(scoring["thresholds"]["maximum_energy_weighted_mirror_error"])
            for values in mirror.values()
            for error in values.values()
        ),
        "population_scores": scores,
    }


def evaluate_v7_t5_typed_spatial_pair_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "lamina_split_protocol",
            "lamina_split_evidence",
            "prior_source_pair_evidence",
            "prior_continuous_moment_evidence",
            "stage1_protocol",
            "scoring_config",
            "source_position_protocol",
        )
    }
    lamina = yaml.safe_load((root / paths["lamina_split_protocol"]).read_text())
    lamina_evidence = json.loads((root / paths["lamina_split_evidence"]).read_text())
    if not all(
        item["polarity"]["passing_condition_count"] == 3
        for item in lamina_evidence["population_consistency"].values()
    ):
        raise ValueError("typed spatial pair precheck requires passed T5 OFF component")
    previous_pair = json.loads((root / paths["prior_source_pair_evidence"]).read_text())
    previous_moment = json.loads(
        (root / paths["prior_continuous_moment_evidence"]).read_text()
    )
    if previous_pair["ordered_precheck_passed"] or previous_moment[
        "control_eligibility_gate_passed"
    ]:
        raise ValueError("typed spatial pair requires preserved predecessor failures")
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    scoring = yaml.safe_load((root / paths["scoring_config"]).read_text())
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("typed spatial pair precheck may consume tuning only")
    probe = LaminaSplitProbe(root, lamina)
    populations = {
        f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    names = np.concatenate([np.full(len(nodes), name) for name, nodes in populations.items()])
    moments = _build_typed_moments(root, probe, targets)
    moments["targets"] = targets
    local = yaml.safe_load((root / lamina["local_edge_config"]).read_text())
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    source_position = yaml.safe_load(
        (root / paths["source_position_protocol"]).read_text()
    )
    positions, _ = _infer_t4_t5_positions(root, probe, source_position)
    finite_positions = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = finite_positions.min(axis=0), finite_positions.max(axis=0)
    assignments = {}
    for population, nodes in populations.items():
        camera = (positions[nodes] - low) / np.maximum(high - low, 1e-12) * (47, 23)
        assignments[population] = np.stack(
            (
                np.argmin(np.abs(x_centers[:, None] - camera[:, 0]), axis=0),
                np.argmin(np.abs(y_centers[:, None] - camera[:, 1]), axis=0),
            ),
            axis=1,
        )
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    ordered = {}
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
                    ordered[(xi, yi, polarity, direction)] = _response(
                        probe, stimulus, moments, "ordered", 0
                    )
    candidates = [
        _score(
            probe,
            ordered,
            populations,
            names,
            moments["valid"],
            float(gain),
            len(x_centers),
            len(y_centers),
            assignments,
            scoring,
        )
        for gain in config["mechanism"]["additive_gains"]
    ]
    minimum = int(config["stop_gate"]["minimum_bilateral_direction_subtypes_to_run_controls"])
    control_eligible = [
        item for item in candidates if len(item["bilateral_direction_subtypes"]) >= minimum
    ]
    strict = [
        item
        for item in candidates
        if item["direction_pass_count"] == len(populations)
        and item["polarity_pass_count"] == len(populations)
        and item["mirror_gate_passed"]
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in paths.values()},
                str(RAW_ADJACENCY): _sha256(root / RAW_ADJACENCY),
                str(ANNOTATIONS): _sha256(root / ANNOTATIONS),
            },
            "condition_id": config["condition_id"],
            "position_count": int(len(x_centers) * len(y_centers)),
            "stimulus_count": len(ordered),
            "candidate_count": len(candidates),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "mechanism": config["source_roles"] | config["mechanism"],
        "source_coverage": {
            "target_count": len(targets),
            "per_source_present_count": {
                name: int(np.count_nonzero(values)) for name, values in moments["present"].items()
            },
            "per_source_coordinate_present_count": {
                name: int(np.count_nonzero(values))
                for name, values in moments["coordinate_present"].items()
            },
            "joint_valid_count": int(np.count_nonzero(moments["valid"])),
            "joint_valid_fraction": float(np.mean(moments["valid"])),
        },
        "ordered_candidates": candidates,
        "control_eligible_candidates": [item["gain"] for item in control_eligible],
        "strict_ordered_candidates": [item["gain"] for item in strict],
        "ordered_precheck_passed": bool(strict),
        "controls_requested": config["controls_after_ordered_gate"],
        "controls_evaluated": False,
        "three_condition_evaluation_performed": False,
        "advance_to_three_tuning_conditions": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            None
            if strict
            else "typed_spatial_pair_produced_no_bilateral_direction_selective_T5_subtype"
        ),
        "boundary": config["boundary"],
    }
