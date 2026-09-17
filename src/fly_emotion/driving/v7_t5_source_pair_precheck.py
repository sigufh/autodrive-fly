"""Tuning-only T5 source-pair dynamics stop/go precheck."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE
from fly_emotion.driving.v7_t5_lamina_split import (
    IMPLEMENTATION as LAMINA_SPLIT_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-source-pair-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_source_pair_precheck.py")


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


def _ordered_trace(probe, stimulus, matrices: dict[str, object]) -> dict[str, np.ndarray]:
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
    previous = {name: np.zeros(matrix.shape[0]) for name, matrix in matrices.items()}
    horizontal = []
    vertical = []
    baseline = []
    for image in stimulus.frames[probe.baseline_frames :]:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous_retina)
        previous_retina = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        positive = np.maximum(state.astype(np.float64), 0.0)
        current = {name: matrix @ positive for name, matrix in matrices.items()}
        horizontal.append(
            current["Tm2"] * previous["Tm9"]
            - current["Tm9"] * previous["Tm2"]
        )
        vertical.append(
            current["Tm9"] * previous["Tm1"]
            - current["Tm1"] * previous["Tm9"]
        )
        baseline.append(current["Tm9"])
        previous = current
    return {
        "baseline_peak": np.max(np.stack(baseline), axis=0),
        "positive_horizontal_reichardt_peak": np.max(
            np.maximum(np.stack(horizontal), 0.0), axis=0
        ),
        "positive_vertical_reichardt_peak": np.max(
            np.maximum(np.stack(vertical), 0.0), axis=0
        ),
    }


def _candidate_scores(
    probe,
    responses: dict,
    populations: dict[str, np.ndarray],
    names: np.ndarray,
    structurally_valid: np.ndarray,
    gain: float,
    x_count: int,
    y_count: int,
    scoring: dict,
    maximum_mirror_error: float,
) -> dict:
    scores = {}
    for population, nodes in populations.items():
        group = np.flatnonzero(names == population)
        subtype, side = population[2], population[-1]
        preferred_direction = (
            HORIZONTAL_PREFERENCE[(subtype, side)]
            if subtype in {"a", "b"}
            else VERTICAL_PREFERENCE[subtype]
        )

        def value(
            polarity: str, direction: str, *, selected_group: np.ndarray = group
        ) -> np.ndarray:
            grid = []
            for yi in range(y_count):
                for xi in range(x_count):
                    result = responses[(xi, yi, polarity, direction)]
                    grid.append(
                        result["baseline_peak"]
                        + gain
                        * (
                            result["positive_horizontal_reichardt_peak"]
                            + result["positive_vertical_reichardt_peak"]
                        )
                    )
            selected = np.max(np.stack(grid), axis=0)[selected_group]
            selected[~structurally_valid[selected_group]] = np.nan
            return selected

        preferred = value("off", preferred_direction)
        direction = strict_contrast_summary(
            probe.graph.body_ids[nodes],
            preferred,
            value("off", OPPOSITE[preferred_direction]),
            scoring["thresholds"],
        )
        polarity = strict_contrast_summary(
            probe.graph.body_ids[nodes],
            preferred,
            value("on", preferred_direction),
            scoring["thresholds"],
        )
        scores[population] = {
            "direction": _compact(direction),
            "polarity": _compact(polarity),
        }
    mirror = {}
    for subtype in "abcd":
        mirror[subtype] = {}
        for head in ("direction", "polarity"):
            left = scores[f"T5{subtype}_L"][head]["median_signed_contrast"]
            right = scores[f"T5{subtype}_R"][head]["median_signed_contrast"]
            error = abs(left - right) if left is not None and right is not None else None
            mirror[subtype][head] = error
    direction_count = sum(item["direction"]["passed"] for item in scores.values())
    polarity_count = sum(item["polarity"]["passed"] for item in scores.values())
    bilateral = [
        subtype
        for subtype in "abcd"
        if scores[f"T5{subtype}_L"]["direction"]["passed"]
        and scores[f"T5{subtype}_R"]["direction"]["passed"]
    ]
    return {
        "gain": gain,
        "direction_pass_count": int(direction_count),
        "polarity_pass_count": int(polarity_count),
        "bilateral_direction_subtypes": bilateral,
        "mirror_absolute_median_contrast_errors": mirror,
        "mirror_gate_passed": all(
            error is not None and error <= maximum_mirror_error
            for heads in mirror.values()
            for error in heads.values()
        ),
        "population_scores": scores,
    }


def evaluate_v7_t5_source_pair_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    lamina_path = Path(config["lamina_split_protocol"])
    lamina = yaml.safe_load((root / lamina_path).read_text(encoding="utf-8"))
    lamina_evidence_path = Path(config["lamina_split_evidence"])
    lamina_evidence = json.loads((root / lamina_evidence_path).read_text(encoding="utf-8"))
    if not all(
        item["polarity"]["passing_condition_count"] == 3
        for item in lamina_evidence["population_consistency"].values()
    ):
        raise ValueError("source-pair precheck requires the passed T5 OFF component")
    axis_path = Path(config["source_axis_evidence"])
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text(encoding="utf-8"))
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    if config["thresholds_from"] != str(scoring_path):
        raise ValueError("source-pair thresholds must come from frozen stage-1 scoring")
    if config["source_pairs"] != {
        "horizontal": {"first": "Tm9", "second": "Tm2"},
        "vertical": {"first": "Tm1", "second": "Tm9"},
    } or config["baseline_source"] != "Tm9":
        raise ValueError("source-pair implementation and frozen mechanism differ")
    if float(config["stop_gate"]["maximum_mirror_error"]) != float(
        scoring["thresholds"]["maximum_energy_weighted_mirror_error"]
    ):
        raise ValueError("source-pair mirror threshold differs from frozen scoring")
    condition = next(
        row for row in stage1["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("source-pair precheck may consume tuning only")

    probe = LaminaSplitProbe(root, lamina)
    populations = {
        f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    names = np.concatenate(
        [np.full(len(nodes), name) for name, nodes in populations.items()]
    )
    source_types = sorted(
        {
            config["baseline_source"],
            *(pair["first"] for pair in config["source_pairs"].values()),
            *(pair["second"] for pair in config["source_pairs"].values()),
        }
    )
    matrices = {
        source_type: probe._normalized_target_inputs(targets, (source_type,))
        for source_type in source_types
    }
    present = {name: np.diff(matrix.indptr) > 0 for name, matrix in matrices.items()}
    structurally_valid = np.logical_and.reduce(list(present.values()))

    local_path = Path(lamina["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text(encoding="utf-8"))
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
                    responses[(xi, yi, polarity, direction)] = _ordered_trace(
                        probe, stimulus, matrices
                    )
    candidates = [
        _candidate_scores(
            probe,
            responses,
            populations,
            names,
            structurally_valid,
            float(gain),
            len(x_centers),
            len(y_centers),
            scoring,
            float(config["stop_gate"]["maximum_mirror_error"]),
        )
        for gain in config["global_gains"]
    ]
    minimum_bilateral = int(
        config["stop_gate"]["minimum_bilateral_direction_subtypes_to_run_controls"]
    )
    advancing = [
        item
        for item in candidates
        if item["direction_pass_count"] == len(populations)
        and item["polarity_pass_count"] == len(populations)
        and item["mirror_gate_passed"]
    ]
    control_eligible = any(
        len(item["bilateral_direction_subtypes"]) >= minimum_bilateral for item in candidates
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
                str(lamina_evidence_path): _sha256(root / lamina_evidence_path),
                str(axis_path): _sha256(root / axis_path),
                str(stage1_path): _sha256(root / stage1_path),
                str(scoring_path): _sha256(root / scoring_path),
                str(local_path): _sha256(root / local_path),
            },
            "condition_id": config["condition_id"],
            "position_count": int(len(x_centers) * len(y_centers)),
            "stimulus_count": len(responses),
            "candidate_count": len(candidates),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "mechanism": {
            "source_pairs": config["source_pairs"],
            "baseline_source": config["baseline_source"],
            "formula": config["formula"],
            "position_reduction": config["position_reduction"],
            "subtype_labels_read_by_dynamics": False,
            "direction_labels_read_by_dynamics": False,
        },
        "source_coverage": {
            "target_count": int(len(targets)),
            "per_source_present_count": {
                name: int(np.count_nonzero(values)) for name, values in present.items()
            },
            "joint_present_count": int(np.count_nonzero(structurally_valid)),
            "joint_present_fraction": float(np.mean(structurally_valid)),
        },
        "candidates": candidates,
        "advancing_gains": [item["gain"] for item in advancing],
        "ordered_precheck_passed": bool(advancing),
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
            if advancing
            else "no_global_gain_produced_any_bilateral_direction_selective_T5_subtype"
        ),
        "boundary": config["boundary"],
    }
