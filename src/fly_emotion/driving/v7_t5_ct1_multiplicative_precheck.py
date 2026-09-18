"""T5 stop/go precheck with multiplicative CT1 sequence modulation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import _infer_t4_t5_positions
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE
from fly_emotion.driving.v7_t5_ct1_source_dynamics_precheck import (
    _compact,
    _trace,
    _variant,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-ct1-multiplicative-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_ct1_multiplicative_precheck.py")


def _score(
    probe,
    responses,
    populations,
    names,
    valid,
    assignments,
    gain,
    x_count,
    y_count,
    scoring,
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
            grid = np.stack(
                [
                    responses[(xi, yi, polarity, direction)]["Tm9_peak"]
                    * (
                        1.0
                        + gain
                        * responses[(xi, yi, polarity, direction)][
                            "positive_CT1_then_fast_peak"
                        ]
                    )
                    for yi in range(y_count)
                    for xi in range(x_count)
                ]
            )[:, selected_group]
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


def evaluate_v7_t5_ct1_multiplicative_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    prior_path = Path(config["prior_additive_evidence"])
    prior = json.loads((root / prior_path).read_text())
    if prior["ordered_precheck_passed"] or prior["strict_ordered_gains"]:
        raise ValueError("multiplicative precheck requires preserved additive failure")
    axis_path = Path(config["axis_calibration"])
    axis = json.loads((root / axis_path).read_text())
    if not axis["authorize_T5_single_condition_precheck"]:
        raise ValueError("multiplicative precheck requires T5-only structural calibration")
    lamina_path = Path(config["lamina_split_protocol"])
    lamina = yaml.safe_load((root / lamina_path).read_text())
    lamina_evidence_path = Path(config["lamina_split_evidence"])
    lamina_evidence = json.loads((root / lamina_evidence_path).read_text())
    if not all(
        item["polarity"]["passing_condition_count"] == 3
        for item in lamina_evidence["population_consistency"].values()
    ):
        raise ValueError("multiplicative precheck requires passed OFF polarity component")
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text())
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("multiplicative precheck may consume tuning only")
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    probe = LaminaSplitProbe(root, lamina)
    populations = {
        f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    names = np.concatenate([np.full(len(nodes), name) for name, nodes in populations.items()])
    fast = probe._normalized_target_inputs(
        targets, tuple(config["source_roles"]["fast_excitation"])
    )
    tm9 = probe._normalized_target_inputs(targets, tuple(config["source_roles"]["enhancer_base"]))
    ct1 = probe._normalized_target_inputs(targets, tuple(config["source_roles"]["slow_inhibition"]))
    valid = (np.diff(fast.indptr) > 0) & (np.diff(tm9.indptr) > 0) & (np.diff(ct1.indptr) > 0)
    source_position_path = Path("configs/driving-v7-lplc2-radial-opponency.yaml")
    positions, _ = _infer_t4_t5_positions(
        root, probe, yaml.safe_load((root / source_position_path).read_text())
    )
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = finite.min(axis=0), finite.max(axis=0)
    local_path = Path(lamina["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text())
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
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
                    ordered[(xi, yi, polarity, direction)] = _trace(
                        probe,
                        _variant(stimulus, "ordered", 0),
                        fast,
                        tm9,
                        ct1,
                    )
    ordered_candidates = [
        _score(
            probe,
            ordered,
            populations,
            names,
            valid,
            assignments,
            float(gain),
            len(x_centers),
            len(y_centers),
            scoring,
        )
        for gain in config["mechanism"]["gains"]
    ]
    minimum = int(config["stop_gate"]["minimum_bilateral_direction_subtypes_to_run_controls"])
    eligible = [
        item
        for item in ordered_candidates
        if len(item["bilateral_direction_subtypes"]) >= minimum
    ]
    strict = [
        item
        for item in ordered_candidates
        if item["direction_pass_count"] == len(populations)
        and item["polarity_pass_count"] == len(populations)
        and item["mirror_gate_passed"]
    ]
    control_results = {}
    control_passing = []
    if eligible:
        for mode in config["controls"]["modes"]:
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
                            responses[(xi, yi, polarity, direction)] = _trace(
                                probe,
                                _variant(
                                    stimulus,
                                    mode,
                                    int(config["controls"]["temporal_shuffle_seed"]),
                                ),
                                fast,
                                tm9,
                                ct1,
                            )
            control_results[mode] = {
                str(item["gain"]): _score(
                    probe,
                    responses,
                    populations,
                    names,
                    valid,
                    assignments,
                    item["gain"],
                    len(x_centers),
                    len(y_centers),
                    scoring,
                )
                for item in eligible
            }
        for item in strict:
            gain = str(item["gain"])
            if all(
                not control_results[mode][gain]["bilateral_direction_subtypes"]
                for mode in control_results
            ):
                control_passing.append(item["gain"])
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(prior_path): _sha256(root / prior_path),
                str(config["prior_additive_protocol"]): _sha256(
                    root / config["prior_additive_protocol"]
                ),
                str(config["source_dynamics_implementation"]): _sha256(
                    root / config["source_dynamics_implementation"]
                ),
                str(axis_path): _sha256(root / axis_path),
                str(lamina_path): _sha256(root / lamina_path),
                str(lamina_evidence_path): _sha256(root / lamina_evidence_path),
                str(stage1_path): _sha256(root / stage1_path),
                str(scoring_path): _sha256(root / scoring_path),
                str(local_path): _sha256(root / local_path),
                str(source_position_path): _sha256(root / source_position_path),
            },
            "condition_id": config["condition_id"],
            "position_count": int(len(x_centers) * len(y_centers)),
            "stimulus_count": len(ordered),
            "candidate_count": len(ordered_candidates),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "source_roles": config["source_roles"],
        "mechanism": config["mechanism"],
        "joint_source_present_count": int(np.count_nonzero(valid)),
        "fixed_target_denominator": len(targets),
        "ordered_candidates": ordered_candidates,
        "control_eligible_gains": [item["gain"] for item in eligible],
        "strict_ordered_gains": [item["gain"] for item in strict],
        "control_passing_strict_gains": control_passing,
        "ordered_precheck_passed": bool(strict),
        "candidate_passed": bool(control_passing),
        "controls_requested": config["controls"]["modes"],
        "controls_evaluated": bool(eligible),
        "control_results": control_results,
        "three_condition_evaluation_performed": False,
        "advance_to_three_tuning_conditions": False,
        "advance_to_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "stop_reason": (
            None
            if control_passing
            else "multiplicative_CT1_candidate_failed_strict_or_control_gate"
        ),
        "boundary": config["boundary"],
    }
