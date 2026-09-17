"""T4-only antisymmetric correlator on held-out-calibrated synapse axes."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_stage1_development import _as_visual_stimulus
from fly_emotion.driving.v7_stage1_nested import build_condition_bundle
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4_synapse_correlator_precheck import (
    IMPLEMENTATION as PRIOR_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4_synapse_correlator_precheck import (
    _compact,
    _synapse_moment_matrices,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE

CONFIG = Path("configs/driving-v7-t4-synapse-antisymmetric-precheck.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_synapse_antisymmetric_precheck.py"
)


def _controlled_stimulus(stimulus: VisualStimulus, mode: str, seed: int) -> VisualStimulus:
    frames = stimulus.frames
    prefix = frames[:2]
    moving = frames[2:]
    if mode == "temporal_shuffle":
        moving = moving[np.random.default_rng(seed).permutation(len(moving))]
    elif mode == "static_sham":
        moving = np.repeat(moving[:1], len(moving), axis=0)
    elif mode != "ordered":
        raise ValueError(f"unknown temporal mode: {mode}")
    return VisualStimulus(
        name=stimulus.name,
        family=stimulus.family,
        polarity=stimulus.polarity,
        direction=stimulus.direction,
        frames=np.concatenate((prefix, moving)),
        mirror_of=stimulus.mirror_of,
    )


def _response(probe, stimulus: VisualStimulus, moments: dict) -> tuple[np.ndarray, ...]:
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
    baseline_target = state[probe.conductance["targets"]].astype(np.float64)
    previous_retina = baseline_values.copy()
    previous = None
    base_values = []
    projected_values = []
    matrices = moments["matrices"]
    axes = moments["axes"]
    for image in stimulus.frames:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous_retina)
        previous_retina = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        source_state = probe._normalized_source_state(state)
        current = {name: matrix @ source_state for name, matrix in matrices.items()}
        base_values.append(
            state[probe.conductance["targets"]].astype(np.float64) - baseline_target
        )
        if previous is not None:
            # Factored sum over all fast-delayed pairs of
            # (x_fast-x_delayed) * (fast_now*delayed_previous
            #                        - delayed_now*fast_previous).
            cross_x = (
                current["fast_x"] * previous["delayed_mass"]
                - current["fast_mass"] * previous["delayed_x"]
                - current["delayed_mass"] * previous["fast_x"]
                + current["delayed_x"] * previous["fast_mass"]
            )
            cross_y = (
                current["fast_y"] * previous["delayed_mass"]
                - current["fast_mass"] * previous["delayed_y"]
                - current["delayed_mass"] * previous["fast_y"]
                + current["delayed_y"] * previous["fast_mass"]
            )
            terms = (
                current["fast_x"] * previous["delayed_mass"],
                current["fast_mass"] * previous["delayed_x"],
                current["delayed_mass"] * previous["fast_x"],
                current["delayed_x"] * previous["fast_mass"],
                current["fast_y"] * previous["delayed_mass"],
                current["fast_mass"] * previous["delayed_y"],
                current["delayed_mass"] * previous["fast_y"],
                current["delayed_y"] * previous["fast_mass"],
            )
            denominator = sum(np.abs(term) for term in terms) + 1e-6
            projected_values.append(
                (axes[:, 0] * cross_x + axes[:, 1] * cross_y) / denominator
            )
        previous = current
    projected = np.stack(projected_values)
    return (
        np.max(np.stack(base_values), axis=0),
        np.max(np.maximum(projected, 0.0), axis=0),
        np.mean(projected, axis=0),
    )


def _run_responses(probe, stimuli, moments: dict, mode: str, seed: int) -> dict:
    return {
        item.identity: _response(
            probe,
            _controlled_stimulus(_as_visual_stimulus(item), mode, seed),
            moments,
        )
        for item in stimuli
    }


def _candidate_response(
    responses: dict,
    lookup: dict,
    reduction: str,
    gain: float,
    target_rows: np.ndarray,
    valid: np.ndarray,
    polarity: str,
    direction: str,
) -> np.ndarray:
    base, positive_peak, signed_mean = responses[lookup[(polarity, direction)]]
    head = positive_peak if reduction == "positive_peak" else signed_mean
    candidate = base + gain * head
    output = np.full(len(target_rows), np.nan)
    present = target_rows >= 0
    output[present] = candidate[target_rows[present]]
    valid_rows = np.maximum(target_rows, 0)
    output[present & ~valid[valid_rows]] = np.nan
    return output


def _score_candidates(
    probe,
    stimuli,
    responses: dict,
    moments: dict,
    scoring: dict,
    specs: list[tuple[str, float]],
) -> list[dict]:
    lookup = {(item.polarity, item.direction): item.identity for item in stimuli}
    populations = {
        name: probe.populations[name]
        for name in scoring["direction_populations"]
        if name.startswith("T4")
    }
    results = []
    for reduction, gain in specs:
        population_scores = {}
        for population, nodes in populations.items():
            preferred = scoring["direction_populations"][population]
            target_rows = np.asarray(
                [
                    moments["target_index"].get(int(body_id), -1)
                    for body_id in probe.graph.body_ids[nodes]
                ]
            )

            value = partial(
                _candidate_response,
                responses,
                lookup,
                reduction,
                gain,
                target_rows,
                moments["valid"],
            )

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
            population_scores[population] = {
                "direction": _compact(direction),
                "polarity": _compact(polarity),
            }
        bilateral = [
            subtype
            for subtype in "abcd"
            if population_scores[f"T4{subtype}_L"]["direction"]["passed"]
            and population_scores[f"T4{subtype}_R"]["direction"]["passed"]
        ]
        mirror_errors = {
            subtype: {
                head: abs(
                    population_scores[f"T4{subtype}_L"][head]["median_signed_contrast"]
                    - population_scores[f"T4{subtype}_R"][head]["median_signed_contrast"]
                )
                for head in ("direction", "polarity")
            }
            for subtype in "abcd"
        }
        mirror_gate = all(
            error
            <= float(scoring["thresholds"]["maximum_energy_weighted_mirror_error"])
            for values in mirror_errors.values()
            for error in values.values()
        )
        direction_count = int(
            sum(item["direction"]["passed"] for item in population_scores.values())
        )
        polarity_count = int(
            sum(item["polarity"]["passed"] for item in population_scores.values())
        )
        results.append(
            {
                "name": f"{reduction}:gain={gain:g}",
                "temporal_reduction": reduction,
                "gain": gain,
                "direction_pass_count": direction_count,
                "polarity_pass_count": polarity_count,
                "bilateral_direction_subtypes": bilateral,
                "mirror_absolute_median_contrast_errors": mirror_errors,
                "mirror_gate_passed": mirror_gate,
                "strict_ordered_gate_passed": bool(
                    direction_count == len(populations)
                    and polarity_count == len(populations)
                    and mirror_gate
                ),
                "population_scores": population_scores,
            }
        )
    return results


def _moving_edges(condition: dict) -> list:
    return [item for item in build_condition_bundle(condition) if item.family == "moving_edge"]


def evaluate_v7_t4_synapse_antisymmetric_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    axis_path = Path(config["synapse_axis_evidence"])
    axis = json.loads((root / axis_path).read_text(encoding="utf-8"))
    prior_path = Path(config["prior_symmetric_evidence"])
    prior = json.loads((root / prior_path).read_text(encoding="utf-8"))
    if not axis["authorize_T4_single_condition_precheck"]:
        raise ValueError("T4 antisymmetric precheck requires held-out axis calibration")
    if axis["authorize_T5_mapping_application"]:
        raise ValueError("T5 mapping must remain forbidden")
    if prior["main_gate_passed"]:
        raise ValueError("antisymmetric correction requires the preserved symmetric failure")
    conductance_path = Path(config["conductance_protocol"])
    conductance = yaml.safe_load((root / conductance_path).read_text(encoding="utf-8"))
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text(encoding="utf-8"))
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    conditions = {item["condition_id"]: item for item in stage1["conditions"]}
    requested = [config["initial_condition_id"], *config["additional_tuning_condition_ids"]]
    if any(conditions[name]["role"] != "tuning" for name in requested):
        raise ValueError("T4 antisymmetric precheck may consume tuning conditions only")

    normalization = _collect_source_normalization(
        root,
        retinal_backend="linear_luminance",
        retinal_geometry=conductance["primary_retinal_geometry"],
        config=conductance,
    )
    probe = MotionSignConductanceProbe(
        root, retinal_backend="linear_luminance", normalization=normalization
    )
    moments = _synapse_moment_matrices(root, probe, config, axis["transforms_by_eye"])
    specs = [
        (reduction, float(gain))
        for reduction in config["correlator"]["temporal_reductions"]
        for gain in config["correlator"]["additive_gains"]
    ]
    initial_stimuli = _moving_edges(conditions[config["initial_condition_id"]])
    ordered_responses = _run_responses(probe, initial_stimuli, moments, "ordered", 0)
    ordered = _score_candidates(
        probe, initial_stimuli, ordered_responses, moments, scoring, specs
    )
    minimum_bilateral = int(
        config["stop_gate"]["minimum_bilateral_direction_subtypes_to_run_controls"]
    )
    eligible = [
        item for item in ordered if len(item["bilateral_direction_subtypes"]) >= minimum_bilateral
    ]
    eligible_names = [item["name"] for item in eligible]
    strict_names = [item["name"] for item in ordered if item["strict_ordered_gate_passed"]]

    control_results = {}
    controls_evaluated = bool(eligible)
    control_passing = []
    if eligible:
        eligible_specs = [(item["temporal_reduction"], item["gain"]) for item in eligible]
        by_mode = {}
        for mode in ("temporal_shuffle", "static_sham"):
            responses = _run_responses(
                probe,
                initial_stimuli,
                moments,
                mode,
                int(config["controls"]["temporal_shuffle_seed"]),
            )
            by_mode[mode] = {
                item["name"]: item
                for item in _score_candidates(
                    probe, initial_stimuli, responses, moments, scoring, eligible_specs
                )
            }
        for name in eligible_names:
            passed = all(
                not by_mode[mode][name]["bilateral_direction_subtypes"]
                for mode in by_mode
            )
            control_results[name] = {
                mode: by_mode[mode][name] for mode in by_mode
            } | {"passed": passed}
            if name in strict_names and passed:
                control_passing.append(name)

    tuning_results = {config["initial_condition_id"]: {item["name"]: item for item in ordered}}
    three_condition_passing = []
    if control_passing:
        passing_specs = [spec for spec in specs if f"{spec[0]}:gain={spec[1]:g}" in control_passing]
        for condition_id in config["additional_tuning_condition_ids"]:
            stimuli = _moving_edges(conditions[condition_id])
            responses = _run_responses(probe, stimuli, moments, "ordered", 0)
            tuning_results[condition_id] = {
                item["name"]: item
                for item in _score_candidates(
                    probe, stimuli, responses, moments, scoring, passing_specs
                )
            }
        three_condition_passing = [
            name
            for name in control_passing
            if all(
                tuning_results[condition_id][name]["strict_ordered_gate_passed"]
                for condition_id in requested
            )
        ]

    all_t4 = {
        name: probe.populations[name]
        for name in scoring["direction_populations"]
        if name.startswith("T4")
    }
    if not eligible:
        stop_reason = "no_bilateral_T4_direction_subtype_passed_antisymmetric_main_gate"
    elif not strict_names:
        stop_reason = "no_candidate_passed_all_T4_ordered_main_gates"
    elif not control_passing:
        stop_reason = "no_strict_ordered_candidate_passed_temporal_controls"
    elif not three_condition_passing:
        stop_reason = "no_controlled_candidate_passed_all_three_tuning_conditions"
    else:
        stop_reason = None
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(PRIOR_IMPLEMENTATION): _sha256(root / PRIOR_IMPLEMENTATION),
                str(axis_path): _sha256(root / axis_path),
                str(prior_path): _sha256(root / prior_path),
                str(conductance_path): _sha256(root / conductance_path),
                str(stage1_path): _sha256(root / stage1_path),
                str(scoring_path): _sha256(root / scoring_path),
                str(config["synapse_spatial_protocol"]): _sha256(
                    root / config["synapse_spatial_protocol"]
                ),
            },
            "initial_condition_id": config["initial_condition_id"],
            "candidate_count": len(specs),
            "T5_evaluated": False,
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "diagnostic_target_count": len(moments["target_bodies"]),
        "diagnostic_valid_target_count": int(np.count_nonzero(moments["valid"])),
        "fixed_T4_population_denominator": int(sum(len(nodes) for nodes in all_t4.values())),
        "ordered_candidates": ordered,
        "control_eligible_candidates": eligible_names,
        "strict_ordered_candidates": strict_names,
        "controls_evaluated": controls_evaluated,
        "control_results": control_results,
        "control_passing_candidates": control_passing,
        "three_condition_evaluation_performed": bool(control_passing),
        "tuning_condition_results": tuning_results,
        "three_condition_passing_candidates": three_condition_passing,
        "advance_to_T4_calibration": bool(three_condition_passing),
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": stop_reason,
        "boundary": config["boundary"],
    }
