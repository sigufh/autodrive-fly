"""T4 synapse-axis correlator sampled at every existing brain substep."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_stage1_development import _as_visual_stimulus
from fly_emotion.driving.v7_stage1_nested import build_condition_bundle
from fly_emotion.driving.v7_t4_synapse_antisymmetric_precheck import _score_candidates
from fly_emotion.driving.v7_t4_synapse_centered_precheck import (
    IMPLEMENTATION as PRIOR_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4_synapse_correlator_precheck import (
    _synapse_moment_matrices,
)

CONFIG = Path("configs/driving-v7-t4-synapse-microstep-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_synapse_microstep_precheck.py")


def _controlled_stimulus(
    stimulus: VisualStimulus, mode: str, seed: int, baseline_frames: int
) -> VisualStimulus:
    prefix = stimulus.frames[:baseline_frames]
    moving = stimulus.frames[baseline_frames:]
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


def _projected_pair(current: dict, previous: dict, axes: np.ndarray) -> np.ndarray:
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
    return (axes[:, 0] * cross_x + axes[:, 1] * cross_y) / denominator


def _response(probe, stimulus: VisualStimulus, moments: dict) -> dict:
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
    baseline_source = probe._normalized_source_state(state).astype(np.float64)
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
            centered_source = probe._normalized_source_state(state) - baseline_source
            current = {name: matrix @ centered_source for name, matrix in matrices.items()}
            if previous is not None:
                projected_values.append(_projected_pair(current, previous, axes))
            previous = current
        base_values.append(
            state[probe.conductance["targets"]].astype(np.float64) - baseline_target
        )
    trace = np.stack(projected_values)
    return {
        "base": np.max(np.stack(base_values), axis=0),
        "positive_peak": np.max(np.maximum(trace, 0.0), axis=0),
        "signed_mean": np.mean(trace, axis=0),
        "trace": trace,
    }


def _run_mode(
    probe, stimuli, moments: dict, mode: str, seed: int, baseline_frames: int
) -> dict:
    return {
        item.identity: _response(
            probe,
            _controlled_stimulus(
                _as_visual_stimulus(item), mode, seed, baseline_frames
            ),
            moments,
        )
        for item in stimuli
    }


def _energy(values: list[np.ndarray], valid: np.ndarray) -> float:
    return float(np.mean(np.abs(np.concatenate([value[:, valid] for value in values], axis=0))))


def evaluate_v7_t4_synapse_microstep_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["v7_contract"])
    contract = yaml.safe_load((root / contract_path).read_text(encoding="utf-8"))
    frozen_leaks = config["frozen_source_leaks_per_substep"]
    actual_leaks = contract["controlled_vision"]["typed_visual_leak_v1"]
    if any(float(actual_leaks[name]) != float(value) for name, value in frozen_leaks.items()):
        raise ValueError("microstep candidate changed frozen type-specific source leaks")
    axis_path = Path(config["synapse_axis_evidence"])
    axis = json.loads((root / axis_path).read_text(encoding="utf-8"))
    prior_path = Path(config["prior_centered_evidence"])
    prior = json.loads((root / prior_path).read_text(encoding="utf-8"))
    if not axis["authorize_T4_single_condition_precheck"]:
        raise ValueError("microstep T4 precheck requires held-out axis calibration")
    if axis["authorize_T5_mapping_application"]:
        raise ValueError("T5 mapping must remain forbidden")
    if prior["control_eligible_candidates"]:
        raise ValueError("microstep follow-up requires preserved centered main-gate failure")
    conductance_path = Path(config["conductance_protocol"])
    conductance = yaml.safe_load((root / conductance_path).read_text(encoding="utf-8"))
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text(encoding="utf-8"))
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("microstep T4 precheck may consume tuning only")
    stimuli = [item for item in build_condition_bundle(condition) if item.family == "moving_edge"]
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
    seed = int(config["temporal_controls"]["temporal_shuffle_seed"])
    modes = {
        mode: _run_mode(
            probe, stimuli, moments, mode, seed, int(conductance["baseline_frames"])
        )
        for mode in config["temporal_controls"]["modes"]
    }
    identities = [item.identity for item in stimuli]
    ordered = [modes["ordered"][name]["trace"] for name in identities]
    shuffled = [modes["temporal_shuffle"][name]["trace"] for name in identities]
    static = [modes["static_sham"][name]["trace"] for name in identities]
    valid = moments["valid"]
    ordered_energy = _energy(ordered, valid)
    static_energy = _energy(static, valid)
    ordered_residual_energy = _energy(
        [value - sham for value, sham in zip(ordered, static, strict=True)], valid
    )
    shuffle_residual_energy = _energy(
        [value - sham for value, sham in zip(shuffled, static, strict=True)], valid
    )
    shuffle_ratio = shuffle_residual_energy / max(ordered_residual_energy, 1e-12)
    static_ratio = static_energy / max(ordered_energy, 1e-12)
    temporal_gates = {
        "shuffle_residual_attenuated": shuffle_ratio
        <= float(
            config["temporal_controls"][
                "maximum_shuffle_to_ordered_residual_energy_ratio"
            ]
        ),
        "static_energy_attenuated": static_ratio
        <= float(config["temporal_controls"]["maximum_static_to_ordered_energy_ratio"]),
    }
    temporal_passed = all(temporal_gates.values())
    candidates = []
    if temporal_passed:
        response_tuples = {
            name: (value["base"], value["positive_peak"], value["signed_mean"])
            for name, value in modes["ordered"].items()
        }
        specs = [
            (reduction, float(gain))
            for reduction in config["correlator"]["temporal_reductions"]
            for gain in config["correlator"]["additive_gains"]
        ]
        candidates = _score_candidates(
            probe, stimuli, response_tuples, moments, scoring, specs
        )
    advancing = [item["name"] for item in candidates if item["strict_ordered_gate_passed"]]
    all_t4 = {
        name: probe.populations[name]
        for name in scoring["direction_populations"]
        if name.startswith("T4")
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(PRIOR_IMPLEMENTATION): _sha256(root / PRIOR_IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
                str(axis_path): _sha256(root / axis_path),
                str(prior_path): _sha256(root / prior_path),
                str(conductance_path): _sha256(root / conductance_path),
                str(stage1_path): _sha256(root / stage1_path),
                str(scoring_path): _sha256(root / scoring_path),
                str(config["synapse_spatial_protocol"]): _sha256(
                    root / config["synapse_spatial_protocol"]
                ),
            },
            "condition_id": config["condition_id"],
            "stimulus_count": len(stimuli),
            "brain_substeps_per_frame": int(probe.brain_substeps),
            "baseline_prefix_frames_in_controls": int(conductance["baseline_frames"]),
            "frozen_source_leaks_per_substep": frozen_leaks,
            "candidate_count_if_temporal_gate_passes": 8,
            "T5_evaluated": False,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "diagnostic_target_count": len(moments["target_bodies"]),
        "diagnostic_valid_target_count": int(np.count_nonzero(valid)),
        "fixed_T4_population_denominator": int(sum(len(nodes) for nodes in all_t4.values())),
        "temporal_identifiability": {
            "ordered_mean_absolute_energy": ordered_energy,
            "static_mean_absolute_energy": static_energy,
            "ordered_minus_static_mean_absolute_energy": ordered_residual_energy,
            "shuffle_minus_static_mean_absolute_energy": shuffle_residual_energy,
            "shuffle_to_ordered_residual_energy_ratio": shuffle_ratio,
            "static_to_ordered_energy_ratio": static_ratio,
            "gates": temporal_gates,
            "passed": temporal_passed,
        },
        "direction_scoring_performed": temporal_passed,
        "ordered_candidates": candidates,
        "advancing_candidates": advancing,
        "advance_to_three_tuning_conditions": bool(advancing),
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            None
            if advancing
            else (
                "microstep_temporal_identifiability_gate_failed"
                if not temporal_passed
                else "no_microstep_candidate_passed_strict_T4_main_gate"
            )
        ),
        "boundary": config["boundary"],
    }
