"""Audit input-energy matching of the T4 temporal-shuffle control."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_stage1_development import _as_visual_stimulus
from fly_emotion.driving.v7_stage1_nested import build_condition_bundle
from fly_emotion.driving.v7_t4_synapse_microstep_precheck import _controlled_stimulus

CONFIG = Path("configs/driving-v7-t4-temporal-shuffle-input-energy-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_temporal_shuffle_input_energy_audit.py"
)


def _frame_multiset_digest(frames: np.ndarray) -> str:
    frame_hashes = sorted(hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames)
    return hashlib.sha256("\n".join(frame_hashes).encode()).hexdigest()


def _input_energies(probe, frames: np.ndarray, source_nodes: dict) -> dict:
    baseline_image = frames[0]
    baseline_retina = probe._sample_retina(baseline_image)[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(
        baseline_retina, baseline_retina, baseline_retina
    )
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    for _ in range(probe.baseline_frames):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = baseline_drive
    baseline_source = probe._normalized_source_state(state).astype(np.float64)
    previous_image = baseline_image
    previous_retina = baseline_retina.copy()
    pixel_total = retinal_total = 0.0
    pixel_count = retinal_count = 0
    source_totals = {source: 0.0 for source in source_nodes}
    source_counts = {source: 0 for source in source_nodes}
    for image in frames[probe.baseline_frames :]:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        retinal_drive = probe._retinal_code(
            sampled, baseline_retina, previous_retina
        )
        pixel_total += float(np.sum(np.abs(image - previous_image)))
        retinal_total += float(np.sum(np.abs(retinal_drive)))
        pixel_count += image.size
        retinal_count += retinal_drive.size
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = retinal_drive
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = retinal_drive
            centered = probe._normalized_source_state(state).astype(np.float64) - baseline_source
            for source, nodes in source_nodes.items():
                source_totals[source] += float(np.sum(np.abs(centered[nodes])))
                source_counts[source] += len(nodes)
        previous_image = image
        previous_retina = sampled
    return {
        "pixel_temporal_difference": pixel_total / pixel_count,
        "R1_R6_drive": retinal_total / retinal_count,
        "centered_source_state": {
            source: source_totals[source] / source_counts[source]
            for source in source_nodes
        },
    }


def _ratio(control: dict, ordered: dict) -> dict:
    return {
        "pixel_temporal_difference": control["pixel_temporal_difference"]
        / ordered["pixel_temporal_difference"],
        "R1_R6_drive": control["R1_R6_drive"] / ordered["R1_R6_drive"],
        "centered_source_state": {
            source: control["centered_source_state"][source]
            / ordered["centered_source_state"][source]
            for source in ordered["centered_source_state"]
        },
    }


def evaluate_v7_t4_temporal_shuffle_input_energy_audit(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    microstep_path = Path(audit["microstep_evidence"])
    microstep_config_path = Path(audit["microstep_config"])
    conductance_path = Path(audit["conductance_config"])
    stage1_path = Path(audit["stage1_protocol"])
    microstep = json.loads((root / microstep_path).read_text(encoding="utf-8"))
    microstep_config = yaml.safe_load(
        (root / microstep_config_path).read_text(encoding="utf-8")
    )
    conductance = yaml.safe_load((root / conductance_path).read_text(encoding="utf-8"))
    stage1 = yaml.safe_load((root / stage1_path).read_text(encoding="utf-8"))
    if microstep["temporal_identifiability"]["passed"]:
        raise ValueError("T4 temporal-shuffle audit requires preserved output-gate failure")
    condition = next(
        item
        for item in stage1["conditions"]
        if item["condition_id"] == microstep_config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("T4 input-energy audit may consume tuning only")
    stimuli = [
        item for item in build_condition_bundle(condition) if item.family == "moving_edge"
    ]
    expected = audit["expected"]
    if len(stimuli) != int(expected["stimulus_count"]):
        raise ValueError("T4 input-energy stimulus count changed")
    normalization = _collect_source_normalization(
        root,
        retinal_backend="linear_luminance",
        retinal_geometry=conductance["primary_retinal_geometry"],
        config=conductance,
    )
    probe = MotionSignConductanceProbe(
        root, retinal_backend="linear_luminance", normalization=normalization
    )
    if probe.brain_substeps != int(expected["brain_substeps_per_frame"]):
        raise ValueError("T4 input-energy brain-substep count changed")
    source_nodes = {
        source: np.flatnonzero(probe.node_types == source)
        for source in audit["source_types"]
    }
    modes = microstep_config["temporal_controls"]["modes"]
    seed = int(microstep_config["temporal_controls"]["temporal_shuffle_seed"])
    baseline_frames = int(conductance["baseline_frames"])
    totals = {
        mode: {
            "pixel_temporal_difference": 0.0,
            "R1_R6_drive": 0.0,
            "centered_source_state": {source: 0.0 for source in source_nodes},
        }
        for mode in modes
    }
    multiset_checks = []
    for stimulus in stimuli:
        frames = {
            mode: _controlled_stimulus(
                _as_visual_stimulus(stimulus), mode, seed, baseline_frames
            ).frames
            for mode in modes
        }
        multiset_checks.append(
            _frame_multiset_digest(frames["ordered"][baseline_frames:])
            == _frame_multiset_digest(frames["temporal_shuffle"][baseline_frames:])
        )
        energies = {
            mode: _input_energies(probe, values, source_nodes)
            for mode, values in frames.items()
        }
        for mode, values in energies.items():
            totals[mode]["pixel_temporal_difference"] += values[
                "pixel_temporal_difference"
            ]
            totals[mode]["R1_R6_drive"] += values["R1_R6_drive"]
            for source, value in values["centered_source_state"].items():
                totals[mode]["centered_source_state"][source] += value
    means = {
        mode: {
            "pixel_temporal_difference": values["pixel_temporal_difference"]
            / len(stimuli),
            "R1_R6_drive": values["R1_R6_drive"] / len(stimuli),
            "centered_source_state": {
                source: value / len(stimuli)
                for source, value in values["centered_source_state"].items()
            },
        }
        for mode, values in totals.items()
    }
    ratios = {
        "temporal_shuffle_to_ordered": _ratio(
            means["temporal_shuffle"], means["ordered"]
        ),
        "static_sham_to_ordered": _ratio(means["static_sham"], means["ordered"]),
    }
    for mode, expected_ratios in (
        ("temporal_shuffle_to_ordered", expected["temporal_shuffle_to_ordered"]),
        ("static_sham_to_ordered", expected["static_sham_to_ordered"]),
    ):
        for metric in ("pixel_temporal_difference", "R1_R6_drive"):
            if not math.isclose(
                ratios[mode][metric], float(expected_ratios[metric]), rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(f"T4 {mode} {metric} ratio changed")
        for source, expected_value in expected_ratios["centered_source_state"].items():
            if not math.isclose(
                ratios[mode]["centered_source_state"][source],
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"T4 {mode} {source} source-state ratio changed")
    frame_multiset = all(multiset_checks)
    retinal_energy_matched = math.isclose(
        ratios["temporal_shuffle_to_ordered"]["R1_R6_drive"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-7,
    )
    tolerance = float(expected["maximum_shuffle_source_energy_relative_error"])
    source_energy_within_tolerance = all(
        abs(value - 1.0) <= tolerance
        for value in ratios["temporal_shuffle_to_ordered"][
            "centered_source_state"
        ].values()
    )
    return {
        "protocol": {
            "name": audit["name"],
            "observed_on": audit["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(microstep_path): _sha256(root / microstep_path),
                str(microstep_config_path): _sha256(root / microstep_config_path),
                str(conductance_path): _sha256(root / conductance_path),
                str(stage1_path): _sha256(root / stage1_path),
            },
            "condition_id": microstep_config["condition_id"],
            "stimulus_count": len(stimuli),
            "brain_substeps_per_frame": probe.brain_substeps,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "mean_absolute_energy_by_mode": means,
        "energy_ratios": ratios,
        "frame_multiset_preserved_for_every_shuffle": frame_multiset,
        "R1_R6_mean_absolute_energy_preserved": retinal_energy_matched,
        "all_source_state_energy_ratios_within_five_percent": (
            source_energy_within_tolerance
        ),
        "original_shuffle_output_ratio": microstep["temporal_identifiability"][
            "shuffle_to_ordered_residual_energy_ratio"
        ],
        "original_shuffle_output_gate_passed": microstep[
            "temporal_identifiability"
        ]["gates"]["shuffle_residual_attenuated"],
        "input_energy_mismatch_explains_original_output_failure": False,
        "original_temporal_identifiability_failure_retained": True,
        "authorize_new_energy_normalized_gate": False,
        "authorize_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            "shuffle_output_exceeds_ordered_despite_matched_R1_R6_and_"
            "near_matched_source_energy"
        ),
        "boundary": audit["boundary"],
    }
