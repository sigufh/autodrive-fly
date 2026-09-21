"""Discover an R1-R6-energy-matched T5 temporal-order control."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_t5_lamina_only_measured_kernel_replacement import (
    _lamina_only_source_matrices,
    _update_lamina_only,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_t5_measured_kernel_full_support_identifiability import (
    _empty_sums,
    _full_candidate_traces,
    _score_sums,
)
from fly_emotion.driving.v7_t5_measured_kernel_identifiability import (
    _population_kernel,
)
from fly_emotion.driving.v7_t5_typed_spatial_pair_precheck import (
    _build_typed_moments,
    _controlled_frames,
)
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path(
    "configs/driving-v7-t5-increment-order-control-discovery-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_increment_order_control_discovery_audit.py"
)


def _multiset_digest(values: np.ndarray) -> str:
    hashes = sorted(hashlib.sha256(value.tobytes()).hexdigest() for value in values)
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


def _increment_order_control_frames(
    frames: np.ndarray, baseline_frames: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    prefix = frames[:baseline_frames]
    start = prefix[-1]
    moving = frames[baseline_frames:]
    increments = np.diff(np.concatenate((start[None], moving)), axis=0)
    order = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            1 + np.random.default_rng(seed).permutation(len(increments) - 1),
        )
    )
    reconstructed = (
        start.astype(np.float64)
        + np.cumsum(increments[order].astype(np.float64), axis=0)
    ).astype(frames.dtype)
    return np.concatenate((prefix, reconstructed)), order


def _retinal_drives(probe: LaminaSplitProbe, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = probe._sample_retina(frames[0])[probe.retinal_permutation]
    previous = baseline.copy()
    drives = []
    for image in frames[probe.baseline_frames :]:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        drives.append(probe._retinal_code(sampled, baseline, previous))
        previous = sampled
    return baseline, np.stack(drives)


def _lamina_sequences_from_drives(
    probe: LaminaSplitProbe,
    baseline: np.ndarray,
    drives: np.ndarray,
    settle_frames: int,
    source_matrices: dict[str, object],
    source_nodes: dict[str, np.ndarray],
    moments: dict,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    zero_drive = probe._retinal_code(baseline, baseline, baseline)
    for _ in range(probe.baseline_frames):
        _update_lamina_only(probe, state, zero_drive)
    sequence = []
    source_energy = {source: 0.0 for source in source_matrices}
    source_count = {source: 0 for source in source_matrices}
    for retinal_drive in [*drives, *([zero_drive] * settle_frames)]:
        preactivation = {}
        for _ in range(probe.brain_substeps):
            preactivation = {
                source: matrix @ (state * probe.source_sign)
                for source, matrix in source_matrices.items()
            }
            _update_lamina_only(probe, state, retinal_drive)
        source_state = np.zeros(probe.graph.node_count, dtype=np.float64)
        for source, values in preactivation.items():
            positive = np.maximum(values, 0.0)
            source_state[source_nodes[source]] = positive
            source_energy[source] += float(np.sum(np.abs(positive)))
            source_count[source] += positive.size
        sequence.append(
            {name: matrix @ source_state for name, matrix in moments["matrices"].items()}
        )
    return (
        {
            name: np.stack([sample[name] for sample in sequence])
            for name in moments["matrices"]
        },
        {source: source_energy[source] / source_count[source] for source in source_matrices},
    )


def _ratio_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def evaluate_v7_t5_increment_order_control_discovery_audit(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_paths = {
        name: Path(audit[name])
        for name in (
            "input_energy_evidence",
            "lamina_only_replacement_evidence",
        )
    }
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }
    if evidence["input_energy_evidence"][
        "energy_matched_temporal_shuffle_control_verified"
    ]:
        raise ValueError("frame-shuffle control unexpectedly became energy matched")
    replacement = evidence["lamina_only_replacement_evidence"]
    measured_path = Path(audit["measured_kernel_config"])
    measured = yaml.safe_load((root / measured_path).read_text(encoding="utf-8"))
    if measured["condition_id"] != audit["discovery_condition_id"]:
        raise ValueError("discovery condition changed")
    candidate_order = [item["name"] for item in measured["candidates"]]
    if candidate_order != audit["required_candidate_order"]:
        raise ValueError("candidate order changed")
    updates = [int(value) for value in audit["tested_brain_updates_per_frame"]]
    if updates != [1, 4]:
        raise ValueError("update-count contract changed")
    settle_frames = int(audit["settle_frames"])
    if settle_frames != 10:
        raise ValueError("settle-window contract changed")

    replacement_config_path = Path(audit["lamina_only_replacement_config"])
    replacement_config = yaml.safe_load(
        (root / replacement_config_path).read_text(encoding="utf-8")
    )
    source_config_path = Path(measured["source_config"])
    source_config = yaml.safe_load((root / source_config_path).read_text(encoding="utf-8"))
    lamina_path = Path(measured["lamina_split_protocol"])
    stage_path = Path(measured["stage1_protocol"])
    local_path = Path(measured["local_edge_config"])
    lamina = yaml.safe_load((root / lamina_path).read_text(encoding="utf-8"))
    stage = yaml.safe_load((root / stage_path).read_text(encoding="utf-8"))
    local = yaml.safe_load((root / local_path).read_text(encoding="utf-8"))
    condition = next(
        row for row in stage["conditions"] if row["condition_id"] == measured["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("discovery audit may consume tuning only")
    control = audit["control"]
    if int(control["seed"]) != int(measured["controls"]["temporal_shuffle_seed"]):
        raise ValueError("control seed changed from the audited shuffle seed")
    if not all(
        control[name]
        for name in (
            "preserve_initialization_increment",
            "reconstruct_images_from_baseline_and_permuted_increments",
            "require_images_within_unit_interval",
            "require_terminal_frame_preserved",
            "require_pixel_increment_multiset_preserved",
            "require_R1_R6_drive_multiset_preserved",
            "require_R1_R6_mean_absolute_energy_preserved",
        )
    ):
        raise ValueError("increment-order control contract changed")

    kernel_samples = int(replacement["replacement_contract"]["kernel_samples"])
    kernels = {}
    source_paths = {}
    for source in measured["source_types"]:
        spec = source_config["white_noise_files"][source]
        source_path = _verify_file(root, spec)
        source_paths[source] = source_path
        kernels[source], _ = _population_kernel(_load_restricted(source_path), kernel_samples)

    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    baseline_frames = int(local["common_background_frames"])
    stimuli = [
        _local_step_edge(
            float(center_x),
            float(center_y),
            float(local["stimulus"]["aperture_radius_pixels"]),
            speed,
            polarity,
            direction,
            baseline_frames,
        )
        for center_x in local["centers"]["x"]
        for center_y in local["centers"]["y"]
        for polarity in ("on", "off")
        for direction in ("left", "right", "up", "down")
    ]
    seed = int(control["seed"])
    input_samples = int(replacement["replacement_contract"]["source_input_samples"])
    expected_shuffled = int(control["shuffle_remaining_increment_count"])
    if input_samples - 1 != expected_shuffled:
        raise ValueError("shuffled increment count changed")

    image_valid = []
    terminal_preserved = []
    terminal_maximum_absolute_errors = []
    pixel_multiset_preserved = []
    retinal_multiset_preserved = []
    retinal_energy_ratios = []
    order = None
    by_updates = {}
    for update_count in updates:
        probe_config = copy.deepcopy(lamina)
        probe_config["brain_substeps_per_frame"] = update_count
        probe = LaminaSplitProbe(root, probe_config)
        populations = {
            f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
            for subtype in "abcd"
            for side in "LR"
        }
        targets = np.concatenate(list(populations.values())).astype(np.int32)
        moments = _build_typed_moments(root, probe, targets)
        valid = moments["valid"] & np.all(np.isfinite(moments["axes"]), axis=1)
        source_nodes = {
            source: np.flatnonzero(probe.node_types == source).astype(np.int32)
            for source in measured["source_types"]
        }
        source_matrices, _ = _lamina_only_source_matrices(
            probe, measured["source_types"], ["L1", "L2", "L3"]
        )
        sums = _empty_sums(candidate_order)
        lamina_ratios = {source: [] for source in measured["source_types"]}
        lamina_totals = {
            mode: {source: 0.0 for source in measured["source_types"]}
            for mode in ("ordered", "increment_order_control")
        }
        element_count = 0
        for stimulus in stimuli:
            control_frames, current_order = _increment_order_control_frames(
                stimulus.frames, baseline_frames, seed
            )
            if order is None:
                order = current_order
            elif not np.array_equal(order, current_order):
                raise ValueError("increment permutation changed across stimuli")
            ordered_increments = np.diff(
                np.concatenate(
                    (stimulus.frames[baseline_frames - 1 : baseline_frames],
                     stimulus.frames[baseline_frames:])
                ),
                axis=0,
            )
            control_increments = np.diff(
                np.concatenate(
                    (control_frames[baseline_frames - 1 : baseline_frames],
                     control_frames[baseline_frames:])
                ),
                axis=0,
            )
            image_valid.append(
                bool(np.all((control_frames >= 0.0) & (control_frames <= 1.0)))
            )
            terminal_error = float(
                np.max(np.abs(control_frames[-1] - stimulus.frames[-1]))
            )
            terminal_maximum_absolute_errors.append(terminal_error)
            terminal_preserved.append(
                terminal_error <= float(control["terminal_frame_absolute_tolerance"])
            )
            pixel_multiset_preserved.append(
                _multiset_digest(ordered_increments)
                == _multiset_digest(control_increments)
            )
            baseline, ordered_drives = _retinal_drives(probe, stimulus.frames)
            control_baseline, control_drives = _retinal_drives(probe, control_frames)
            if not np.array_equal(baseline, control_baseline):
                raise ValueError("control baseline changed")
            retinal_multiset_preserved.append(
                _multiset_digest(ordered_drives) == _multiset_digest(control_drives)
            )
            retinal_energy_ratios.append(
                float(np.mean(np.abs(control_drives)) / np.mean(np.abs(ordered_drives)))
            )
            static_frames = _controlled_frames(stimulus, "static_sham", seed)
            static_baseline, static_drives = _retinal_drives(probe, static_frames)
            if not np.array_equal(baseline, static_baseline):
                raise ValueError("static baseline changed")
            sequences = {}
            energies = {}
            for mode, drives in (
                ("ordered", ordered_drives),
                ("increment_order_control", control_drives),
                ("static_sham", static_drives),
            ):
                sequences[mode], energies[mode] = _lamina_sequences_from_drives(
                    probe,
                    baseline,
                    drives,
                    settle_frames,
                    source_matrices,
                    source_nodes,
                    moments,
                )
            for source in measured["source_types"]:
                ordered_energy = energies["ordered"][source]
                control_energy = energies["increment_order_control"][source]
                lamina_totals["ordered"][source] += ordered_energy
                lamina_totals["increment_order_control"][source] += control_energy
                lamina_ratios[source].append(control_energy / ordered_energy)
            traces = {
                mode: _full_candidate_traces(values, kernels, moments["axes"])
                for mode, values in sequences.items()
            }
            for name in candidate_order:
                ordered = traces["ordered"][name][:, valid]
                shuffled = traces["increment_order_control"][name][:, valid]
                static = traces["static_sham"][name][:, valid]
                sums[name]["ordered"] += float(np.sum(np.abs(ordered)))
                sums[name]["static"] += float(np.sum(np.abs(static)))
                sums[name]["ordered_residual"] += float(np.sum(np.abs(ordered - static)))
                sums[name]["shuffle_residual"] += float(np.sum(np.abs(shuffled - static)))
            element_count += next(iter(traces["ordered"].values())).shape[0] * int(
                np.count_nonzero(valid)
            )
        by_updates[str(update_count)] = {
            "lamina_only_source_energy_ratio": {
                source: (
                    lamina_totals["increment_order_control"][source]
                    / lamina_totals["ordered"][source]
                )
                for source in measured["source_types"]
            },
            "lamina_only_source_per_stimulus_ratio_summary": {
                source: _ratio_summary(values) for source, values in lamina_ratios.items()
            },
            "candidate_output_results": _score_sums(
                sums, element_count, replacement_config["controls"]
            ),
            "scored_element_count_per_candidate": element_count,
        }

    if order is None:
        raise ValueError("no increment permutation generated")
    fixed_points = int(np.count_nonzero(order == np.arange(len(order))))
    preserved_forward = int(np.count_nonzero(np.diff(order) == 1))
    preserved_reverse = int(np.count_nonzero(np.diff(order) == -1))
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        **{str(path): _sha256(root / path) for path in evidence_paths.values()},
        str(replacement_config_path): _sha256(root / replacement_config_path),
        str(Path(audit["lamina_only_replacement_implementation"])): _sha256(
            root / audit["lamina_only_replacement_implementation"]
        ),
        str(measured_path): _sha256(root / measured_path),
        str(Path(audit["measured_kernel_implementation"])): _sha256(
            root / audit["measured_kernel_implementation"]
        ),
        str(Path(audit["full_support_implementation"])): _sha256(
            root / audit["full_support_implementation"]
        ),
        str(source_config_path): _sha256(root / source_config_path),
        str(lamina_path): _sha256(root / lamina_path),
        str(stage_path): _sha256(root / stage_path),
        str(local_path): _sha256(root / local_path),
        **{
            source_config["white_noise_files"][source]["path"]: _sha256(path)
            for source, path in source_paths.items()
        },
    }
    return {
        "protocol": {
            "name": audit["name"],
            "observed_on": audit["observed_on"],
            "dependencies_sha256": dependencies,
            "discovery_condition_id": measured["condition_id"],
            "stimulus_count": len(stimuli),
            "settle_frames": settle_frames,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "increment_order": order.tolist(),
        "increment_order_fixed_point_count": fixed_points,
        "preserved_forward_adjacency_count": preserved_forward,
        "preserved_reverse_adjacency_count": preserved_reverse,
        "control_images_within_unit_interval": all(image_valid),
        "terminal_frame_preserved_for_every_control": all(terminal_preserved),
        "terminal_frame_maximum_absolute_error": max(
            terminal_maximum_absolute_errors
        ),
        "pixel_increment_multiset_preserved_for_every_control": all(
            pixel_multiset_preserved
        ),
        "R1_R6_drive_multiset_preserved_for_every_control": all(
            retinal_multiset_preserved
        ),
        "R1_R6_mean_absolute_energy_ratio_summary": _ratio_summary(
            retinal_energy_ratios
        ),
        "R1_R6_mean_absolute_energy_preserved_within_tolerance": all(
            abs(ratio - 1.0) <= float(control["energy_ratio_absolute_tolerance"])
            for ratio in retinal_energy_ratios
        ),
        "by_brain_updates_per_frame": by_updates,
        "existing_temporal_gate_replaced": False,
        "independent_condition_evaluation_performed": False,
        "new_acceptance_threshold_defined": False,
        "direction_scoring_authorized": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "post_hoc_control_discovery_requires_frozen_independent_evaluation",
        "boundary": audit["boundary"],
    }
