"""Evaluate the preregistered T5 increment-order control replication."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_t5_increment_order_control_discovery_audit import (
    _increment_order_control_frames,
    _lamina_only_source_matrices,
    _lamina_sequences_from_drives,
    _multiset_digest,
    _ratio_summary,
    _retinal_drives,
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

CONFIG = Path("configs/driving-v7-t5-increment-order-control-replication.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_increment_order_control_replication.py"
)


def _git_sha256(root: Path, revision: str, path: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def evaluate_v7_t5_increment_order_control_replication(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    prereg_path = Path(audit["preregistration_evidence"])
    if _sha256(root / prereg_path) != audit["preregistration_sha256"]:
        raise ValueError("preregistration artifact hash changed")
    revision = audit["preregistration_commit"]
    if _git_sha256(root, revision, str(prereg_path)) != audit["preregistration_sha256"]:
        raise ValueError("preregistration was not frozen at the declared commit")
    prereg = json.loads((root / prereg_path).read_text(encoding="utf-8"))
    if not prereg["replication_protocol_frozen"]:
        raise ValueError("replication protocol is not frozen")
    if prereg["protocol"]["replication_outputs_observed"]:
        raise ValueError("preregistration already contains replication outputs")
    if prereg["replication_condition_ids"] != audit["replication_condition_ids"]:
        raise ValueError("replication conditions differ from preregistration")

    discovery_implementation_path = Path(audit["discovery_implementation"])
    frozen_discovery = next(
        item
        for item in prereg["protocol"]["frozen_inputs"]
        if item["path"] == str(discovery_implementation_path)
    )
    if _sha256(root / discovery_implementation_path) != frozen_discovery["sha256"]:
        raise ValueError("discovery implementation changed after preregistration")

    replacement_path = Path(audit["lamina_only_replacement_evidence"])
    replacement = json.loads((root / replacement_path).read_text(encoding="utf-8"))
    replacement_config_path = Path(audit["lamina_only_replacement_config"])
    replacement_config = yaml.safe_load(
        (root / replacement_config_path).read_text(encoding="utf-8")
    )
    measured_path = Path(audit["measured_kernel_config"])
    measured = yaml.safe_load((root / measured_path).read_text(encoding="utf-8"))
    candidate_order = [item["name"] for item in measured["candidates"]]
    if candidate_order != prereg["candidate_order"]:
        raise ValueError("candidate order differs from preregistration")
    updates = [int(value) for value in prereg["tested_brain_updates_per_frame"]]
    settle_frames = int(prereg["settle_frames"])
    seed = int(prereg["control"]["seed"])
    thresholds = {
        "maximum_shuffle_to_ordered_residual_energy_ratio": float(
            prereg["inherited_output_thresholds"][
                "maximum_control_to_ordered_residual_energy_ratio"
            ]
        ),
        "maximum_static_to_ordered_energy_ratio": float(
            prereg["inherited_output_thresholds"][
                "maximum_static_to_ordered_energy_ratio"
            ]
        ),
    }
    if thresholds != {
        key: float(value) for key, value in replacement_config["controls"].items()
    }:
        raise ValueError("replication thresholds differ from inherited controls")

    source_config_path = Path(measured["source_config"])
    source_config = yaml.safe_load((root / source_config_path).read_text(encoding="utf-8"))
    lamina_path = Path(measured["lamina_split_protocol"])
    stage_path = Path(measured["stage1_protocol"])
    local_path = Path(measured["local_edge_config"])
    lamina = yaml.safe_load((root / lamina_path).read_text(encoding="utf-8"))
    stage = yaml.safe_load((root / stage_path).read_text(encoding="utf-8"))
    local = yaml.safe_load((root / local_path).read_text(encoding="utf-8"))
    conditions = {row["condition_id"]: row for row in stage["conditions"]}
    kernel_samples = int(replacement["replacement_contract"]["kernel_samples"])
    kernels = {}
    source_paths = {}
    for source in measured["source_types"]:
        spec = source_config["white_noise_files"][source]
        source_path = _verify_file(root, spec)
        source_paths[source] = source_path
        kernels[source], _ = _population_kernel(_load_restricted(source_path), kernel_samples)

    results = {}
    all_input_checks = []
    for condition_id in prereg["replication_condition_ids"]:
        condition = conditions[condition_id]
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
        condition_results = {}
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
            image_valid = []
            terminal_preserved = []
            pixel_multiset_preserved = []
            retinal_multiset_preserved = []
            retinal_energy_ratios = []
            orders = []
            element_count = 0
            for stimulus in stimuli:
                control_frames, order = _increment_order_control_frames(
                    stimulus.frames, baseline_frames, seed
                )
                orders.append(order)
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
                image_valid.append(bool(np.all((control_frames >= 0) & (control_frames <= 1))))
                terminal_error = float(
                    np.max(np.abs(control_frames[-1] - stimulus.frames[-1]))
                )
                terminal_preserved.append(
                    terminal_error
                    <= float(prereg["input_validity"]["terminal_frame_absolute_tolerance"])
                )
                pixel_multiset_preserved.append(
                    _multiset_digest(ordered_increments) == _multiset_digest(control_increments)
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
                    controlled = traces["increment_order_control"][name][:, valid]
                    static = traces["static_sham"][name][:, valid]
                    sums[name]["ordered"] += float(np.sum(np.abs(ordered)))
                    sums[name]["static"] += float(np.sum(np.abs(static)))
                    sums[name]["ordered_residual"] += float(np.sum(np.abs(ordered - static)))
                    sums[name]["shuffle_residual"] += float(np.sum(np.abs(controlled - static)))
                element_count += next(iter(traces["ordered"].values())).shape[0] * int(
                    np.count_nonzero(valid)
                )
            if not all(np.array_equal(orders[0], order) for order in orders[1:]):
                raise ValueError("increment order changed within a condition")
            retinal_tolerance = float(
                prereg["input_validity"]["R1_R6_energy_ratio_absolute_tolerance"]
            )
            input_gates = {
                "control_images_within_unit_interval": all(image_valid),
                "terminal_frame_preserved": all(terminal_preserved),
                "pixel_increment_multiset_preserved": all(pixel_multiset_preserved),
                "R1_R6_drive_multiset_preserved": all(retinal_multiset_preserved),
                "R1_R6_mean_absolute_energy_preserved": all(
                    abs(ratio - 1.0) <= retinal_tolerance
                    for ratio in retinal_energy_ratios
                ),
            }
            all_input_checks.extend(input_gates.values())
            condition_results[str(update_count)] = {
                "increment_order": orders[0].tolist(),
                "input_gates": input_gates,
                "R1_R6_mean_absolute_energy_ratio_summary": _ratio_summary(
                    retinal_energy_ratios
                ),
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
                    sums, element_count, thresholds
                ),
                "scored_element_count_per_candidate": element_count,
            }
        results[condition_id] = {
            "role": condition["role"],
            "edge_speed_pixels_per_frame": speed,
            "stimulus_count": len(stimuli),
            "by_brain_updates_per_frame": condition_results,
        }

    candidate_replication = {
        name: all(
            results[condition]["by_brain_updates_per_frame"][str(update)][
                "candidate_output_results"
            ][name]["passed"]
            for condition in prereg["replication_condition_ids"]
            for update in updates
        )
        for name in candidate_order
    }
    input_validity_passed = all(all_input_checks)
    same_candidate_passed = any(candidate_replication.values())
    replication_passed = input_validity_passed and same_candidate_passed
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(prereg_path): _sha256(root / prereg_path),
        str(discovery_implementation_path): _sha256(root / discovery_implementation_path),
        str(replacement_path): _sha256(root / replacement_path),
        str(replacement_config_path): _sha256(root / replacement_config_path),
        str(measured_path): _sha256(root / measured_path),
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
            "preregistration_commit": revision,
            "preregistration_sha256": audit["preregistration_sha256"],
            "discovery_condition_excluded": True,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "condition_results": results,
        "input_validity_passed_every_condition_and_update": input_validity_passed,
        "candidate_replication_passed_every_condition_and_update": candidate_replication,
        "same_candidate_passed_every_condition_and_update": same_candidate_passed,
        "replication_gate_passed": replication_passed,
        "direction_scoring_authorized": replication_passed,
        "direction_scoring_performed": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "replication_temporal_gate_passed_direction_scoring_required"
            if replication_passed
            else "no_candidate_passed_preregistered_replication_gate"
        ),
        "boundary": audit["boundary"],
    }
