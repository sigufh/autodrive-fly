"""Test measured T5 kernel controls across diagnostic and standard substeps."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_t5_measured_kernel_identifiability import (
    _candidate_traces,
    _energy,
    _population_kernel,
    _source_sequences,
)
from fly_emotion.driving.v7_t5_typed_spatial_pair_precheck import (
    _build_typed_moments,
)
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path(
    "configs/driving-v7-t5-measured-kernel-substep-sensitivity.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_measured_kernel_substep_sensitivity.py"
)


def _score_traces(config: dict, traces: dict, valid: np.ndarray) -> dict:
    results = {}
    for candidate in config["candidates"]:
        name = candidate["name"]
        ordered = traces["ordered"][name]
        shuffled = traces["temporal_shuffle"][name]
        static = traces["static_sham"][name]
        ordered_energy = _energy(ordered, valid)
        static_energy = _energy(static, valid)
        ordered_residual = _energy(
            [value - sham for value, sham in zip(ordered, static, strict=True)], valid
        )
        shuffle_residual = _energy(
            [value - sham for value, sham in zip(shuffled, static, strict=True)], valid
        )
        shuffle_ratio = shuffle_residual / max(ordered_residual, 1e-12)
        static_ratio = static_energy / max(ordered_energy, 1e-12)
        gates = {
            "shuffle_residual_attenuated": shuffle_ratio
            <= float(
                config["controls"][
                    "maximum_shuffle_to_ordered_residual_energy_ratio"
                ]
            ),
            "static_energy_attenuated": static_ratio
            <= float(config["controls"]["maximum_static_to_ordered_energy_ratio"]),
        }
        results[name] = {
            "ordered_mean_absolute_energy": ordered_energy,
            "static_mean_absolute_energy": static_energy,
            "ordered_minus_static_mean_absolute_energy": ordered_residual,
            "shuffle_minus_static_mean_absolute_energy": shuffle_residual,
            "shuffle_to_ordered_residual_energy_ratio": shuffle_ratio,
            "static_to_ordered_energy_ratio": static_ratio,
            "gates": gates,
            "passed": all(gates.values()),
        }
    return results


def evaluate_v7_t5_measured_kernel_substep_sensitivity(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    reference_config_path = Path(audit["reference_config"])
    reference_evidence_path = Path(audit["reference_evidence"])
    reference_implementation_path = Path(audit["reference_implementation"])
    v7_contract_path = Path(audit["v7_contract"])
    config = yaml.safe_load((root / reference_config_path).read_text(encoding="utf-8"))
    reference = json.loads((root / reference_evidence_path).read_text(encoding="utf-8"))
    v7 = yaml.safe_load((root / v7_contract_path).read_text(encoding="utf-8"))
    candidate_order = [item["name"] for item in config["candidates"]]
    if candidate_order != audit["required_candidate_order"]:
        raise ValueError("measured-kernel candidate order changed")
    if list(reference["candidate_results"]) != candidate_order:
        raise ValueError("reference measured-kernel results changed")

    paths = {
        name: Path(config[name])
        for name in (
            "source_kernel_evidence",
            "source_config",
            "stimulus_coordinate_evidence",
            "stimulus_coordinate_protocol",
            "lamina_split_protocol",
            "typed_spatial_implementation",
            "stage1_protocol",
            "local_edge_config",
        )
    }
    coordinates = json.loads(
        (root / paths["stimulus_coordinate_evidence"]).read_text(encoding="utf-8")
    )
    lamina = yaml.safe_load(
        (root / paths["lamina_split_protocol"]).read_text(encoding="utf-8")
    )
    reference_updates = int(audit["reference_brain_updates_per_frame"])
    standard_updates = int(audit["standard_offline_brain_updates_per_frame"])
    if audit["tested_brain_updates_per_frame"] != [reference_updates, standard_updates]:
        raise ValueError("tested update-count order changed")
    if not audit["cross_substep_gate"][
        "require_same_candidate_to_pass_every_update_count"
    ]:
        raise ValueError("cross-substep candidate identity gate must remain enabled")
    if not audit["cross_substep_gate"][
        "evaluate_direction_only_after_cross_substep_temporal_identifiability"
    ]:
        raise ValueError("direction-scoring stop gate must remain enabled")
    if int(lamina["brain_substeps_per_frame"]) != reference_updates:
        raise ValueError("reference diagnostic update count changed")
    if int(v7["controlled_vision"]["brain_substeps_per_frame"]) != standard_updates:
        raise ValueError("standard v7 update count changed")
    if int(coordinates["time_coordinates"]["neural_substeps_per_frame"]) != standard_updates:
        raise ValueError("coordinate-contract update count changed")

    kernel_evidence = json.loads(
        (root / paths["source_kernel_evidence"]).read_text(encoding="utf-8")
    )
    source_config = yaml.safe_load(
        (root / paths["source_config"]).read_text(encoding="utf-8")
    )
    kernel_length = min(kernel_evidence["aggregate"]["kernel_lengths"])
    kernels = {}
    source_paths = {}
    for source in config["source_types"]:
        spec = source_config["white_noise_files"][source]
        source_path = _verify_file(root, spec)
        source_paths[source] = source_path
        kernels[source], _ = _population_kernel(
            _load_restricted(source_path), kernel_length
        )

    sensitivity_lamina = copy.deepcopy(lamina)
    sensitivity_lamina["brain_substeps_per_frame"] = standard_updates
    probe = LaminaSplitProbe(root, sensitivity_lamina)
    populations = {
        f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    moments = _build_typed_moments(root, probe, targets)
    valid = moments["valid"] & np.all(np.isfinite(moments["axes"]), axis=1)
    if len(targets) != int(reference["fixed_target_denominator"]):
        raise ValueError("fixed target denominator changed across update counts")
    if int(np.count_nonzero(valid)) != int(reference["valid_target_count"]):
        raise ValueError("valid target count changed across update counts")

    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    local = yaml.safe_load((root / paths["local_edge_config"]).read_text())
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("substep sensitivity may consume tuning only")
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    stimuli = [
        _local_step_edge(
            float(center_x),
            float(center_y),
            float(local["stimulus"]["aperture_radius_pixels"]),
            speed,
            polarity,
            direction,
            int(local["common_background_frames"]),
        )
        for center_x in x_centers
        for center_y in y_centers
        for polarity in ("on", "off")
        for direction in ("left", "right", "up", "down")
    ]
    modes = config["controls"]["modes"]
    seed = int(config["controls"]["temporal_shuffle_seed"])
    traces = {mode: {name: [] for name in candidate_order} for mode in modes}
    for stimulus in stimuli:
        for mode in modes:
            sequence = _source_sequences(probe, stimulus, moments, mode, seed)
            candidate_traces = _candidate_traces(sequence, kernels, moments["axes"])
            if list(candidate_traces) != candidate_order:
                raise ValueError("implemented measured-kernel candidates changed")
            for name, trace in candidate_traces.items():
                traces[mode][name].append(trace)
    standard_results = _score_traces(config, traces, valid)

    candidate_results = {}
    for name in candidate_order:
        reference_result = reference["candidate_results"][name]
        standard_result = standard_results[name]
        candidate_results[name] = {
            "by_brain_updates_per_frame": {
                str(reference_updates): reference_result,
                str(standard_updates): standard_result,
            },
            "passed_every_update_count": bool(
                reference_result["passed"] and standard_result["passed"]
            ),
        }
    cross_substep_passed = any(
        item["passed_every_update_count"] for item in candidate_results.values()
    )
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(reference_evidence_path): _sha256(root / reference_evidence_path),
        str(reference_config_path): _sha256(root / reference_config_path),
        str(reference_implementation_path): _sha256(root / reference_implementation_path),
        str(v7_contract_path): _sha256(root / v7_contract_path),
        **{str(path): _sha256(root / path) for path in paths.values()},
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
            "condition_id": config["condition_id"],
            "stimulus_count_per_mode": len(stimuli),
            "position_count": int(len(x_centers) * len(y_centers)),
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "timebase_sensitivity": {
            "stimulus_frame_interval_milliseconds": coordinates["time_coordinates"][
                "frame_interval_milliseconds"
            ],
            "kernel_sample_interval_milliseconds": config["kernel"][
                "sample_interval_milliseconds"
            ],
            "tested_brain_updates_per_frame": audit[
                "tested_brain_updates_per_frame"
            ],
            "reference_diagnostic_updates_per_frame": reference_updates,
            "standard_offline_updates_per_frame": standard_updates,
            "solver_interval_biologically_calibrated": False,
        },
        "fixed_target_denominator": len(targets),
        "valid_target_count": int(np.count_nonzero(valid)),
        "candidate_results": candidate_results,
        "standard_substep_negative_result_reproduced": all(
            not item["passed"] for item in standard_results.values()
        ),
        "cross_substep_temporal_identifiability_passed": cross_substep_passed,
        "direction_scoring_authorized": cross_substep_passed,
        "direction_scoring_performed": False,
        "authorize_T5_functional_precheck": False,
        "authorize_physical_source_dynamics_transfer": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "cross_substep_temporal_gate_passed_direction_scoring_required"
            if cross_substep_passed
            else "no_measured_kernel_candidate_passed_both_substep_resolutions"
        ),
        "boundary": audit["boundary"],
    }
