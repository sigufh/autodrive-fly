"""Test measured T5 source kernels against temporal shuffle/static controls."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_kohn_portes_t5_source_kernel_audit import (
    _author_rescaled_temporal,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_t5_typed_spatial_pair_precheck import (
    _build_typed_moments,
    _controlled_frames,
)
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-measured-kernel-identifiability.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_measured_kernel_identifiability.py")


def _population_kernel(records: list[dict], length: int) -> tuple[np.ndarray, int]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in records:
        grouped[str(record["recording_id"])].append(_author_rescaled_temporal(record, length))
    kernel = np.mean([np.mean(values, axis=0) for values in grouped.values()], axis=0)
    scale = float(np.sum(np.abs(kernel)))
    if not np.isfinite(kernel).all() or scale <= 0.0:
        raise ValueError("invalid population source kernel")
    return kernel / scale, len(grouped)


def _convolve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(
        lambda trace: np.convolve(trace, kernel, mode="full")[: len(trace)],
        0,
        values,
    )


def _validate_timebase(config: dict, coordinates: dict, lamina: dict) -> dict:
    kernel_ms = float(config["kernel"]["sample_interval_milliseconds"])
    frame_ms = float(coordinates["time_coordinates"]["frame_interval_milliseconds"])
    samples_per_frame = int(config["kernel"]["convolution_samples_per_stimulus_frame"])
    brain_updates = int(lamina["brain_substeps_per_frame"])
    gates = {
        "offline_time_coordinate_contract_complete": bool(
            coordinates["offline_time_coordinate_contract_complete"]
        ),
        "coordinate_contract_applies_to_offline_v7_evaluation": bool(
            coordinates["time_coordinates"][
                "applied_to_offline_v7_controlled_visual_evaluation"
            ]
        ),
        "kernel_sample_interval_matches_stimulus_frame_interval": bool(
            np.isclose(kernel_ms, frame_ms)
        ),
        "one_kernel_sample_per_stimulus_frame": samples_per_frame == 1,
        "probe_uses_declared_single_brain_update_per_frame": brain_updates == 1,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"measured-kernel timebase contract failed: {failed}")
    return {
        "kernel_sample_interval_milliseconds": kernel_ms,
        "stimulus_frame_interval_milliseconds": frame_ms,
        "convolution_samples_per_stimulus_frame": samples_per_frame,
        "probe_brain_updates_per_frame": brain_updates,
        "kernel_to_stimulus_frame_alignment_verified": True,
        "probe_brain_update_interval_milliseconds": None,
        "probe_brain_update_interval_biologically_calibrated": False,
        "external_recording_to_probe_solver_alignment_verified": False,
        "gates": gates,
    }


def _source_sequences(probe, stimulus, moments: dict, mode: str, seed: int) -> dict:
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
    sequence = []
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
        sequence.append({name: matrix @ positive for name, matrix in moments["matrices"].items()})
    return {name: np.stack([sample[name] for sample in sequence]) for name in moments["matrices"]}


def _candidate_traces(
    sequences: dict[str, np.ndarray], kernels: dict[str, np.ndarray], axes: np.ndarray
) -> dict[str, np.ndarray]:
    filtered = {
        f"{source}_{component}": _convolve(sequences[f"{source}_{component}"], kernels[source])
        for source in kernels
        for component in ("mass", "x", "y")
    }
    summed = {
        component: sum(filtered[f"{source}_{component}"] for source in kernels)
        for component in ("mass", "x", "y")
    }
    summed_projection = (summed["x"] * axes[:, 0] + summed["y"] * axes[:, 1]) / (
        np.abs(summed["x"]) + np.abs(summed["y"]) + 1e-9
    )

    fast = {
        component: sum(filtered[f"{source}_{component}"] for source in ("Tm1", "Tm2", "Tm4")) / 3.0
        for component in ("mass", "x", "y")
    }
    tm9 = {component: filtered[f"Tm9_{component}"] for component in ("mass", "x", "y")}
    x_terms = (fast["x"] * tm9["mass"], fast["mass"] * tm9["x"])
    y_terms = (fast["y"] * tm9["mass"], fast["mass"] * tm9["y"])
    centroid_projection = (
        (x_terms[0] - x_terms[1]) * axes[:, 0] + (y_terms[0] - y_terms[1]) * axes[:, 1]
    ) / (sum(np.abs(term) for term in (*x_terms, *y_terms)) + 1e-9)

    previous = {
        name: np.concatenate((np.zeros_like(values[:1]), values[:-1]))
        for name, values in filtered.items()
    }

    def reichardt(first: str, second: str, component: str) -> np.ndarray:
        terms = (
            filtered[f"{first}_{component}"] * previous[f"{second}_mass"],
            filtered[f"{first}_mass"] * previous[f"{second}_{component}"],
            filtered[f"{second}_{component}"] * previous[f"{first}_mass"],
            filtered[f"{second}_mass"] * previous[f"{first}_{component}"],
        )
        return (terms[0] - terms[1] - terms[2] + terms[3]) / (
            sum(np.abs(term) for term in terms) + 1e-9
        )

    horizontal = reichardt("Tm2", "Tm9", "x")
    vertical = reichardt("Tm9", "Tm1", "y")
    lagged_reichardt = horizontal * axes[:, 0] + vertical * axes[:, 1]
    temporal_difference = np.diff(
        lagged_reichardt, axis=0, prepend=np.zeros_like(lagged_reichardt[:1])
    )
    return {
        "summed_filtered_source_centroid_projection": summed_projection,
        "fast_pool_vs_Tm9_centroid_difference": centroid_projection,
        "temporal_difference_filtered_Tm_pair_reichardt": temporal_difference,
    }


def _energy(traces: list[np.ndarray], valid: np.ndarray) -> float:
    values = np.concatenate([trace[:, valid] for trace in traces], axis=0)
    if not np.isfinite(values).all():
        raise ValueError("non-finite measured-kernel diagnostic values")
    return float(np.mean(np.abs(values)))


def evaluate_v7_t5_measured_kernel_identifiability(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
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
    kernel_evidence = json.loads((root / paths["source_kernel_evidence"]).read_text())
    if not kernel_evidence["Tm1_Tm2_Tm4_Tm9_voltage_derived_temporal_kernels_verified"]:
        raise ValueError("four verified Tm kernel sets are required")
    source_config = yaml.safe_load((root / paths["source_config"]).read_text())
    coordinates = json.loads(
        (root / paths["stimulus_coordinate_evidence"]).read_text()
    )
    lamina = yaml.safe_load((root / paths["lamina_split_protocol"]).read_text())
    timebase = _validate_timebase(config, coordinates, lamina)
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    local = yaml.safe_load((root / paths["local_edge_config"]).read_text())
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("measured-kernel identifiability may consume tuning only")

    source_types = config["source_types"]
    kernel_length = min(kernel_evidence["aggregate"]["kernel_lengths"])
    kernels = {}
    recording_counts = {}
    source_paths = {}
    for source in source_types:
        spec = source_config["white_noise_files"][source]
        path = _verify_file(root, spec)
        source_paths[source] = path
        kernels[source], recording_counts[source] = _population_kernel(
            _load_restricted(path), kernel_length
        )

    probe = LaminaSplitProbe(root, lamina)
    populations = {
        f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    moments = _build_typed_moments(root, probe, targets)
    finite_axis = np.all(np.isfinite(moments["axes"]), axis=1)
    valid = moments["valid"] & finite_axis

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
    traces = {mode: {candidate["name"]: [] for candidate in config["candidates"]} for mode in modes}
    for stimulus in stimuli:
        for mode in modes:
            sequence = _source_sequences(probe, stimulus, moments, mode, seed)
            for name, trace in _candidate_traces(sequence, kernels, moments["axes"]).items():
                traces[mode][name].append(trace)

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
            <= float(config["controls"]["maximum_shuffle_to_ordered_residual_energy_ratio"]),
            "static_energy_attenuated": static_ratio
            <= float(config["controls"]["maximum_static_to_ordered_energy_ratio"]),
        }
        results[name] = {
            "formula": candidate["formula"],
            "ordered_mean_absolute_energy": ordered_energy,
            "static_mean_absolute_energy": static_energy,
            "ordered_minus_static_mean_absolute_energy": ordered_residual,
            "shuffle_minus_static_mean_absolute_energy": shuffle_residual,
            "shuffle_to_ordered_residual_energy_ratio": shuffle_ratio,
            "static_to_ordered_energy_ratio": static_ratio,
            "gates": gates,
            "passed": all(gates.values()),
        }
    temporal_passed = any(item["passed"] for item in results.values())
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        **{str(path): _sha256(root / path) for path in paths.values()},
        **{
            source_config["white_noise_files"][source]["path"]: _sha256(path)
            for source, path in source_paths.items()
        },
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "condition_id": config["condition_id"],
            "stimulus_count_per_mode": len(stimuli),
            "position_count": int(len(x_centers) * len(y_centers)),
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "kernel_summary": {
            "source_order": source_types,
            "recording_id_counts": recording_counts,
            "kernel_length": kernel_length,
            "sample_interval_milliseconds": config["kernel"]["sample_interval_milliseconds"],
            "normalization": config["kernel"]["normalization"],
        },
        "timebase_contract": timebase,
        "fixed_target_denominator": len(targets),
        "valid_target_count": int(np.count_nonzero(valid)),
        "invalid_source_or_axis_target_count": int(len(targets) - np.count_nonzero(valid)),
        "candidate_results": results,
        "temporal_identifiability_passed": temporal_passed,
        "direction_scoring_performed": temporal_passed,
        "authorize_T5_functional_precheck": False,
        "authorize_physical_source_dynamics_transfer": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "measured_kernel_source_readouts_failed_temporal_controls",
        "boundary": config["boundary"],
    }
