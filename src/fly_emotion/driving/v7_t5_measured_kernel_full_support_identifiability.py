"""Test measured T5 kernels over their complete zero-tailed FIR support."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import fftconvolve

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_t5_measured_kernel_identifiability import (
    _population_kernel,
    _source_sequences,
)
from fly_emotion.driving.v7_t5_typed_spatial_pair_precheck import (
    _build_typed_moments,
)
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path(
    "configs/driving-v7-t5-measured-kernel-full-support-identifiability.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_measured_kernel_full_support_identifiability.py"
)


def _full_convolve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return fftconvolve(values, kernel[:, None], mode="full", axes=0)


def _full_candidate_traces(
    sequences: dict[str, np.ndarray], kernels: dict[str, np.ndarray], axes: np.ndarray
) -> dict[str, np.ndarray]:
    filtered = {
        f"{source}_{component}": _full_convolve(
            sequences[f"{source}_{component}"], kernels[source]
        )
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
        component: sum(
            filtered[f"{source}_{component}"] for source in ("Tm1", "Tm2", "Tm4")
        )
        / 3.0
        for component in ("mass", "x", "y")
    }
    tm9 = {component: filtered[f"Tm9_{component}"] for component in ("mass", "x", "y")}
    x_terms = (fast["x"] * tm9["mass"], fast["mass"] * tm9["x"])
    y_terms = (fast["y"] * tm9["mass"], fast["mass"] * tm9["y"])
    centroid_projection = (
        (x_terms[0] - x_terms[1]) * axes[:, 0]
        + (y_terms[0] - y_terms[1]) * axes[:, 1]
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
    lagged = horizontal * axes[:, 0] + vertical * axes[:, 1]
    temporal_difference = np.diff(lagged, axis=0, prepend=np.zeros_like(lagged[:1]))
    return {
        "summed_filtered_source_centroid_projection": summed_projection,
        "fast_pool_vs_Tm9_centroid_difference": centroid_projection,
        "temporal_difference_filtered_Tm_pair_reichardt": temporal_difference,
    }


def _empty_sums(candidate_order: list[str]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "ordered": 0.0,
            "static": 0.0,
            "ordered_residual": 0.0,
            "shuffle_residual": 0.0,
        }
        for name in candidate_order
    }


def _score_sums(sums: dict, count: int, controls: dict) -> dict:
    results = {}
    for name, values in sums.items():
        means = {key: value / count for key, value in values.items()}
        shuffle_ratio = means["shuffle_residual"] / max(
            means["ordered_residual"], 1e-12
        )
        static_ratio = means["static"] / max(means["ordered"], 1e-12)
        gates = {
            "shuffle_residual_attenuated": shuffle_ratio
            <= float(controls["maximum_shuffle_to_ordered_residual_energy_ratio"]),
            "static_energy_attenuated": static_ratio
            <= float(controls["maximum_static_to_ordered_energy_ratio"]),
        }
        results[name] = {
            "ordered_mean_absolute_energy": means["ordered"],
            "static_mean_absolute_energy": means["static"],
            "ordered_minus_static_mean_absolute_energy": means["ordered_residual"],
            "shuffle_minus_static_mean_absolute_energy": means["shuffle_residual"],
            "shuffle_to_ordered_residual_energy_ratio": shuffle_ratio,
            "static_to_ordered_energy_ratio": static_ratio,
            "gates": gates,
            "passed": all(gates.values()),
        }
    return results


def evaluate_v7_t5_measured_kernel_full_support_identifiability(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_path = Path(audit["measured_kernel_evidence"])
    measured_config_path = Path(audit["measured_kernel_config"])
    measured_implementation_path = Path(audit["measured_kernel_implementation"])
    substep_path = Path(audit["substep_sensitivity_evidence"])
    support_path = Path(audit["support_coverage_evidence"])
    v7_path = Path(audit["v7_contract"])
    evidence = json.loads((root / evidence_path).read_text(encoding="utf-8"))
    support = json.loads((root / support_path).read_text(encoding="utf-8"))
    measured = yaml.safe_load((root / measured_config_path).read_text(encoding="utf-8"))
    v7 = yaml.safe_load((root / v7_path).read_text(encoding="utf-8"))
    candidate_order = [item["name"] for item in measured["candidates"]]
    if candidate_order != audit["required_candidate_order"]:
        raise ValueError("full-support candidate order changed")
    if support["full_kernel_support_evaluated"]:
        raise ValueError("support audit unexpectedly claims full-kernel evaluation")
    full = audit["full_support"]
    kernel_length = int(evidence["kernel_summary"]["kernel_length"] )
    input_samples = int(support["window"]["post_baseline_trace_samples"] )
    output_samples = input_samples + kernel_length - 1
    if (
        input_samples != int(full["expected_input_samples"])
        or kernel_length != int(full["expected_kernel_samples"])
        or output_samples != int(full["expected_output_samples"])
    ):
        raise ValueError("full-support convolution length contract changed")
    updates = [int(value) for value in audit["tested_brain_updates_per_frame"]]
    if updates != [1, 4]:
        raise ValueError("full-support update-count contract changed")

    paths = {
        name: Path(measured[name])
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
    source_config = yaml.safe_load((root / paths["source_config"]).read_text())
    lamina = yaml.safe_load((root / paths["lamina_split_protocol"]).read_text())
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    local = yaml.safe_load((root / paths["local_edge_config"]).read_text())
    condition = next(
        row for row in stage1["conditions"] if row["condition_id"] == measured["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("full-support audit may consume tuning only")
    if int(v7["controlled_vision"]["brain_substeps_per_frame"]) != updates[-1]:
        raise ValueError("standard v7 update count changed")

    kernels = {}
    source_paths = {}
    for source in measured["source_types"]:
        spec = source_config["white_noise_files"][source]
        source_path = _verify_file(root, spec)
        source_paths[source] = source_path
        kernels[source], _ = _population_kernel(
            _load_restricted(source_path), kernel_length
        )

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
        for center_x in local["centers"]["x"]
        for center_y in local["centers"]["y"]
        for polarity in ("on", "off")
        for direction in ("left", "right", "up", "down")
    ]
    modes = measured["controls"]["modes"]
    seed = int(measured["controls"]["temporal_shuffle_seed"] )
    by_updates = {}
    valid_count = None
    fixed_denominator = None
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
        current_valid_count = int(np.count_nonzero(valid))
        if fixed_denominator is None:
            fixed_denominator = len(targets)
            valid_count = current_valid_count
        if len(targets) != fixed_denominator or current_valid_count != valid_count:
            raise ValueError("target denominator changed across update counts")
        sums = _empty_sums(candidate_order)
        element_count = 0
        for stimulus in stimuli:
            mode_traces = {}
            for mode in modes:
                sequences = _source_sequences(probe, stimulus, moments, mode, seed)
                if {len(values) for values in sequences.values()} != {input_samples}:
                    raise ValueError("source-drive input length changed")
                mode_traces[mode] = _full_candidate_traces(
                    sequences, kernels, moments["axes"]
                )
            for name in candidate_order:
                ordered = mode_traces["ordered"][name][:, valid]
                shuffled = mode_traces["temporal_shuffle"][name][:, valid]
                static = mode_traces["static_sham"][name][:, valid]
                if ordered.shape[0] != output_samples:
                    raise ValueError("full-support output length changed")
                arrays = (ordered, shuffled, static)
                if not all(np.isfinite(array).all() for array in arrays):
                    raise ValueError("non-finite full-support diagnostic values")
                sums[name]["ordered"] += float(np.sum(np.abs(ordered)))
                sums[name]["static"] += float(np.sum(np.abs(static)))
                sums[name]["ordered_residual"] += float(
                    np.sum(np.abs(ordered - static))
                )
                sums[name]["shuffle_residual"] += float(
                    np.sum(np.abs(shuffled - static))
                )
            element_count += output_samples * current_valid_count
        by_updates[str(update_count)] = {
            "candidate_results": _score_sums(sums, element_count, audit["controls"]),
            "scored_element_count_per_candidate": element_count,
        }

    candidates = {}
    for name in candidate_order:
        passed_all = all(
            by_updates[str(update)]["candidate_results"][name]["passed"]
            for update in updates
        )
        candidates[name] = {
            "by_brain_updates_per_frame": {
                str(update): by_updates[str(update)]["candidate_results"][name]
                for update in updates
            },
            "passed_every_update_count": passed_all,
        }
    cross_substep_passed = any(
        item["passed_every_update_count"] for item in candidates.values()
    )
    all_candidates_failed_every_update_count = all(
        not by_updates[str(update)]["candidate_results"][name]["passed"]
        for update in updates
        for name in candidate_order
    )
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(evidence_path): _sha256(root / evidence_path),
        str(measured_config_path): _sha256(root / measured_config_path),
        str(measured_implementation_path): _sha256(root / measured_implementation_path),
        str(substep_path): _sha256(root / substep_path),
        str(support_path): _sha256(root / support_path),
        str(v7_path): _sha256(root / v7_path),
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
            "condition_id": measured["condition_id"],
            "stimulus_count_per_mode": len(stimuli),
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "full_support_contract": {
            "source_input_samples": input_samples,
            "kernel_samples": kernel_length,
            "zero_tail_samples": kernel_length - 1,
            "output_samples": output_samples,
            "source_drive_extension": full["source_drive_extension"],
            "neural_network_advanced_during_zero_tail": False,
            "tested_brain_updates_per_frame": updates,
            "fixed_target_denominator": fixed_denominator,
            "valid_target_count": valid_count,
        },
        "candidate_results": candidates,
        "full_kernel_support_evaluated": True,
        "all_candidates_failed_every_update_count": (
            all_candidates_failed_every_update_count
        ),
        "cross_substep_temporal_identifiability_passed": cross_substep_passed,
        "direction_scoring_authorized": cross_substep_passed,
        "direction_scoring_performed": False,
        "authorize_T5_functional_precheck": False,
        "authorize_physical_source_dynamics_transfer": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "full_support_temporal_gate_passed_direction_scoring_required"
            if cross_substep_passed
            else "no_full_support_candidate_passed_both_substep_resolutions"
        ),
        "boundary": audit["boundary"],
    }
