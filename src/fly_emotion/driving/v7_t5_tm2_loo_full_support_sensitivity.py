"""Test full-support T5 controls across Tm2 recording-ID jackknife kernels."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import fftconvolve

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_kohn_portes_t5_source_kernel_audit import (
    _author_rescaled_temporal,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_t5_measured_kernel_full_support_identifiability import (
    _score_sums,
)
from fly_emotion.driving.v7_t5_measured_kernel_identifiability import (
    _population_kernel,
    _source_sequences,
)
from fly_emotion.driving.v7_t5_typed_spatial_pair_precheck import (
    _build_typed_moments,
)
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-tm2-loo-full-support-sensitivity.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_tm2_loo_full_support_sensitivity.py"
)


def _batched_full_convolve(values: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    return fftconvolve(values[None, :, :], kernels[:, :, None], mode="full", axes=1)


def _candidate_variants(
    sequences: dict[str, np.ndarray],
    fixed_kernels: dict[str, np.ndarray],
    tm2_kernels: np.ndarray,
    axes: np.ndarray,
) -> dict[str, np.ndarray]:
    filtered = {}
    for source in fixed_kernels:
        for component in ("mass", "x", "y"):
            values = sequences[f"{source}_{component}"]
            if source == "Tm2":
                filtered[f"{source}_{component}"] = _batched_full_convolve(
                    values, tm2_kernels
                )
            else:
                filtered[f"{source}_{component}"] = fftconvolve(
                    values, fixed_kernels[source][:, None], mode="full", axes=0
                )[None, :, :]

    summed_x = sum(filtered[f"{source}_x"] for source in fixed_kernels)
    summed_y = sum(filtered[f"{source}_y"] for source in fixed_kernels)
    summed_projection = (
        summed_x * axes[None, :, 0] + summed_y * axes[None, :, 1]
    ) / (np.abs(summed_x) + np.abs(summed_y) + 1e-9)

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
    centroid = (
        (x_terms[0] - x_terms[1]) * axes[None, :, 0]
        + (y_terms[0] - y_terms[1]) * axes[None, :, 1]
    ) / (sum(np.abs(term) for term in (*x_terms, *y_terms)) + 1e-9)

    previous = {
        name: np.concatenate((np.zeros_like(values[:, :1]), values[:, :-1]), axis=1)
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

    lagged = (
        reichardt("Tm2", "Tm9", "x") * axes[None, :, 0]
        + reichardt("Tm9", "Tm1", "y") * axes[None, :, 1]
    )
    temporal_difference = np.diff(
        lagged, axis=1, prepend=np.zeros_like(lagged[:, :1])
    )
    return {
        "summed_filtered_source_centroid_projection": summed_projection,
        "fast_pool_vs_Tm9_centroid_difference": centroid,
        "temporal_difference_filtered_Tm_pair_reichardt": temporal_difference,
    }


def evaluate_v7_t5_tm2_loo_full_support_sensitivity(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    robustness_path = Path(audit["population_kernel_robustness_evidence"])
    reference_path = Path(audit["reference_full_support_evidence"])
    reference_config_path = Path(audit["reference_full_support_config"])
    measured_config_path = Path(audit["measured_kernel_config"])
    measured_implementation_path = Path(audit["measured_kernel_implementation"])
    full_implementation_path = Path(audit["full_support_implementation"])
    v7_path = Path(audit["v7_contract"])
    robustness = json.loads((root / robustness_path).read_text(encoding="utf-8"))
    reference = json.loads((root / reference_path).read_text(encoding="utf-8"))
    reference_config = yaml.safe_load(
        (root / reference_config_path).read_text(encoding="utf-8")
    )
    measured = yaml.safe_load((root / measured_config_path).read_text(encoding="utf-8"))
    v7 = yaml.safe_load((root / v7_path).read_text(encoding="utf-8"))
    sensitivity_source = audit["sensitivity_source"]
    if robustness["failing_sources"] != [sensitivity_source]:
        raise ValueError("sensitivity must remain restricted to the sole failing source")
    if robustness["passing_sources"] != audit["expected_other_sources_passing_robustness_gate"]:
        raise ValueError("passing-source robustness inventory changed")
    candidate_order = [item["name"] for item in measured["candidates"]]
    if candidate_order != audit["required_candidate_order"]:
        raise ValueError("Tm2 LOO candidate order changed")
    if reference_config["controls"] != audit["controls"]:
        raise ValueError("Tm2 LOO control thresholds changed")
    updates = [int(value) for value in audit["tested_brain_updates_per_frame"]]
    if updates != [1, 4] or int(v7["controlled_vision"]["brain_substeps_per_frame"]) != 4:
        raise ValueError("Tm2 LOO update-count contract changed")

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
        raise ValueError("Tm2 LOO sensitivity may consume tuning only")
    full_contract = reference["full_support_contract"]
    input_samples = int(full_contract["source_input_samples"] )
    kernel_length = int(full_contract["kernel_samples"] )
    output_samples = int(full_contract["output_samples"] )

    fixed_kernels = {}
    source_paths = {}
    tm2_records = None
    for source in measured["source_types"]:
        spec = source_config["white_noise_files"][source]
        path = _verify_file(root, spec)
        source_paths[source] = path
        records = _load_restricted(path)
        fixed_kernels[source], _ = _population_kernel(records, kernel_length)
        if source == sensitivity_source:
            tm2_records = records
    if tm2_records is None:
        raise ValueError("Tm2 records missing")
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in tm2_records:
        grouped[str(record["recording_id"])].append(
            _author_rescaled_temporal(record, kernel_length)
        )
    recording_ids = sorted(grouped)
    if len(recording_ids) != int(audit["expected_fold_count"]):
        raise ValueError("Tm2 LOO fold count changed")
    per_id = {name: np.mean(values, axis=0) for name, values in grouped.items()}
    tm2_kernels = []
    for held_out in recording_ids:
        kernel = np.mean(
            [values for name, values in per_id.items() if name != held_out], axis=0
        )
        kernel /= np.sum(np.abs(kernel))
        tm2_kernels.append(kernel)
    tm2_kernel_bank = np.stack(tm2_kernels)

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
    fold_count = len(recording_ids)
    by_updates = {}
    fixed_denominator = None
    valid_count = None
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
        valid_indices = np.flatnonzero(valid)
        current_valid_count = len(valid_indices)
        if fixed_denominator is None:
            fixed_denominator = len(targets)
            valid_count = current_valid_count
        if len(targets) != fixed_denominator or current_valid_count != valid_count:
            raise ValueError("target denominator changed across update counts")
        sums = {
            name: {
                metric: np.zeros(fold_count)
                for metric in (
                    "ordered",
                    "static",
                    "ordered_residual",
                    "shuffle_residual",
                )
            }
            for name in candidate_order
        }
        element_count = 0
        for stimulus in stimuli:
            sequences_by_mode = {
                mode: _source_sequences(probe, stimulus, moments, mode, seed)
                for mode in modes
            }
            for start in range(0, current_valid_count, int(audit["target_chunk_size"])):
                selected = valid_indices[start : start + int(audit["target_chunk_size"])]
                mode_candidates = {}
                for mode in modes:
                    sequences = {
                        name: values[:, selected]
                        for name, values in sequences_by_mode[mode].items()
                    }
                    if {len(values) for values in sequences.values()} != {input_samples}:
                        raise ValueError("source-drive input length changed")
                    mode_candidates[mode] = _candidate_variants(
                        sequences, fixed_kernels, tm2_kernel_bank, moments["axes"][selected]
                    )
                for name in candidate_order:
                    ordered = mode_candidates["ordered"][name]
                    shuffled = mode_candidates["temporal_shuffle"][name]
                    static = mode_candidates["static_sham"][name]
                    if ordered.shape[1] != output_samples:
                        raise ValueError("full-support output length changed")
                    if not all(np.isfinite(array).all() for array in (ordered, shuffled, static)):
                        raise ValueError("non-finite Tm2 LOO diagnostic values")
                    sums[name]["ordered"] += np.sum(np.abs(ordered), axis=(1, 2))
                    sums[name]["static"] += np.sum(np.abs(static), axis=(1, 2))
                    sums[name]["ordered_residual"] += np.sum(
                        np.abs(ordered - static), axis=(1, 2)
                    )
                    sums[name]["shuffle_residual"] += np.sum(
                        np.abs(shuffled - static), axis=(1, 2)
                    )
                element_count += output_samples * len(selected)
        folds = {}
        for fold_index, recording_id in enumerate(recording_ids):
            fold_sums = {
                name: {metric: float(values[fold_index]) for metric, values in metrics.items()}
                for name, metrics in sums.items()
            }
            folds[recording_id] = {
                "candidate_results": _score_sums(
                    fold_sums, element_count, audit["controls"]
                )
            }
        by_updates[str(update_count)] = {
            "folds": folds,
            "scored_element_count_per_candidate_per_fold": element_count,
        }

    candidates = {}
    for name in candidate_order:
        fold_results = {}
        for recording_id in recording_ids:
            by_update = {
                str(update): by_updates[str(update)]["folds"][recording_id][
                    "candidate_results"
                ][name]
                for update in updates
            }
            fold_results[recording_id] = {
                "by_brain_updates_per_frame": by_update,
                "passed_every_update_count": all(
                    item["passed"] for item in by_update.values()
                ),
            }
        candidates[name] = {
            "folds": fold_results,
            "evaluation_count": len(recording_ids) * len(updates),
            "passed_evaluation_count": sum(
                item["passed"]
                for fold in fold_results.values()
                for item in fold["by_brain_updates_per_frame"].values()
            ),
            "shuffle_ratio_range": [
                min(
                    item["shuffle_to_ordered_residual_energy_ratio"]
                    for fold in fold_results.values()
                    for item in fold["by_brain_updates_per_frame"].values()
                ),
                max(
                    item["shuffle_to_ordered_residual_energy_ratio"]
                    for fold in fold_results.values()
                    for item in fold["by_brain_updates_per_frame"].values()
                ),
            ],
            "static_ratio_range": [
                min(
                    item["static_to_ordered_energy_ratio"]
                    for fold in fold_results.values()
                    for item in fold["by_brain_updates_per_frame"].values()
                ),
                max(
                    item["static_to_ordered_energy_ratio"]
                    for fold in fold_results.values()
                    for item in fold["by_brain_updates_per_frame"].values()
                ),
            ],
            "passed_every_fold_and_update_count": all(
                item["passed_every_update_count"] for item in fold_results.values()
            ),
        }
    robust_passed = any(
        item["passed_every_fold_and_update_count"] for item in candidates.values()
    )
    all_failed = all(
        not by_updates[str(update)]["folds"][recording_id]["candidate_results"][
            name
        ]["passed"]
        for update in updates
        for recording_id in recording_ids
        for name in candidate_order
    )
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(robustness_path): _sha256(root / robustness_path),
        str(reference_path): _sha256(root / reference_path),
        str(reference_config_path): _sha256(root / reference_config_path),
        str(measured_config_path): _sha256(root / measured_config_path),
        str(measured_implementation_path): _sha256(root / measured_implementation_path),
        str(full_implementation_path): _sha256(root / full_implementation_path),
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
        "sensitivity_contract": {
            "source": sensitivity_source,
            "held_out_recording_ids": recording_ids,
            "fold_count": fold_count,
            "tested_brain_updates_per_frame": updates,
            "source_input_samples": input_samples,
            "kernel_samples": kernel_length,
            "output_samples": output_samples,
            "fixed_target_denominator": fixed_denominator,
            "valid_target_count": valid_count,
        },
        "candidate_results": candidates,
        "all_candidates_failed_every_fold_and_update_count": all_failed,
        "evaluation_count": len(candidate_order) * len(recording_ids) * len(updates),
        "passed_evaluation_count": sum(
            item["passed_evaluation_count"] for item in candidates.values()
        ),
        "robust_temporal_identifiability_passed": robust_passed,
        "direction_scoring_authorized": robust_passed,
        "direction_scoring_performed": False,
        "independent_biological_validation_performed": False,
        "authorize_T5_functional_precheck": False,
        "authorize_physical_source_dynamics_transfer": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "Tm2_LOO_temporal_gate_passed_direction_scoring_required"
            if robust_passed
            else "no_candidate_passed_every_Tm2_LOO_fold_and_update_count"
        ),
        "boundary": audit["boundary"],
    }
