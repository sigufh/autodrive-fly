"""Test author row-weighted T5 kernels over complete zero-tailed support."""

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
from fly_emotion.driving.v7_kohn_portes_t5_source_kernel_audit import (
    _author_rescaled_temporal,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_t5_measured_kernel_full_support_identifiability import (
    _empty_sums,
    _full_candidate_traces,
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

CONFIG = Path(
    "configs/driving-v7-t5-author-row-weighted-full-support-sensitivity.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_author_row_weighted_full_support_sensitivity.py"
)


def _row_weighted_population_kernel(records: list[dict], length: int) -> np.ndarray:
    kernel = np.mean(
        [_author_rescaled_temporal(record, length) for record in records], axis=0
    )
    scale = float(np.sum(np.abs(kernel)))
    if not np.isfinite(kernel).all() or scale <= 0.0:
        raise ValueError("invalid row-weighted population kernel")
    return kernel / scale


def evaluate_v7_t5_author_row_weighted_full_support_sensitivity(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    aggregation_path = Path(audit["aggregation_semantics_evidence"])
    reference_path = Path(audit["reference_full_support_evidence"])
    reference_config_path = Path(audit["reference_full_support_config"])
    measured_config_path = Path(audit["measured_kernel_config"])
    measured_implementation_path = Path(audit["measured_kernel_implementation"])
    full_implementation_path = Path(audit["full_support_implementation"])
    v7_path = Path(audit["v7_contract"])
    aggregation = json.loads((root / aggregation_path).read_text(encoding="utf-8"))
    reference = json.loads((root / reference_path).read_text(encoding="utf-8"))
    reference_config = yaml.safe_load(
        (root / reference_config_path).read_text(encoding="utf-8")
    )
    measured = yaml.safe_load((root / measured_config_path).read_text(encoding="utf-8"))
    v7 = yaml.safe_load((root / v7_path).read_text(encoding="utf-8"))
    candidate_order = [item["name"] for item in measured["candidates"]]
    if candidate_order != audit["required_candidate_order"]:
        raise ValueError("row-weighted candidate order changed")
    if not aggregation["current_no_baseline_matches_author_default"]:
        raise ValueError("author default baseline semantics are not verified")
    if aggregation["author_row_weighted_aggregation_exactly_reproduced"]:
        raise ValueError("row-weighted sensitivity is no longer distinct")
    if not reference["full_kernel_support_evaluated"]:
        raise ValueError("reference full-support evaluation is incomplete")
    if reference_config["controls"] != audit["controls"]:
        raise ValueError("row-weighted control thresholds changed")

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
        raise ValueError("row-weighted sensitivity may consume tuning only")
    updates = [int(value) for value in audit["tested_brain_updates_per_frame"]]
    if updates != [1, 4]:
        raise ValueError("row-weighted update-count contract changed")
    if int(v7["controlled_vision"]["brain_substeps_per_frame"]) != updates[-1]:
        raise ValueError("standard v7 update count changed")

    full_contract = reference["full_support_contract"]
    input_samples = int(full_contract["source_input_samples"] )
    kernel_length = int(full_contract["kernel_samples"] )
    output_samples = int(full_contract["output_samples"] )
    row_kernels = {}
    reference_kernels = {}
    source_paths = {}
    kernel_comparison = {}
    for source in measured["source_types"]:
        spec = source_config["white_noise_files"][source]
        source_path = _verify_file(root, spec)
        source_paths[source] = source_path
        records = _load_restricted(source_path)
        row_kernels[source] = _row_weighted_population_kernel(records, kernel_length)
        reference_kernels[source], recording_count = _population_kernel(
            records, kernel_length
        )
        difference = row_kernels[source] - reference_kernels[source]
        kernel_comparison[source] = {
            "payload_row_count": len(records),
            "recording_id_count": recording_count,
            "kernels_exactly_equal": bool(
                np.array_equal(row_kernels[source], reference_kernels[source])
            ),
            "correlation": float(
                np.corrcoef(row_kernels[source], reference_kernels[source])[0, 1]
            ),
            "maximum_absolute_difference": float(np.max(np.abs(difference))),
        }
    changed_sources = [
        source
        for source, item in kernel_comparison.items()
        if not item["kernels_exactly_equal"]
    ]
    if changed_sources != ["Tm1"]:
        raise ValueError("author row weighting must change only Tm1")

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
                    sequences, row_kernels, moments["axes"]
                )
            for name in candidate_order:
                ordered = mode_traces["ordered"][name][:, valid]
                shuffled = mode_traces["temporal_shuffle"][name][:, valid]
                static = mode_traces["static_sham"][name][:, valid]
                if ordered.shape[0] != output_samples:
                    raise ValueError("full-support output length changed")
                if not all(np.isfinite(array).all() for array in (ordered, shuffled, static)):
                    raise ValueError("non-finite row-weighted diagnostic values")
                sums[name]["ordered"] += float(np.sum(np.abs(ordered)))
                sums[name]["static"] += float(np.sum(np.abs(static)))
                sums[name]["ordered_residual"] += float(np.sum(np.abs(ordered - static)))
                sums[name]["shuffle_residual"] += float(np.sum(np.abs(shuffled - static)))
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
    all_failed = all(
        not by_updates[str(update)]["candidate_results"][name]["passed"]
        for update in updates
        for name in candidate_order
    )
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(aggregation_path): _sha256(root / aggregation_path),
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
        "aggregation_contract": {
            "variant": audit["aggregation"]["variant"],
            "reference": audit["aggregation"]["reference"],
            "author_default_baseline_none_preserved": True,
            "changed_sources": changed_sources,
            "kernel_comparison": kernel_comparison,
        },
        "full_support_contract": {
            "source_input_samples": input_samples,
            "kernel_samples": kernel_length,
            "output_samples": output_samples,
            "tested_brain_updates_per_frame": updates,
            "fixed_target_denominator": fixed_denominator,
            "valid_target_count": valid_count,
        },
        "candidate_results": candidates,
        "all_candidates_failed_every_update_count": all_failed,
        "cross_substep_temporal_identifiability_passed": cross_substep_passed,
        "direction_scoring_authorized": cross_substep_passed,
        "direction_scoring_performed": False,
        "authorize_T5_functional_precheck": False,
        "authorize_physical_source_dynamics_transfer": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "row_weighted_temporal_gate_passed_direction_scoring_required"
            if cross_substep_passed
            else "author_row_weighting_did_not_rescue_full_support_temporal_controls"
        ),
        "boundary": audit["boundary"],
    }
