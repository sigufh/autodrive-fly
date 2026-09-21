"""Test measured T5 kernels on isolated L1/L2/L3 feed-forward drive."""

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

CONFIG = Path("configs/driving-v7-t5-lamina-only-measured-kernel-replacement.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_lamina_only_measured_kernel_replacement.py"
)


def _lamina_only_source_matrices(
    probe: LaminaSplitProbe, source_types: list[str], lamina_types: list[str]
) -> tuple[dict[str, object], dict[str, dict[str, int | float]]]:
    lamina_mask = np.isin(probe.node_types, lamina_types).astype(np.float32)
    matrices = {}
    coverage = {}
    for source in source_types:
        nodes = np.flatnonzero(probe.node_types == source).astype(np.int32)
        matrix = probe.adjacency[nodes].multiply(lamina_mask).tocsr()
        matrix.eliminate_zeros()
        covered = np.diff(matrix.indptr) > 0
        matrices[source] = matrix
        coverage[source] = {
            "source_node_count": len(nodes),
            "source_node_with_lamina_input_count": int(np.count_nonzero(covered)),
            "source_node_without_lamina_input_count": int(np.count_nonzero(~covered)),
            "source_node_with_lamina_input_fraction": float(np.mean(covered)),
        }
    return matrices, coverage


def _update_lamina_only(
    probe: LaminaSplitProbe, state: np.ndarray, retinal_drive: np.ndarray
) -> None:
    drive = np.zeros(probe.graph.node_count, dtype=np.float32)
    drive[probe.retina.node_indices] = retinal_drive
    positive = np.maximum(drive, 0.0)
    negative = np.maximum(-drive, 0.0)
    for nodes, matrix, is_on in probe.lamina_projection.values():
        target = np.tanh(matrix @ (positive if is_on else negative)).astype(np.float32)
        state[nodes] = (1.0 - probe.leak[nodes]) * state[nodes] + probe.leak[nodes] * target


def _lamina_only_source_sequences(
    probe: LaminaSplitProbe,
    stimulus,
    moments: dict,
    source_matrices: dict[str, object],
    source_nodes: dict[str, np.ndarray],
    mode: str,
    seed: int,
) -> dict[str, np.ndarray]:
    frames = _controlled_frames(stimulus, mode, seed)
    lamina_state = np.zeros(probe.graph.node_count, dtype=np.float32)
    baseline_image = frames[0]
    baseline_values = probe._sample_retina(baseline_image)[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(baseline_values, baseline_values, baseline_values)
    for _ in range(probe.baseline_frames):
        _update_lamina_only(probe, lamina_state, baseline_drive)
    previous_retina = baseline_values.copy()
    sequence = []
    for image in frames[probe.baseline_frames :]:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        retinal_drive = probe._retinal_code(sampled, baseline_values, previous_retina)
        previous_retina = sampled
        source_preactivation = {}
        for _ in range(probe.brain_substeps):
            source_preactivation = {
                source: matrix @ (lamina_state * probe.source_sign)
                for source, matrix in source_matrices.items()
            }
            _update_lamina_only(probe, lamina_state, retinal_drive)
        source_state = np.zeros(probe.graph.node_count, dtype=np.float64)
        for source, values in source_preactivation.items():
            source_state[source_nodes[source]] = np.maximum(values, 0.0)
        sequence.append(
            {name: matrix @ source_state for name, matrix in moments["matrices"].items()}
        )
    return {
        name: np.stack([sample[name] for sample in sequence])
        for name in moments["matrices"]
    }


def evaluate_v7_t5_lamina_only_measured_kernel_replacement(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    cascade_path = Path(audit["cascade_evidence"])
    inputs_path = Path(audit["source_input_evidence"])
    reference_path = Path(audit["reference_full_support_evidence"])
    reference_config_path = Path(audit["reference_full_support_config"])
    measured_config_path = Path(audit["measured_kernel_config"])
    measured_implementation_path = Path(audit["measured_kernel_implementation"])
    full_implementation_path = Path(audit["full_support_implementation"])
    lamina_implementation_path = Path(audit["lamina_split_implementation"])
    v7_path = Path(audit["v7_contract"])
    cascade = json.loads((root / cascade_path).read_text(encoding="utf-8"))
    input_evidence = json.loads((root / inputs_path).read_text(encoding="utf-8"))
    reference = json.loads((root / reference_path).read_text(encoding="utf-8"))
    reference_config = yaml.safe_load(
        (root / reference_config_path).read_text(encoding="utf-8")
    )
    measured = yaml.safe_load((root / measured_config_path).read_text(encoding="utf-8"))
    v7 = yaml.safe_load((root / v7_path).read_text(encoding="utf-8"))
    replacement = audit["replacement"]
    if not cascade["typed_recurrent_source_state_then_measured_kernel_cascade_verified"]:
        raise ValueError("reference cascade must be verified")
    if not input_evidence["source_dynamics_replacement_requires_explicit_intervention"]:
        raise ValueError("source replacement intervention boundary changed")
    if replacement["retained_source_types"] != ["L1", "L2", "L3"]:
        raise ValueError("lamina-only retained source set changed")
    if replacement["replaced_source_types"] != measured["source_types"]:
        raise ValueError("Tm replacement source set changed")
    if not all(
        replacement[name]
        for name in (
            "preserve_synchronous_prior_lamina_state_order",
            "remove_Tm_leak",
            "remove_Tm_tanh",
            "remove_non_lamina_visual_inputs",
            "remove_Tm_T4_T5_recurrent_or_feedback_inputs",
            "remove_CT1_input",
            "uncovered_Tm_source_nodes_remain_zero",
        )
    ):
        raise ValueError("lamina-only replacement intervention changed")
    candidate_order = [item["name"] for item in measured["candidates"]]
    if candidate_order != audit["required_candidate_order"]:
        raise ValueError("lamina-only candidate order changed")
    if reference_config["controls"] != audit["controls"]:
        raise ValueError("lamina-only control thresholds changed")
    updates = [int(value) for value in audit["tested_brain_updates_per_frame"]]
    if updates != [1, 4] or int(v7["controlled_vision"]["brain_substeps_per_frame"]) != 4:
        raise ValueError("lamina-only update-count contract changed")

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
        raise ValueError("lamina-only replacement may consume tuning only")
    full_contract = reference["full_support_contract"]
    input_samples = int(full_contract["source_input_samples"] )
    kernel_length = int(full_contract["kernel_samples"] )
    output_samples = int(full_contract["output_samples"] )
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
    fixed_denominator = None
    valid_count = None
    coverage = None
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
        source_nodes = {
            source: np.flatnonzero(probe.node_types == source).astype(np.int32)
            for source in measured["source_types"]
        }
        source_matrices, current_coverage = _lamina_only_source_matrices(
            probe, measured["source_types"], replacement["retained_source_types"]
        )
        if coverage is None:
            coverage = current_coverage
        elif coverage != current_coverage:
            raise ValueError("lamina-only source coverage changed across update counts")
        sums = _empty_sums(candidate_order)
        element_count = 0
        for stimulus in stimuli:
            mode_traces = {}
            for mode in modes:
                sequences = _lamina_only_source_sequences(
                    probe,
                    stimulus,
                    moments,
                    source_matrices,
                    source_nodes,
                    mode,
                    seed,
                )
                if {len(values) for values in sequences.values()} != {input_samples}:
                    raise ValueError("lamina-only source-drive input length changed")
                mode_traces[mode] = _full_candidate_traces(
                    sequences, kernels, moments["axes"]
                )
            for name in candidate_order:
                ordered = mode_traces["ordered"][name][:, valid]
                shuffled = mode_traces["temporal_shuffle"][name][:, valid]
                static = mode_traces["static_sham"][name][:, valid]
                if ordered.shape[0] != output_samples:
                    raise ValueError("full-support output length changed")
                if not all(np.isfinite(array).all() for array in (ordered, shuffled, static)):
                    raise ValueError("non-finite lamina-only diagnostic values")
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
        by_update = {
            str(update): by_updates[str(update)]["candidate_results"][name]
            for update in updates
        }
        candidates[name] = {
            "by_brain_updates_per_frame": by_update,
            "passed_every_update_count": all(
                result["passed"] for result in by_update.values()
            ),
        }
    temporal_passed = any(
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
        str(cascade_path): _sha256(root / cascade_path),
        str(inputs_path): _sha256(root / inputs_path),
        str(reference_path): _sha256(root / reference_path),
        str(reference_config_path): _sha256(root / reference_config_path),
        str(measured_config_path): _sha256(root / measured_config_path),
        str(measured_implementation_path): _sha256(root / measured_implementation_path),
        str(full_implementation_path): _sha256(root / full_implementation_path),
        str(lamina_implementation_path): _sha256(root / lamina_implementation_path),
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
        "replacement_contract": {
            **replacement,
            "source_input_coverage": coverage,
            "source_input_samples": input_samples,
            "kernel_samples": kernel_length,
            "output_samples": output_samples,
            "tested_brain_updates_per_frame": updates,
            "fixed_target_denominator": fixed_denominator,
            "valid_target_count": valid_count,
        },
        "candidate_results": candidates,
        "all_candidates_failed_every_update_count": all_failed,
        "temporal_identifiability_passed": temporal_passed,
        "direction_scoring_authorized": temporal_passed,
        "direction_scoring_performed": False,
        "authorize_T5_functional_precheck": False,
        "authorize_physical_source_dynamics_transfer": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "lamina_only_temporal_gate_passed_direction_scoring_required"
            if temporal_passed
            else "lamina_only_measured_kernel_replacement_failed_temporal_controls"
        ),
        "boundary": audit["boundary"],
    }
