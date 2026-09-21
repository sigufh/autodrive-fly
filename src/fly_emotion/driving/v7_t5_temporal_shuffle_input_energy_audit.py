"""Audit input-energy matching of the T5 temporal-shuffle control."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t5_lamina_only_measured_kernel_replacement import (
    _lamina_only_source_matrices,
    _update_lamina_only,
)
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_t5_typed_spatial_pair_precheck import _controlled_frames
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-temporal-shuffle-input-energy-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_temporal_shuffle_input_energy_audit.py"
)


def _frame_multiset_digest(frames: np.ndarray) -> str:
    frame_hashes = sorted(hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames)
    return hashlib.sha256("\n".join(frame_hashes).encode()).hexdigest()


def _input_energies(
    probe: LaminaSplitProbe,
    frames: np.ndarray,
    source_matrices: dict[str, object],
) -> dict:
    baseline_image = frames[0]
    baseline_retina = probe._sample_retina(baseline_image)[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(
        baseline_retina, baseline_retina, baseline_retina
    )
    lamina_state = np.zeros(probe.graph.node_count, dtype=np.float32)
    for _ in range(probe.baseline_frames):
        _update_lamina_only(probe, lamina_state, baseline_drive)
    previous_image = baseline_image
    previous_retina = baseline_retina.copy()
    pixel_total = 0.0
    retinal_total = 0.0
    source_totals = {source: 0.0 for source in source_matrices}
    pixel_count = retinal_count = 0
    source_counts = {source: 0 for source in source_matrices}
    for image in frames[probe.baseline_frames :]:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        retinal_drive = probe._retinal_code(sampled, baseline_retina, previous_retina)
        pixel_total += float(np.sum(np.abs(image - previous_image)))
        retinal_total += float(np.sum(np.abs(retinal_drive)))
        pixel_count += image.size
        retinal_count += retinal_drive.size
        source_preactivation = {}
        for _ in range(probe.brain_substeps):
            source_preactivation = {
                source: matrix @ (lamina_state * probe.source_sign)
                for source, matrix in source_matrices.items()
            }
            _update_lamina_only(probe, lamina_state, retinal_drive)
        for source, values in source_preactivation.items():
            positive = np.maximum(values, 0.0)
            source_totals[source] += float(np.sum(np.abs(positive)))
            source_counts[source] += positive.size
        previous_image = image
        previous_retina = sampled
    return {
        "pixel_temporal_difference": pixel_total / pixel_count,
        "R1_R6_signed_frame_difference": retinal_total / retinal_count,
        "lamina_only_Tm_preactivation": {
            source: source_totals[source] / source_counts[source]
            for source in source_matrices
        },
    }


def _ratio(shuffled: float, ordered: float) -> float:
    return shuffled / max(ordered, np.finfo(np.float64).tiny)


def evaluate_v7_t5_temporal_shuffle_input_energy_audit(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    measured_config_path = Path(audit["measured_kernel_config"])
    measured_implementation_path = Path(audit["measured_kernel_implementation"])
    replacement_path = Path(audit["lamina_only_replacement_evidence"])
    replacement_implementation_path = Path(
        audit["lamina_only_replacement_implementation"]
    )
    measured = yaml.safe_load((root / measured_config_path).read_text(encoding="utf-8"))
    replacement = json.loads((root / replacement_path).read_text(encoding="utf-8"))
    if replacement["replacement_contract"]["retained_source_types"] != [
        "L1",
        "L2",
        "L3",
    ]:
        raise ValueError("lamina-only replacement source set changed")
    if measured["controls"]["modes"] != [
        "ordered",
        "temporal_shuffle",
        "static_sham",
    ]:
        raise ValueError("temporal control mode order changed")
    updates = [int(value) for value in audit["tested_brain_updates_per_frame"]]
    if updates != [1, 4]:
        raise ValueError("input-energy update-count contract changed")
    if (
        audit["lamina_source_metric"]
        != "positive_half_wave_of_signed_target_normalized_preactivation"
    ):
        raise ValueError("lamina source-energy metric changed")

    lamina_path = Path(measured["lamina_split_protocol"])
    stage1_path = Path(measured["stage1_protocol"])
    local_path = Path(measured["local_edge_config"])
    lamina = yaml.safe_load((root / lamina_path).read_text())
    stage1 = yaml.safe_load((root / stage1_path).read_text())
    local = yaml.safe_load((root / local_path).read_text())
    condition = next(
        row for row in stage1["conditions"] if row["condition_id"] == measured["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("input-energy audit may consume tuning only")
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
    if len(stimuli) != int(audit["expected_stimulus_count"]):
        raise ValueError("input-energy stimulus count changed")
    seed = int(measured["controls"]["temporal_shuffle_seed"])
    frame_multiset_preserved = []
    by_updates = {}
    for update_count in updates:
        probe_config = copy.deepcopy(lamina)
        probe_config["brain_substeps_per_frame"] = update_count
        probe = LaminaSplitProbe(root, probe_config)
        source_matrices, _ = _lamina_only_source_matrices(
            probe, measured["source_types"], ["L1", "L2", "L3"]
        )
        totals = {
            mode: {
                "pixel_temporal_difference": 0.0,
                "R1_R6_signed_frame_difference": 0.0,
                "lamina_only_Tm_preactivation": {
                    source: 0.0 for source in measured["source_types"]
                },
            }
            for mode in measured["controls"]["modes"]
        }
        per_stimulus_ratios = {
            "pixel_temporal_difference": [],
            "R1_R6_signed_frame_difference": [],
            **{
                f"lamina_only_{source}_preactivation": []
                for source in measured["source_types"]
            },
        }
        for stimulus in stimuli:
            frames = {
                mode: _controlled_frames(stimulus, mode, seed)
                for mode in measured["controls"]["modes"]
            }
            frame_multiset_preserved.append(
                _frame_multiset_digest(frames["ordered"][probe.baseline_frames :])
                == _frame_multiset_digest(
                    frames["temporal_shuffle"][probe.baseline_frames :]
                )
            )
            energies = {
                mode: _input_energies(probe, mode_frames, source_matrices)
                for mode, mode_frames in frames.items()
            }
            for mode, values in energies.items():
                totals[mode]["pixel_temporal_difference"] += values[
                    "pixel_temporal_difference"
                ]
                totals[mode]["R1_R6_signed_frame_difference"] += values[
                    "R1_R6_signed_frame_difference"
                ]
                for source, value in values["lamina_only_Tm_preactivation"].items():
                    totals[mode]["lamina_only_Tm_preactivation"][source] += value
            per_stimulus_ratios["pixel_temporal_difference"].append(
                _ratio(
                    energies["temporal_shuffle"]["pixel_temporal_difference"],
                    energies["ordered"]["pixel_temporal_difference"],
                )
            )
            per_stimulus_ratios["R1_R6_signed_frame_difference"].append(
                _ratio(
                    energies["temporal_shuffle"]["R1_R6_signed_frame_difference"],
                    energies["ordered"]["R1_R6_signed_frame_difference"],
                )
            )
            for source in measured["source_types"]:
                per_stimulus_ratios[f"lamina_only_{source}_preactivation"].append(
                    _ratio(
                        energies["temporal_shuffle"]["lamina_only_Tm_preactivation"][
                            source
                        ],
                        energies["ordered"]["lamina_only_Tm_preactivation"][source],
                    )
                )
        count = len(stimuli)
        means = {
            mode: {
                "pixel_temporal_difference": values["pixel_temporal_difference"]
                / count,
                "R1_R6_signed_frame_difference": values[
                    "R1_R6_signed_frame_difference"
                ]
                / count,
                "lamina_only_Tm_preactivation": {
                    source: value / count
                    for source, value in values["lamina_only_Tm_preactivation"].items()
                },
            }
            for mode, values in totals.items()
        }
        aggregate_ratios = {
            "pixel_temporal_difference": _ratio(
                means["temporal_shuffle"]["pixel_temporal_difference"],
                means["ordered"]["pixel_temporal_difference"],
            ),
            "R1_R6_signed_frame_difference": _ratio(
                means["temporal_shuffle"]["R1_R6_signed_frame_difference"],
                means["ordered"]["R1_R6_signed_frame_difference"],
            ),
            "lamina_only_Tm_preactivation": {
                source: _ratio(
                    means["temporal_shuffle"]["lamina_only_Tm_preactivation"][source],
                    means["ordered"]["lamina_only_Tm_preactivation"][source],
                )
                for source in measured["source_types"]
            },
        }
        static_ratios = {
            "pixel_temporal_difference": _ratio(
                means["static_sham"]["pixel_temporal_difference"],
                means["ordered"]["pixel_temporal_difference"],
            ),
            "R1_R6_signed_frame_difference": _ratio(
                means["static_sham"]["R1_R6_signed_frame_difference"],
                means["ordered"]["R1_R6_signed_frame_difference"],
            ),
            "lamina_only_Tm_preactivation": {
                source: _ratio(
                    means["static_sham"]["lamina_only_Tm_preactivation"][source],
                    means["ordered"]["lamina_only_Tm_preactivation"][source],
                )
                for source in measured["source_types"]
            },
        }
        by_updates[str(update_count)] = {
            "mean_absolute_energy_by_mode": means,
            "temporal_shuffle_to_ordered_energy_ratio": aggregate_ratios,
            "static_sham_to_ordered_energy_ratio": static_ratios,
            "per_stimulus_ratio_summary": {
                name: {
                    "minimum": float(np.min(values)),
                    "median": float(np.median(values)),
                    "maximum": float(np.max(values)),
                }
                for name, values in per_stimulus_ratios.items()
            },
        }
    frame_multiset_all = all(frame_multiset_preserved)
    retinal_energy_exact = all(
        result["temporal_shuffle_to_ordered_energy_ratio"][
            "R1_R6_signed_frame_difference"
        ]
        == 1.0
        for result in by_updates.values()
    )
    lamina_energy_exact = all(
        ratio == 1.0
        for result in by_updates.values()
        for ratio in result["temporal_shuffle_to_ordered_energy_ratio"][
            "lamina_only_Tm_preactivation"
        ].values()
    )
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(measured_config_path): _sha256(root / measured_config_path),
        str(measured_implementation_path): _sha256(root / measured_implementation_path),
        str(replacement_path): _sha256(root / replacement_path),
        str(replacement_implementation_path): _sha256(
            root / replacement_implementation_path
        ),
        str(lamina_path): _sha256(root / lamina_path),
        str(stage1_path): _sha256(root / stage1_path),
        str(local_path): _sha256(root / local_path),
    }
    return {
        "protocol": {
            "name": audit["name"],
            "observed_on": audit["observed_on"],
            "dependencies_sha256": dependencies,
            "condition_id": measured["condition_id"],
            "stimulus_count": len(stimuli),
            "input_energy_metric": audit["input_energy_metric"],
            "lamina_source_metric": audit["lamina_source_metric"],
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "frame_multiset_preserved_for_every_shuffle": frame_multiset_all,
        "by_brain_updates_per_frame": by_updates,
        "retinal_temporal_energy_exactly_preserved": retinal_energy_exact,
        "lamina_only_source_energy_exactly_preserved": lamina_energy_exact,
        "temporal_shuffle_input_energy_amplified": all(
            result["temporal_shuffle_to_ordered_energy_ratio"][
                "R1_R6_signed_frame_difference"
            ]
            > 1.0
            for result in by_updates.values()
        ),
        "energy_matched_temporal_shuffle_control_verified": bool(
            frame_multiset_all and retinal_energy_exact and lamina_energy_exact
        ),
        "existing_output_ratios_recomputed": False,
        "existing_output_ratios_invalidated": False,
        "equal_energy_temporal_selectivity_interpretation_authorized": False,
        "authorize_new_energy_normalized_gate": False,
        "authorize_T5_functional_precheck": False,
        "direction_scoring_authorized": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "shuffle_preserves_frames_but_not_retinal_or_lamina_input_energy",
        "boundary": audit["boundary"],
    }
