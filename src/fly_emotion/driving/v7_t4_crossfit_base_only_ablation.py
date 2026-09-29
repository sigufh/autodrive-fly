"""Locate the frozen T4 failure with a zero-gain source-sequence ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_t4_synapse_antisymmetric_precheck import (
    _moving_edges,
    _run_responses,
    _score_candidates,
)
from fly_emotion.driving.v7_t4_synapse_crossfit_precheck import (
    IMPLEMENTATION as CROSS_FIT_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4_synapse_crossfit_precheck import _crossfit_moments

CONFIG = Path("configs/driving-v7-t4-crossfit-base-only-ablation.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_crossfit_base_only_ablation.py"
)


def _passing_populations(result: dict, head: str) -> list[str]:
    return sorted(
        population
        for population, score in result["population_scores"].items()
        if score[head]["passed"]
    )


def _compact(result: dict) -> dict:
    return {
        "direction_pass_count": result["direction_pass_count"],
        "polarity_pass_count": result["polarity_pass_count"],
        "direction_passing_populations": _passing_populations(result, "direction"),
        "polarity_passing_populations": _passing_populations(result, "polarity"),
        "bilateral_direction_subtypes": result["bilateral_direction_subtypes"],
        "mirror_gate_passed": result["mirror_gate_passed"],
        "population_scores": result["population_scores"],
    }


def evaluate_v7_t4_crossfit_base_only_ablation(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "frozen_candidate_family_evidence",
            "source_sequence_evidence",
            "crossfit_axis_evidence",
            "crossfit_axis_protocol",
            "synapse_spatial_protocol",
            "conductance_protocol",
            "stage1_protocol",
            "scoring_config",
        )
    }
    family = json.loads((root / paths["frozen_candidate_family_evidence"]).read_text())
    if not family["frozen_candidate_grid_failed"]:
        raise ValueError("base-only ablation requires frozen candidate-grid failure")
    if family["authorize_post_hoc_additive_gain_expansion"]:
        raise ValueError("post-hoc additive gain expansion must remain forbidden")
    sequence = json.loads((root / paths["source_sequence_evidence"]).read_text())
    if sequence["authorize_new_functional_candidate"]:
        raise ValueError("base-only ablation requires preserved sequence failure")
    crossfit = json.loads((root / paths["crossfit_axis_evidence"]).read_text())
    if not crossfit["T4_synapse_crossfit_axis_passed"]:
        raise ValueError("base-only ablation requires passed structural cross-fit axis")

    crossfit_protocol = yaml.safe_load(
        (root / paths["crossfit_axis_protocol"]).read_text()
    )
    config["crossfit_split_seed"] = crossfit_protocol["split_seed"]
    conductance = yaml.safe_load((root / paths["conductance_protocol"]).read_text())
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    scoring = yaml.safe_load((root / paths["scoring_config"]).read_text())
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("base-only ablation may consume tuning only")
    if float(config["source_sequence_gain"]) != 0.0:
        raise ValueError("base-only ablation requires zero source-sequence gain")

    normalization = _collect_source_normalization(
        root,
        retinal_backend="linear_luminance",
        retinal_geometry=conductance["primary_retinal_geometry"],
        config=conductance,
    )
    probe = MotionSignConductanceProbe(
        root, retinal_backend="linear_luminance", normalization=normalization
    )
    moments = _crossfit_moments(root, probe, config, crossfit)
    stimuli = _moving_edges(condition)
    results = {}
    for mode in config["modes"]:
        responses = _run_responses(
            probe, stimuli, moments, mode, int(config["temporal_shuffle_seed"])
        )
        scored = _score_candidates(
            probe,
            stimuli,
            responses,
            moments,
            scoring,
            [
                (reduction, float(config["source_sequence_gain"]))
                for reduction in config["temporal_reductions"]
            ],
        )
        by_reduction = {
            result["temporal_reduction"]: _compact(result) for result in scored
        }
        first, second = (
            by_reduction[reduction]
            for reduction in config["temporal_reductions"]
        )
        results[mode] = {
            "by_reduction": by_reduction,
            "reduction_invariant": first == second,
        }

    ordered = results["ordered"]["by_reduction"][
        config["temporal_reductions"][0]
    ]
    dominant_set = max(
        family["direction_pass_set_frequency"],
        key=family["direction_pass_set_frequency"].get,
    ).split("|")
    base_controls_have_direction_passes = any(
        result["direction_pass_count"] > 0
        for mode in ("temporal_shuffle", "static_sham")
        for result in results[mode]["by_reduction"].values()
    )
    base_control_maximum = max(
        result["direction_pass_count"]
        for mode in ("temporal_shuffle", "static_sham")
        for result in results[mode]["by_reduction"].values()
    )
    source_sequence_ordered = sequence["mode_reduction_results"]["ordered"][
        "positive_peak"
    ]
    source_sequence_control_maximum = max(
        result["direction_pass_count"]
        for mode in ("temporal_shuffle", "static_sham")
        for result in sequence["mode_reduction_results"][mode].values()
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(CROSS_FIT_IMPLEMENTATION): _sha256(root / CROSS_FIT_IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in paths.values()},
            },
            "condition_id": config["condition_id"],
            "stimulus_count_per_mode": len(stimuli),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "fixed_T4_population_denominator": sequence[
            "fixed_T4_population_denominator"
        ],
        "valid_target_count": int(np.count_nonzero(moments["valid"])),
        "source_sequence_gain": 0.0,
        "mode_results": results,
        "ordered_direction_passing_populations": ordered[
            "direction_passing_populations"
        ],
        "ordered_polarity_passing_populations": ordered[
            "polarity_passing_populations"
        ],
        "ordered_reproduces_dominant_frozen_family_pass_set": (
            ordered["direction_passing_populations"] == dominant_set
        ),
        "base_only_ordered_direction_pass_count": ordered[
            "direction_pass_count"
        ],
        "base_only_control_maximum_direction_pass_count": base_control_maximum,
        "base_only_control_specificity_at_direction_gate": (
            ordered["direction_pass_count"] > 0 and base_control_maximum == 0
        ),
        "source_sequence_only_ordered_direction_passing_populations": (
            _passing_populations(source_sequence_ordered, "direction")
        ),
        "base_only_controls_have_direction_passes": base_controls_have_direction_passes,
        "source_sequence_only_control_maximum_direction_pass_count": (
            source_sequence_control_maximum
        ),
        "source_sequence_controls_have_direction_passes": (
            source_sequence_control_maximum > 0
        ),
        "gate_level_unilateral_pass_pattern_present_in_conductance_base": (
            ordered["direction_passing_populations"] == dominant_set
        ),
        "current_source_sequence_overlay_expands_bilateral_direction_passes": False,
        "base_only_is_new_candidate": False,
        "authorize_base_only_candidate": False,
        "authorize_source_sequence_removal": False,
        "authorize_direction_or_subtype_label_change": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "base_only_ablation_is_diagnostic_not_a_functional_candidate",
        "boundary": config["boundary"],
    }
