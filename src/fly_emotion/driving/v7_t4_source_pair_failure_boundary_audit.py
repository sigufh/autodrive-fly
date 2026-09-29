"""Bound the orthogonal ordered/control failures of the T4 source-pair audit."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path(
    "configs/driving-v7-t4-source-pair-failure-boundary-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_source_pair_failure_boundary_audit.py"
)


def evaluate_v7_t4_source_pair_failure_boundary_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_path = Path(config["source_pair_evidence"])
    evidence = json.loads((root / evidence_path).read_text(encoding="utf-8"))
    if evidence["authorize_source_pair_candidate"]:
        raise ValueError("failure boundary requires preserved source-pair rejection")
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    pair_name = config["focus_pair"]
    if pair_name not in evidence["pair_results"]:
        raise ValueError("focus T4 source pair is not in the frozen complete grid")
    pair = evidence["pair_results"][pair_name]
    reduction = config["focus_reduction"]
    ordered = pair["mode_results"][config["ordered_mode"]][reduction]
    populations = sorted(ordered["per_population_direction_median_signed_contrast"])
    ordered_direction_missing = sorted(
        set(populations) - set(ordered["direction_passing_populations"])
    )
    ordered_polarity_missing = sorted(
        set(populations) - set(ordered["polarity_passing_populations"])
    )
    control_bilateral = sorted(
        {
            subtype
            for mode in config["control_modes"]
            for subtype in pair["mode_results"][mode][reduction][
                "bilateral_direction_subtypes"
            ]
        }
    )
    control_populations = sorted(
        {f"T4{subtype}_{side}" for subtype in control_bilateral for side in "LR"}
    )
    thresholds = scoring["thresholds"]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(evidence_path): _sha256(root / evidence_path),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "focus_pair": pair_name,
        "focus_reduction": reduction,
        "ordered_direction_missing_populations": ordered_direction_missing,
        "ordered_polarity_missing_populations": ordered_polarity_missing,
        "ordered_missing_population_metrics": {
            population: {
                "direction_median_signed_contrast": ordered[
                    "per_population_direction_median_signed_contrast"
                ][population],
                "direction_positive_cell_fraction": ordered[
                    "per_population_direction_positive_cell_fraction"
                ][population],
            }
            for population in sorted(
                set(ordered_direction_missing) | set(ordered_polarity_missing)
            )
        },
        "frozen_direction_thresholds": {
            "minimum_median_signed_contrast": thresholds[
                "minimum_median_signed_contrast"
            ],
            "minimum_positive_cell_fraction": thresholds[
                "minimum_positive_cell_fraction"
            ],
        },
        "control_bilateral_direction_subtypes": control_bilateral,
        "control_bilateral_populations": control_populations,
        "ordered_missing_and_control_bilateral_populations_overlap": sorted(
            set(ordered_direction_missing) & set(control_populations)
        ),
        "ordered_failure_and_control_failure_are_population_orthogonal": (
            not set(ordered_direction_missing) & set(control_populations)
        ),
        "authorize_post_hoc_T4d_L_repair": False,
        "authorize_ignore_temporal_shuffle_T4a": False,
        "authorize_source_pair_candidate": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "Tm3_C3_ordered_T4d_L_failure_is_orthogonal_to_shuffle_T4a_bilateral_failure"
        ),
        "boundary": config["boundary"],
    }
