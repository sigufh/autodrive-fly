"""Audit saturation and population bias across frozen T4 candidate families."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t4-frozen-candidate-family-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_frozen_candidate_family_audit.py"
)


def evaluate_v7_t4_frozen_candidate_family_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    expected_mapping = config["expected_direction_populations"]
    observed_mapping = {
        name: direction
        for name, direction in scoring["direction_populations"].items()
        if name.startswith("T4")
    }
    if observed_mapping != expected_mapping:
        raise ValueError("T4 direction or subtype labels changed")

    expected_names = {
        f"{reduction}:gain={float(gain):g}"
        for reduction in config["expected_candidate_grid"]["temporal_reductions"]
        for gain in config["expected_candidate_grid"]["additive_gains"]
    }
    family_results = {}
    all_candidates = []
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(scoring_path): _sha256(root / scoring_path),
    }
    for family, specification in config["candidate_families"].items():
        evidence_path = Path(specification["evidence"])
        evidence = json.loads((root / evidence_path).read_text(encoding="utf-8"))
        candidates = evidence[specification["result_key"]]
        names = {candidate["name"] for candidate in candidates}
        if len(candidates) != len(expected_names) or names != expected_names:
            raise ValueError(f"T4 {family} candidate grid changed")
        if evidence["protocol"].get("target_activity_injection") is not False:
            raise ValueError(f"T4 {family} target-input boundary changed")
        rows = []
        for candidate in candidates:
            scores = candidate["population_scores"]
            if set(scores) != set(expected_mapping):
                raise ValueError(f"T4 {family} population denominator changed")
            passed = sorted(
                population
                for population, result in scores.items()
                if result["direction"]["passed"]
            )
            row = {
                "family": family,
                "name": candidate["name"],
                "temporal_reduction": candidate["temporal_reduction"],
                "gain": float(candidate["gain"]),
                "direction_pass_count": int(candidate["direction_pass_count"]),
                "polarity_pass_count": int(candidate["polarity_pass_count"]),
                "direction_passing_populations": passed,
                "bilateral_direction_subtypes": candidate[
                    "bilateral_direction_subtypes"
                ],
                "mirror_gate_passed": bool(candidate["mirror_gate_passed"]),
            }
            if len(passed) != row["direction_pass_count"]:
                raise ValueError(f"T4 {family} direction pass count changed")
            rows.append(row)
            all_candidates.append(row)
        family_results[family] = {
            "candidate_count": len(rows),
            "candidates": rows,
        }
        dependencies[str(evidence_path)] = _sha256(root / evidence_path)

    retinal_path = Path(config["retinal_symmetry_evidence"])
    retinal = json.loads((root / retinal_path).read_text(encoding="utf-8"))
    axis_path = Path(config["crossfit_axis_evidence"])
    axis = json.loads((root / axis_path).read_text(encoding="utf-8"))
    dependencies[str(retinal_path)] = _sha256(root / retinal_path)
    dependencies[str(axis_path)] = _sha256(root / axis_path)

    population_frequency = Counter(
        population
        for candidate in all_candidates
        for population in candidate["direction_passing_populations"]
    )
    pass_set_frequency = Counter(
        "|".join(candidate["direction_passing_populations"]) or "none"
        for candidate in all_candidates
    )
    all_polarity = all(candidate["polarity_pass_count"] == 8 for candidate in all_candidates)
    no_bilateral = all(
        not candidate["bilateral_direction_subtypes"] for candidate in all_candidates
    )
    frozen_grid_failed = bool(
        len(all_candidates) == 32
        and all_polarity
        and no_bilateral
        and max(candidate["direction_pass_count"] for candidate in all_candidates) <= 2
    )
    retinal_control_did_not_rescue = bool(
        retinal["control_exact_mirror_drive"]
        and not retinal["retinal_sampling_imbalance_explains_direction_failure"]
        and retinal["maximum_balanced_ordered_direction_pass_count"]
        <= retinal["maximum_reference_ordered_direction_pass_count"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "candidate_outputs_reused_without_recomputation": True,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "candidate_family_count": len(family_results),
        "candidate_count": len(all_candidates),
        "candidate_grid": config["expected_candidate_grid"],
        "direction_population_mapping": observed_mapping,
        "family_results": family_results,
        "direction_passing_population_frequency": {
            population: population_frequency.get(population, 0)
            for population in expected_mapping
        },
        "direction_pass_set_frequency": dict(sorted(pass_set_frequency.items())),
        "maximum_direction_pass_count": max(
            candidate["direction_pass_count"] for candidate in all_candidates
        ),
        "all_candidates_pass_all_polarity_populations": all_polarity,
        "all_candidates_lack_bilateral_direction_subtypes": no_bilateral,
        "mirror_gate_passing_candidate_count": sum(
            candidate["mirror_gate_passed"] for candidate in all_candidates
        ),
        "crossfit_anatomical_axis_gate_passed": axis[
            "T4_synapse_crossfit_axis_passed"
        ],
        "crossfit_axis_minimum_population_accuracy": min(
            item["accuracy"] for item in axis["out_of_fold_by_population"].values()
        ),
        "exact_mirror_retinal_control": {
            "exact_mirror_drive": retinal["control_exact_mirror_drive"],
            "reference_maximum_ordered_direction_pass_count": retinal[
                "maximum_reference_ordered_direction_pass_count"
            ],
            "balanced_maximum_ordered_direction_pass_count": retinal[
                "maximum_balanced_ordered_direction_pass_count"
            ],
            "retinal_sampling_imbalance_explains_direction_failure": retinal[
                "retinal_sampling_imbalance_explains_direction_failure"
            ],
            "did_not_rescue_ordered_direction": retinal_control_did_not_rescue,
        },
        "frozen_candidate_grid_failed": frozen_grid_failed,
        "authorize_post_hoc_additive_gain_expansion": False,
        "authorize_direction_or_subtype_label_change": False,
        "authorize_new_T4_functional_candidate": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "frozen_32_candidate_grid_has_unilateral_T4a_R_T4c_R_pattern_and_"
            "exact_mirror_retina_does_not_rescue_ordered_direction"
            if frozen_grid_failed and retinal_control_did_not_rescue
            else "frozen_candidate_family_audit_requires_review"
        ),
        "boundary": config["boundary"],
    }
