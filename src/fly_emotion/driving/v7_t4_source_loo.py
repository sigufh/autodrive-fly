"""Run the preregistered T4 source-sequence leave-one-out audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_t4_crossfit_sequence_identifiability import _score
from fly_emotion.driving.v7_t4_synapse_antisymmetric_precheck import (
    _moving_edges,
    _run_responses,
)
from fly_emotion.driving.v7_t4_synapse_crossfit_precheck import (
    IMPLEMENTATION as CROSS_FIT_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4_synapse_crossfit_precheck import _crossfit_moments

CONFIG = Path("configs/driving-v7-t4-source-loo.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_source_loo.py")


def _git_sha256(root: Path, revision: str, path: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def _compact(result: dict) -> dict:
    return {
        "direction_pass_count": result["direction_pass_count"],
        "polarity_pass_count": result["polarity_pass_count"],
        "direction_passing_populations": sorted(
            population
            for population, score in result["population_scores"].items()
            if score["direction"]["passed"]
        ),
        "polarity_passing_populations": sorted(
            population
            for population, score in result["population_scores"].items()
            if score["polarity"]["passed"]
        ),
        "bilateral_direction_subtypes": result["bilateral_direction_subtypes"],
        "per_population_direction_median_signed_contrast": {
            population: score["direction"]["median_signed_contrast"]
            for population, score in result["population_scores"].items()
        },
        "per_population_direction_positive_cell_fraction": {
            population: score["direction"]["positive_cell_fraction"]
            for population, score in result["population_scores"].items()
        },
    }


def _ablated_moments(moments: dict, node_types: np.ndarray, source: str) -> tuple[dict, dict]:
    keep = (node_types != source).astype(np.float64)[None, :]
    matrices = {}
    removed = {}
    for name, matrix in moments["matrices"].items():
        masked = matrix.multiply(keep).tocsr()
        masked.eliminate_zeros()
        matrices[name] = masked
        removed[name] = int(matrix.nnz - masked.nnz)
    return {**moments, "matrices": matrices}, removed


def evaluate_v7_t4_source_loo(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    prereg_path = Path(config["preregistration_evidence"])
    prereg_sha = config["preregistration_sha256"]
    if _sha256(root / prereg_path) != prereg_sha:
        raise ValueError("T4 source LOO preregistration hash changed")
    revision = config["preregistration_commit"]
    if _git_sha256(root, revision, str(prereg_path)) != prereg_sha:
        raise ValueError("T4 source LOO preregistration was not frozen at commit")
    prereg = json.loads((root / prereg_path).read_text(encoding="utf-8"))
    if not prereg["ablation_protocol_frozen"]:
        raise ValueError("T4 source LOO protocol is not frozen")
    if prereg["protocol"]["ablation_outputs_observed"]:
        raise ValueError("T4 source LOO preregistration already contains outputs")

    paths = {
        name: Path(config[name])
        for name in (
            "crossfit_axis_evidence",
            "crossfit_axis_protocol",
            "synapse_spatial_protocol",
            "conductance_protocol",
            "stage1_protocol",
            "scoring_config",
        )
    }
    crossfit = json.loads((root / paths["crossfit_axis_evidence"]).read_text())
    if not crossfit["T4_synapse_crossfit_axis_passed"]:
        raise ValueError("T4 source LOO requires passed structural cross-fit axis")
    crossfit_protocol = yaml.safe_load(
        (root / paths["crossfit_axis_protocol"]).read_text()
    )
    execution = {
        "crossfit_split_seed": crossfit_protocol["split_seed"],
        "synapse_spatial_protocol": str(paths["synapse_spatial_protocol"]),
        "source_groups": prereg["source_groups"],
    }
    conductance = yaml.safe_load((root / paths["conductance_protocol"]).read_text())
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    scoring = yaml.safe_load((root / paths["scoring_config"]).read_text())
    if scoring["thresholds"] != prereg["scoring_contract"]["thresholds"]:
        raise ValueError("T4 source LOO scoring thresholds changed")
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == prereg["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("T4 source LOO may consume tuning only")

    normalization = _collect_source_normalization(
        root,
        retinal_backend="linear_luminance",
        retinal_geometry=conductance["primary_retinal_geometry"],
        config=conductance,
    )
    probe = MotionSignConductanceProbe(
        root, retinal_backend="linear_luminance", normalization=normalization
    )
    moments = _crossfit_moments(root, probe, execution, crossfit)
    fixed_denominator = sum(
        len(probe.populations[name])
        for name in scoring["direction_populations"]
        if name.startswith("T4")
    )
    if fixed_denominator != prereg["scoring_contract"]["fixed_T4_population_denominator"]:
        raise ValueError("T4 source LOO fixed population denominator changed")
    if int(np.count_nonzero(moments["valid"])) != prereg["scoring_contract"][
        "valid_crossfit_target_count"
    ]:
        raise ValueError("T4 source LOO valid target count changed")

    stimuli = _moving_edges(condition)
    reduction_indices = {"positive_peak": 1, "signed_mean": 2}
    conditions = {}
    for condition_name in prereg["ablation_conditions"]:
        if condition_name == "intact":
            selected_moments = moments
            removed = {name: 0 for name in moments["matrices"]}
        else:
            source = condition_name.removeprefix("drop_")
            if source not in prereg["source_types"]:
                raise ValueError(f"unknown T4 source ablation: {condition_name}")
            selected_moments, removed = _ablated_moments(
                moments, probe.node_types, source
            )
            if sum(removed.values()) == 0:
                raise ValueError(f"T4 source ablation removed no entries: {source}")
        mode_results = {}
        for mode in prereg["modes"]:
            responses = _run_responses(
                probe,
                stimuli,
                selected_moments,
                mode,
                int(prereg["temporal_shuffle_seed"]),
            )
            mode_results[mode] = {
                reduction: _compact(
                    _score(
                        probe,
                        stimuli,
                        responses,
                        selected_moments,
                        scoring,
                        reduction_indices[reduction],
                    )
                )
                for reduction in prereg["temporal_reductions"]
            }
        controls_maximum = max(
            result["direction_pass_count"]
            for mode in ("temporal_shuffle", "static_sham")
            for result in mode_results[mode].values()
        )
        ordered_maximum = max(
            result["direction_pass_count"]
            for result in mode_results["ordered"].values()
        )
        ordered_polarity_maximum = max(
            result["polarity_pass_count"]
            for result in mode_results["ordered"].values()
        )
        ordered_bilateral = sorted(
            {
                subtype
                for result in mode_results["ordered"].values()
                for subtype in result["bilateral_direction_subtypes"]
            }
        )
        control_bilateral = sorted(
            {
                subtype
                for mode in ("temporal_shuffle", "static_sham")
                for result in mode_results[mode].values()
                for subtype in result["bilateral_direction_subtypes"]
            }
        )
        conditions[condition_name] = {
            "removed_matrix_nonzero_count": removed,
            "mode_results": mode_results,
            "ordered_maximum_direction_pass_count": ordered_maximum,
            "ordered_maximum_polarity_pass_count": ordered_polarity_maximum,
            "control_maximum_direction_pass_count": controls_maximum,
            "ordered_bilateral_direction_subtypes": ordered_bilateral,
            "control_bilateral_direction_subtypes": control_bilateral,
            "ordered_minus_max_control_direction_pass_count": (
                ordered_maximum - controls_maximum
            ),
        }

    frozen_sequence = json.loads(
        (root / next(
            Path(item["path"])
            for item in prereg["protocol"]["frozen_inputs"]
            if item["path"].endswith("crossfit-sequence-identifiability.json")
        )).read_text()
    )
    intact_matches = all(
        conditions["intact"]["mode_results"][mode][reduction][
            "direction_pass_count"
        ]
        == frozen_sequence["mode_reduction_results"][mode][reduction][
            "direction_pass_count"
        ]
        and conditions["intact"]["mode_results"][mode][reduction][
            "polarity_pass_count"
        ]
        == frozen_sequence["mode_reduction_results"][mode][reduction][
            "polarity_pass_count"
        ]
        and conditions["intact"]["mode_results"][mode][reduction][
            "bilateral_direction_subtypes"
        ]
        == frozen_sequence["mode_reduction_results"][mode][reduction][
            "bilateral_direction_subtypes"
        ]
        for mode in prereg["modes"]
        for reduction in prereg["temporal_reductions"]
    )
    if not intact_matches:
        raise ValueError("T4 source LOO intact result differs from frozen evidence")

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(CROSS_FIT_IMPLEMENTATION): _sha256(root / CROSS_FIT_IMPLEMENTATION),
                str(prereg_path): prereg_sha,
                **{str(path): _sha256(root / path) for path in paths.values()},
            },
            "preregistration_commit": revision,
            "preregistration_sha256": prereg_sha,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "fixed_T4_population_denominator": fixed_denominator,
        "valid_target_count": int(np.count_nonzero(moments["valid"])),
        "condition_results": conditions,
        "condition_summary": {
            name: {
                key: result[key]
                for key in (
                    "ordered_maximum_direction_pass_count",
                    "ordered_maximum_polarity_pass_count",
                    "control_maximum_direction_pass_count",
                    "ordered_minus_max_control_direction_pass_count",
                    "ordered_bilateral_direction_subtypes",
                    "control_bilateral_direction_subtypes",
                )
            }
            for name, result in conditions.items()
        },
        "intact_matches_frozen_source_sequence_evidence": intact_matches,
        "ablation_evaluated": True,
        "descriptive_ablation_only": True,
        "authorize_source_removal": False,
        "any_source_removal_satisfies_all_eight_direction_and_polarity_populations": False,
        "any_source_removal_has_ordered_bilateral_subtype_absent_from_controls": bool(
            any(
                result["ordered_bilateral_direction_subtypes"]
                and not result["control_bilateral_direction_subtypes"]
                for name, result in conditions.items()
                if name != "intact"
            )
        ),
        "authorize_new_T4_functional_candidate": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "source_LOO_is_diagnostic_and_does_not_authorize_source_removal",
        "boundary": config["boundary"],
    }
