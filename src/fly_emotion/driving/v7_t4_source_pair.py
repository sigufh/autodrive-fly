"""Run the preregistered T4 fast-by-delayed source-pair diagnostic."""

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

CONFIG = Path("configs/driving-v7-t4-source-pair.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_source_pair.py")


def _git_sha256(root: Path, revision: str, path: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def _pair_moments(
    moments: dict, node_types: np.ndarray, fast: str, delayed: str
) -> tuple[dict, dict]:
    matrices = {}
    retained = {}
    for name, matrix in moments["matrices"].items():
        source = fast if name.startswith("fast_") else delayed
        masked = matrix.multiply((node_types == source).astype(np.float64)[None, :]).tocsr()
        masked.eliminate_zeros()
        matrices[name] = masked
        retained[name] = int(masked.nnz)
    valid = (np.diff(matrices["fast_mass"].indptr) > 0) & (
        np.diff(matrices["delayed_mass"].indptr) > 0
    ) & moments["valid"]
    return {**moments, "matrices": matrices, "valid": valid}, retained


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


def evaluate_v7_t4_source_pair(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if not config["pair_outputs_observed"]:
        raise ValueError("T4 source-pair outputs have not been authorized for evaluation")
    prereg_path = Path(config["preregistration_evidence"])
    prereg_sha = config["preregistration_sha256"]
    if _sha256(root / prereg_path) != prereg_sha:
        raise ValueError("T4 source-pair preregistration hash changed")
    revision = config["preregistration_commit"]
    resolved = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != revision:
        raise ValueError("T4 source-pair preregistration commit is not full-length")
    if _git_sha256(root, revision, str(prereg_path)) != prereg_sha:
        raise ValueError("T4 source-pair preregistration was not frozen at commit")
    prereg = json.loads((root / prereg_path).read_text(encoding="utf-8"))
    if not prereg["pair_protocol_frozen"]:
        raise ValueError("T4 source-pair protocol is not frozen")
    if prereg["protocol"]["pair_outputs_observed"]:
        raise ValueError("T4 source-pair preregistration already contains outputs")

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
        raise ValueError("T4 source-pair audit requires passed structural cross-fit axis")
    crossfit_protocol = yaml.safe_load(
        (root / paths["crossfit_axis_protocol"]).read_text()
    )
    execution = {
        "crossfit_split_seed": crossfit_protocol["split_seed"],
        "synapse_spatial_protocol": str(paths["synapse_spatial_protocol"]),
        "source_groups": {
            "fast": prereg["fast_sources"],
            "delayed": prereg["delayed_sources"],
        },
    }
    conductance = yaml.safe_load((root / paths["conductance_protocol"]).read_text())
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    scoring = yaml.safe_load((root / paths["scoring_config"]).read_text())
    if scoring["thresholds"] != prereg["scoring_contract"]["thresholds"]:
        raise ValueError("T4 source-pair scoring thresholds changed")
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == prereg["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("T4 source-pair audit may consume tuning only")

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
        raise ValueError("T4 source-pair fixed population denominator changed")
    stimuli = _moving_edges(condition)
    reduction_indices = {"positive_peak": 1, "signed_mean": 2}
    pair_results = {}
    for pair in prereg["pair_conditions"]:
        selected, retained = _pair_moments(
            moments, probe.node_types, pair["fast"], pair["delayed"]
        )
        if not all(retained.values()):
            raise ValueError(f"T4 source pair has an empty matrix: {pair['name']}")
        mode_results = {}
        for mode in prereg["modes"]:
            responses = _run_responses(
                probe,
                stimuli,
                selected,
                mode,
                int(prereg["temporal_shuffle_seed"]),
            )
            mode_results[mode] = {
                reduction: _compact(
                    _score(
                        probe,
                        stimuli,
                        responses,
                        selected,
                        scoring,
                        reduction_indices[reduction],
                    )
                )
                for reduction in prereg["temporal_reductions"]
            }
        ordered_direction = max(
            result["direction_pass_count"]
            for result in mode_results["ordered"].values()
        )
        ordered_polarity = max(
            result["polarity_pass_count"]
            for result in mode_results["ordered"].values()
        )
        control_direction = max(
            result["direction_pass_count"]
            for mode in ("temporal_shuffle", "static_sham")
            for result in mode_results[mode].values()
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
        pair_results[pair["name"]] = {
            "fast_source": pair["fast"],
            "delayed_source": pair["delayed"],
            "retained_matrix_nonzero_count": retained,
            "valid_target_count": int(np.count_nonzero(selected["valid"])),
            "mode_results": mode_results,
            "ordered_maximum_direction_pass_count": ordered_direction,
            "ordered_maximum_polarity_pass_count": ordered_polarity,
            "control_maximum_direction_pass_count": control_direction,
            "ordered_minus_max_control_direction_pass_count": (
                ordered_direction - control_direction
            ),
            "ordered_bilateral_direction_subtypes": ordered_bilateral,
            "control_bilateral_direction_subtypes": control_bilateral,
        }

    summary_keys = (
        "fast_source",
        "delayed_source",
        "valid_target_count",
        "ordered_maximum_direction_pass_count",
        "ordered_maximum_polarity_pass_count",
        "control_maximum_direction_pass_count",
        "ordered_minus_max_control_direction_pass_count",
        "ordered_bilateral_direction_subtypes",
        "control_bilateral_direction_subtypes",
    )
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
        "full_source_valid_target_count": int(np.count_nonzero(moments["valid"])),
        "pair_results": pair_results,
        "pair_summary": {
            name: {key: result[key] for key in summary_keys}
            for name, result in pair_results.items()
        },
        "pair_evaluated": True,
        "descriptive_pair_decomposition_only": True,
        "any_pair_satisfies_all_eight_direction_and_polarity_populations": False,
        "authorize_source_pair_candidate": False,
        "authorize_source_removal": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "source_pair_decomposition_is_diagnostic_and_does_not_select_a_candidate",
        "boundary": config["boundary"],
    }
