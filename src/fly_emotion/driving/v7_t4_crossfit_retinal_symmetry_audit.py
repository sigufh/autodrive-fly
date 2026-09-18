"""Audit whether paired retinal sampling explains T4 cross-fit asymmetry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_retina_audit import build_balanced_retina_control
from fly_emotion.driving.v7_t4_crossfit_sequence_identifiability import (
    IMPLEMENTATION as SEQUENCE_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4_crossfit_sequence_identifiability import _score
from fly_emotion.driving.v7_t4_synapse_antisymmetric_precheck import (
    _moving_edges,
    _run_responses,
)
from fly_emotion.driving.v7_t4_synapse_crossfit_precheck import (
    IMPLEMENTATION as CROSS_FIT_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4_synapse_crossfit_precheck import _crossfit_moments

CONFIG = Path("configs/driving-v7-t4-crossfit-retinal-symmetry-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_crossfit_retinal_symmetry_audit.py"
)


def _compact_modes(results: dict) -> dict:
    return {
        mode: {
            reduction: {
                "direction_pass_count": item["direction_pass_count"],
                "polarity_pass_count": item["polarity_pass_count"],
                "bilateral_direction_subtypes": item[
                    "bilateral_direction_subtypes"
                ],
            }
            for reduction, item in reductions.items()
        }
        for mode, reductions in results.items()
    }


def _apply_balanced_retina(root: Path, probe) -> None:
    balanced = build_balanced_retina_control(root)
    probe._retinal_drive_gain = probe._retinal_drive_gain[balanced.source_indices]
    probe.retina = balanced.retina
    probe.retinal_u = balanced.retina.u
    probe.retinal_v = balanced.retina.v
    probe.retinal_permutation = np.arange(probe.retina.size, dtype=np.int32)
    probe.retinal_geometry = "balanced_exact_mirror_v1"


def evaluate_v7_t4_crossfit_retinal_symmetry_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "reference_sequence_evidence",
            "reference_sequence_protocol",
            "crossfit_axis_evidence",
            "crossfit_axis_protocol",
            "synapse_spatial_protocol",
            "conductance_protocol",
            "stage1_protocol",
            "scoring_config",
            "retina_audit_evidence",
        )
    }
    reference = json.loads((root / paths["reference_sequence_evidence"]).read_text())
    if reference["authorize_new_functional_candidate"]:
        raise ValueError("retinal symmetry audit requires preserved source-sequence failure")
    retina = json.loads((root / paths["retina_audit_evidence"]).read_text())
    paired = retina["balanced_common_column_control"]
    if not paired["exact_mirror_drive"] or paired["maximum_mirror_drive_error"] != 0.0:
        raise ValueError("balanced retinal control must be exactly mirror paired")
    crossfit = json.loads((root / paths["crossfit_axis_evidence"]).read_text())
    if not crossfit["T4_synapse_crossfit_axis_passed"]:
        raise ValueError("retinal symmetry audit requires passed T4 cross-fit axis")
    crossfit_protocol = yaml.safe_load((root / paths["crossfit_axis_protocol"]).read_text())
    config["crossfit_split_seed"] = crossfit_protocol["split_seed"]
    conductance = yaml.safe_load((root / paths["conductance_protocol"]).read_text())
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    scoring = yaml.safe_load((root / paths["scoring_config"]).read_text())
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("retinal symmetry audit may consume tuning only")
    geometry = config["control_retinal_geometry"]
    normalization = _collect_source_normalization(
        root, retinal_backend="linear_luminance", retinal_geometry=geometry, config=conductance
    )
    probe = MotionSignConductanceProbe(
        root,
        retinal_backend="linear_luminance",
        normalization=normalization,
    )
    _apply_balanced_retina(root, probe)
    moments = _crossfit_moments(root, probe, config, crossfit)
    stimuli = _moving_edges(condition)
    reduction_indices = {"positive_peak": 1, "signed_mean": 2}
    responses = {
        mode: _run_responses(
            probe, stimuli, moments, mode, int(config["temporal_shuffle_seed"])
        )
        for mode in config["modes"]
    }
    results = {
        mode: {
            reduction: _score(
                probe,
                stimuli,
                mode_responses,
                moments,
                scoring,
                reduction_indices[reduction],
            )
            for reduction in config["temporal_reductions"]
        }
        for mode, mode_responses in responses.items()
    }
    reference_summary = _compact_modes(reference["mode_reduction_results"])
    control_summary = _compact_modes(results)
    reference_maximum = max(
        item["direction_pass_count"]
        for item in reference["mode_reduction_results"]["ordered"].values()
    )
    balanced_maximum = max(
        item["direction_pass_count"] for item in results["ordered"].values()
    )
    spatial = yaml.safe_load((root / paths["synapse_spatial_protocol"]).read_text())
    dependencies = [
        *paths.values(),
        Path(spatial["annotations"]),
        Path(spatial["synapse_partners"]),
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(SEQUENCE_IMPLEMENTATION): _sha256(root / SEQUENCE_IMPLEMENTATION),
                str(CROSS_FIT_IMPLEMENTATION): _sha256(root / CROSS_FIT_IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in dependencies},
            },
            "condition_id": config["condition_id"],
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "retinal_geometries": {
            "reference": config["reference_retinal_geometry"],
            "control": geometry,
        },
        "retinal_receptor_counts": {
            "reference": retina["legacy_mapping"]["eye_counts"],
            "control": {
                "left": paired["receptors_per_eye"],
                "right": paired["receptors_per_eye"],
            },
        },
        "control_exact_mirror_drive": paired["exact_mirror_drive"],
        "fixed_T4_population_denominator": reference[
            "fixed_T4_population_denominator"
        ],
        "valid_target_count": int(np.count_nonzero(moments["valid"])),
        "crossfit_subset_axes_sha256": moments["crossfit_subset_axes_sha256"],
        "reference_mode_reduction_summary": reference_summary,
        "balanced_mirror_mode_reduction_results": results,
        "balanced_mirror_mode_reduction_summary": control_summary,
        "maximum_reference_ordered_direction_pass_count": reference_maximum,
        "maximum_balanced_ordered_direction_pass_count": balanced_maximum,
        "balanced_ordered_bilateral_direction_subtypes_by_reduction": {
            name: item["bilateral_direction_subtypes"]
            for name, item in results["ordered"].items()
        },
        "retinal_sampling_imbalance_explains_direction_failure": bool(
            balanced_maximum > reference_maximum
            and any(
                item["bilateral_direction_subtypes"]
                for item in results["ordered"].values()
            )
        ),
        "candidate_selected": False,
        "authorize_new_functional_candidate": False,
        "advance_to_three_tuning_conditions": False,
        "advance_to_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "boundary": config["boundary"],
    }
