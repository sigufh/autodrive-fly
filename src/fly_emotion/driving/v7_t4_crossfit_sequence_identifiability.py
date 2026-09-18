"""Post-failure direction audit of the T4 cross-fit source sequence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4_synapse_antisymmetric_precheck import (
    _compact,
    _moving_edges,
    _run_responses,
)
from fly_emotion.driving.v7_t4_synapse_crossfit_precheck import (
    IMPLEMENTATION as CROSS_FIT_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4_synapse_crossfit_precheck import _crossfit_moments
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE

CONFIG = Path("configs/driving-v7-t4-crossfit-sequence-identifiability.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_crossfit_sequence_identifiability.py"
)


def _score(
    probe, stimuli, responses, moments: dict, scoring: dict, reduction_index: int
) -> dict:
    lookup = {(item.polarity, item.direction): item.identity for item in stimuli}
    populations = {
        name: probe.populations[name]
        for name in scoring["direction_populations"]
        if name.startswith("T4")
    }
    scores = {}
    for population, nodes in populations.items():
        preferred = scoring["direction_populations"][population]
        target_rows = np.asarray(
            [
                moments["target_index"].get(int(body_id), -1)
                for body_id in probe.graph.body_ids[nodes]
            ]
        )

        def value(
            polarity: str,
            direction: str,
            *,
            selected_rows: np.ndarray = target_rows,
        ) -> np.ndarray:
            sequence = responses[lookup[(polarity, direction)]][reduction_index]
            output = np.full(len(selected_rows), np.nan)
            present = selected_rows >= 0
            output[present] = sequence[selected_rows[present]]
            valid_rows = np.maximum(selected_rows, 0)
            output[present & ~moments["valid"][valid_rows]] = np.nan
            return output

        preferred_response = value("on", preferred)
        scores[population] = {
            "direction": _compact(
                strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("on", OPPOSITE[preferred]),
                    scoring["thresholds"],
                )
            ),
            "polarity": _compact(
                strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("off", preferred),
                    scoring["thresholds"],
                )
            ),
        }
    return {
        "direction_pass_count": int(
            sum(item["direction"]["passed"] for item in scores.values())
        ),
        "polarity_pass_count": int(
            sum(item["polarity"]["passed"] for item in scores.values())
        ),
        "bilateral_direction_subtypes": [
            subtype
            for subtype in "abcd"
            if scores[f"T4{subtype}_L"]["direction"]["passed"]
            and scores[f"T4{subtype}_R"]["direction"]["passed"]
        ],
        "population_scores": scores,
    }


def evaluate_v7_t4_crossfit_sequence_identifiability(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "base_evidence",
            "base_protocol",
            "crossfit_axis_evidence",
            "crossfit_axis_protocol",
            "synapse_spatial_protocol",
            "conductance_protocol",
            "stage1_protocol",
            "scoring_config",
        )
    }
    base = json.loads((root / paths["base_evidence"]).read_text())
    if base["candidate_passed"]:
        raise ValueError("T4 sequence audit requires preserved cross-fit candidate failure")
    crossfit = json.loads((root / paths["crossfit_axis_evidence"]).read_text())
    if not crossfit["T4_synapse_crossfit_axis_passed"]:
        raise ValueError("T4 sequence audit requires passed cross-fit axis")
    crossfit_protocol = yaml.safe_load((root / paths["crossfit_axis_protocol"]).read_text())
    config["crossfit_split_seed"] = crossfit_protocol["split_seed"]
    conductance = yaml.safe_load((root / paths["conductance_protocol"]).read_text())
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    scoring = yaml.safe_load((root / paths["scoring_config"]).read_text())
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("T4 sequence audit may consume tuning only")
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
                str(CROSS_FIT_IMPLEMENTATION): _sha256(
                    root / CROSS_FIT_IMPLEMENTATION
                ),
                **{str(path): _sha256(root / path) for path in dependencies},
            },
            "condition_id": config["condition_id"],
            "stimulus_count_per_mode": len(stimuli),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "fixed_T4_population_denominator": base["fixed_T4_population_denominator"],
        "valid_target_count": int(np.count_nonzero(moments["valid"])),
        "crossfit_subset_axes_sha256": moments["crossfit_subset_axes_sha256"],
        "mode_reduction_results": results,
        "maximum_ordered_direction_pass_count": max(
            item["direction_pass_count"] for item in results["ordered"].values()
        ),
        "ordered_bilateral_direction_subtypes_by_reduction": {
            name: item["bilateral_direction_subtypes"]
            for name, item in results["ordered"].items()
        },
        "candidate_selected": False,
        "authorize_new_functional_candidate": False,
        "advance_to_three_tuning_conditions": False,
        "advance_to_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "boundary": config["boundary"],
    }
