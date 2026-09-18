"""Post-failure reduction audit for the cross-fit T5 CT1 axis sequence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import _infer_t4_t5_positions
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE
from fly_emotion.driving.v7_t5_ct1_axis_aware_precheck import (
    IMPLEMENTATION as AXIS_AWARE_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t5_ct1_axis_aware_precheck import (
    _compact,
    _terminal_moments,
    _trace,
)
from fly_emotion.driving.v7_t5_ct1_source_dynamics_precheck import _variant
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-ct1-axis-sequence-identifiability.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_ct1_axis_sequence_identifiability.py"
)


def _score_reduction(
    probe, responses, reduction, populations, names, valid, assignments, x_count, y_count, scoring
) -> dict:
    scores = {}
    for population, nodes in populations.items():
        group = np.flatnonzero(names == population)
        subtype, side = population[2], population[-1]
        preferred = (
            HORIZONTAL_PREFERENCE[(subtype, side)]
            if subtype in {"a", "b"}
            else VERTICAL_PREFERENCE[subtype]
        )

        def value(
            polarity: str,
            direction: str,
            *,
            selected_group: np.ndarray = group,
            selected_population: str = population,
        ) -> np.ndarray:
            grid = np.stack(
                [
                    responses[(xi, yi, polarity, direction)][reduction]
                    for yi in range(y_count)
                    for xi in range(x_count)
                ]
            )[:, selected_group]
            assignment = assignments[selected_population]
            rows = assignment[:, 1] * x_count + assignment[:, 0]
            output = grid[rows, np.arange(len(selected_group))]
            output[~valid[selected_group]] = np.nan
            return output

        preferred_response = value("off", preferred)
        scores[population] = {
            "direction": _compact(
                strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("off", OPPOSITE[preferred]),
                    scoring["thresholds"],
                )
            ),
            "polarity": _compact(
                strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("on", preferred),
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
            if scores[f"T5{subtype}_L"]["direction"]["passed"]
            and scores[f"T5{subtype}_R"]["direction"]["passed"]
        ],
        "population_scores": scores,
    }


def evaluate_v7_t5_ct1_axis_sequence_identifiability(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "base_evidence",
            "base_protocol",
            "crossfit_axis_evidence",
            "crossfit_axis_protocol",
            "terminal_axis_evidence",
            "terminal_axis_protocol",
            "lamina_split_protocol",
            "stage1_protocol",
            "scoring_config",
            "annotations",
            "synapse_partners",
        )
    }
    base = json.loads((root / paths["base_evidence"]).read_text())
    if base["candidate_passed"]:
        raise ValueError("identifiability audit requires preserved candidate failure")
    crossfit = json.loads((root / paths["crossfit_axis_evidence"]).read_text())
    if not crossfit["T5_CT1_crossfit_axis_passed"]:
        raise ValueError("identifiability audit requires passed cross-fit axis")
    terminal = json.loads((root / paths["terminal_axis_evidence"]).read_text())
    crossfit_protocol = yaml.safe_load((root / paths["crossfit_axis_protocol"]).read_text())
    config["crossfit_split_seed"] = crossfit_protocol["split_seed"]
    lamina = yaml.safe_load((root / paths["lamina_split_protocol"]).read_text())
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    scoring = yaml.safe_load((root / paths["scoring_config"]).read_text())
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("identifiability audit may consume tuning only")
    probe = LaminaSplitProbe(root, lamina)
    populations = {
        f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    names = np.concatenate([np.full(len(nodes), name) for name, nodes in populations.items()])
    moments = _terminal_moments(root, probe, config, targets, terminal, crossfit)
    if moments["crossfit_axes_sha256"] != crossfit["out_of_fold_predictions_sha256"]:
        raise ValueError("identifiability audit did not reproduce cross-fit target axes")
    tm9 = probe._normalized_target_inputs(
        targets, tuple(config["source_roles"]["enhancer_base"])
    )
    valid = moments["valid"] & (np.diff(tm9.indptr) > 0)
    source_position_path = Path("configs/driving-v7-lplc2-radial-opponency.yaml")
    source_position = yaml.safe_load((root / source_position_path).read_text())
    positions, _ = _infer_t4_t5_positions(root, probe, source_position)
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = finite.min(axis=0), finite.max(axis=0)
    local_path = Path(lamina["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text())
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    assignments = {}
    for population, nodes in populations.items():
        camera = (positions[nodes] - low) / np.maximum(high - low, 1e-12) * (47, 23)
        assignments[population] = np.stack(
            (
                np.argmin(np.abs(x_centers[:, None] - camera[:, 0]), axis=0),
                np.argmin(np.abs(y_centers[:, None] - camera[:, 1]), axis=0),
            ),
            axis=1,
        )
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    responses = {mode: {} for mode in config["modes"]}
    for xi, center_x in enumerate(x_centers):
        for yi, center_y in enumerate(y_centers):
            for polarity in ("on", "off"):
                for direction in ("left", "right", "up", "down"):
                    stimulus = _local_step_edge(
                        float(center_x),
                        float(center_y),
                        float(local["stimulus"]["aperture_radius_pixels"]),
                        speed,
                        polarity,
                        direction,
                        int(local["common_background_frames"]),
                    )
                    for mode in config["modes"]:
                        responses[mode][(xi, yi, polarity, direction)] = _trace(
                            probe,
                            _variant(
                                stimulus, mode, int(config["temporal_shuffle_seed"])
                            ),
                            moments,
                            tm9,
                            float(config["reverse_order_coefficient"]),
                        )
    results = {
        mode: {
            reduction: _score_reduction(
                probe,
                mode_responses,
                reduction,
                populations,
                names,
                valid,
                assignments,
                len(x_centers),
                len(y_centers),
                scoring,
            )
            for reduction in config["temporal_reductions"]
        }
        for mode, mode_responses in responses.items()
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(AXIS_AWARE_IMPLEMENTATION): _sha256(root / AXIS_AWARE_IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in paths.values()},
                str(local_path): _sha256(root / local_path),
                str(source_position_path): _sha256(root / source_position_path),
            },
            "condition_id": config["condition_id"],
            "position_count": int(len(x_centers) * len(y_centers)),
            "stimulus_count_per_mode": len(responses["ordered"]),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "fixed_target_denominator": len(targets),
        "valid_target_count": int(np.count_nonzero(valid)),
        "crossfit_axes_sha256": moments["crossfit_axes_sha256"],
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
