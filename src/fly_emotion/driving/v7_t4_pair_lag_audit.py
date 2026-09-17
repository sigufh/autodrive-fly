"""Frozen-lag audit for the T4 continuous source-pair precheck."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4_continuous_pair_precheck import (
    IMPLEMENTATION as BASE_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4_continuous_pair_precheck import _build_projection, _compact
from fly_emotion.driving.v7_t4_source_resolved import (
    IMPLEMENTATION as LAG_SOURCE_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t4-pair-lag-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_pair_lag_audit.py")


def _pair_lag_traces(probe, stimulus, projection: dict, lags: list[int]) -> dict:
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    baseline_image = stimulus.frames[0]
    baseline_values = probe._sample_retina(baseline_image)[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(baseline_values, baseline_values, baseline_values)
    for _ in range(probe.baseline_frames):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = baseline_drive
    baseline_state = state.astype(np.float64)
    previous_retina = baseline_values.copy()
    sequence = []
    for image in stimulus.frames[probe.baseline_frames :]:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous_retina)
        previous_retina = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        positive = np.maximum(state.astype(np.float64) - baseline_state, 0.0)
        sequence.append(
            {name: matrix @ positive for name, matrix in projection["matrices"].items()}
        )

    outputs = {}
    axes = projection["axes"]
    for lag in lags:
        projected = []
        for index in range(lag, len(sequence)):
            current = sequence[index]
            previous = sequence[index - lag]
            cross_x = -(
                current["offset_x_moment"] * previous["direct_mass"]
                - current["offset_mass"] * previous["direct_x_moment"]
                - current["direct_x_moment"] * previous["offset_mass"]
                + current["direct_mass"] * previous["offset_x_moment"]
            )
            cross_y = -(
                current["offset_y_moment"] * previous["direct_mass"]
                - current["offset_mass"] * previous["direct_y_moment"]
                - current["direct_y_moment"] * previous["offset_mass"]
                + current["direct_mass"] * previous["offset_y_moment"]
            )
            terms = (
                current["offset_x_moment"] * previous["direct_mass"],
                current["offset_mass"] * previous["direct_x_moment"],
                current["direct_x_moment"] * previous["offset_mass"],
                current["direct_mass"] * previous["offset_x_moment"],
                current["offset_y_moment"] * previous["direct_mass"],
                current["offset_mass"] * previous["direct_y_moment"],
                current["direct_y_moment"] * previous["offset_mass"],
                current["direct_mass"] * previous["offset_y_moment"],
            )
            denominator = sum(np.abs(term) for term in terms) + 1e-9
            projected.append((axes[:, 0] * cross_x + axes[:, 1] * cross_y) / denominator)
        outputs[lag] = np.stack(projected)
    return outputs


def _reduce(values: np.ndarray, reduction: str) -> np.ndarray:
    if reduction == "signed_mean":
        return np.mean(values, axis=0)
    if reduction == "maximum":
        return np.max(values, axis=0)
    if reduction == "positive_mean":
        return np.mean(np.maximum(values, 0.0), axis=0)
    raise ValueError(f"unknown temporal reduction: {reduction}")


def evaluate_v7_t4_pair_lag_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    base_path = Path(config["base_protocol"])
    base = yaml.safe_load((root / base_path).read_text(encoding="utf-8"))
    evidence_path = Path(config["base_evidence"])
    evidence = json.loads((root / evidence_path).read_text(encoding="utf-8"))
    if evidence["control_eligibility_gate_passed"]:
        raise ValueError("lag audit is only for a failed base candidate")
    lag_source_path = Path(config["lag_source_protocol"])
    lag_source = yaml.safe_load((root / lag_source_path).read_text(encoding="utf-8"))
    frozen_lags = lag_source["hybrid_pilot"]["lags_substeps"]
    if config["lags_frames"] != frozen_lags:
        raise ValueError("lag audit differs from the previously frozen lag range")
    nested_path = Path(base["stage1_protocol"])
    nested = yaml.safe_load((root / nested_path).read_text(encoding="utf-8"))
    scoring_path = Path(base["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    local_path = Path(base["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text(encoding="utf-8"))
    audit_path = Path(base["source_audit_evidence"])
    source_audit = json.loads((root / audit_path).read_text(encoding="utf-8"))
    condition = next(
        row for row in nested["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("T4 pair lag audit may consume tuning only")

    probe = MassBalancedVisualProbe(root, local)
    populations = {
        f"T4{subtype}_{side}": probe.populations[f"T4{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    names = np.concatenate([np.full(len(nodes), name) for name, nodes in populations.items()])
    projection = _build_projection(root, probe, base, source_audit, targets)
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    responses = {}
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
                    responses[(xi, yi, polarity, direction)] = _pair_lag_traces(
                        probe, stimulus, projection, config["lags_frames"]
                    )
    candidates = {}
    for lag in config["lags_frames"]:
        for reduction in config["temporal_reductions"]:
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
                    selected_lag: int = lag,
                    selected_reduction: str = reduction,
                ) -> np.ndarray:
                    grid = [
                        _reduce(
                            responses[(xi, yi, polarity, direction)][selected_lag],
                            selected_reduction,
                        )
                        for yi in range(len(y_centers))
                        for xi in range(len(x_centers))
                    ]
                    output = np.max(np.stack(grid), axis=0)[selected_group]
                    output[~projection["valid"][selected_group]] = np.nan
                    return output

                preferred_response = value("on", preferred)
                direction = strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("on", OPPOSITE[preferred]),
                    scoring["thresholds"],
                )
                polarity = strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("off", preferred),
                    scoring["thresholds"],
                )
                scores[population] = {
                    "direction": _compact(direction),
                    "polarity": _compact(polarity),
                }
            bilateral = [
                subtype
                for subtype in "abcd"
                if scores[f"T4{subtype}_L"]["direction"]["passed"]
                and scores[f"T4{subtype}_R"]["direction"]["passed"]
            ]
            candidates[f"lag={lag}:{reduction}"] = {
                "lag_frames": lag,
                "temporal_reduction": reduction,
                "direction_pass_count": int(
                    sum(item["direction"]["passed"] for item in scores.values())
                ),
                "polarity_pass_count": int(
                    sum(item["polarity"]["passed"] for item in scores.values())
                ),
                "bilateral_direction_subtypes": bilateral,
                "population_scores": scores,
            }
    minimum = int(config["stop_gate"]["minimum_bilateral_direction_subtypes_to_expand"])
    advancing = [
        name
        for name, result in candidates.items()
        if len(result["bilateral_direction_subtypes"]) >= minimum
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(base_path): _sha256(root / base_path),
                str(BASE_IMPLEMENTATION): _sha256(root / BASE_IMPLEMENTATION),
                str(evidence_path): _sha256(root / evidence_path),
                str(lag_source_path): _sha256(root / lag_source_path),
                str(LAG_SOURCE_IMPLEMENTATION): _sha256(root / LAG_SOURCE_IMPLEMENTATION),
                str(nested_path): _sha256(root / nested_path),
                str(scoring_path): _sha256(root / scoring_path),
                str(local_path): _sha256(root / local_path),
                str(audit_path): _sha256(root / audit_path),
            },
            "condition_id": config["condition_id"],
            "lag_count": len(config["lags_frames"]),
            "candidate_count": len(candidates),
            "stimulus_count": len(responses),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "lags_frames": config["lags_frames"],
        "temporal_reductions": config["temporal_reductions"],
        "candidates": candidates,
        "advancing_candidates": advancing,
        "lag_audit_passed": bool(advancing),
        "three_condition_evaluation_performed": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            None
            if advancing
            else "no_frozen_lag_produced_any_bilateral_direction_selective_T4_subtype"
        ),
        "boundary": config["boundary"],
    }
