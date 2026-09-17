"""Audit whether label-free temporal consistency rejects shuffled motion frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_three_hop_moment import (
    IMPLEMENTATION as THREE_HOP_IMPLEMENTATION,
)
from fly_emotion.driving.v7_three_hop_moment import _energy, _local_step_edge, _path_matrix

CONFIG = Path("configs/driving-v7-three-hop-temporal-consistency.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_three_hop_temporal_consistency.py")


def _metrics(horizontal: np.ndarray, vertical: np.ndarray) -> dict[str, np.ndarray]:
    vectors = np.stack((horizontal, vertical), axis=2)
    mean_vector = np.mean(vectors, axis=0)
    mean_magnitude = np.linalg.norm(mean_vector, axis=1)
    path_length = np.mean(np.linalg.norm(vectors, axis=2), axis=0)
    coherence = mean_magnitude / (path_length + 1e-9)
    sign_persistence = np.linalg.norm(np.mean(np.sign(vectors), axis=0), axis=1) / np.sqrt(2.0)
    return {
        "normalized_mean_vector_coherence": coherence,
        "vector_sign_consistency": sign_persistence,
        "mean_vector_magnitude": mean_magnitude,
        "coherence_weighted_magnitude": mean_magnitude * coherence,
        "squared_coherence_weighted_magnitude": mean_magnitude * coherence**2,
        "sign_persistence_weighted_magnitude": mean_magnitude * sign_persistence,
    }


def evaluate_v7_three_hop_temporal_consistency(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    three_hop_path = Path(config["three_hop_protocol"])
    three_hop = yaml.safe_load((root / three_hop_path).read_text(encoding="utf-8"))
    evidence_path = Path(config["three_hop_evidence"])
    evidence = json.loads((root / evidence_path).read_text(encoding="utf-8"))
    if evidence["temporal_shuffle_control_passed"]:
        raise ValueError("temporal-consistency audit is only for the failed shuffle control")
    frozen_threshold = float(
        three_hop["controls"][
            "maximum_shuffle_to_ordered_energy_ratio_for_diagnostic_attenuation"
        ]
    )
    if float(config["maximum_shuffle_to_ordered_energy_ratio"]) != frozen_threshold:
        raise ValueError("temporal-consistency threshold differs from frozen three-hop protocol")
    local_path = Path(config["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text(encoding="utf-8"))
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text(encoding="utf-8"))
    condition = next(
        row for row in stage1["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("temporal-consistency audit may consume tuning only")

    probe = MassBalancedVisualProbe(root, local)
    receptor_x = probe.retinal_u.astype(np.float64) * 47.0
    receptor_y = probe.retinal_v.astype(np.float64) * 23.0
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])
    modes = ("ordered", "temporal_shuffle")
    results = {}
    for family, settings in config["families"].items():
        populations = {
            f"{family}{subtype}_{side}": probe.populations[f"{family}{subtype}_{side}"]
            for subtype in "abcd"
            for side in "LR"
        }
        targets = np.concatenate(list(populations.values())).astype(np.int32)
        names = np.concatenate(
            [np.full(len(nodes), name) for name, nodes in populations.items()]
        )
        matrix, path_summary = _path_matrix(probe, targets, tuple(settings["sources"]))
        values = {
            metric: {mode: [] for mode in modes} for metric in config["metrics"]
        }
        for xi, center_x in enumerate(x_centers):
            for yi, center_y in enumerate(y_centers):
                for direction in ("left", "right", "up", "down"):
                    stimulus = _local_step_edge(
                        float(center_x),
                        float(center_y),
                        float(three_hop["stimulus"]["aperture_radius_pixels"]),
                        speed,
                        str(settings["channel"]),
                        direction,
                        int(three_hop["stimulus"]["common_background_frames"]),
                    )
                    baseline_count = int(three_hop["stimulus"]["common_background_frames"])
                    prefix = stimulus.frames[:baseline_count]
                    moving = stimulus.frames[baseline_count:]
                    order = np.random.default_rng(
                        int(config["temporal_shuffle_seed"]) + xi * 1000 + yi * 100
                    ).permutation(len(moving))
                    frames_by_mode = {
                        "ordered": stimulus.frames,
                        "temporal_shuffle": np.concatenate((prefix, moving[order])),
                    }
                    for mode, frames in frames_by_mode.items():
                        variant = VisualStimulus(
                            f"{stimulus.name}:{mode}",
                            stimulus.family,
                            stimulus.polarity,
                            stimulus.direction,
                            frames,
                        )
                        horizontal, vertical = _energy(
                            probe,
                            variant,
                            matrix,
                            receptor_x,
                            receptor_y,
                            str(settings["channel"]),
                        )
                        metrics = _metrics(horizontal, vertical)
                        for metric in config["metrics"]:
                            values[metric][mode].append(metrics[metric])
        family_results = {}
        for metric in config["metrics"]:
            population_results = {}
            for population in populations:
                group = np.flatnonzero(names == population)
                ordered = float(np.mean(np.stack(values[metric]["ordered"])[:, group]))
                shuffled = float(
                    np.mean(np.stack(values[metric]["temporal_shuffle"])[:, group])
                )
                ratio = shuffled / max(ordered, 1e-12)
                population_results[population] = {
                    "ordered_mean_energy": ordered,
                    "temporal_shuffle_mean_energy": shuffled,
                    "shuffle_to_ordered_energy_ratio": ratio,
                    "attenuation_gate_passed": ratio <= frozen_threshold,
                }
            family_results[metric] = {
                "population_results": population_results,
                "attenuated_population_count": int(
                    sum(item["attenuation_gate_passed"] for item in population_results.values())
                ),
                "minimum_shuffle_to_ordered_ratio": min(
                    item["shuffle_to_ordered_energy_ratio"]
                    for item in population_results.values()
                ),
                "median_shuffle_to_ordered_ratio": float(
                    np.median(
                        [
                            item["shuffle_to_ordered_energy_ratio"]
                            for item in population_results.values()
                        ]
                    )
                ),
                "maximum_shuffle_to_ordered_ratio": max(
                    item["shuffle_to_ordered_energy_ratio"]
                    for item in population_results.values()
                ),
                "passed": all(
                    item["attenuation_gate_passed"] for item in population_results.values()
                ),
            }
        results[family] = {
            "path_summary": path_summary,
            "metrics": family_results,
        }
    passing = [
        f"{family}:{metric}"
        for family, family_result in results.items()
        for metric, result in family_result["metrics"].items()
        if result["passed"]
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(three_hop_path): _sha256(root / three_hop_path),
                str(THREE_HOP_IMPLEMENTATION): _sha256(root / THREE_HOP_IMPLEMENTATION),
                str(evidence_path): _sha256(root / evidence_path),
                str(local_path): _sha256(root / local_path),
                str(stage1_path): _sha256(root / stage1_path),
            },
            "condition_id": config["condition_id"],
            "metric_count": len(config["metrics"]),
            "stimulus_count_per_family": int(len(x_centers) * len(y_centers) * 4),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "maximum_shuffle_to_ordered_energy_ratio": frozen_threshold,
        "families": results,
        "passing_family_metrics": passing,
        "temporal_consistency_gate_passed": bool(passing),
        "authorize_new_motion_field_candidate": False,
        "advance_to_three_tuning_conditions": False,
        "advance_to_LPLC_mechanism_repair": False,
        "stop_reason": (
            None
            if passing
            else "all_label_free_consistency_metrics_amplified_temporal_shuffle"
        ),
        "boundary": config["boundary"],
    }
