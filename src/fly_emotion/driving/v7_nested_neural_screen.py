from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import V7VisualProbe
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina
from fly_emotion.driving.v7_stage1_development import (
    _as_visual_stimulus,
    run_signed_cell_responses,
    score_development,
)
from fly_emotion.driving.v7_stage1_nested import (
    CONFIG as NESTED_CONFIG,
)
from fly_emotion.driving.v7_stage1_nested import (
    build_condition_bundle,
)

CONFIG = Path("configs/driving-v7-nested-neural-screen.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_nested_neural_screen.py")


class MassBalancedVisualProbe(V7VisualProbe):
    def __init__(self, root: Path, config: dict):
        super().__init__(
            root,
            brain_substeps=int(config["brain_substeps_per_frame"]),
            baseline_frames=int(config["baseline_frames"]),
            retinal_backend=config["retinal_backend"],
            retinal_geometry="legacy_proxy_v2",
            dynamics_backend=config["dynamics_backend"],
        )
        balanced = build_mass_balanced_retina(root)
        nodes = np.searchsorted(self.graph.body_ids, balanced.body_ids).astype(np.int32)
        if not np.array_equal(self.graph.body_ids[nodes], balanced.body_ids):
            raise ValueError("mass-balanced receptor IDs do not align with the graph")
        self.retina = type(self.retina)(
            nodes,
            balanced.body_ids,
            balanced.x / 47.0,
            balanced.y / 23.0,
            balanced.side,
            4,
        )
        self.retinal_u, self.retinal_v = self.retina.u, self.retina.v
        self.retinal_permutation = np.arange(len(nodes), dtype=np.int32)
        self.mass_gain = balanced.mass * (len(nodes) / (2 * 24 * 24))

    def _retinal_code(
        self, values: np.ndarray, baseline: np.ndarray, previous: np.ndarray
    ) -> np.ndarray:
        return super()._retinal_code(values, baseline, previous) * self.mass_gain


def _pass_count(section: dict) -> int:
    return sum(bool(item["passed"]) for item in section.values())


def _joint_target_coverage(
    condition_scores: dict[str, dict], condition_ids: list[str], section: str, population: str
) -> dict:
    scores = [condition_scores[name][section][population] for name in condition_ids]
    body_ids = np.asarray(scores[0]["cell_ids"], dtype=np.int64)
    if len(np.unique(body_ids)) != len(body_ids):
        raise ValueError("population body IDs must be unique")
    valid_by_condition = []
    success_by_condition = []
    for score in scores:
        ids = np.asarray(score["cell_ids"], dtype=np.int64)
        if not np.array_equal(ids, body_ids):
            raise ValueError("population body IDs are misaligned across conditions")
        contrast = np.asarray(
            [value if value is not None else np.nan for value in score["contrast"]],
            dtype=np.float64,
        )
        valid = np.isfinite(contrast)
        valid_by_condition.append(valid)
        success_by_condition.append(valid & (contrast >= 0.10))
    valid_matrix = np.stack(valid_by_condition)
    success_matrix = np.stack(success_by_condition)
    joint_valid = np.all(valid_matrix, axis=0)
    all_condition_success = np.all(success_matrix, axis=0)
    total = len(body_ids)
    per_condition = {
        name: {
            "valid_count": int(np.count_nonzero(valid_matrix[index])),
            "valid_fraction": float(np.mean(valid_matrix[index])),
            "all_target_success_count": int(np.count_nonzero(success_matrix[index])),
            "all_target_success_fraction": float(np.mean(success_matrix[index])),
        }
        for index, name in enumerate(condition_ids)
    }
    return {
        "target_count": total,
        "body_ids_sha256": hashlib.sha256(body_ids.tobytes()).hexdigest(),
        "per_condition": per_condition,
        "joint_valid_count": int(np.count_nonzero(joint_valid)),
        "joint_valid_fraction": float(np.mean(joint_valid)),
        "all_condition_success_count": int(np.count_nonzero(all_condition_success)),
        "all_condition_success_fraction": float(np.mean(all_condition_success)),
        "joint_invalid_body_ids": body_ids[~joint_valid].tolist(),
        "all_condition_failure_body_ids": body_ids[~all_condition_success].tolist(),
        "passed": bool(
            all(item["valid_fraction"] >= 0.80 for item in per_condition.values())
            and np.mean(joint_valid) >= 0.80
            and np.mean(all_condition_success) >= 0.60
        ),
    }


def _compact_condition_score(score: dict) -> dict:
    body_ids = np.asarray(score["cell_ids"], dtype=np.int64)
    contrasts = np.asarray(
        [value if value is not None else np.nan for value in score["contrast"]],
        dtype=np.float64,
    )
    valid = np.isfinite(contrasts)
    success = valid & (contrasts >= 0.10)
    return {
        "cell_count": int(len(body_ids)),
        "valid_cell_count": int(np.count_nonzero(valid)),
        "valid_cell_fraction": float(np.mean(valid)),
        "median_signed_contrast": score["median_signed_contrast"],
        "positive_cell_fraction_valid_only": score["positive_cell_fraction"],
        "all_target_success_count": int(np.count_nonzero(success)),
        "all_target_success_fraction": float(np.mean(success)),
        "invalid_body_ids": body_ids[~valid].tolist(),
        "below_margin_or_invalid_body_ids": body_ids[~success].tolist(),
        "gates": score["gates"],
        "passed": score["passed"],
    }


def evaluate_v7_nested_neural_screen(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_config = yaml.safe_load((root / NESTED_CONFIG).read_text())
    nested_path = Path(config["nested_protocol"])
    if not (root / nested_path).exists():
        raise ValueError("nested protocol evidence is missing")
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    conditions_by_id = {item["condition_id"]: item for item in nested_config["conditions"]}
    condition_ids = list(config["condition_ids"])
    if any(conditions_by_id[name]["role"] != "tuning" for name in condition_ids):
        raise ValueError("nested neural screen may consume tuning conditions only")
    if len(condition_ids) != int(config["minimum_consistent_conditions"]):
        raise ValueError("nested neural screen requires exactly three tuning conditions")
    probe = MassBalancedVisualProbe(root, config)
    stimuli_by_condition = {
        name: build_condition_bundle(conditions_by_id[name]) for name in condition_ids
    }
    responses = {}
    for stimuli in stimuli_by_condition.values():
        for stimulus in stimuli:
            responses[stimulus.identity] = run_signed_cell_responses(
                probe, _as_visual_stimulus(stimulus)
            )
    per_condition = {
        name: score_development(probe, stimuli, responses, scoring)
        for name, stimuli in stimuli_by_condition.items()
    }
    all_stimuli = [item for name in condition_ids for item in stimuli_by_condition[name]]
    aggregate = score_development(probe, all_stimuli, responses, scoring)
    population_consistency = {}
    target_coverage = {}
    for section in ("direction", "polarity", "looming"):
        population_consistency[section] = {
            population: {
                "passing_condition_count": sum(
                    per_condition[name][section][population]["passed"] for name in condition_ids
                ),
                "required_condition_count": len(condition_ids),
                "passed": all(
                    per_condition[name][section][population]["passed"] for name in condition_ids
                ),
            }
            for population in aggregate[section]
        }
        target_coverage[section] = {
            population: _joint_target_coverage(per_condition, condition_ids, section, population)
            for population in aggregate[section]
        }
    summaries = {
        name: {
            "direction_pass_count": _pass_count(score["direction"]),
            "polarity_pass_count": _pass_count(score["polarity"]),
            "looming_pass_count": _pass_count(score["looming"]),
            "mirror_pass_count": _pass_count(score["mirror"]),
            "mirror_total": len(score["mirror"]),
            "passed": score["passed"],
        }
        for name, score in per_condition.items()
    }
    summaries["aggregate"] = {
        "direction_pass_count": _pass_count(aggregate["direction"]),
        "polarity_pass_count": _pass_count(aggregate["polarity"]),
        "looming_pass_count": _pass_count(aggregate["looming"]),
        "mirror_pass_count": _pass_count(aggregate["mirror"]),
        "mirror_total": len(aggregate["mirror"]),
        "passed": aggregate["passed"],
    }
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(NESTED_CONFIG): _sha256(root / NESTED_CONFIG),
                str(nested_path): _sha256(root / nested_path),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_ids": condition_ids,
            "stimulus_count": len(all_stimuli),
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "candidate": {
            "retina": "mass_balanced_exact_pixel_reconstruction",
            "retinal_backend": config["retinal_backend"],
            "dynamics_backend": config["dynamics_backend"],
            "receptor_count": len(probe.retina.node_indices),
        },
        "summaries": summaries,
        "population_consistency": population_consistency,
        "target_coverage": target_coverage,
        "per_condition_scores": {
            name: {
                section: {
                    population: _compact_condition_score(score)
                    for population, score in values.items()
                }
                for section, values in per_condition[name].items()
                if section in {"direction", "polarity", "looming"}
            }
            for name in condition_ids
        },
        "aggregate_scores": aggregate,
        "strict_neural_gates_passed": aggregate["passed"]
        and all(
            item["passed"]
            for section in population_consistency.values()
            for item in section.values()
        ),
        "calibration_evaluated": False,
        "final_evaluated": False,
        "advance_to_visual_gate": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
