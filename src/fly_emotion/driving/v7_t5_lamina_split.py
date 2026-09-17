"""Tuning-only T5 diagnostic with explicit R1-R6-to-lamina ON/OFF separation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import (
    _coverage,
    _infer_t4_t5_positions,
    _layer_traces,
)
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    IMPLEMENTATION as LOCAL_EDGE_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    OPPOSITE,
)
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-lamina-split.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_lamina_split.py")


class LaminaSplitProbe(MassBalancedVisualProbe):
    def __init__(self, root: Path, config: dict):
        super().__init__(root, config)
        receptor_mask = np.zeros(self.graph.node_count, dtype=np.float32)
        receptor_mask[self.retina.node_indices] = 1.0
        self.lamina_projection = {}
        for cell_type in (
            *config["lamina_split"]["ON_targets"],
            *config["lamina_split"]["OFF_targets"],
        ):
            nodes = np.flatnonzero(self.node_types == cell_type).astype(np.int32)
            matrix = self.adjacency[nodes, :].multiply(receptor_mask).tocsr()
            totals = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
            scale = np.zeros_like(totals, dtype=np.float64)
            np.divide(1.0, totals, out=scale, where=totals > 0.0)
            self.lamina_projection[cell_type] = (
                nodes,
                (sparse.diags(scale) @ matrix).tocsr(),
                cell_type in config["lamina_split"]["ON_targets"],
            )

    def _advance(
        self, state: np.ndarray, drive: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        updated = super()._advance(state, drive, history)
        positive, negative = np.maximum(drive, 0.0), np.maximum(-drive, 0.0)
        for nodes, matrix, is_on in self.lamina_projection.values():
            target = np.tanh(matrix @ (positive if is_on else negative)).astype(np.float32)
            updated[nodes] = (1.0 - self.leak[nodes]) * state[nodes] + self.leak[nodes] * target
        return updated


def _compact(score: dict) -> dict:
    return {
        key: score[key]
        for key in (
            "cell_count",
            "valid_cell_count",
            "valid_cell_fraction",
            "median_signed_contrast",
            "positive_cell_fraction",
            "invalid_cell_ids",
            "cell_ids",
            "contrast",
            "gates",
            "passed",
        )
    }


def evaluate_v7_t5_lamina_split(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    local_edge_path = Path(config["local_edge_config"])
    local = yaml.safe_load((root / local_edge_path).read_text())
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text())
    source_path = Path(config["source_protocol"])
    source = yaml.safe_load((root / source_path).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    conditions = {row["condition_id"]: row for row in stage1["conditions"]}
    if any(conditions[name]["role"] != "tuning" for name in config["condition_ids"]):
        raise ValueError("T5 lamina split may consume tuning only")
    probe = LaminaSplitProbe(root, config)
    positions, _ = _infer_t4_t5_positions(root, probe, source)
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = finite.min(axis=0), finite.max(axis=0)
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    populations = {
        f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    assignments = {}
    for name, nodes in populations.items():
        camera = (positions[nodes] - low) / np.maximum(high - low, 1e-12) * (47, 23)
        assignments[name] = np.stack(
            (
                np.argmin(np.abs(x_centers[:, None] - camera[:, 0]), axis=0),
                np.argmin(np.abs(y_centers[:, None] - camera[:, 1]), axis=0),
            ),
            axis=1,
        )
    per_condition = {}
    raw = {name: {"direction": [], "polarity": []} for name in populations}
    for condition_id in config["condition_ids"]:
        speed = float(
            conditions[condition_id]["generator_parameters"]["edge_speed_pixels_per_frame"]
        )
        responses = {}
        for xi, cx in enumerate(x_centers):
            for yi, cy in enumerate(y_centers):
                for polarity in ("on", "off"):
                    for direction in ("left", "right", "up", "down"):
                        stimulus = _local_step_edge(
                            float(cx),
                            float(cy),
                            float(local["stimulus"]["aperture_radius_pixels"]),
                            speed,
                            polarity,
                            direction,
                            int(local["common_background_frames"]),
                        )
                        traces = _layer_traces(probe, stimulus, populations)
                        responses[(xi, yi, polarity, direction)] = {
                            name: np.max(values, axis=0) for name, values in traces.items()
                        }
        condition_scores = {}
        for name, nodes in populations.items():
            subtype, side = name[2], name[-1]
            preferred_direction = (
                HORIZONTAL_PREFERENCE[(subtype, side)]
                if subtype in {"a", "b"}
                else VERTICAL_PREFERENCE[subtype]
            )
            assignment = assignments[name]
            columns = np.arange(len(nodes))

            def value(
                polarity: str,
                direction: str,
                *,
                selected_responses: dict = responses,
                selected_name: str = name,
                selected_assignment: np.ndarray = assignment,
                selected_columns: np.ndarray = columns,
            ) -> np.ndarray:
                grid = np.stack(
                    [
                        selected_responses[(xi, yi, polarity, direction)][selected_name]
                        for yi in range(len(y_centers))
                        for xi in range(len(x_centers))
                    ]
                )
                rows = selected_assignment[:, 1] * len(x_centers) + selected_assignment[:, 0]
                return grid[rows, selected_columns]

            preferred = value("off", preferred_direction)
            direction_score = strict_contrast_summary(
                probe.graph.body_ids[nodes],
                preferred,
                value("off", OPPOSITE[preferred_direction]),
                scoring["thresholds"],
            )
            polarity_score = strict_contrast_summary(
                probe.graph.body_ids[nodes],
                preferred,
                value("on", preferred_direction),
                scoring["thresholds"],
            )
            raw[name]["direction"].append(direction_score)
            raw[name]["polarity"].append(polarity_score)
            condition_scores[name] = {
                "direction": _compact(direction_score),
                "polarity": _compact(polarity_score),
            }
        per_condition[condition_id] = condition_scores
    consistency = {}
    for name, heads in raw.items():
        consistency[name] = {}
        for head, scores in heads.items():
            coverage = _coverage(scores, config["thresholds"])
            consistency[name][head] = {
                "passing_condition_count": int(sum(score["passed"] for score in scores)),
                "required_condition_count": len(scores),
                "coverage": coverage,
                "passed": bool(all(score["passed"] for score in scores) and coverage["passed"]),
            }
    mirror = {}
    for condition_id, condition_scores in per_condition.items():
        condition_mirror = {}
        for subtype in "abcd":
            pair = {}
            for head in ("direction", "polarity"):
                left = condition_scores[f"T5{subtype}_L"][head]["median_signed_contrast"]
                right = condition_scores[f"T5{subtype}_R"][head]["median_signed_contrast"]
                error = abs(left - right) if left is not None and right is not None else None
                pair[head] = {
                    "absolute_median_contrast_error": error,
                    "passed": bool(
                        error is not None
                        and error <= float(config["thresholds"]["maximum_mirror_error"])
                    ),
                }
            condition_mirror[f"T5{subtype}_L<->T5{subtype}_R"] = pair
        mirror[condition_id] = condition_mirror
    response_passed = all(
        result["passed"] for heads in consistency.values() for result in heads.values()
    )
    mirror_passed = all(
        result["passed"]
        for condition in mirror.values()
        for pair in condition.values()
        for result in pair.values()
    )
    passed = bool(response_passed and mirror_passed)
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(local_edge_path): _sha256(root / local_edge_path),
                str(LOCAL_EDGE_IMPLEMENTATION): _sha256(root / LOCAL_EDGE_IMPLEMENTATION),
                str(stage1_path): _sha256(root / stage1_path),
                str(source_path): _sha256(root / source_path),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_ids": config["condition_ids"],
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "lamina_split": config["lamina_split"],
        "per_condition": per_condition,
        "population_consistency": consistency,
        "mirror": mirror,
        "mirror_gate_passed": mirror_passed,
        "strict_T5_gates_passed": bool(passed),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
