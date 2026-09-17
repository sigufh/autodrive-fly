"""Anatomy-only audit of direct T5 fast-to-delayed source displacement axes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import _node_annotations
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_source_audit import CARDINAL_ORDER, CARDINAL_VECTORS
from fly_emotion.driving.v7_t5_lamina_split import (
    IMPLEMENTATION as LAMINA_SPLIT_IMPLEMENTATION,
)

CONFIG = Path("configs/driving-v7-t5-source-axis-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_source_axis_audit.py")
ANNOTATIONS = Path("data/raw/malecns-v1.0/body-annotations.feather")
RAW_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_raw.npz")
BODY_IDS = Path("data/processed/malecns-v1.0/body_ids.npy")


def _centroid(
    source_nodes: np.ndarray,
    weights: np.ndarray,
    coordinates: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    located = selected & np.all(np.isfinite(coordinates[source_nodes]), axis=1)
    return (
        (
            np.average(coordinates[source_nodes[located]], axis=0, weights=weights[located])
            if np.any(located)
            else np.full(2, np.nan, dtype=np.float64)
        ),
        int(np.count_nonzero(located)),
        float(np.sum(weights[located])),
    )


def _unit_median(vectors: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    if not np.any(valid):
        return None
    normalized = vectors[valid] / np.linalg.norm(vectors[valid], axis=1, keepdims=True)
    center = np.median(normalized, axis=0)
    norm = np.linalg.norm(center)
    return center / norm if norm > 1e-12 else None


def evaluate_v7_t5_source_axis_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    lamina_evidence_path = Path(config["lamina_split_evidence"])
    lamina_evidence = json.loads((root / lamina_evidence_path).read_text())
    if not all(
        result["polarity"]["passing_condition_count"] == 3
        for result in lamina_evidence["population_consistency"].values()
    ):
        raise ValueError("T5 source-axis audit requires the frozen lamina polarity component")
    lamina_path = Path(config["lamina_split_protocol"])
    lamina = yaml.safe_load((root / lamina_path).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    calibration_path = Path(config["optic_hex_axis_evidence"])
    calibration = json.loads((root / calibration_path).read_text())
    if calibration["protocol"]["T5_used_for_fit_or_model_selection"]:
        raise ValueError("T5 axis comparison requires a T4-only calibration")
    probe = MassBalancedVisualProbe(root, lamina)
    raw = load_graph(root / "data/processed/malecns-v1.0", normalized=False).adjacency
    _, coordinates = _node_annotations(root, probe)
    fast_types = tuple(config["source_groups"]["fast"])
    delayed_types = tuple(config["source_groups"]["delayed"])
    thresholds = config["thresholds"]
    records = []
    summaries = {}
    median_axes = {}
    for subtype in "abcd":
        for side in "LR":
            name = f"T5{subtype}_{side}"
            target_nodes = probe.populations[name]
            vectors = np.full((len(target_nodes), 2), np.nan, dtype=np.float64)
            expected_name = scoring["direction_populations"][name]
            expected = CARDINAL_VECTORS[expected_name]
            population_records = []
            for index, target in enumerate(target_nodes):
                row = raw.getrow(int(target))
                sources = row.indices
                weights = np.abs(row.data).astype(np.float64)
                fast, fast_count, fast_weight = _centroid(
                    sources,
                    weights,
                    coordinates,
                    np.isin(probe.node_types[sources], fast_types),
                )
                delayed, delayed_count, delayed_weight = _centroid(
                    sources,
                    weights,
                    coordinates,
                    np.isin(probe.node_types[sources], delayed_types),
                )
                if side == "L":
                    fast[0] *= -1.0
                    delayed[0] *= -1.0
                axis = delayed - fast
                norm = float(np.linalg.norm(axis))
                valid = bool(np.all(np.isfinite(axis)) and norm > thresholds["minimum_axis_norm"])
                if valid:
                    vectors[index] = axis
                    unit = axis / norm
                    cosine = float(unit @ expected)
                    cardinal_vectors = np.stack([CARDINAL_VECTORS[x] for x in CARDINAL_ORDER])
                    predicted = CARDINAL_ORDER[
                        int(np.argmax(unit @ cardinal_vectors.T))
                    ]
                else:
                    cosine = None
                    predicted = None
                population_records.append(
                    {
                        "body_id": int(probe.graph.body_ids[target]),
                        "fast_located_source_count": fast_count,
                        "delayed_located_source_count": delayed_count,
                        "fast_located_source_weight": fast_weight,
                        "delayed_located_source_weight": delayed_weight,
                        "fast_centroid": fast.tolist() if np.all(np.isfinite(fast)) else None,
                        "delayed_centroid": (
                            delayed.tolist() if np.all(np.isfinite(delayed)) else None
                        ),
                        "fast_to_delayed_axis": axis.tolist() if valid else None,
                        "axis_norm": norm if np.isfinite(norm) else None,
                        "expected_direction": expected_name,
                        "expected_cosine": cosine,
                        "predicted_cardinal_direction": predicted,
                    }
                )
            valid = np.all(np.isfinite(vectors), axis=1)
            normalized = np.full_like(vectors, np.nan)
            normalized[valid] = vectors[valid] / np.linalg.norm(
                vectors[valid], axis=1, keepdims=True
            )
            cosines = normalized @ expected
            positive = valid & (cosines > 0.0)
            exact = np.asarray(
                [row["predicted_cardinal_direction"] == expected_name for row in population_records]
            )
            median_axis = _unit_median(vectors, valid)
            median_axes[name] = median_axis
            gates = {
                "valid_cell_fraction": float(np.mean(valid))
                >= float(thresholds["minimum_valid_cell_fraction"]),
                "median_expected_cosine": bool(
                    np.any(valid)
                    and np.median(cosines[valid])
                    >= float(thresholds["minimum_median_expected_cosine"])
                ),
                "positive_alignment_fraction": float(np.mean(positive))
                >= float(thresholds["minimum_positive_alignment_fraction"]),
            }
            summaries[name] = {
                "cell_count": int(len(target_nodes)),
                "valid_axis_count": int(np.count_nonzero(valid)),
                "valid_axis_fraction": float(np.mean(valid)),
                "median_expected_cosine": (
                    float(np.median(cosines[valid])) if np.any(valid) else None
                ),
                "positive_alignment_fraction_all_cells": float(np.mean(positive)),
                "exact_cardinal_accuracy_all_cells": float(np.mean(exact)),
                "median_unit_axis": median_axis.tolist() if median_axis is not None else None,
                "gates": gates,
                "passed": bool(all(gates.values())),
            }
            records.extend(population_records)
    mirrors = {}
    for subtype in "abcd":
        left = median_axes[f"T5{subtype}_L"]
        right = median_axes[f"T5{subtype}_R"]
        reflected_right = None if right is None else np.asarray((-right[0], right[1]))
        error = (
            None
            if left is None or reflected_right is None
            else float(np.linalg.norm(left - reflected_right))
        )
        mirrors[subtype] = {
            "left_median_unit_axis": left.tolist() if left is not None else None,
            "reflected_right_median_unit_axis": (
                reflected_right.tolist() if reflected_right is not None else None
            ),
            "error": error,
            "passed": bool(
                error is not None
                and error <= float(thresholds["maximum_population_mirror_axis_error"])
            ),
        }
    opponent_axes = {}
    for side in "LR":
        for first, second in (("a", "b"), ("c", "d")):
            first_axis = median_axes[f"T5{first}_{side}"]
            second_axis = median_axes[f"T5{second}_{side}"]
            cosine = (
                None
                if first_axis is None or second_axis is None
                else float(first_axis @ second_axis)
            )
            opponent_axes[f"T5{first}_{side}<->T5{second}_{side}"] = {
                "median_axis_cosine": cosine,
                "opposite_angle_degrees": (
                    None
                    if cosine is None
                    else float(180.0 - np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                ),
                "descriptive_only": True,
            }
    population_gate = all(item["passed"] for item in summaries.values())
    mirror_gate = all(item["passed"] for item in mirrors.values())
    strict = bool(population_gate and mirror_gate)
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(lamina_evidence_path): _sha256(root / lamina_evidence_path),
                str(lamina_path): _sha256(root / lamina_path),
                str(LAMINA_SPLIT_IMPLEMENTATION): _sha256(root / LAMINA_SPLIT_IMPLEMENTATION),
                str(scoring_path): _sha256(root / scoring_path),
                str(calibration_path): _sha256(root / calibration_path),
                str(ANNOTATIONS): _sha256(root / ANNOTATIONS),
                str(RAW_ADJACENCY): _sha256(root / RAW_ADJACENCY),
                str(BODY_IDS): _sha256(root / BODY_IDS),
            },
            "axis_definition": config["axis_definition"],
            "coordinate_convention": config["coordinate_convention"],
            "target_count": len(records),
            "parameter_fit": False,
            "sign_search": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "population_summaries": summaries,
        "population_mirror": mirrors,
        "within_eye_opponent_axis_pairs": opponent_axes,
        "independent_T4_axis_calibration": {
            "fit_source": calibration["protocol"]["fit_source"],
            "T5_used_for_fit_or_model_selection": calibration["protocol"][
                "T5_used_for_fit_or_model_selection"
            ],
            "held_out_T4_accuracy": calibration["held_out_T4"]["accuracy"],
            "held_out_T4_median_angle_error_degrees": calibration["held_out_T4"][
                "median_angle_error_degrees"
            ],
            "zero_shot_T5_accuracy": calibration["zero_shot_T5"]["accuracy"],
            "zero_shot_T5_median_angle_error_degrees": calibration["zero_shot_T5"][
                "median_angle_error_degrees"
            ],
            "cross_eye_maximum_error_degrees": calibration["cross_eye_mirror"][
                "maximum_degrees"
            ],
            "preregistered_gates": calibration["preregistered_gates"],
            "axis_calibration_passed": calibration["axis_calibration_pass"],
            "transform_application_authorized": False,
        },
        "population_gate_passed": population_gate,
        "mirror_gate_passed": mirror_gate,
        "strict_source_axis_gate_passed": strict,
        "authorize_dynamic_axis_candidate": strict,
        "target_records": records,
        "boundary": config["boundary"],
    }
