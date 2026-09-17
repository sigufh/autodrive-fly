"""Freeze parameter-matched baseline architectures without training them."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-parameter-matched-baselines.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_parameter_matched_baselines.py")


def parameter_count(kind: str, inputs: int, outputs: int, width: int | None = None) -> int:
    if kind == "linear":
        return inputs * outputs + outputs
    if width is None or width < 1:
        raise ValueError("nonlinear baseline requires a positive width")
    if kind == "one_hidden_layer_mlp":
        return inputs * width + width + width * outputs + outputs
    if kind == "single_layer_gru":
        recurrent = 3 * width * inputs + 3 * width * width + 6 * width
        output = width * outputs + outputs
        return recurrent + output
    raise ValueError(f"unknown baseline architecture: {kind}")


def _nearest_width(kind: str, inputs: int, outputs: int, budget: int, search: dict) -> dict:
    candidates = []
    for width in range(int(search["minimum_width"]), int(search["maximum_width"]) + 1):
        count = parameter_count(kind, inputs, outputs, width)
        candidates.append((abs(count - budget), width, count))
    _, width, count = min(candidates)
    return {
        "hidden_width": width,
        "trainable_parameter_count": count,
        "absolute_parameter_difference": abs(count - budget),
        "relative_parameter_difference": abs(count - budget) / budget,
    }


def evaluate_v7_parameter_matched_baselines(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    reference_config_path = Path(config["reference_config"])
    reference_config = yaml.safe_load((root / reference_config_path).read_text())
    reference_path = Path(config["reference_evidence"])
    reference = json.loads((root / reference_path).read_text())
    split_path = Path(config["navigation_split"])
    split = yaml.safe_load((root / split_path).read_text())
    feature_dimension = int(reference["model"]["feature_dimension_per_parity"])
    output_dimension = len(reference["model"]["output_names"])
    coefficient_lengths = [len(values) for values in reference["model"]["coefficients"]]
    if coefficient_lengths != [feature_dimension + 1] * output_dimension:
        raise ValueError("reference linear model parameter shape changed")
    budget = sum(coefficient_lengths)
    architectures = {
        "linear": {
            "hidden_width": None,
            "trainable_parameter_count": parameter_count(
                "linear", feature_dimension, output_dimension
            ),
            "absolute_parameter_difference": 0,
            "relative_parameter_difference": 0.0,
            "parameter_tensors": {
                "input_to_output_weight": feature_dimension * output_dimension,
                "output_bias": output_dimension,
            },
        },
        "one_hidden_layer_mlp": _nearest_width(
            "one_hidden_layer_mlp",
            feature_dimension,
            output_dimension,
            budget,
            config["search"],
        ),
        "single_layer_gru": _nearest_width(
            "single_layer_gru",
            feature_dimension,
            output_dimension,
            budget,
            config["search"],
        ),
    }
    mlp_width = architectures["one_hidden_layer_mlp"]["hidden_width"]
    architectures["one_hidden_layer_mlp"]["parameter_tensors"] = {
        "input_to_hidden_weight": feature_dimension * mlp_width,
        "hidden_bias": mlp_width,
        "hidden_to_output_weight": mlp_width * output_dimension,
        "output_bias": output_dimension,
    }
    gru_width = architectures["single_layer_gru"]["hidden_width"]
    architectures["single_layer_gru"]["parameter_tensors"] = {
        "input_hidden_weights_three_gates": 3 * gru_width * feature_dimension,
        "hidden_hidden_weights_three_gates": 3 * gru_width * gru_width,
        "input_hidden_biases_three_gates": 3 * gru_width,
        "hidden_hidden_biases_three_gates": 3 * gru_width,
        "hidden_to_output_weight": gru_width * output_dimension,
        "output_bias": output_dimension,
    }
    count_gate = all(
        result["relative_parameter_difference"]
        <= float(config["gate"]["maximum_relative_parameter_difference"])
        for result in architectures.values()
    )
    conditions = split["conditions"]
    roles = {
        role: [item["condition_id"] for item in conditions if item["role"] == role]
        for role in ("tuning", "calibration", "external_final")
    }
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(reference_config_path): _sha256(root / reference_config_path),
                str(reference_path): _sha256(root / reference_path),
                str(split_path): _sha256(root / split_path),
            },
            "input_dimension": feature_dimension,
            "output_dimension": output_dimension,
            "reference_trainable_parameter_budget": budget,
            "preprocessing_parameter_counted": False,
            "preprocessing_shared_identically": True,
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "architectures": architectures,
        "parameter_count_gate_passed": bool(count_gate),
        "data_contract": {
            "input_features": (
                "same frozen 450-dimensional parity-aware structured neural features"
            ),
            "outputs": reference_config["readout"]["outputs"],
            "tuning_conditions": roles["tuning"],
            "cross_validation": "leave_one_tuning_mirror_pair_out",
            "calibration_conditions": roles["calibration"],
            "external_final_conditions": roles["external_final"],
            "calibration_access_authorized": False,
            "external_final_access_authorized": False,
        },
        "training_contract": {
            **config["training_budget"],
            "selection_metric": (
                "tuning_cross_validation_task_collision_exit_window_steering_drift_mirror"
            ),
            "models_trained": False,
            "reason_not_run": "controlled visual response gate has not passed",
        },
        "protocol_ready": bool(count_gate),
        "evaluation_performed": False,
        "advance_to_calibration": False,
        "advance_to_final": False,
        "boundary": config["boundary"],
    }
