"""Strictly split fitting of shared conductances to T4 direction/ON phenotypes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

from fly_emotion.driving.v7 import (
    HORIZONTAL_PREFERENCE,
    V7_IMPLEMENTATION,
    VERTICAL_PREFERENCE,
    V7Contract,
    V7VisualProbe,
    VisualStimulus,
    _moving_edge,
)
from fly_emotion.driving.v7_branched import (
    BRANCHED_IMPLEMENTATION,
    SOURCE_AUDIT,
    V7BranchedT4Probe,
)
from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    CONDUCTANCE_IMPLEMENTATION,
    SOURCE_TYPES,
    SourceNormalization,
    _collect_source_normalization,
)

FIT_CONFIG = Path("configs/driving-v7-t4-fit.yaml")
FIT_IMPLEMENTATION = Path("src/fly_emotion/driving/v7_fit.py")
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class T4FeatureBattery:
    body_ids: np.ndarray
    populations: np.ndarray
    expected_directions: np.ndarray
    conditions: dict[str, dict[str, np.ndarray]]
    sha256: str


def _scaled_edge(
    *, width: int, height: int, frames: int, direction: str, polarity: str, contrast: float
) -> np.ndarray:
    axis = "x" if direction in {"left", "right"} else "y"
    sign = 1 if direction in {"right", "down"} else -1
    base = _moving_edge(
        width=width,
        height=height,
        frames=frames,
        axis=axis,
        direction=sign,
        bright=polarity == "on",
    )
    return np.clip(0.5 + (base - 0.5) * contrast / 0.84, 0.0, 1.0).astype(np.float32)


def build_t4_fit_stimuli(config: dict, split: str) -> list[VisualStimulus]:
    settings = config["stimulus_split"]
    output = []
    for index, condition in enumerate(settings[split]):
        for polarity in ("on", "off"):
            for direction in ("left", "right", "up", "down"):
                name = f"{split}_{index}_{polarity}_{direction}"
                output.append(
                    VisualStimulus(
                        name,
                        "fit_edge",
                        polarity,
                        direction,
                        _scaled_edge(
                            width=int(settings["width"]),
                            height=int(settings["height"]),
                            frames=int(condition["frames"]),
                            direction=direction,
                            polarity=polarity,
                            contrast=float(condition["contrast"]),
                        ),
                    )
                )
    return output


def _cell_split(
    body_ids: np.ndarray, populations: np.ndarray, config: dict
) -> dict[str, np.ndarray]:
    split = config["cell_split"]
    assignments = np.full(len(body_ids), "", dtype=object)
    for population in sorted(set(populations)):
        group = np.flatnonzero(populations == population)
        hashes = np.asarray(
            [
                hashlib.sha256(f"{split['seed']}:{int(body_ids[index])}".encode()).digest()
                for index in group
            ],
            dtype="S32",
        )
        ordered = group[np.argsort(hashes, kind="stable")]
        train_end = int(np.floor(len(group) * float(split["train_fraction"])))
        validation_end = train_end + int(np.floor(len(group) * float(split["validation_fraction"])))
        assignments[ordered[:train_end]] = "train"
        assignments[ordered[train_end:validation_end]] = "validation"
        assignments[ordered[validation_end:]] = "test"
    return {name: assignments == name for name in ("train", "validation", "test")}


def _t4_targets(probe: V7BranchedT4Probe) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = probe.branched["targets"]
    body_ids = probe.graph.body_ids[targets]
    node_populations = np.full(probe.graph.node_count, "", dtype=object)
    for name, nodes in probe.populations.items():
        if name.startswith("T4"):
            node_populations[nodes] = name
    populations = node_populations[targets].astype(str)
    if np.any(populations == ""):
        raise ValueError("every fitted T4 target must have a population label")
    # Avoid repeated linear membership scans above in future callers by asserting uniqueness.
    if len(np.unique(body_ids)) != len(body_ids):
        raise ValueError("T4 target body IDs must be unique")
    expected = np.asarray(
        [
            HORIZONTAL_PREFERENCE[(population[2], population[-1])]
            if population[2] in {"a", "b"}
            else VERTICAL_PREFERENCE[population[2]]
            for population in populations
        ],
        dtype=str,
    )
    return targets, body_ids, populations, expected


def _source_matrices(probe: V7BranchedT4Probe, targets: np.ndarray) -> dict:
    return {
        source_type: probe._normalized_target_inputs(targets, (source_type,))
        for source_type in SOURCE_TYPES
    }


def _extract_condition(
    probe: V7BranchedT4Probe,
    stimulus: VisualStimulus,
    normalization: SourceNormalization,
    matrices: dict,
) -> dict[str, np.ndarray]:
    probe.correlator = None
    probe.dynamics_backend = "typed_visual_subgraph_v1"
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    baseline = probe._sample_retina(stimulus.frames[0])[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(baseline, baseline, baseline)
    for _ in range(probe.baseline_frames):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        for _ in range(probe.brain_substeps):
            state = V7VisualProbe._advance(probe, state, drive, history)
            state[probe.retina.node_indices] = baseline_drive
    previous = baseline.copy()
    traces = {source_type: [] for source_type in SOURCE_TYPES}
    for image in stimulus.frames:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline, previous)
        previous = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = V7VisualProbe._advance(probe, state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        normalized = np.clip(
            (state - normalization.low) / np.maximum(normalization.high - normalization.low, 1e-6),
            0.0,
            1.0,
        )
        for source_type in SOURCE_TYPES:
            traces[source_type].append(matrices[source_type] @ normalized)
    return {
        source_type: np.asarray(values, dtype=np.float32) for source_type, values in traces.items()
    }


def extract_t4_feature_battery(root: Path, config: dict, split: str) -> T4FeatureBattery:
    retinal_backend = str(config["retinal_backend"])
    retinal_geometry = str(config["retinal_geometry"])
    normalization_config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text(encoding="utf-8"))
    normalization = _collect_source_normalization(
        root,
        retinal_backend=retinal_backend,
        retinal_geometry=retinal_geometry,
        config=normalization_config,
    )
    probe = V7BranchedT4Probe(
        root,
        retinal_backend=retinal_backend,
        retinal_geometry=retinal_geometry,
        brain_substeps=int(config["brain_substeps_per_frame"]),
        baseline_frames=int(config["baseline_frames"]),
    )
    targets, body_ids, populations, expected = _t4_targets(probe)
    matrices = _source_matrices(probe, targets)
    conditions = {}
    digest = hashlib.sha256()
    for stimulus in build_t4_fit_stimuli(config, split):
        traces = _extract_condition(probe, stimulus, normalization, matrices)
        conditions[stimulus.name] = traces
        digest.update(stimulus.sha256.encode())
        for source_type in SOURCE_TYPES:
            digest.update(traces[source_type].tobytes())
    return T4FeatureBattery(body_ids, populations, expected, conditions, digest.hexdigest())


def _unpack(parameters: np.ndarray) -> dict:
    return {
        "gains": dict(zip(SOURCE_TYPES, parameters[:5], strict=True)),
        "thresholds": dict(zip(SOURCE_TYPES, parameters[5:10], strict=True)),
        "leak_reversal": float(parameters[10]),
        "leak_conductance": float(parameters[11]),
    }


def _response_energy(features: dict[str, np.ndarray], parameters: np.ndarray) -> np.ndarray:
    parsed = _unpack(parameters)
    reversal = {"Mi9": -71.0, "Tm3": -21.0, "Mi1": -21.0, "Mi4": -68.0, "C3": -68.0}
    shape = next(iter(features.values())).shape
    numerator = np.full(shape, parsed["leak_reversal"] * parsed["leak_conductance"])
    denominator = np.full(shape, parsed["leak_conductance"])
    for source_type in SOURCE_TYPES:
        conductance = parsed["gains"][source_type] * np.maximum(
            features[source_type] - parsed["thresholds"][source_type], 0.0
        )
        numerator += reversal[source_type] * conductance
        denominator += conductance
    voltage = numerator / np.maximum(denominator, 1e-12)
    baseline = voltage[0]
    return np.mean(np.maximum(voltage - baseline, 0.0), axis=0)


def _battery_metrics(battery: T4FeatureBattery, mask: np.ndarray, parameters: np.ndarray) -> dict:
    energies = {
        name: _response_energy(features, parameters)
        for name, features in battery.conditions.items()
    }
    groups = sorted({name.rsplit("_", 2)[0] for name in energies})
    direction_contrasts = []
    polarity_contrasts = []
    correct = []
    polarity_correct = []
    by_population = {}
    for population in sorted(set(battery.populations[mask])):
        local = mask & (battery.populations == population)
        expected = str(battery.expected_directions[np.flatnonzero(local)[0]])
        opposite = OPPOSITE[expected]
        population_direction = []
        population_polarity = []
        for group in groups:
            preferred = energies[f"{group}_on_{expected}"][local]
            opposed = energies[f"{group}_on_{opposite}"][local]
            off = energies[f"{group}_off_{expected}"][local]
            direction = (preferred - opposed) / (np.abs(preferred) + np.abs(opposed) + 1e-9)
            polarity = (preferred - off) / (np.abs(preferred) + np.abs(off) + 1e-9)
            direction_contrasts.extend(direction.tolist())
            polarity_contrasts.extend(polarity.tolist())
            correct.extend((direction > 0).tolist())
            polarity_correct.extend((polarity > 0).tolist())
            population_direction.extend(direction.tolist())
            population_polarity.extend(polarity.tolist())
        by_population[population] = {
            "cells_x_conditions": len(population_direction),
            "direction_accuracy": float(np.mean(np.asarray(population_direction) > 0)),
            "median_direction_contrast": float(np.median(population_direction)),
            "on_off_accuracy": float(np.mean(np.asarray(population_polarity) > 0)),
            "median_on_off_contrast": float(np.median(population_polarity)),
        }
    return {
        "cells": int(np.count_nonzero(mask)),
        "direction_accuracy": float(np.mean(correct)),
        "median_direction_contrast": float(np.median(direction_contrasts)),
        "on_off_accuracy": float(np.mean(polarity_correct)),
        "median_on_off_contrast": float(np.median(polarity_contrasts)),
        "population_direction_fraction": float(
            np.mean([value["median_direction_contrast"] > 0 for value in by_population.values()])
        ),
        "by_population": by_population,
    }


def _fit_subset(
    body_ids: np.ndarray, populations: np.ndarray, train: np.ndarray, config: dict
) -> np.ndarray:
    selected = np.zeros(len(body_ids), dtype=bool)
    count = int(config["cell_split"]["fit_cells_per_population"])
    for population in sorted(set(populations[train])):
        group = np.flatnonzero(train & (populations == population))
        hashes = np.asarray(
            [hashlib.sha256(f"fit:{int(body_ids[index])}".encode()).digest() for index in group],
            dtype="S32",
        )
        selected[group[np.argsort(hashes, kind="stable")[:count]]] = True
    return selected


def fit_v7_t4_conductance(root: Path) -> dict:
    config_path = root / FIT_CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("deployment_enabled") or config.get("driving_data_allowed"):
        raise ValueError("T4 fit must remain isolated from deployment and driving data")
    train_battery = extract_t4_feature_battery(root, config, "train")
    splits = _cell_split(train_battery.body_ids, train_battery.populations, config)
    fit_mask = _fit_subset(
        train_battery.body_ids, train_battery.populations, splits["train"], config
    )
    published = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text(encoding="utf-8"))[
        "published_single_compartment_model"
    ]
    start = np.asarray(
        [published["source_parameters"][name]["gain"] for name in SOURCE_TYPES]
        + [published["source_parameters"][name]["threshold"] for name in SOURCE_TYPES]
        + [
            published["reversal_potentials_millivolts"]["leak"],
            published["leak_conductance"],
        ],
        dtype=float,
    )
    bounds = (
        [tuple(config["parameter_bounds"]["gains"])] * 5
        + [tuple(config["parameter_bounds"]["thresholds"])] * 5
        + [tuple(config["parameter_bounds"]["leak_reversal_millivolts"])]
        + [tuple(config["parameter_bounds"]["leak_conductance"])]
    )
    optimization = config["optimization"]

    def objective(parameters: np.ndarray) -> float:
        metrics = _battery_metrics(train_battery, fit_mask, parameters)
        temperature = float(optimization["margin_temperature"])
        direction_loss = 1.0 - 0.5 * (
            1.0 + np.tanh(temperature * metrics["median_direction_contrast"])
        )
        polarity_loss = 1.0 - 0.5 * (1.0 + np.tanh(temperature * metrics["median_on_off_contrast"]))
        widths = np.asarray([high - low for low, high in bounds])
        regularization = float(np.mean(((parameters - start) / widths) ** 2))
        return float(
            float(optimization["direction_loss_weight"]) * direction_loss
            + float(optimization["polarity_loss_weight"]) * polarity_loss
            + float(optimization["parameter_regularization"]) * regularization
        )

    result = minimize(
        objective,
        start,
        method=str(optimization["method"]),
        bounds=bounds,
        options={"maxiter": int(optimization["max_iterations"])},
    )
    fitted = np.asarray(result.x, dtype=float)
    train_metrics = _battery_metrics(train_battery, splits["train"], fitted)
    validation_battery = extract_t4_feature_battery(root, config, "validation")
    validation_metrics = _battery_metrics(validation_battery, splits["validation"], fitted)
    gates_config = config["validation_gates"]
    gates = {
        "direction_accuracy": validation_metrics["direction_accuracy"]
        >= gates_config["minimum_direction_accuracy"],
        "direction_contrast": validation_metrics["median_direction_contrast"]
        >= gates_config["minimum_median_direction_contrast"],
        "on_off_accuracy": validation_metrics["on_off_accuracy"]
        >= gates_config["minimum_on_off_accuracy"],
        "on_off_contrast": validation_metrics["median_on_off_contrast"]
        >= gates_config["minimum_median_on_off_contrast"],
        "population_direction_fraction": validation_metrics["population_direction_fraction"]
        >= gates_config["minimum_population_direction_fraction"],
    }
    validation_passed = all(gates.values())
    test_result = {
        "evaluated": False,
        "reason": "validation gates failed; one-time test remains sealed",
    }
    if validation_passed:
        test_battery = extract_t4_feature_battery(root, config, "test")
        test_result = {
            "evaluated": True,
            "feature_sha256": test_battery.sha256,
            "metrics": _battery_metrics(test_battery, splits["test"], fitted),
        }
    contract = V7Contract.load(root)
    return {
        "protocol": {
            "version": 7,
            "name": config["name"],
            "deployment_enabled": False,
            "driving_data_used": False,
            "config_sha256": _sha256(config_path),
            "implementation_sha256": _sha256(root / FIT_IMPLEMENTATION),
            "v7_config_sha256": contract.sha256,
            "v7_implementation_sha256": _sha256(root / V7_IMPLEMENTATION),
            "branched_implementation_sha256": _sha256(root / BRANCHED_IMPLEMENTATION),
            "conductance_implementation_sha256": _sha256(root / CONDUCTANCE_IMPLEMENTATION),
            "source_audit_sha256": _sha256(root / SOURCE_AUDIT),
            "direction_labels_used_for_training": True,
            "direction_labels_used_by_runtime_dynamics": False,
            "measured_voltage_data_used": False,
            "supervision": config["supervision"],
            "one_time_test_policy": config["test_policy"],
        },
        "split_counts": {name: int(np.count_nonzero(mask)) for name, mask in splits.items()},
        "split_body_id_overlap": {
            "train_validation": int(
                np.intersect1d(
                    train_battery.body_ids[splits["train"]],
                    train_battery.body_ids[splits["validation"]],
                ).size
            ),
            "train_test": int(
                np.intersect1d(
                    train_battery.body_ids[splits["train"]],
                    train_battery.body_ids[splits["test"]],
                ).size
            ),
        },
        "fit_cells": int(np.count_nonzero(fit_mask)),
        "train_feature_sha256": train_battery.sha256,
        "validation_feature_sha256": validation_battery.sha256,
        "published_start_parameters": _unpack(start),
        "fitted_parameters": _unpack(fitted),
        "optimization": {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "initial_objective": objective(start),
            "final_objective": objective(fitted),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_gates": gates,
        "validation_passed": validation_passed,
        "test": test_result,
        "original_visual_battery_evaluated": False,
        "advance_to_central_complex": False,
        "stop_reason": (
            "validation passed; one-time test evaluated but original battery remains gated"
            if validation_passed
            else "held-out cell and stimulus validation failed; test remains sealed"
        ),
    }
