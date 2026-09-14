from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.driving.v7 import V7Contract
from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    SOURCE_TYPES,
    V7PublishedConductanceProbe,
    _collect_source_normalization,
)
from fly_emotion.driving.v7_stability import CHECKPOINTS

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_gain_audit.py")
FINITE_DIFFERENCE_STEPS = (1e-3, 1e-4, 1e-5)


def interface_output(probe: V7PublishedConductanceProbe, state: np.ndarray) -> np.ndarray:
    model = probe.conductance["model"]
    output = model["output_normalization"]
    voltage = probe._conductance_voltage(probe._normalized_source_state(state))
    scale = output["excitatory_reversal_millivolts"] - output["baseline_millivolts"]
    return np.clip((voltage - output["baseline_millivolts"]) / scale, -1.0, 1.0)


def interface_jacobian(probe: V7PublishedConductanceProbe, state: np.ndarray) -> tuple:
    model = probe.conductance["model"]
    normalization = probe.conductance["normalization"]
    span = np.maximum(normalization.high - normalization.low, 1e-6).astype(np.float64)
    raw_normalized = (state - normalization.low) / span
    interior = (raw_normalized > 0.0) & (raw_normalized < 1.0)
    normalized = probe._normalized_source_state(state)
    parameters = model["source_parameters"]
    matrices = probe.conductance["matrices"]
    signals = {name: matrix @ normalized for name, matrix in matrices.items()}
    total_g = np.full(len(probe.conductance["targets"]), model["leak_conductance"], dtype=float)
    for name, values in signals.items():
        total_g += parameters[name]["gain"] * np.maximum(values - parameters[name]["threshold"], 0)
    voltage = probe._conductance_voltage(normalized)
    output = model["output_normalization"]
    scale = output["excitatory_reversal_millivolts"] - output["baseline_millivolts"]
    raw_output = (voltage - output["baseline_millivolts"]) / scale
    output_interior = (raw_output > -1.0) & (raw_output < 1.0)
    jacobian = sparse.csr_matrix(next(iter(matrices.values())).shape, dtype=np.float64)
    contributions = {}
    for name, matrix in matrices.items():
        param = parameters[name]
        active = signals[name] > param["threshold"]
        reversal = model["reversal_potentials_millivolts"][param["reversal"]]
        coefficient = (
            output_interior * active * param["gain"] * (reversal - voltage) / (total_g * scale)
        )
        derivative = (
            matrix.multiply(coefficient[:, None]).multiply((interior / span)[None, :]).tocsr()
        )
        jacobian += derivative
        contributions[name] = np.asarray(abs(derivative).sum(axis=1)).ravel()
    jacobian.eliminate_zeros()
    return (
        jacobian,
        contributions,
        {
            "normalization_interior": interior,
            "raw_normalized": raw_normalized,
            "signals": signals,
            "output_interior": output_interior,
            "raw_output": raw_output,
        },
    )


def quantiles(values: np.ndarray) -> dict:
    return {str(q): float(np.quantile(values, q)) for q in (0.0, 0.5, 0.9, 0.99, 1.0)}


def audit_interface_snapshot(probe: V7PublishedConductanceProbe, state: np.ndarray) -> dict:
    jacobian, contributions, detail = interface_jacobian(probe, state)
    row_gain = np.asarray(abs(jacobian).sum(axis=1)).ravel()
    normalization = probe.conductance["normalization"]
    span = np.maximum(normalization.high - normalization.low, 1e-6)
    source_groups = {}
    sides = probe._node_labels(
        probe.root / "data/raw/malecns-v1.0/body-annotations.feather", "somaSide"
    )
    for kind in SOURCE_TYPES:
        for side in ("L", "R"):
            mask = (probe.node_types == kind) & (sides == side)
            source_groups[f"{kind}_{side}"] = {
                "cells": int(mask.sum()),
                "calibration_span_quantiles": quantiles(span[mask]),
                "unclipped_normalization_gain_quantiles": quantiles(1.0 / span[mask]),
                "normalization_interior_fraction": float(
                    np.mean(detail["normalization_interior"][mask])
                ),
                "at_or_below_lower_clip_fraction": float(
                    np.mean(detail["raw_normalized"][mask] <= 0)
                ),
                "at_or_above_upper_clip_fraction": float(
                    np.mean(detail["raw_normalized"][mask] >= 1)
                ),
            }
    targets = probe.conductance["targets"]
    target_groups = {}
    for name, nodes in probe.populations.items():
        if not name.startswith("T4"):
            continue
        mask = np.isin(targets, nodes)
        target_groups[name] = {
            "cells": int(mask.sum()),
            "row_l1_gain_quantiles": quantiles(row_gain[mask]),
            "fraction_row_gain_above_one": float(np.mean(row_gain[mask] > 1)),
            "leak_scaled_row_l1_gain_quantiles": quantiles(
                row_gain[mask] * probe.leak[targets[mask]]
            ),
            "branch_row_l1_gain_medians": {
                kind: float(np.median(values[mask])) for kind, values in contributions.items()
            },
            "output_clip_interior_fraction": float(np.mean(detail["output_interior"][mask])),
        }
    rng = np.random.default_rng(20260915)
    direction = np.zeros(len(state), dtype=np.float64)
    active_sources = normalization.source_mask & detail["normalization_interior"]
    direction[active_sources] = (
        rng.choice((-1.0, 1.0), int(active_sources.sum())) * span[active_sources]
    )
    predicted = jacobian @ direction
    finite_difference = []
    for epsilon in FINITE_DIFFERENCE_STEPS:
        plus = state.astype(np.float64) + epsilon * direction
        minus = state.astype(np.float64) - epsilon * direction
        measured = (interface_output(probe, plus) - interface_output(probe, minus)) / (2 * epsilon)
        valid_sources = ~active_sources | (
            (minus > normalization.low)
            & (minus < normalization.high)
            & (plus > normalization.low)
            & (plus < normalization.high)
        )
        crossed_sources = ~valid_sources
        smooth = np.ones(len(targets), dtype=bool)
        for kind, matrix in probe.conductance["matrices"].items():
            affected = np.asarray(matrix @ crossed_sources.astype(float)).ravel() > 0
            smooth &= ~affected
            plus_signal = matrix @ probe._normalized_source_state(plus)
            minus_signal = matrix @ probe._normalized_source_state(minus)
            threshold = probe.conductance["model"]["source_parameters"][kind]["threshold"]
            smooth &= (plus_signal > threshold) == (minus_signal > threshold)
            smooth &= detail["signals"][kind] != threshold
        for perturbed in (plus, minus):
            voltage = probe._conductance_voltage(probe._normalized_source_state(perturbed))
            output = probe.conductance["model"]["output_normalization"]
            value = (voltage - output["baseline_millivolts"]) / (
                output["excitatory_reversal_millivolts"] - output["baseline_millivolts"]
            )
            smooth &= ((value > -1) & (value < 1)) == detail["output_interior"]
        smooth &= (detail["raw_output"] != -1) & (detail["raw_output"] != 1)
        error = np.abs(predicted - measured)
        finite_difference.append(
            {
                "epsilon_in_calibration_span_units": epsilon,
                "smooth_targets": int(smooth.sum()),
                "excluded_crossing_or_kink_targets": int((~smooth).sum()),
                "maximum_absolute_error_smooth": float(error[smooth].max())
                if smooth.any()
                else None,
                "median_absolute_error_smooth": float(np.median(error[smooth]))
                if smooth.any()
                else None,
                "maximum_absolute_prediction_smooth": float(np.abs(predicted[smooth]).max())
                if smooth.any()
                else None,
            }
        )
    top = np.lexsort((probe.graph.body_ids[targets], -row_gain))[:10]
    return {
        "interface_scope": "upstream checkpoint to T4 target drive; not a full timestep",
        "source_populations": source_groups,
        "target_populations": target_groups,
        "row_l1_gain_quantiles": quantiles(row_gain),
        "finite_difference": finite_difference,
        "top_targets": [
            {
                "body_id": int(probe.graph.body_ids[targets[index]]),
                "row_l1_gain": float(row_gain[index]),
                "by_source_type": {
                    kind: float(values[index]) for kind, values in contributions.items()
                },
            }
            for index in top
        ],
    }


def evaluate_v7_normalization_gain(root: Path) -> dict:
    contract = V7Contract.load(root)
    config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text())
    reference_path = Path("artifacts/v7-background-stability.json")
    reference = json.loads((root / reference_path).read_text())
    hashes = dict(reference["protocol"]["dependencies_sha256"])
    for relative, expected in hashes.items():
        with (root / relative).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"stale background dependency: {relative}")
    results = {}
    for backend in ("linear_luminance", "signed_frame_difference"):
        normalization = _collect_source_normalization(
            root,
            retinal_backend=backend,
            retinal_geometry=config["primary_retinal_geometry"],
            config=config,
        )
        if normalization.sha256 != reference["results"][backend]["normalization_sha256"]:
            raise ValueError("normalization does not match background baseline")
        probe = V7PublishedConductanceProbe(
            root, retinal_backend=backend, normalization=normalization
        )
        state = np.zeros(probe.graph.node_count, dtype=np.float32)
        history = [state.copy()]
        sampled = probe._sample_retina(np.full((24, 48), 0.5, dtype=np.float32))
        encoded = probe._retinal_code(sampled, sampled, sampled)
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = encoded
        snapshots = {}
        for step in range(1, CHECKPOINTS[-1] + 1):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = encoded
            if step in (128, 2048):
                state_hash = hashlib.sha256(state.tobytes()).hexdigest()
                expected_hash = reference["results"][backend]["models"][
                    "V7PublishedConductanceProbe"
                ]["checkpoints"][str(step)]["state_sha256"]
                if state_hash != expected_hash:
                    raise ValueError("fixed-state replay differs from stability audit")
                snapshots[str(step)] = {
                    "state_sha256": state_hash,
                    **audit_interface_snapshot(probe, state),
                }
        results[backend] = {"normalization_sha256": normalization.sha256, "snapshots": snapshots}
    for path in (IMPLEMENTATION, reference_path):
        with (root / path).open("rb") as source:
            hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-normalization-local-gain-v1",
            "advance_allowed": False,
            "dependencies_sha256": hashes,
            "v7_config_sha256": contract.sha256,
            "parameter_fitting": False,
            "driving_data_used": False,
            "linearization": "real-arithmetic derivative of normalize/threshold/conductance/output",
            "float32": "runtime normalize/aggregate uses float32; numerical errors recorded",
            "bound": "row absolute derivative sum for independently bounded upstream perturbations",
            "scope": "not the closed-loop Jacobian, spectral radius, or proof of instability",
            "kinks": "zero slope at kinks; crossing targets excluded only in derivative checks",
            "state_protocol": "gray 0.5; zero initial state; checkpoints 128 and 2048 microsteps",
        },
        "results": results,
        "limitations": [
            "Large local gain alone cannot establish a causal source of feedback instability.",
            "Finite differences use one fixed random direction, not every possible perturbation.",
            "Calibration ranges were fitted without direction labels but are proxy state units.",
            "No normalizer, model parameter, topology, visual gate or deployed policy changed.",
        ],
        "advance_to_central_complex": False,
    }
