from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import V7Contract
from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    V7PublishedConductanceProbe,
    _collect_source_normalization,
)
from fly_emotion.driving.v7_gain_audit import interface_jacobian

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_perturbation.py")
SNAPSHOTS = (128, 2048)
HORIZON = 32
SEEDS = (20260915, 20260916)
EPSILONS = (1e-3, 1e-4)


def full_update_jvp(
    probe: V7PublishedConductanceProbe,
    state: np.ndarray,
    drive: np.ndarray,
    vectors: np.ndarray,
    *,
    unit_normalization_slope: bool = False,
) -> np.ndarray:
    if vectors.ndim != 2 or vectors.shape[0] != len(state):
        raise ValueError("tangent vectors must have shape (nodes, directions)")
    activation = np.tanh(1.8 * (probe.adjacency @ (state * probe.source_sign)) + drive)
    upstream = ((1.0 - probe.leak) * state + probe.leak * activation).astype(np.float32)
    propagated = (1.0 - probe.leak[:, None]) * vectors + (
        1.8 * probe.leak * (1.0 - activation.astype(np.float64) ** 2)
    )[:, None] * (probe.adjacency @ (probe.source_sign[:, None] * vectors))
    interface, _, _ = interface_jacobian(probe, upstream)
    interface_vectors = propagated
    if unit_normalization_slope:
        norm = probe.conductance["normalization"]
        span = np.maximum(norm.high - norm.low, 1e-6)
        interface_vectors = propagated * span[:, None]
    targets = probe.conductance["targets"]
    target_variation = interface @ interface_vectors
    propagated[targets] = (1.0 - probe.leak[targets, None]) * vectors[targets] + (
        probe.leak[targets, None] * target_variation
    )
    propagated[probe.retina.node_indices] = 0.0
    return propagated


def advance_with_clamp(
    probe: V7PublishedConductanceProbe, state: np.ndarray, drive: np.ndarray
) -> np.ndarray:
    updated = probe._advance(state, drive, [state.copy()])
    updated[probe.retina.node_indices] = drive[probe.retina.node_indices]
    return updated


def perturbation_directions(probe: V7PublishedConductanceProbe) -> np.ndarray:
    norm = probe.conductance["normalization"]
    selected = norm.source_mask.copy()
    selected[probe.retina.node_indices] = False
    span = np.maximum(norm.high - norm.low, 1e-6)
    vectors = np.zeros((probe.graph.node_count, len(SEEDS)), dtype=np.float64)
    for index, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        vectors[selected, index] = rng.choice((-1.0, 1.0), int(selected.sum())) * span[selected]
    return vectors


def _norms(values: np.ndarray) -> dict:
    return {"l2": float(np.linalg.norm(values)), "linf": float(np.max(np.abs(values)))}


def replay_perturbations(
    probe: V7PublishedConductanceProbe, state: np.ndarray, drive: np.ndarray
) -> dict:
    directions = perturbation_directions(probe)
    initial = directions.copy()
    counterfactual = directions.copy()
    observed = {}
    for epsilon in EPSILONS:
        plus = [
            (state.astype(np.float64) + epsilon * directions[:, i]).astype(np.float32)
            for i in range(len(SEEDS))
        ]
        minus = [
            (state.astype(np.float64) - epsilon * directions[:, i]).astype(np.float32)
            for i in range(len(SEEDS))
        ]
        tangent = np.column_stack(
            [
                (p.astype(np.float64) - m.astype(np.float64)) / (2 * epsilon)
                for p, m in zip(plus, minus, strict=True)
            ]
        )
        observed[epsilon] = {
            "plus": plus,
            "minus": minus,
            "tangent": tangent,
            "initial": tangent.copy(),
            "trace": [],
        }
    linear_trace = []
    nominal = state.copy()
    targets = probe.conductance["targets"]
    for step in range(1, HORIZON + 1):
        directions = full_update_jvp(probe, nominal, drive, directions)
        counterfactual = full_update_jvp(
            probe, nominal, drive, counterfactual, unit_normalization_slope=True
        )
        linear_trace.append(
            {
                "step": step,
                "directions": [
                    {
                        "seed": seed,
                        "full_l2_gain": float(
                            np.linalg.norm(directions[:, index]) / np.linalg.norm(initial[:, index])
                        ),
                        "unit_normalization_slope_l2_gain": float(
                            np.linalg.norm(counterfactual[:, index])
                            / np.linalg.norm(initial[:, index])
                        ),
                        "full_linf": float(np.max(np.abs(directions[:, index]))),
                    }
                    for index, seed in enumerate(SEEDS)
                ],
            }
        )
        for epsilon, record in observed.items():
            record["tangent"] = full_update_jvp(probe, nominal, drive, record["tangent"])
            comparisons = []
            for index, seed in enumerate(SEEDS):
                record["plus"][index] = advance_with_clamp(probe, record["plus"][index], drive)
                record["minus"][index] = advance_with_clamp(probe, record["minus"][index], drive)
                separation = record["plus"][index].astype(np.float64) - record["minus"][
                    index
                ].astype(np.float64)
                measured = separation / (2 * epsilon)
                predicted = record["tangent"][:, index]
                denominator = np.linalg.norm(record["initial"][:, index])
                comparisons.append(
                    {
                        "seed": seed,
                        "nonlinear_l2_gain": float(np.linalg.norm(measured) / denominator),
                        "tangent_l2_gain": float(np.linalg.norm(predicted) / denominator),
                        "derivative_error_l2": float(np.linalg.norm(measured - predicted)),
                        "relative_derivative_error": float(
                            np.linalg.norm(measured - predicted)
                            / max(np.linalg.norm(predicted), 1e-30)
                        ),
                        "actual_pair_separation": _norms(separation),
                        "T4_pair_separation_l2": float(np.linalg.norm(separation[targets])),
                        "retinal_pair_separation_linf": float(
                            np.max(np.abs(separation[probe.retina.node_indices]))
                        ),
                    }
                )
            record["trace"].append({"step": step, "directions": comparisons})
        nominal = advance_with_clamp(probe, nominal, drive)
    return {
        "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "nominal_final_state_sha256": hashlib.sha256(nominal.tobytes()).hexdigest(),
        "direction_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "initial_direction_norms": [_norms(initial[:, i]) for i in range(len(SEEDS))],
        "linear_propagation": linear_trace,
        "nonlinear_replays": {
            str(epsilon): {
                "realized_initial_half_difference_norms": [
                    _norms(record["initial"][:, i]) for i in range(len(SEEDS))
                ],
                "initial_pair_separation_norms": [
                    _norms(2 * epsilon * record["initial"][:, i]) for i in range(len(SEEDS))
                ],
                "initial_direction_rounding_error_l2": [
                    float(np.linalg.norm(record["initial"][:, i] - initial[:, i]))
                    for i in range(len(SEEDS))
                ],
                "trace": record["trace"],
            }
            for epsilon, record in observed.items()
        },
    }


def evaluate_v7_perturbation(root: Path) -> dict:
    V7Contract.load(root)
    config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text())
    reference_path = Path("artifacts/v7-normalization-gain.json")
    reference = json.loads((root / reference_path).read_text())
    hashes = dict(reference["protocol"]["dependencies_sha256"])
    for path, expected in hashes.items():
        with (root / path).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"stale gain audit dependency: {path}")
    results = {}
    for backend in ("linear_luminance", "signed_frame_difference"):
        normalization = _collect_source_normalization(
            root,
            retinal_backend=backend,
            retinal_geometry=config["primary_retinal_geometry"],
            config=config,
        )
        if normalization.sha256 != reference["results"][backend]["normalization_sha256"]:
            raise ValueError("normalization differs from frozen audit")
        probe = V7PublishedConductanceProbe(
            root, retinal_backend=backend, normalization=normalization
        )
        sampled = probe._sample_retina(np.full((24, 48), 0.5, dtype=np.float32))
        drive = np.zeros(probe.graph.node_count, dtype=np.float32)
        drive[probe.retina.node_indices] = probe._retinal_code(sampled, sampled, sampled)
        state = np.zeros_like(drive)
        snapshots = {}
        for step in range(1, max(SNAPSHOTS) + 1):
            state = advance_with_clamp(probe, state, drive)
            if step in SNAPSHOTS:
                expected = reference["results"][backend]["snapshots"][str(step)]["state_sha256"]
                if hashlib.sha256(state.tobytes()).hexdigest() != expected:
                    raise ValueError("nominal trajectory differs from frozen snapshot")
                snapshots[str(step)] = replay_perturbations(probe, state, drive)
        results[backend] = {"normalization_sha256": normalization.sha256, "snapshots": snapshots}
    for path in (IMPLEMENTATION, reference_path):
        with (root / path).open("rb") as source:
            hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-full-update-perturbation-v1",
            "advance_allowed": False,
            "dependencies_sha256": hashes,
            "snapshots": list(SNAPSHOTS),
            "horizon_substeps": HORIZON,
            "seeds": list(SEEDS),
            "epsilon_relative_to_source_span": list(EPSILONS),
            "perturbation": "five source types, fixed random signs; retinal clamp unperturbed",
            "linearization": "time-varying mixed-update Jacobian along the unperturbed trajectory",
            "counterfactual": "unit slope in tangent interface only; nominal state unchanged",
            "rounding": "finite-replay tangent initialized from realized float32 half-difference",
            "parameter_fitting": False,
            "driving_data_used": False,
        },
        "results": results,
        "limitations": [
            "Finite-time amplification for two directions is not an asymptotic Lyapunov exponent.",
            "Clipping and threshold crossings can invalidate local linear predictions.",
            "Float32 roundoff can dominate very small trajectory differences.",
            "Unit-slope tangent is an algebraic diagnostic, not a deployable counterfactual model.",
            "No model, normalizer, published visual score, or default policy was modified.",
        ],
        "advance_to_central_complex": False,
    }
