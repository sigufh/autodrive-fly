from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-target-fit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_target_fit.py")


def authorize_optimizer_conditions(condition_roles: list[str]) -> bool:
    """Fail closed unless an optimizer sees exactly the three tuning bundles."""
    return len(condition_roles) == 3 and all(role == "tuning" for role in condition_roles)


def simulate_t5_target_head(
    fast: np.ndarray,
    tm9: np.ndarray,
    ct1: np.ndarray,
    *,
    fast_weight: np.ndarray,
    delayed_weight: np.ndarray,
    leak: np.ndarray,
    tm9_effect: float,
    ct1_effect: float,
) -> tuple[np.ndarray, dict]:
    """Replay a target head while retaining Tm9 and CT1 as separate components."""
    fast = np.asarray(fast, dtype=np.float64)
    tm9 = np.asarray(tm9, dtype=np.float64)
    ct1 = np.asarray(ct1, dtype=np.float64)
    if fast.shape != tm9.shape or fast.shape != ct1.shape or fast.ndim != 3:
        raise ValueError("features must share condition-by-target-by-time shape")
    target_count = fast.shape[1]
    wf = np.asarray(fast_weight, dtype=np.float64)
    wd = np.asarray(delayed_weight, dtype=np.float64)
    alpha = np.asarray(leak, dtype=np.float64)
    if any(value.shape != (target_count,) for value in (wf, wd, alpha)):
        raise ValueError("each parameter must have one value per target")
    if np.any(wf < 0) or np.any(wd < 0) or np.any((alpha <= 0) | (alpha > 1)):
        raise ValueError("weights must be nonnegative and leak must be in (0, 1]")
    tm9_component = tm9 * float(tm9_effect)
    ct1_component = ct1 * float(ct1_effect)
    drive = wf[None, :, None] * fast + wd[None, :, None] * (tm9_component + ct1_component)
    output = np.empty_like(drive)
    state = np.zeros(drive.shape[:2], dtype=np.float64)
    for time_index in range(drive.shape[2]):
        state = (1.0 - alpha[None, :]) * state + alpha[None, :] * np.tanh(drive[:, :, time_index])
        output[:, :, time_index] = state
    return output, {
        "tm9_component": tm9_component,
        "ct1_component": ct1_component,
        "components_merged_before_source_specific_transform": False,
    }


def target_grid_search(
    fast: np.ndarray,
    tm9: np.ndarray,
    ct1: np.ndarray,
    desired: np.ndarray,
    *,
    grid: dict[str, list[float]],
    tm9_effect: float,
    ct1_effect: float,
    regularization_strength: float = 0.01,
    reference: tuple[float, float, float] = (1.0, 1.0, 0.5),
) -> dict:
    """Select one deterministic (fast, delayed, leak) tuple for every target."""
    desired = np.asarray(desired, dtype=np.float64)
    if desired.shape != np.asarray(fast).shape:
        raise ValueError("desired traces must match feature shape")
    candidates = list(itertools.product(grid["fast_weight"], grid["delayed_weight"], grid["leak"]))
    target_count = desired.shape[1]
    best_key: list[tuple[float, float, tuple[float, float, float]] | None] = [None] * target_count
    best = np.zeros((target_count, 3), dtype=np.float64)
    for candidate in candidates:
        wf, wd, alpha = (float(value) for value in candidate)
        predicted, _ = simulate_t5_target_head(
            fast,
            tm9,
            ct1,
            fast_weight=np.full(target_count, wf),
            delayed_weight=np.full(target_count, wd),
            leak=np.full(target_count, alpha),
            tm9_effect=tm9_effect,
            ct1_effect=ct1_effect,
        )
        residual = predicted - desired
        absolute = np.abs(residual)
        huber = np.where(absolute <= 0.1, 0.5 * residual**2, 0.1 * (absolute - 0.05))
        per_target = np.median(np.mean(huber, axis=2), axis=0)
        distance = (
            (wf - reference[0]) ** 2 + (wd - reference[1]) ** 2 + 4 * (alpha - reference[2]) ** 2
        )
        loss = per_target + regularization_strength * distance
        for target_index in range(target_count):
            key = (float(loss[target_index]), float(distance), (wf, wd, alpha))
            if best_key[target_index] is None or key < best_key[target_index]:
                best_key[target_index] = key
                best[target_index] = candidate
    return {
        "fast_weight": best[:, 0],
        "delayed_weight": best[:, 1],
        "leak": best[:, 2],
        "loss": np.asarray([item[0] for item in best_key if item is not None]),
        "candidate_count": len(candidates),
        "target_count": target_count,
    }


def bootstrap_condition_consistency(
    effects: np.ndarray,
    *,
    replicates: int,
    seed: int,
    minimum_effect: float,
    minimum_conditions: int,
    minimum_probability: float,
) -> dict:
    """Paired cluster bootstrap; each sampled target carries every condition."""
    values = np.asarray(effects, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("effects must be nonempty target-by-condition data")
    if not np.all(np.isfinite(values)):
        raise ValueError("effects must be finite; missing clusters cannot be silently dropped")
    if minimum_conditions > values.shape[1]:
        raise ValueError("minimum condition count exceeds available conditions")
    point = np.median(values, axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, values.shape[1]), dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, values.shape[0], values.shape[0])
        boot[index] = np.median(values[sampled], axis=0)
    pass_counts = np.sum(boot >= minimum_effect, axis=1)
    probability = float(np.mean(pass_counts >= minimum_conditions))
    overall = np.median(boot, axis=1)
    lower, upper = np.percentile(overall, [2.5, 97.5])
    gates = {
        "point_condition_consistency": int(np.sum(point >= minimum_effect)) >= minimum_conditions,
        "bootstrap_condition_consistency": probability >= minimum_probability,
        "overall_lower_confidence_bound": float(lower) >= minimum_effect,
    }
    return {
        "independent_cluster_count": int(values.shape[0]),
        "condition_count": int(values.shape[1]),
        "bootstrap_replicates": replicates,
        "point_condition_medians": point.tolist(),
        "point_passing_condition_count": int(np.sum(point >= minimum_effect)),
        "probability_at_least_minimum_conditions": probability,
        "overall_median_bootstrap_95_ci": [float(lower), float(upper)],
        "gates": gates,
        "passed": all(gates.values()),
    }


def score_lplc1(
    near_collision: np.ndarray,
    matched_miss: np.ndarray,
    back_to_front: np.ndarray,
    front_to_back: np.ndarray,
) -> dict:
    """Keep LPLC1 trajectory and object-direction contrasts separate."""
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (near_collision, matched_miss, back_to_front, front_to_back)
    ]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("LPLC1 paired observations must have equal shape")
    return {
        "mechanism": "near_collision_and_relative_background_motion",
        "near_collision_minus_matched_miss": arrays[0] - arrays[1],
        "back_to_front_minus_front_to_back": arrays[2] - arrays[3],
        "shared_looming_comparator_used": False,
    }


def score_lplc2(
    outward: np.ndarray, inward: np.ndarray, motion_free: np.ndarray, translation: np.ndarray
) -> dict:
    """Return each LPLC2 radial-opponency control instead of collapsing them."""
    values = [
        np.asarray(value, dtype=np.float64) for value in (outward, inward, motion_free, translation)
    ]
    if any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("LPLC2 paired observations must have equal shape")
    return {
        "mechanism": "local_radial_outward_motion",
        "outward_minus_inward": values[0] - values[1],
        "outward_minus_motion_free_darkening": values[0] - values[2],
        "outward_minus_wide_field_translation": values[0] - values[3],
        "controls_collapsed_with_max": False,
    }


def score_lc4(angular_velocity: np.ndarray, response: np.ndarray, angular_size: np.ndarray) -> dict:
    """Fit LC4 velocity response only within a fixed angular-size probe stratum."""
    velocity = np.asarray(angular_velocity, dtype=np.float64)
    activity = np.asarray(response, dtype=np.float64)
    size = np.asarray(angular_size, dtype=np.float64)
    if velocity.ndim != 1 or activity.shape != velocity.shape or size.shape != velocity.shape:
        raise ValueError("LC4 velocity, response and size must be equal one-dimensional arrays")
    if velocity.size < 3 or not np.allclose(size, size[0]):
        raise ValueError("LC4 score requires at least three fixed-size velocity probes")
    denominator = float(velocity @ velocity)
    if denominator <= 0:
        raise ValueError("LC4 angular velocities must have nonzero energy")
    slope = float((velocity @ activity) / denominator)
    fitted = slope * velocity
    residual = float(np.sum((activity - fitted) ** 2))
    total = float(np.sum((activity - np.mean(activity)) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else None
    return {
        "mechanism": "angular_velocity_at_fixed_angular_size",
        "through_origin_slope": slope,
        "r_squared": r_squared,
        "angular_size": float(size[0]),
        "terminal_size_gate_used": False,
    }


def evaluate_v7_target_fit_contract(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_path = Path(config["nested_protocol"])
    structure_path = Path(config["structure_evidence"])
    mechanism_path = Path(config["mechanism_evidence"])
    nested = json.loads((root / nested_path).read_text())
    structure = json.loads((root / structure_path).read_text())
    mechanism = json.loads((root / mechanism_path).read_text())
    if nested["final_authorized"] or nested["calibration_authorized"]:
        raise ValueError("nested holdouts unexpectedly authorized")
    if structure["structural_findings"]["T5_target_count"] != config["t5"]["declared_target_count"]:
        raise ValueError("T5 target denominator drifted")
    if any(
        item["ready_for_typed_development_fit"] for item in mechanism["target_contracts"].values()
    ):
        raise ValueError("looming mechanism audit unexpectedly authorizes fitting")
    grid = config["t5"]["grid"]
    candidate_count = len(grid["fast_weight"]) * len(grid["delayed_weight"]) * len(grid["leak"])
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(nested_path): _sha256(root / nested_path),
                str(structure_path): _sha256(root / structure_path),
                str(mechanism_path): _sha256(root / mechanism_path),
            },
            "real_fit_performed": False,
            "runtime_modified": False,
        },
        "t5_contract": config["t5"],
        "candidate_count_per_target": candidate_count,
        "declared_parameter_count": 3 * config["t5"]["declared_target_count"],
        "looming_readouts": config["looming_readouts"],
        "statistics": config["statistics"],
        "execution": config["execution"],
        "boundary": config["boundary"],
        "calibration_authorized": False,
        "final_authorized": False,
        "advance_to_visual_gate": False,
        "advance_to_central_complex": False,
    }
