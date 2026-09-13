from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np

from fly_emotion.driving.engine import DrivingEngine


def run_episode(
    engine: DrivingEngine,
    seed: int,
    *,
    learning: bool,
    explore: bool,
    safety_constraints: bool = True,
) -> dict:
    engine.reset(seed, keep_learning=True)
    while not engine.env.done:
        engine.step(
            learning=learning,
            explore=explore,
            safety_constraints=safety_constraints,
            include_activity=False,
        )
    return {
        "seed": seed,
        "distance": engine.env.y,
        "return": engine.env.total_reward,
        "steps": engine.env.steps,
        "success": engine.env.y >= engine.env.road_length,
        "terminal_reason": engine.env.terminal_reason,
        **engine.control_summary(),
    }


def summarize(episodes: list[dict]) -> dict:
    summary = {
        "episodes": len(episodes),
        "mean_distance": mean(item["distance"] for item in episodes),
        "mean_return": mean(item["return"] for item in episodes),
        "success_rate": mean(float(item["success"]) for item in episodes),
        "details": episodes,
    }
    for key in (
        "mean_abs_steering",
        "mean_abs_steering_change",
        "far_mean_abs_steering",
        "steering_sign_changes",
        "max_abs_lateral",
        "constraint_rate",
        "mean_abs_constraint",
    ):
        summary[key] = mean(item[key] for item in episodes)
    summary["road_exit_rate"] = mean(
        float(item["terminal_reason"] == "road_boundary") for item in episodes
    )
    summary["obstacle_collision_rate"] = mean(
        float(item["terminal_reason"] == "obstacle") for item in episodes
    )
    return summary


def paired_statistics(frozen: list[dict], learned: list[dict], seed: int) -> dict:
    frozen_distance = np.asarray([item["distance"] for item in frozen])
    learned_distance = np.asarray([item["distance"] for item in learned])
    frozen_return = np.asarray([item["return"] for item in frozen])
    learned_return = np.asarray([item["return"] for item in learned])
    distance_delta = learned_distance - frozen_distance
    return_delta = learned_return - frozen_return
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(frozen), size=(50_000, len(frozen)))

    def interval(values: np.ndarray) -> list[float]:
        means = values[samples].mean(axis=1)
        return np.quantile(means, [0.025, 0.975]).tolist()

    return {
        "distance": {
            "frozen_mean": float(frozen_distance.mean()),
            "learned_mean": float(learned_distance.mean()),
            "delta_mean": float(distance_delta.mean()),
            "frozen_median": float(np.median(frozen_distance)),
            "learned_median": float(np.median(learned_distance)),
            "paired_wins": int(np.count_nonzero(distance_delta > 0)),
            "paired_losses": int(np.count_nonzero(distance_delta < 0)),
            "bootstrap_95_ci": interval(distance_delta),
        },
        "return": {
            "frozen_mean": float(frozen_return.mean()),
            "learned_mean": float(learned_return.mean()),
            "delta_mean": float(return_delta.mean()),
            "bootstrap_95_ci": interval(return_delta),
        },
    }


def evaluate(
    root: Path,
    *,
    train_episodes: int = 48,
    evaluation_seeds: int = 32,
    evaluation_start: int = 200,
    seed: int = 20260912,
) -> dict:
    frozen = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=False)
    learned = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=False)
    frozen_exposure = []
    training = []
    for index in range(train_episodes):
        training_seed = 10_000 + index
        frozen_exposure.append(
            run_episode(frozen, training_seed, learning=False, explore=True)
        )
        training.append(
            run_episode(learned, training_seed, learning=True, explore=True)
        )
    test_seeds = range(evaluation_start, evaluation_start + evaluation_seeds)
    baseline = [
        run_episode(frozen, index, learning=False, explore=False) for index in test_seeds
    ]
    candidate_path = root / "artifacts/checkpoints/driving-policy.candidate.npz"
    learned.policy.save(candidate_path, learned.graph.body_ids)
    deployed = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=False)
    deployed.policy.load(candidate_path, deployed.graph.body_ids)
    deployed.checkpoint_loaded = True
    learned_eval = [
        run_episode(deployed, index, learning=False, explore=False) for index in test_seeds
    ]
    baseline_distance = mean(item["distance"] for item in baseline)
    learned_distance = mean(item["distance"] for item in learned_eval)
    paired = paired_statistics(baseline, learned_eval, seed)
    distance_interval = paired["distance"]["bootstrap_95_ci"]
    return_interval = paired["return"]["bootstrap_95_ci"]
    report = {
        "protocol": {
            "seed": seed,
            "train_episodes": train_episodes,
            "evaluation_seeds": evaluation_seeds,
            "evaluation_start": evaluation_start,
            "training_seed_range": [10_000, 10_000 + train_episodes - 1],
            "test_seed_range": [evaluation_start, evaluation_start + evaluation_seeds - 1],
            "matched_exposure": True,
            "selection": "none; all paired evaluation seeds reported",
        },
        "frozen": summarize(baseline),
        "frozen_exposure": summarize(frozen_exposure),
        "training": summarize(training),
        "learned": summarize(learned_eval),
        "delta_mean_distance": learned_distance - baseline_distance,
        "delta_mean_return": paired["return"]["delta_mean"],
        "paired_statistics": paired,
        "success_gate": {
            "distance_positive": paired["distance"]["delta_mean"] > 0,
            "return_positive": paired["return"]["delta_mean"] > 0,
            "distance_ci_excludes_zero": distance_interval[0] > 0,
            "return_ci_excludes_zero": return_interval[0] > 0,
            "road_exit_not_worse": (
                summarize(learned_eval)["road_exit_rate"]
                <= summarize(baseline)["road_exit_rate"]
            ),
            "far_steering_not_worse": (
                summarize(learned_eval)["far_mean_abs_steering"]
                <= summarize(baseline)["far_mean_abs_steering"]
            ),
            "road_exit_rate_at_most_10pct": (
                summarize(learned_eval)["road_exit_rate"] <= 0.10
            ),
            "far_mean_abs_steering_at_most_0_15": (
                summarize(learned_eval)["far_mean_abs_steering"] <= 0.15
            ),
            "steering_change_not_worse": (
                summarize(learned_eval)["mean_abs_steering_change"]
                <= summarize(baseline)["mean_abs_steering_change"]
            ),
        },
        "plasticity": deployed.policy.summary(),
        "interpretation": (
            "This report evaluates dopamine plasticity after behavioural stabilization. "
            "A checkpoint is published only when every task, uncertainty and stability "
            "gate passes; otherwise the negative result remains the evidence."
        ),
    }
    if all(report["success_gate"].values()):
        candidate_path.replace(root / "artifacts/checkpoints/driving-policy.npz")
        report["published_checkpoint"] = "artifacts/checkpoints/driving-policy.npz"
    else:
        report["published_checkpoint"] = None
        candidate_path.unlink(missing_ok=True)
    return report


def write_evaluation(root: Path, output: Path, **kwargs) -> dict:
    report = evaluate(root, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def calibrate_stable_policy(
    root: Path, *, episodes: int = 48, evaluation_seeds: int = 32,
    evaluation_start: int = 200, seed: int = 20260912
) -> dict:
    engine = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=False)
    exposure = [
        run_episode(
            engine, 10_000 + index, learning=False, explore=True,
            safety_constraints=True,
        )
        for index in range(episodes)
    ]
    checkpoint = root / "artifacts/checkpoints/driving-policy.npz"
    candidate = checkpoint.with_name("driving-policy.calibrated-candidate.npz")
    engine.policy.save(
        candidate, engine.graph.body_ids, checkpoint_kind="frozen_calibrated"
    )
    deployed = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=False)
    deployed.policy.load(candidate, deployed.graph.body_ids)
    test_seeds = range(evaluation_start, evaluation_start + evaluation_seeds)
    evaluation = [
        run_episode(
            deployed, value, learning=False, explore=False, safety_constraints=True
        )
        for value in test_seeds
    ]
    summary = summarize(evaluation)
    gates = {
        "road_exit_rate_at_most_10pct": summary["road_exit_rate"] <= 0.10,
        "far_mean_abs_steering_at_most_0_15": (
            summary["far_mean_abs_steering"] <= 0.15
        ),
        "mean_abs_steering_change_at_most_0_03": (
            summary["mean_abs_steering_change"] <= 0.03
        ),
    }
    published = all(gates.values())
    if published:
        candidate.replace(checkpoint)
    else:
        candidate.unlink(missing_ok=True)
    checkpoint_sha256 = (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest() if published else None
    )
    return {
        "calibration_seed_range": [10_000, 10_000 + episodes - 1],
        "test_seed_range": [evaluation_start, evaluation_start + evaluation_seeds - 1],
        "learning": False,
        "explore": True,
        "checkpoint": str(checkpoint.relative_to(root)) if published else None,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_kind": deployed.policy.checkpoint_kind,
        "fresh_engine_loaded": True,
        "published": published,
        "gates": gates,
        "deployed": summary,
        "exposure": summarize(exposure),
    }


def evaluate_constraints(
    root: Path, *, evaluation_seeds: int = 32, evaluation_start: int = 200,
    seed: int = 20260913,
) -> dict:
    disabled = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=True)
    enabled = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=True)
    if (
        disabled.policy.checkpoint_kind != "frozen_calibrated"
        or enabled.policy.checkpoint_kind != "frozen_calibrated"
    ):
        raise ValueError(
            "constraint evaluation requires a frozen_calibrated checkpoint"
        )
    test_seeds = range(evaluation_start, evaluation_start + evaluation_seeds)
    without_constraint = [
        run_episode(
            disabled, value, learning=False, explore=False, safety_constraints=False
        )
        for value in test_seeds
    ]
    with_constraint = [
        run_episode(
            enabled, value, learning=False, explore=False, safety_constraints=True
        )
        for value in test_seeds
    ]
    return {
        "protocol": {
            "checkpoint_kind": enabled.policy.checkpoint_kind,
            "checkpoint_sha256": hashlib.sha256(
                enabled.policy_checkpoint.read_bytes()
            ).hexdigest(),
            "test_seed_range": [evaluation_start, evaluation_start + evaluation_seeds - 1],
            "learning": False, "explore": False,
            "only_variable": "lane_constraint",
        },
        "constraint_off": summarize(without_constraint),
        "constraint_on": summarize(with_constraint),
        "paired": paired_statistics(without_constraint, with_constraint, seed),
    }
