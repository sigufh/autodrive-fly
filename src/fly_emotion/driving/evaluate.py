from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np

from fly_emotion.driving.engine import DrivingEngine


def run_episode(
    engine: DrivingEngine, seed: int, *, learning: bool, explore: bool
) -> dict:
    engine.reset(seed, keep_learning=True)
    while not engine.env.done:
        engine.step(learning=learning, explore=explore, include_activity=False)
    return {
        "seed": seed,
        "distance": engine.env.y,
        "return": engine.env.total_reward,
        "steps": engine.env.steps,
        "success": engine.env.y >= engine.env.road_length,
    }


def summarize(episodes: list[dict]) -> dict:
    return {
        "episodes": len(episodes),
        "mean_distance": mean(item["distance"] for item in episodes),
        "mean_return": mean(item["return"] for item in episodes),
        "success_rate": mean(float(item["success"]) for item in episodes),
        "details": episodes,
    }


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
    learned_eval = [
        run_episode(learned, index, learning=False, explore=False) for index in test_seeds
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
        },
        "plasticity": learned.policy.summary(),
        "interpretation": (
            "A positive delta is preliminary task evidence, not biological validation; "
            "report failures and variance across seeds."
        ),
    }
    if all(report["success_gate"].values()):
        learned.policy.save(
            root / "artifacts/checkpoints/driving-policy.npz", learned.graph.body_ids
        )
        report["published_checkpoint"] = "artifacts/checkpoints/driving-policy.npz"
    else:
        report["published_checkpoint"] = None
    return report


def write_evaluation(root: Path, output: Path, **kwargs) -> dict:
    report = evaluate(root, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
