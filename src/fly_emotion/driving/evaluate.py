from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from statistics import mean

import numpy as np

from fly_emotion.driving.engine import NEURAL_POLICY_VERSION, POLICY_VERSION, DrivingEngine
from fly_emotion.driving.environment import DrivingEnvironment

EVALUATION_PROTOCOL_VERSION = 4
POST_PASS_WINDOW_STEPS = 30
POST_PASS_METRIC_PROTOCOL_VERSION = 2


def validate_mirror_protocol(count: int, start: int, *, train_episodes: int = 0) -> None:
    if count < 2 or count % 2 or start < 0 or start % 2:
        raise ValueError("evaluation requires complete adjacent even/odd mirror pairs")
    if train_episodes < 0 or train_episodes % 2:
        raise ValueError("calibration/training requires complete mirror pairs")
    if set(range(start // 2, (start + count) // 2)) & set(
        range(5_000, 5_000 + train_episodes // 2)
    ):
        raise ValueError("training and evaluation pair identities overlap")


def summarize_post_pass_windows(
    trace: np.ndarray,
    pass_steps: list[int] | tuple[int, ...],
    terminal_reason: str | None,
    *,
    window_steps: int = POST_PASS_WINDOW_STEPS,
) -> dict:
    """Measure post-pass control without treating early crashes as stability."""
    if trace.ndim != 2 or trace.shape[1] < 4:
        raise ValueError("control trace must have at least four columns")
    if window_steps < 2:
        raise ValueError("post-pass window must contain at least two steps")
    all_raw_steering: list[float] = []
    all_steering: list[float] = []
    all_lateral_drift: list[float] = []
    complete_raw_steering: list[float] = []
    complete_steering: list[float] = []
    complete_lateral_drift: list[float] = []
    complete_drift_per_metre: list[float] = []
    failure_truncated_windows = 0
    censored_windows = 0
    for step in pass_steps:
        start = max(int(step) - 1, 0)
        window = trace[start : min(len(trace), start + window_steps)]
        if not len(window):
            continue
        raw_steering = float(np.mean(np.abs(window[:, 0])))
        steering = float(np.mean(np.abs(window[:, 1])))
        lateral_drift = float(abs(window[-1, 2] - window[0, 2]))
        all_raw_steering.append(raw_steering)
        all_steering.append(steering)
        all_lateral_drift.append(lateral_drift)
        if len(window) == window_steps:
            complete_raw_steering.append(raw_steering)
            complete_steering.append(steering)
            complete_lateral_drift.append(lateral_drift)
            forward_distance = float(max(window[-1, 3] - window[0, 3], 0.0))
            if forward_distance > 1e-6:
                complete_drift_per_metre.append(lateral_drift / forward_distance)
        elif terminal_reason in {"obstacle", "road_boundary"}:
            failure_truncated_windows += 1
        else:
            censored_windows += 1
    window_count = len(all_steering)
    complete_windows = len(complete_steering)
    return {
        # Legacy variable-length metrics are retained for report compatibility.
        "post_pass_mean_abs_raw_steering": (
            mean(all_raw_steering) if all_raw_steering else 0.0
        ),
        "post_pass_mean_abs_steering": mean(all_steering) if all_steering else 0.0,
        "post_pass_mean_lateral_drift": (
            mean(all_lateral_drift) if all_lateral_drift else 0.0
        ),
        "post_pass_window_count": window_count,
        "post_pass_complete_window_count": complete_windows,
        "post_pass_failure_truncated_window_count": failure_truncated_windows,
        "post_pass_censored_window_count": censored_windows,
        "post_pass_complete_window_rate": (
            complete_windows / window_count if window_count else 0.0
        ),
        "post_pass_30_step_early_failure_rate": (
            failure_truncated_windows / window_count if window_count else 0.0
        ),
        "post_pass_complete_mean_abs_raw_steering": (
            mean(complete_raw_steering) if complete_raw_steering else 0.0
        ),
        "post_pass_complete_mean_abs_steering": (
            mean(complete_steering) if complete_steering else 0.0
        ),
        "post_pass_complete_mean_lateral_drift": (
            mean(complete_lateral_drift) if complete_lateral_drift else 0.0
        ),
        "post_pass_complete_mean_lateral_drift_per_metre": (
            mean(complete_drift_per_metre) if complete_drift_per_metre else 0.0
        ),
        "post_pass_complete_drift_per_metre_count": len(complete_drift_per_metre),
    }


def evaluate_city_alpha(root: Path, *, seeds: tuple[int, ...] = (0, 1, 7)) -> dict:
    """Run the fixed, small city-alpha smoke protocol.

    This is intentionally separate from the v5 mirror-release protocol.  It
    checks city route/rule integration, not city-scale generalization.
    """
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("city alpha requires non-negative seeds")
    episodes = []
    for seed in seeds:
        engine = DrivingEngine(root, seed=seed, top_k=1, scenario="city")
        engine.reset(seed, keep_learning=False, scenario="city")
        while not engine.env.done:
            state = engine.step(learning=False, explore=False, include_activity=False)
        city = state["environment"]["city"]
        episodes.append(
            {
                "seed": seed,
                "terminal_reason": state["environment"]["terminal_reason"],
                "distance": state["environment"]["vehicle"]["y"],
                "actors_passed": state["environment"]["obstacles_passed"],
                "traffic_violations": city["violations"],
                "mean_abs_steering_change": state["control_statistics"]["mean_abs_steering_change"],
            }
        )
    return {
        "protocol": {
            "scenario": "city_alpha",
            "seeds": list(seeds),
            "learning": False,
            "explore": False,
            "checkpoint_kind": engine.policy.checkpoint_kind,
            "checkpoint_sha256": hashlib.sha256(engine.policy_checkpoint.read_bytes()).hexdigest(),
            "claim_boundary": "integration smoke protocol; not a city generalization benchmark",
        },
        "summary": {
            "success_rate": mean(item["terminal_reason"] == "success" for item in episodes),
            "mean_distance": mean(item["distance"] for item in episodes),
            "mean_actors_passed": mean(item["actors_passed"] for item in episodes),
            "traffic_violation_rate": mean(bool(item["traffic_violations"]) for item in episodes),
            "mean_abs_steering_change": mean(item["mean_abs_steering_change"] for item in episodes),
        },
        "episodes": episodes,
    }


def evaluation_contract(root: Path) -> dict:
    sources = Path(__file__).parent
    return {
        "policy_version": POLICY_VERSION,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "brain_substeps": 4,
        "learning": False,
        "explore": False,
        "implementation_sha256": {
            name: hashlib.sha256((sources / name).read_bytes()).hexdigest()
            for name in ("engine.py", "environment.py", "retina.py")
        },
        "asset_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "data/processed/malecns-v1.0/body_ids.npy",
                "data/processed/malecns-v1.0/adjacency_target_norm.npz",
                "data/processed/malecns-v1.0/retina_map.npz",
                "data/raw/malecns-v1.0/body-neurotransmitters.feather",
            )
        },
    }


def run_episode(
    engine: DrivingEngine,
    seed: int,
    *,
    learning: bool,
    explore: bool,
    safety_constraints: bool = True,
    control_mode: str = "assisted",
    curriculum_stage: str = "full",
    sensory_profile: str | None = None,
) -> dict:
    engine.reset(
        seed,
        keep_learning=True,
        control_mode=control_mode,
        curriculum_stage=curriculum_stage,
        sensory_profile=sensory_profile,
    )
    controls = []
    while not engine.env.done:
        engine.step(
            learning=learning,
            explore=explore,
            safety_constraints=safety_constraints,
            include_activity=False,
        )
        controls.append(
            [
                engine.last_raw_action["steering"],
                engine.env.steering,
                engine.env.x,
                engine.env.y,
                engine.last_action["drive"],
                engine.last_action["reverse"],
            ]
        )
    trace = np.asarray(controls)
    post_pass = summarize_post_pass_windows(
        trace, list(engine.env.obstacle_pass_steps.values()), engine.env.terminal_reason
    )
    first = engine.env.obstacles[0]
    before_first = trace[:, 3] < first.y - first.radius - engine.env.vehicle_radius
    return {
        "seed": seed,
        "pair_seed": engine.env.pair_seed,
        "mirror": engine.env.mirror,
        "first_obstacle_side": "left" if first.x < 0 else "right",
        "distance": engine.env.y,
        "return": engine.env.total_reward,
        "steps": engine.env.steps,
        "success": engine.env.terminal_reason == "success",
        "terminal_reason": engine.env.terminal_reason,
        "obstacles_passed": engine.env.obstacles_passed,
        "first_obstacle_passed": 0 in engine.env.passed_obstacle_indices,
        "mean_signed_raw_steering": float(trace[:, 0].mean()),
        "mean_signed_steering": float(trace[:, 1].mean()),
        "right_turn_fraction": float(np.mean(trace[:, 1] > 0.055)),
        "left_turn_fraction": float(np.mean(trace[:, 1] < -0.055)),
        "pre_first_mean_signed_steering": (
            float(trace[before_first, 1].mean()) if np.any(before_first) else 0.0
        ),
        "collision_before_first_pass": (
            engine.env.terminal_reason in {"obstacle", "road_boundary"}
            and 0 not in engine.env.passed_obstacle_indices
        ),
        "mean_drive": float(trace[:, 4].mean()),
        "mean_reverse": float(trace[:, 5].mean()),
        "reverse_fraction": float(np.mean(trace[:, 5] > 0.05)),
        "negative_speed_fraction": float(
            np.mean(
                np.asarray([point[1] for point in engine.env.trajectory[1:]])
                < np.asarray([point[1] for point in engine.env.trajectory[:-1]])
            )
        ),
        **post_pass,
        "control_trace": controls,
        "control_mode": control_mode,
        "curriculum_stage": curriculum_stage,
        "sensory_profile": engine.sensory_profile,
        **engine.control_summary(),
    }


def train_neural_curriculum(
    root: Path,
    *,
    train_episodes: int = 24,
    evaluation_start: int = 600,
    evaluation_seeds: int = 8,
    seed: int = 20260914,
    publish: bool = False,
    stage: str = "single",
    resume: bool = False,
    sensory_profile: str = "front",
) -> dict:
    """Train direct neural control on a mirrored reward-only curriculum."""
    if stage not in {"single", "triple", "nine"}:
        raise ValueError(f"unknown neural curriculum stage: {stage}")
    validate_mirror_protocol(evaluation_seeds, evaluation_start, train_episodes=train_episodes)
    frozen = DrivingEngine(
        root,
        seed=seed,
        top_k=1,
        load_checkpoint=resume,
        control_mode="neural",
        sensory_profile=sensory_profile,
    )
    learned = DrivingEngine(
        root,
        seed=seed,
        top_k=1,
        load_checkpoint=resume,
        control_mode="neural",
        sensory_profile=sensory_profile,
    )
    if resume and (not frozen.checkpoint_loaded or not learned.checkpoint_loaded):
        raise ValueError("resume requested without a compatible neural-v6 checkpoint")
    for policy in (frozen.policy, learned.policy):
        policy.configure_neural_curriculum()
        if stage == "nine":
            # Long single-obstacle credit encouraged prolonged turns. On the
            # full road, faster traces let immediate stability cost terminate
            # the turn while still spanning obstacle clearance.
            policy.learning_rate = 0.006
            policy.exploration_sigma = 0.12
            policy.eligibility_decay = 0.98

    frozen_exposure, training = [], []
    for index in range(train_episodes):
        training_seed = 10_000 + index
        frozen_exposure.append(
            run_episode(
                frozen,
                training_seed,
                learning=False,
                explore=True,
                safety_constraints=False,
                control_mode="neural",
                curriculum_stage=stage,
                sensory_profile=sensory_profile,
            )
        )
        training.append(
            run_episode(
                learned,
                training_seed,
                learning=True,
                explore=True,
                safety_constraints=False,
                control_mode="neural",
                curriculum_stage=stage,
                sensory_profile=sensory_profile,
            )
        )

    test_seeds = range(evaluation_start, evaluation_start + evaluation_seeds)
    baseline = [
        run_episode(
            frozen,
            value,
            learning=False,
            explore=False,
            safety_constraints=False,
            control_mode="neural",
            curriculum_stage=stage,
            sensory_profile=sensory_profile,
        )
        for value in test_seeds
    ]
    candidate = root / (
        f"artifacts/checkpoints/driving-policy.neural-v6.{stage}.{sensory_profile}.candidate.npz"
    )
    published = root / "artifacts/checkpoints/driving-policy.neural-v6.npz"
    learned.policy.save(
        candidate,
        learned.graph.body_ids,
        checkpoint_kind="neural_curriculum_v6",
        format_version=NEURAL_POLICY_VERSION,
    )
    deployed = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=False, control_mode="neural")
    deployed.policy.load(candidate, deployed.graph.body_ids)
    learned_eval = [
        run_episode(
            deployed,
            value,
            learning=False,
            explore=False,
            safety_constraints=False,
            control_mode="neural",
            curriculum_stage=stage,
            sensory_profile=sensory_profile,
        )
        for value in test_seeds
    ]
    frozen_summary, learned_summary = summarize(baseline), summarize(learned_eval)
    shuffled_episodes = []
    for item in learned_eval:
        env = DrivingEnvironment()
        env.reset(item["seed"])
        env.configure_curriculum(stage)
        trace = np.asarray(item["control_trace"], dtype=float)
        rng = np.random.default_rng(seed + item["pair_seed"])
        order = rng.permutation(len(trace))
        step = 0
        while not env.done:
            row = trace[order[step % len(order)]]
            reverse = float(row[5])
            drive = float(row[4])
            throttle = float(np.clip((drive + reverse) / max(1.0 - reverse, 1e-6), 0, 1))
            env.step(float(row[1]), throttle, reverse)
            step += 1
        shuffled_episodes.append(
            {
                "seed": item["seed"],
                "terminal_reason": env.terminal_reason,
                "distance": env.y,
                "obstacles_passed": env.obstacles_passed,
                "success": env.terminal_reason == "success",
            }
        )
    shuffled_summary = {
        "controller": "learned_action_trace_time_shuffled",
        "success_rate": mean(float(item["success"]) for item in shuffled_episodes),
        "mean_distance": mean(item["distance"] for item in shuffled_episodes),
        "mean_obstacles_passed": mean(item["obstacles_passed"] for item in shuffled_episodes),
        "details": shuffled_episodes,
    }
    gates = {
        "first_pass_at_least_70pct": learned_summary["first_obstacle_pass_rate"] >= 0.70,
        "completion_meets_stage_target": learned_summary["success_rate"]
        >= (0.50 if stage == "single" else 0.25),
        "completion_exceeds_frozen": (
            learned_summary["success_rate"] > frozen_summary["success_rate"]
        ),
        "distance_exceeds_frozen": learned_summary["mean_distance"]
        > frozen_summary["mean_distance"],
        "obstacles_exceed_frozen": (
            learned_summary["mean_obstacles_passed"] > frozen_summary["mean_obstacles_passed"]
        ),
        "post_pass_steering_below_frozen": (
            learned_summary["post_pass_complete_mean_abs_steering"]
            < frozen_summary["post_pass_complete_mean_abs_steering"]
        ),
        "post_pass_drift_below_frozen": (
            learned_summary["post_pass_complete_mean_lateral_drift_per_metre"]
            < frozen_summary["post_pass_complete_mean_lateral_drift_per_metre"]
        ),
        "post_pass_complete_window_rate_not_lower": (
            learned_summary["post_pass_complete_window_rate"]
            >= frozen_summary["post_pass_complete_window_rate"]
        ),
        "post_pass_early_failure_rate_not_higher": (
            learned_summary["post_pass_30_step_early_failure_rate"]
            <= frozen_summary["post_pass_30_step_early_failure_rate"]
        ),
        "post_pass_complete_windows_available": (
            learned_summary["post_pass_complete_window_count"] > 0
            and frozen_summary["post_pass_complete_window_count"] > 0
        ),
        "road_exit_below_frozen": (
            learned_summary["road_exit_rate"] < frozen_summary["road_exit_rate"]
        ),
        "completion_exceeds_shuffled_actions": (
            learned_summary["success_rate"] > shuffled_summary["success_rate"]
        ),
        "distance_exceeds_shuffled_actions": (
            learned_summary["mean_distance"] > shuffled_summary["mean_distance"]
        ),
        "zero_action_override": (
            learned_summary["constraint_rate"] == 0 and learned_summary["mean_abs_constraint"] == 0
        ),
    }
    released = bool(publish and all(gates.values()))
    if released:
        if resume and published.exists():
            previous = published.with_name("driving-policy.neural-v6.single.npz")
            if not previous.exists():
                shutil.copy2(published, previous)
        candidate.replace(published)
    return {
        "protocol_version": NEURAL_POLICY_VERSION,
        "post_pass_metric_protocol_version": POST_PASS_METRIC_PROTOCOL_VERSION,
        "post_pass_window_steps": POST_PASS_WINDOW_STEPS,
        "curriculum_stage": stage,
        "sensory_profile": sensory_profile,
        "resumed_from_published_v6": resume,
        "training_seed_range": [10_000, 10_000 + train_episodes - 1],
        "test_seed_range": [evaluation_start, evaluation_start + evaluation_seeds - 1],
        "learning_signal": "environment_reward_only",
        "action_override": False,
        "published": released,
        "checkpoint": str(published.relative_to(root)) if released else None,
        "candidate_checkpoint": str(candidate.relative_to(root)) if not released else None,
        "candidate_sha256": hashlib.sha256(
            (published if released else candidate).read_bytes()
        ).hexdigest(),
        "gates": gates,
        "frozen": frozen_summary,
        "frozen_exposure": summarize(frozen_exposure),
        "training": summarize(training),
        "learned": learned_summary,
        "shuffled_action_baseline": shuffled_summary,
        "plasticity": deployed.policy.summary(),
    }


def evaluate_neural_decision_baseline(
    root: Path, *, start: int = 400, count: int = 8, seed: int = 20260912
) -> dict:
    """Compare the preserved v5 assisted baseline to direct neural control.

    Both arms use the same frozen learned-v5 checkpoint and exact road seeds.
    In the neural arm, vehicle actions come only from DNp20/DNpe017/MDN output
    through the actuator adapter; no obstacle, road or rule controller may
    override those outputs.
    """
    if count < 1 or start < 0:
        raise ValueError("neural-decision comparison requires a positive seed range")
    seeds = range(start, start + count)
    arms = {}
    for mode in ("assisted", "neural"):
        engine = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=True, control_mode=mode)
        episodes = [
            run_episode(
                engine,
                value,
                learning=False,
                explore=False,
                safety_constraints=True,
                control_mode=mode,
            )
            for value in seeds
        ]
        arms[mode] = summarize(episodes)
    neural = arms["neural"]
    if neural["constraint_rate"] != 0 or neural["mean_abs_constraint"] != 0:
        raise AssertionError("neural decision evidence contains an action override")
    return {
        "protocol": {
            "scenario": "highway_random_obstacles",
            "seeds": list(seeds),
            "learning": False,
            "explore": False,
            "checkpoint_sha256": hashlib.sha256(
                (root / "artifacts/checkpoints/driving-policy.npz").read_bytes()
            ).hexdigest(),
            "claim_boundary": (
                "Assisted is a preserved engineering baseline; neural actions are direct "
                "DNp20/DNpe017/MDN-to-vehicle mappings and are not claimed to be trained avoidance."
            ),
        },
        "assisted_baseline": arms["assisted"],
        "neural_decision": arms["neural"],
        "delta_neural_minus_assisted": {
            "mean_distance": neural["mean_distance"] - arms["assisted"]["mean_distance"],
            "mean_obstacles_passed": (
                neural["mean_obstacles_passed"] - arms["assisted"]["mean_obstacles_passed"]
            ),
            "success_rate": neural["success_rate"] - arms["assisted"]["success_rate"],
        },
    }


def evaluate_neural_transfer(
    root: Path,
    *,
    count: int = 8,
    triple_start: int = 700,
    full_start: int = 420,
) -> dict:
    """Evaluate the published v6 checkpoint without further learning."""
    results = {}
    for stage, start in (("triple", triple_start), ("full", full_start)):
        engine = DrivingEngine(
            root, seed=20260914, top_k=1, load_checkpoint=True, control_mode="neural"
        )
        if not engine.checkpoint_loaded or engine.policy.checkpoint_kind != "neural_curriculum_v6":
            raise ValueError("neural transfer requires the published v6 checkpoint")
        episodes = [
            run_episode(
                engine,
                value,
                learning=False,
                explore=False,
                safety_constraints=False,
                control_mode="neural",
                curriculum_stage=stage,
            )
            for value in range(start, start + count)
        ]
        summary = summarize(episodes)
        results[stage] = {
            key: summary[key]
            for key in (
                "episodes",
                "mean_distance",
                "success_rate",
                "mean_obstacles_passed",
                "first_obstacle_pass_rate",
                "road_exit_rate",
                "obstacle_collision_rate",
                "timeout_rate",
                "constraint_rate",
                "mean_abs_constraint",
            )
        }
        results[stage]["details"] = [
            {
                "seed": item["seed"],
                "terminal_reason": item["terminal_reason"],
                "distance": item["distance"],
                "obstacles_passed": item["obstacles_passed"],
            }
            for item in episodes
        ]
    checkpoint = root / "artifacts/checkpoints/driving-policy.neural-v6.npz"
    return {
        "protocol": {
            "learning": False,
            "explore": False,
            "action_override": False,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "triple_seed_range": [triple_start, triple_start + count - 1],
            "full_seed_range": [full_start, full_start + count - 1],
        },
        **results,
    }


def evaluate_neural_motor_adaptation(root: Path, *, start: int = 900, count: int = 8) -> dict:
    """Pair frozen v6 roads with/without environment-blind DNp20 adaptation."""
    arms = {}
    for name, rate in (("no_adaptation", 0.0), ("adaptation_0_08", 0.08)):
        engine = DrivingEngine(
            root, seed=20260914, top_k=1, load_checkpoint=True, control_mode="neural"
        )
        engine.neural_motor_adapter.adaptation_rate = rate
        episodes = [
            run_episode(
                engine,
                seed,
                learning=False,
                explore=False,
                safety_constraints=False,
                control_mode="neural",
                curriculum_stage="nine",
            )
            for seed in range(start, start + count)
        ]
        arms[name] = summarize(episodes)
    baseline, adapted = arms["no_adaptation"], arms["adaptation_0_08"]
    gates = {
        "completion_not_worse": adapted["success_rate"] >= baseline["success_rate"],
        "obstacles_not_worse": (
            adapted["mean_obstacles_passed"] >= baseline["mean_obstacles_passed"]
        ),
        "post_pass_steering_lower": (
            adapted["post_pass_complete_mean_abs_steering"]
            < baseline["post_pass_complete_mean_abs_steering"]
        ),
        "post_pass_drift_lower": (
            adapted["post_pass_complete_mean_lateral_drift_per_metre"]
            < baseline["post_pass_complete_mean_lateral_drift_per_metre"]
        ),
        "post_pass_complete_window_rate_not_lower": (
            adapted["post_pass_complete_window_rate"]
            >= baseline["post_pass_complete_window_rate"]
        ),
        "post_pass_early_failure_rate_not_higher": (
            adapted["post_pass_30_step_early_failure_rate"]
            <= baseline["post_pass_30_step_early_failure_rate"]
        ),
        "post_pass_complete_windows_available": (
            adapted["post_pass_complete_window_count"] > 0
            and baseline["post_pass_complete_window_count"] > 0
        ),
        "max_lateral_lower": adapted["max_abs_lateral"] < baseline["max_abs_lateral"],
        "zero_action_override": (
            adapted["constraint_rate"] == 0 and adapted["mean_abs_constraint"] == 0
        ),
    }
    return {
        "protocol": {
            "seed_range": [start, start + count - 1],
            "learning": False,
            "explore": False,
            "action_override": False,
            "post_pass_metric_protocol_version": POST_PASS_METRIC_PROTOCOL_VERSION,
            "post_pass_window_steps": POST_PASS_WINDOW_STEPS,
            "only_variable": "environment_blind_DNp20_motor_adaptation_rate",
            "checkpoint_sha256": hashlib.sha256(
                (root / "artifacts/checkpoints/driving-policy.neural-v6.npz").read_bytes()
            ).hexdigest(),
        },
        "gates": gates,
        **arms,
    }


def evaluate_sensory_ablation(
    root: Path,
    *,
    train_episodes: int = 4,
    evaluation_start: int = 960,
    evaluation_seeds: int = 4,
    seed: int = 20260914,
) -> dict:
    """Matched small-sample screen of anatomy-backed sensory profiles."""
    published = root / "artifacts/checkpoints/driving-policy.neural-v6.npz"
    if not published.exists():
        raise FileNotFoundError("sensory ablation requires the published neural-v6 checkpoint")
    profiles = ("front", "panorama", "panorama_flow", "panorama_flow_body")
    arms = {}
    metrics = (
        "mean_distance",
        "success_rate",
        "mean_obstacles_passed",
        "road_exit_rate",
        "obstacle_collision_rate",
        "post_pass_mean_abs_steering",
        "post_pass_mean_lateral_drift",
        "post_pass_complete_window_count",
        "post_pass_complete_window_rate",
        "post_pass_30_step_early_failure_rate",
        "post_pass_complete_mean_abs_steering",
        "post_pass_complete_mean_lateral_drift",
        "post_pass_complete_mean_lateral_drift_per_metre",
        "constraint_rate",
    )
    for profile in profiles:
        report = train_neural_curriculum(
            root,
            train_episodes=train_episodes,
            evaluation_start=evaluation_start,
            evaluation_seeds=evaluation_seeds,
            seed=seed,
            publish=False,
            stage="nine",
            resume=True,
            sensory_profile=profile,
        )
        arms[profile] = {
            "candidate_sha256": report["candidate_sha256"],
            "gates": report["gates"],
            "frozen": {key: report["frozen"][key] for key in metrics},
            "learned": {key: report["learned"][key] for key in metrics},
        }
    front = arms["front"]["learned"]
    for profile in profiles[1:]:
        learned = arms[profile]["learned"]
        arms[profile]["versus_front"] = {
            "completion_delta": learned["success_rate"] - front["success_rate"],
            "distance_delta": learned["mean_distance"] - front["mean_distance"],
            "obstacles_delta": (learned["mean_obstacles_passed"] - front["mean_obstacles_passed"]),
            "post_pass_complete_steering_delta": (
                learned["post_pass_complete_mean_abs_steering"]
                - front["post_pass_complete_mean_abs_steering"]
            ),
            "post_pass_complete_drift_per_metre_delta": (
                learned["post_pass_complete_mean_lateral_drift_per_metre"]
                - front["post_pass_complete_mean_lateral_drift_per_metre"]
            ),
            "post_pass_early_failure_rate_delta": (
                learned["post_pass_30_step_early_failure_rate"]
                - front["post_pass_30_step_early_failure_rate"]
            ),
        }
    return {
        "protocol": {
            "starting_checkpoint": str(published.relative_to(root)),
            "starting_checkpoint_sha256": hashlib.sha256(published.read_bytes()).hexdigest(),
            "training_seed_range": [10_000, 10_000 + train_episodes - 1],
            "test_seed_range": [evaluation_start, evaluation_start + evaluation_seeds - 1],
            "matched_training_and_test_seeds": True,
            "learning_signal": "environment_reward_only",
            "action_override": False,
            "post_pass_metric_protocol_version": POST_PASS_METRIC_PROTOCOL_VERSION,
            "post_pass_window_steps": POST_PASS_WINDOW_STEPS,
            "claim_boundary": (
                "two-pair screening ablation; no sensory profile is deployed "
                "without a larger held-out confirmation"
            ),
        },
        "anatomy": DrivingEngine(root, top_k=1, load_checkpoint=False).sensory_projection.summary(),
        "arms": arms,
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
        "mean_signed_raw_steering",
        "mean_signed_steering",
        "right_turn_fraction",
        "left_turn_fraction",
        "pre_first_mean_signed_steering",
        "mean_drive",
        "mean_reverse",
        "reverse_fraction",
        "negative_speed_fraction",
        "reverse_gate_fraction",
        "post_pass_mean_abs_raw_steering",
        "post_pass_mean_abs_steering",
        "post_pass_mean_lateral_drift",
    ):
        summary[key] = mean(item[key] for item in episodes)
    summary["road_exit_rate"] = mean(
        float(item["terminal_reason"] == "road_boundary") for item in episodes
    )
    summary["obstacle_collision_rate"] = mean(
        float(item["terminal_reason"] == "obstacle") for item in episodes
    )
    summary["mean_obstacles_passed"] = mean(item["obstacles_passed"] for item in episodes)
    summary["first_obstacle_pass_rate"] = mean(
        float(item["first_obstacle_passed"]) for item in episodes
    )
    summary["collision_before_first_pass_rate"] = mean(
        float(item["collision_before_first_pass"]) for item in episodes
    )
    summary["mean_steps"] = mean(item["steps"] for item in episodes)
    summary["timeout_rate"] = mean(float(item["terminal_reason"] == "timeout") for item in episodes)
    post_pass_windows = sum(item.get("post_pass_window_count", 0) for item in episodes)
    complete_windows = sum(item.get("post_pass_complete_window_count", 0) for item in episodes)
    failure_windows = sum(
        item.get("post_pass_failure_truncated_window_count", 0) for item in episodes
    )
    censored_windows = sum(item.get("post_pass_censored_window_count", 0) for item in episodes)
    drift_per_metre_windows = sum(
        item.get("post_pass_complete_drift_per_metre_count", 0) for item in episodes
    )

    def weighted_post_pass(key: str, count_key: str, count: int) -> float:
        if not count:
            return 0.0
        return sum(item.get(key, 0.0) * item.get(count_key, 0) for item in episodes) / count

    summary.update(
        {
            "post_pass_window_count": post_pass_windows,
            "post_pass_complete_window_count": complete_windows,
            "post_pass_failure_truncated_window_count": failure_windows,
            "post_pass_censored_window_count": censored_windows,
            "post_pass_complete_window_rate": (
                complete_windows / post_pass_windows if post_pass_windows else 0.0
            ),
            "post_pass_30_step_early_failure_rate": (
                failure_windows / post_pass_windows if post_pass_windows else 0.0
            ),
            "post_pass_complete_mean_abs_raw_steering": weighted_post_pass(
                "post_pass_complete_mean_abs_raw_steering",
                "post_pass_complete_window_count",
                complete_windows,
            ),
            "post_pass_complete_mean_abs_steering": weighted_post_pass(
                "post_pass_complete_mean_abs_steering",
                "post_pass_complete_window_count",
                complete_windows,
            ),
            "post_pass_complete_mean_lateral_drift": weighted_post_pass(
                "post_pass_complete_mean_lateral_drift",
                "post_pass_complete_window_count",
                complete_windows,
            ),
            "post_pass_complete_mean_lateral_drift_per_metre": weighted_post_pass(
                "post_pass_complete_mean_lateral_drift_per_metre",
                "post_pass_complete_drift_per_metre_count",
                drift_per_metre_windows,
            ),
            "post_pass_complete_drift_per_metre_count": drift_per_metre_windows,
        }
    )
    return summary


def mirror_summary(episodes: list[dict]) -> dict:
    by_seed = {item["seed"]: item for item in episodes}
    if len(by_seed) != len(episodes) or any(seed ^ 1 not in by_seed for seed in by_seed):
        raise ValueError("mirror summary requires unique complete pairs")
    pairs = []
    for seed in sorted(by_seed):
        if seed % 2:
            continue
        left, right = by_seed[seed], by_seed[seed + 1]
        length = min(left["steps"], right["steps"])
        a = np.asarray(left["control_trace"])[:length]
        b = np.asarray(right["control_trace"])[:length]
        pairs.append(
            {
                "pair_seed": seed // 2,
                "common_steps": length,
                "raw_steering_mirror_mae": float(np.mean(np.abs(a[:, 0] + b[:, 0]))),
                "steering_mirror_mae": float(np.mean(np.abs(a[:, 1] + b[:, 1]))),
                "lateral_mirror_mae": float(np.mean(np.abs(a[:, 2] + b[:, 2]))),
                "distance_gap": abs(left["distance"] - right["distance"]),
                "steps_gap": abs(left["steps"] - right["steps"]),
                "drive_mirror_mae": float(np.mean(np.abs(a[:, 4] - b[:, 4]))),
                "reverse_mirror_mae": float(np.mean(np.abs(a[:, 5] - b[:, 5]))),
                "first_pass_agrees": left["first_obstacle_passed"]
                == right["first_obstacle_passed"],
            }
        )
    return {
        "pair_count": len(pairs),
        "raw_steering_mirror_mae": mean(p["raw_steering_mirror_mae"] for p in pairs),
        "steering_mirror_mae": mean(p["steering_mirror_mae"] for p in pairs),
        "mean_distance_gap": mean(p["distance_gap"] for p in pairs),
        "max_steps_gap": max(p["steps_gap"] for p in pairs),
        "by_first_obstacle_side": {
            side: summarize([item for item in episodes if item["first_obstacle_side"] == side])
            for side in ("left", "right")
        },
        "details": pairs,
    }


def usability_gates(summary: dict) -> dict:
    # Absolute task gates prevent early collision or stopping from looking stable.
    return {
        "completion_at_least_50pct": summary["success_rate"] >= 0.5,
        "first_pass_at_least_75pct": summary["first_obstacle_pass_rate"] >= 0.75,
        "mean_obstacles_passed_at_least_4_5": summary["mean_obstacles_passed"] >= 4.5,
        "early_collision_at_most_25pct": summary["collision_before_first_pass_rate"] <= 0.25,
        "timeout_at_most_10pct": summary["timeout_rate"] <= 0.1,
        "road_exit_at_most_10pct": summary["road_exit_rate"] <= 0.1,
        "forward_drive_at_least_0_35": summary["mean_drive"] >= 0.35,
        "reverse_fraction_at_most_10pct": summary["reverse_fraction"] <= 0.10,
        "negative_speed_at_most_5pct": summary["negative_speed_fraction"] <= 0.05,
    }


def paired_statistics(frozen: list[dict], learned: list[dict], seed: int) -> dict:
    if not frozen or len(frozen) != len(learned):
        raise ValueError("paired statistics require equally sized nonempty episodes")
    if any(a.get("seed") != b.get("seed") for a, b in zip(frozen, learned, strict=True)):
        raise ValueError("paired statistics require identical ordered seeds")
    frozen_distance = np.asarray([item["distance"] for item in frozen])
    learned_distance = np.asarray([item["distance"] for item in learned])
    frozen_return = np.asarray([item["return"] for item in frozen])
    learned_return = np.asarray([item["return"] for item in learned])
    distance_delta = learned_distance - frozen_distance
    return_delta = learned_return - frozen_return
    rng = np.random.default_rng(seed)
    pair_ids = [item.get("pair_seed", index) for index, item in enumerate(frozen)]
    clusters = [np.flatnonzero(np.asarray(pair_ids) == key) for key in sorted(set(pair_ids))]
    samples = rng.integers(0, len(clusters), size=(50_000, len(clusters)))

    def interval(values: np.ndarray) -> list[float]:
        cluster_values = np.asarray([values[indices].mean() for indices in clusters])
        means = cluster_values[samples].mean(axis=1)
        return np.quantile(means, [0.025, 0.975]).tolist()

    return {
        "bootstrap_unit": "road_pair",
        "independent_units": len(clusters),
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
    evaluation_start: int = 400,
    seed: int = 20260912,
) -> dict:
    validate_mirror_protocol(evaluation_seeds, evaluation_start, train_episodes=train_episodes)
    if train_episodes == 0:
        raise ValueError("training exposure must be positive")
    frozen = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=False)
    learned = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=False)
    frozen_exposure = []
    training = []
    for index in range(train_episodes):
        training_seed = 10_000 + index
        frozen_exposure.append(run_episode(frozen, training_seed, learning=False, explore=True))
        training.append(run_episode(learned, training_seed, learning=True, explore=True))
    test_seeds = range(evaluation_start, evaluation_start + evaluation_seeds)
    baseline = [run_episode(frozen, index, learning=False, explore=False) for index in test_seeds]
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
            **usability_gates(summarize(learned_eval)),
            "distance_positive": paired["distance"]["delta_mean"] > 0,
            "return_positive": paired["return"]["delta_mean"] > 0,
            "distance_ci_excludes_zero": distance_interval[0] > 0,
            "return_ci_excludes_zero": return_interval[0] > 0,
            "road_exit_not_worse": (
                summarize(learned_eval)["road_exit_rate"] <= summarize(baseline)["road_exit_rate"]
            ),
            "far_steering_not_worse": (
                summarize(learned_eval)["far_mean_abs_steering"]
                <= summarize(baseline)["far_mean_abs_steering"]
            ),
            "road_exit_rate_at_most_10pct": (summarize(learned_eval)["road_exit_rate"] <= 0.10),
            "far_mean_abs_steering_at_most_0_15": (
                summarize(learned_eval)["far_mean_abs_steering"] <= 0.15
            ),
            "steering_change_not_worse": (
                summarize(learned_eval)["mean_abs_steering_change"]
                <= summarize(baseline)["mean_abs_steering_change"]
            ),
        },
        "plasticity": deployed.policy.summary(),
        "frozen_mirror": mirror_summary(baseline),
        "learned_mirror": mirror_summary(learned_eval),
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
    root: Path,
    *,
    episodes: int = 48,
    evaluation_seeds: int = 32,
    evaluation_start: int = 400,
    seed: int = 20260912,
) -> dict:
    validate_mirror_protocol(evaluation_seeds, evaluation_start, train_episodes=episodes)
    if episodes == 0:
        raise ValueError("calibration exposure must be positive")
    engine = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=False)
    exposure = [
        run_episode(
            engine,
            10_000 + index,
            learning=True,
            explore=True,
            safety_constraints=True,
        )
        for index in range(episodes)
    ]
    checkpoint = root / "artifacts/checkpoints/driving-policy.npz"
    candidate = checkpoint.with_name("driving-policy.calibrated-candidate.npz")
    engine.policy.save(candidate, engine.graph.body_ids, checkpoint_kind="learned_v5")
    deployed = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=False)
    deployed.policy.load(candidate, deployed.graph.body_ids)
    test_seeds = range(evaluation_start, evaluation_start + evaluation_seeds)
    evaluation = [
        run_episode(deployed, value, learning=False, explore=False, safety_constraints=True)
        for value in test_seeds
    ]
    summary = summarize(evaluation)
    mirrored = mirror_summary(evaluation)
    gates = {
        "road_exit_rate_at_most_10pct": summary["road_exit_rate"] <= 0.10,
        "far_mean_abs_steering_at_most_0_15": (summary["far_mean_abs_steering"] <= 0.15),
        # A traversable obstacle course necessarily has turning and recovery.
        # The 0.06 bound stays below the actuator's 0.12 per-step hard limit
        # while not rejecting safe, complete trajectories as "unstable".
        "mean_abs_steering_change_at_most_0_06": (summary["mean_abs_steering_change"] <= 0.06),
        "raw_mirror_error_at_most_1e_6": mirrored["raw_steering_mirror_mae"] <= 1e-6,
        "executed_mirror_error_at_most_1e_6": mirrored["steering_mirror_mae"] <= 1e-6,
        "drive_mirror_error_at_most_1e_6": max(
            item["drive_mirror_mae"] for item in mirrored["details"]
        )
        <= 1e-6,
        "reverse_mirror_error_at_most_1e_6": max(
            item["reverse_mirror_mae"] for item in mirrored["details"]
        )
        <= 1e-6,
        **usability_gates(summary),
    }
    baseline = straight_baseline(evaluation_start, evaluation_seeds)
    gates["distance_exceeds_straight_baseline"] = (
        summary["mean_distance"] > baseline["mean_distance"]
    )
    gates["obstacles_exceed_straight_baseline"] = (
        summary["mean_obstacles_passed"] > baseline["mean_obstacles_passed"]
    )
    published = all(gates.values())
    if published:
        candidate.replace(checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if published else None
    return {
        "calibration_seed_range": [10_000, 10_000 + episodes - 1],
        "test_seed_range": [evaluation_start, evaluation_start + evaluation_seeds - 1],
        "protocol_version": POLICY_VERSION,
        "evaluation_contract": evaluation_contract(root),
        "calibration_learning": True,
        "calibration_explore": True,
        "evaluation_learning": False,
        "evaluation_explore": False,
        "evaluation_safety_constraints": True,
        "checkpoint": str(checkpoint.relative_to(root)) if published else None,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_kind": deployed.policy.checkpoint_kind,
        "candidate_checkpoint": str(candidate.relative_to(root)) if not published else None,
        "candidate_sha256": hashlib.sha256(
            (checkpoint if published else candidate).read_bytes()
        ).hexdigest(),
        "fresh_engine_loaded": True,
        "published": published,
        "gates": gates,
        "deployed": summary,
        "mirror": mirrored,
        "straight_baseline": baseline,
        "interpretation": (
            "Complete held-out mirror pairs; no held-out tuning. Stability alone "
            "does not establish obstacle avoidance. Failed candidates are retained "
            "for reproducibility but are not published as the default policy."
        ),
        "exposure": summarize(exposure),
    }


def straight_baseline(start: int, count: int) -> dict:
    episodes = []
    env = DrivingEnvironment()
    for seed in range(start, start + count):
        env.reset(seed)
        while not env.done:
            env.step(0.0, 0.62)
        episodes.append(
            {
                "seed": seed,
                "distance": env.y,
                "success": env.terminal_reason == "success",
                "obstacles_passed": env.obstacles_passed,
                "first_obstacle_passed": 0 in env.passed_obstacle_indices,
            }
        )
    return {
        "controller": "zero_steering_constant_0_62_throttle",
        "mean_distance": mean(item["distance"] for item in episodes),
        "success_rate": mean(float(item["success"]) for item in episodes),
        "first_obstacle_pass_rate": mean(float(item["first_obstacle_passed"]) for item in episodes),
        "mean_obstacles_passed": mean(item["obstacles_passed"] for item in episodes),
        "details": episodes,
    }


def evaluate_constraints(
    root: Path,
    *,
    evaluation_seeds: int = 32,
    evaluation_start: int = 400,
    seed: int = 20260913,
    checkpoint: Path | None = None,
    reference_report: Path | None = None,
) -> dict:
    validate_mirror_protocol(evaluation_seeds, evaluation_start)
    test_seeds = range(evaluation_start, evaluation_start + evaluation_seeds)
    cached = None
    selected_checkpoint = checkpoint or root / "artifacts/checkpoints/driving-policy.npz"
    checkpoint_hash = hashlib.sha256(selected_checkpoint.read_bytes()).hexdigest()
    if reference_report is not None:
        reference = json.loads(reference_report.read_text())
        if (
            reference.get("protocol_version") != POLICY_VERSION
            or reference.get("evaluation_contract") != evaluation_contract(root)
            or reference.get("candidate_sha256") != checkpoint_hash
            or reference.get("evaluation_learning") is not False
            or reference.get("evaluation_explore") is not False
            or reference.get("evaluation_safety_constraints") is not True
            or reference.get("test_seed_range") != [evaluation_start, test_seeds.stop - 1]
            or [item["seed"] for item in reference["deployed"]["details"]] != list(test_seeds)
        ):
            raise ValueError("reference report does not match checkpoint and evaluation protocol")
        cached = reference["deployed"]["details"]
        for item in cached:
            trace = np.asarray(item["control_trace"], dtype=float)
            if (
                item["pair_seed"] != item["seed"] // 2
                or item["mirror"] != (1 if item["seed"] % 2 == 0 else -1)
                or item["first_obstacle_side"] != ("left" if item["seed"] % 2 == 0 else "right")
                or item["steps"] < 1
                or trace.shape != (item["steps"], 6)
                or not np.isfinite(trace).all()
                or trace[-1, 3] != item["distance"]
            ):
                raise ValueError("reference report contains invalid episode evidence")
    disabled = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=checkpoint is None)
    enabled = DrivingEngine(root, seed=seed, top_k=1, load_checkpoint=checkpoint is None)
    if checkpoint is not None:
        for engine in (disabled, enabled):
            engine.policy.load(checkpoint, engine.graph.body_ids)
    if (
        disabled.policy.checkpoint_kind != "frozen_calibrated"
        or enabled.policy.checkpoint_kind != "frozen_calibrated"
    ):
        raise ValueError("constraint evaluation requires a frozen_calibrated checkpoint")
    without_constraint = [
        run_episode(disabled, value, learning=False, explore=False, safety_constraints=False)
        for value in test_seeds
    ]
    with_constraint = (
        cached
        if cached is not None
        else [
            run_episode(enabled, value, learning=False, explore=False, safety_constraints=True)
            for value in test_seeds
        ]
    )
    return {
        "protocol": {
            "checkpoint_kind": enabled.policy.checkpoint_kind,
            "checkpoint_sha256": hashlib.sha256(
                (checkpoint or enabled.policy_checkpoint).read_bytes()
            ).hexdigest(),
            "test_seed_range": [evaluation_start, evaluation_start + evaluation_seeds - 1],
            "learning": False,
            "explore": False,
            "only_variable": "lane_constraint",
            "evaluation_contract": evaluation_contract(root),
            "constraint_on_reference": str(reference_report) if cached is not None else None,
            "reference_report_sha256": (
                hashlib.sha256(reference_report.read_bytes()).hexdigest()
                if cached is not None
                else None
            ),
        },
        "constraint_off": summarize(without_constraint),
        "constraint_on": summarize(with_constraint),
        "constraint_off_mirror": mirror_summary(without_constraint),
        "constraint_on_mirror": mirror_summary(with_constraint),
        "paired": paired_statistics(without_constraint, with_constraint, seed),
    }
