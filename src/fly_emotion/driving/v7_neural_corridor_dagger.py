from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_heading_ring import EPGPENPEGHeadingRing
from fly_emotion.driving.v7_neural_channels import (
    StructuredNeuralFeatures,
    _predict,
)
from fly_emotion.driving.v7_neural_corridor import (
    _current_mean_features,
    _episode,
    _fit,
    _goal,
    _predict_safety,
    _safety_targets,
)
from fly_emotion.driving.v7_neural_dynamics_local import BackendLocalColumnFeatures
from fly_emotion.driving.v7_neural_episode_cv import (
    _collect_teacher_features,
    _feature_mask,
    _fit_from_cache,
)
from fly_emotion.driving.v7_r1r6_local import (
    local_visual_channels,
    reconstruct_image,
)
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina
from fly_emotion.driving.v7_visual_corridor_goal import obstacle_distance_profile

CONFIG = Path("configs/driving-v7-neural-corridor-dagger.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_corridor_dagger.py")


def _collect_candidate_trajectory(
    root: Path,
    local_features,
    global_features,
    local_model: dict,
    global_model: dict,
    teacher: dict,
    heading_config: dict,
    corridor_config: dict,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    teacher_config = yaml.safe_load((root / "configs/driving-v7-r1r6-local.yaml").read_text())
    retina = build_mass_balanced_retina(root)
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    local_features.reset()
    global_features.reset()
    ring = EPGPENPEGHeadingRing(heading_config)
    decoded_heading = ring.decode()
    command = 0.0
    goal = 0.0
    local_rows, safety_targets = [], []
    global_even_rows, global_odd_rows, global_targets = [], [], []
    while not environment.done:
        local_even, local_odd = local_features.step(image)
        global_even, global_odd = global_features.step(image)
        reconstructed = reconstruct_image(retina, retina.encode(image))
        channels = local_visual_channels(reconstructed, teacher_config)
        profile = obstacle_distance_profile(reconstructed, teacher_config)
        local_rows.append(_current_mean_features(local_features, local_even, local_odd))
        safety_targets.append(_safety_targets(profile, 6))
        global_even_rows.append(global_even)
        global_odd_rows.append(global_odd)
        global_targets.append(
            [channels["danger"], channels["obstacle_asymmetry"], channels["road_centroid"]]
        )
        danger, _asymmetry, road = _predict(global_model, global_even, global_odd)
        safety = _predict_safety(
            local_model, _current_mean_features(local_features, local_even, local_odd)
        )
        if danger >= float(corridor_config["corridor"]["active_danger_threshold"]):
            goal, _changed = _goal(
                safety,
                int(corridor_config["corridor"]["width_bins"]),
                goal,
                float(corridor_config["corridor"]["switch_margin"]),
            )
        else:
            goal *= 0.85
        target = np.tanh(
            float(corridor_config["corridor"]["goal_gain"]) * danger * goal
            + float(corridor_config["fixed_policy"]["road_gain"]) * road
            - float(corridor_config["fixed_policy"]["heading_gain"]) * decoded_heading
        )
        command = (
            float(corridor_config["fixed_policy"]["smoothing_previous"]) * command
            + float(corridor_config["fixed_policy"]["smoothing_current"]) * target
        )
        image, _, _ = environment.step(
            command, float(corridor_config["fixed_policy"]["throttle"]), 0.0
        )
        decoded_heading = ring.step(
            environment.last_yaw_rate, environment.dt, cue_visible=False
        )
    return (
        {"features": np.asarray(local_rows), "targets": np.asarray(safety_targets)},
        {
            "even": np.asarray(global_even_rows),
            "odd": np.asarray(global_odd_rows),
            "targets": np.asarray(global_targets),
        },
        {
            "seed": seed,
            "terminal_reason": environment.terminal_reason,
            "obstacles_passed": environment.obstacles_passed,
            "steps": environment.steps,
        },
    )


def _round_key(result: dict) -> tuple:
    return (
        -result["held_out_success_count"],
        -result["held_out_obstacles_passed"],
        result["aggregation_rounds"],
    )


def evaluate_v7_neural_corridor_dagger(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_path = Path(config["nested_protocol"])
    base_path = Path(config["base_neural_config"])
    local_path = Path(config["local_column_config"])
    corridor_path = Path(config["neural_corridor_config"])
    dynamics_path = Path(config["dynamics_evidence"])
    nested = yaml.safe_load((root / nested_path).read_text())
    base = yaml.safe_load((root / base_path).read_text())
    local = yaml.safe_load((root / local_path).read_text())
    corridor = yaml.safe_load((root / corridor_path).read_text())
    heading = yaml.safe_load((root / corridor["heading_config"]).read_text())
    dynamics = json.loads((root / dynamics_path).read_text())
    if dynamics["selected_backend"] != config["selected_backend"]:
        raise ValueError("DAgger backend differs from frozen dynamics screen")
    teacher = json.loads((root / base["adapter_source"]).read_text())["selected_candidate"]
    pairs = [[int(seed) for seed in pair] for pair in config["tuning_pairs"]]
    protocol_pairs = [
        [int(seed) for seed in item["mirror_pair_seeds"]]
        for item in nested["conditions"]
        if item["role"] == "tuning"
    ]
    if pairs != protocol_pairs:
        raise ValueError("DAgger pairs differ from nested protocol")
    seeds = [seed for pair in pairs for seed in pair]
    backend = config["selected_backend"]
    modified_base = deepcopy(base)
    modified_base["input"]["visual_graph"] = backend
    # Teacher-policy caches provide round zero and are generated for tuning pairs only.
    from fly_emotion.driving.v7_neural_corridor import _collect
    from fly_emotion.driving.v7_neural_dynamics_local import _safety_metrics

    local_features, local_teacher_cache = _collect(
        root,
        modified_base,
        seeds,
        teacher,
        6,
        local_feature_config={**local, "base_neural_config": str(base_path)},
    )
    if backend != base["input"]["visual_graph"]:
        # Use the exact backend-specific cache already frozen by the dynamics screen.
        local_features = BackendLocalColumnFeatures(root, base, local, backend)
        local_teacher_cache = {}
        teacher_config = yaml.safe_load(
            (root / "configs/driving-v7-r1r6-local.yaml").read_text()
        )
        retina = build_mass_balanced_retina(root)
        from fly_emotion.driving.v7_neural_channels import _teacher_step

        for seed in seeds:
            environment = DrivingEnvironment()
            image = environment.reset(seed)
            local_features.reset()
            command = 0.0
            rows, targets = [], []
            while not environment.done:
                even, odd = local_features.step(image)
                rows.append(_current_mean_features(local_features, even, odd))
                reconstructed = reconstruct_image(retina, retina.encode(image))
                targets.append(
                    _safety_targets(
                        obstacle_distance_profile(reconstructed, teacher_config), 6
                    )
                )
                image, command, _channels = _teacher_step(
                    environment, image, command, retina, teacher_config, teacher
                )
            local_teacher_cache[seed] = {
                "features": np.asarray(rows),
                "targets": np.asarray(targets),
            }
    global_features, global_teacher_cache = _collect_teacher_features(
        root, modified_base, seeds, teacher
    )
    global_variant = {
        "name": "current_mean",
        "statistics": ["mean"],
        "temporal_terms": ["current"],
    }
    global_mask = _feature_mask(modified_base, global_features, global_variant)
    round_results = {rounds: [] for rounds in config["aggregation_rounds"]}
    for held_out in pairs:
        train = [seed for seed in seeds if seed not in held_out]
        local_cache = {seed: local_teacher_cache[seed] for seed in train}
        global_cache = {seed: global_teacher_cache[seed] for seed in train}
        fit_keys = list(train)
        for round_index in config["aggregation_rounds"]:
            local_model = _fit(local_cache, fit_keys, float(config["ridge_alpha"]))
            global_model = _fit_from_cache(
                global_features,
                global_cache,
                fit_keys,
                float(config["ridge_alpha"]),
                feature_mask=global_mask,
                feature_variant=global_variant["name"],
            )
            evaluation_episodes = [
                _episode(
                    local_features,
                    global_model,
                    local_model,
                    teacher,
                    heading,
                    corridor,
                    seed,
                    global_features=StructuredNeuralFeatures(root, modified_base),
                )
                for seed in held_out
            ]
            round_results[int(round_index)].append(
                {
                    "training_seeds": train,
                    "held_out_seeds": held_out,
                    "fit_keys": fit_keys.copy(),
                    "safety_metrics_on_teacher_held_out": _safety_metrics(
                        local_model, local_teacher_cache, held_out
                    ),
                    "success_count": sum(item["success"] for item in evaluation_episodes),
                    "total_obstacles_passed": sum(
                        item["obstacles_passed"] for item in evaluation_episodes
                    ),
                    "episodes": evaluation_episodes,
                }
            )
            if round_index == max(config["aggregation_rounds"]):
                continue
            for seed in train:
                key = (int(round_index) + 1) * 100_000 + seed
                local_view = BackendLocalColumnFeatures(root, base, local, backend)
                global_view = StructuredNeuralFeatures(root, modified_base)
                local_item, global_item, _summary = _collect_candidate_trajectory(
                    root,
                    local_view,
                    global_view,
                    local_model,
                    global_model,
                    teacher,
                    heading,
                    corridor,
                    seed,
                )
                local_cache[key] = local_item
                global_cache[key] = global_item
                fit_keys.append(key)
    summaries = []
    for rounds in config["aggregation_rounds"]:
        folds = round_results[int(rounds)]
        summaries.append(
            {
                "aggregation_rounds": int(rounds),
                "held_out_success_count": sum(item["success_count"] for item in folds),
                "held_out_obstacles_passed": sum(
                    item["total_obstacles_passed"] for item in folds
                ),
                "folds": folds,
            }
        )
    selected = min(summaries, key=_round_key)
    passed = selected["held_out_success_count"] == len(seeds) and selected[
        "held_out_obstacles_passed"
    ] == 9 * len(seeds)
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(nested_path): _sha256(root / nested_path),
                str(base_path): _sha256(root / base_path),
                str(local_path): _sha256(root / local_path),
                str(corridor_path): _sha256(root / corridor_path),
                str(dynamics_path): _sha256(root / dynamics_path),
            },
            "tuning_only": True,
            "held_out_pairs_never_labeled_for_fit": True,
            "teacher_actions_injected": False,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "round_summaries": summaries,
        "selected_rounds": selected["aggregation_rounds"],
        "selected_summary": selected,
        "cross_validation_passed": bool(passed),
        "advance_to_full_tuning": bool(passed),
        "advance_to_calibration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
