from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.sensory import AnatomySensoryProjection
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_neural_channels import StructuredNeuralFeatures
from fly_emotion.driving.v7_neural_corridor import (
    _collect,
    _episode,
    _fit,
    _predict_safety,
)
from fly_emotion.driving.v7_neural_episode_cv import (
    _collect_teacher_features,
    _feature_mask,
    _fit_from_cache,
)

CONFIG = Path("configs/driving-v7-neural-dynamics-local.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_dynamics_local.py")


class BackendLocalColumnFeatures(StructuredNeuralFeatures):
    def __init__(self, root: Path, base: dict, local: dict, backend: str):
        modified = deepcopy(base)
        modified["input"]["visual_graph"] = backend
        super().__init__(root, modified)
        bins = int(local["features"]["T4_T5_spatial_bins"])
        projection = AnatomySensoryProjection.from_annotations(
            self.probe.graph.body_ids,
            root / "data/raw/malecns-v1.0/body-annotations.feather",
            adjacency=self.probe.graph.adjacency,
            flow_regions=bins,
        )
        self.groups = []
        self.group_names = []
        for region, (positive, negative) in enumerate(
            zip(projection.flow_positive_bins, projection.flow_negative_bins, strict=True)
        ):
            self.groups.extend((positive, negative))
            self.group_names.extend(
                (f"front_to_back_{region}", f"back_to_front_{region}")
            )
        for name in local["features"]["looming_populations"]:
            self.groups.append(self.probe.populations[name])
            self.group_names.append(name)
        self.motion_bin_counts = [len(group) for group in self.groups[: 2 * bins]]
        self.unmapped_flow_count = int(len(projection.flow_unmapped))
        index = {name: position for position, name in enumerate(self.group_names)}
        mirror = []
        for name in self.group_names:
            if name.startswith("front_to_back_"):
                region = int(name.rsplit("_", 1)[1])
                counterpart = f"back_to_front_{bins - 1 - region}"
            elif name.startswith("back_to_front_"):
                region = int(name.rsplit("_", 1)[1])
                counterpart = f"front_to_back_{bins - 1 - region}"
            else:
                counterpart = name[:-1] + ("R" if name.endswith("L") else "L")
            mirror.append(index[counterpart])
        self.group_mirror = np.asarray(mirror, dtype=np.int32)
        self.stat_count = len(local["features"]["per_group_statistics"])
        self.stat_mirror = np.concatenate(
            [np.arange(self.stat_count) + self.stat_count * value for value in self.group_mirror]
        )


def _safety_metrics(model: dict, cache: dict, held_out: list[int]) -> dict:
    targets = np.concatenate([cache[seed]["targets"] for seed in held_out])
    predictions = np.concatenate(
        [
            np.stack([_predict_safety(model, row) for row in cache[seed]["features"]])
            for seed in held_out
        ]
    )
    errors = targets - predictions
    denominator = np.sum((targets - np.mean(targets, axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum(errors**2, axis=0) / np.maximum(denominator, 1e-12)
    return {
        "mean_absolute_error": float(np.mean(np.abs(errors))),
        "per_bin_r2": r2.tolist(),
        "mean_r2": float(np.mean(r2)),
    }


def _key(result: dict) -> tuple:
    return (
        result["held_out_safety_mae"],
        -result["held_out_safety_mean_r2"],
        result["backend"],
    )


def evaluate_v7_neural_dynamics_local(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_path = Path(config["nested_protocol"])
    base_path = Path(config["base_neural_config"])
    local_path = Path(config["local_column_config"])
    corridor_path = Path(config["neural_corridor_config"])
    nested = yaml.safe_load((root / nested_path).read_text())
    base = yaml.safe_load((root / base_path).read_text())
    local = yaml.safe_load((root / local_path).read_text())
    corridor = yaml.safe_load((root / corridor_path).read_text())
    heading = yaml.safe_load((root / corridor["heading_config"]).read_text())
    teacher = __import__("json").loads((root / base["adapter_source"]).read_text())[
        "selected_candidate"
    ]
    pairs = [[int(seed) for seed in pair] for pair in config["tuning_pairs"]]
    protocol_pairs = [
        [int(seed) for seed in item["mirror_pair_seeds"]]
        for item in nested["conditions"]
        if item["role"] == "tuning"
    ]
    if pairs != protocol_pairs:
        raise ValueError("dynamics screen pairs differ from nested protocol")
    seeds = [seed for pair in pairs for seed in pair]
    results = []
    caches = {}
    feature_views = {}
    for backend in config["dynamics_backends"]:
        modified_base = deepcopy(base)
        modified_base["input"]["visual_graph"] = backend
        local_features = BackendLocalColumnFeatures(root, base, local, backend)
        # Reuse the common collection loop by presenting the prebuilt local feature view.
        _, local_cache = _collect(
            root,
            modified_base,
            seeds,
            teacher,
            int(corridor["targets"]["local_safety_bins"]),
            local_feature_config={**local, "base_neural_config": str(base_path)},
        )
        # _collect cannot override the backend through the file-backed local config;
        # replace it with traces from the backend-specific view when they differ.
        if backend != base["input"]["visual_graph"]:
            from fly_emotion.driving.environment import DrivingEnvironment
            from fly_emotion.driving.v7_neural_channels import _teacher_step
            from fly_emotion.driving.v7_neural_corridor import _current_mean_features
            from fly_emotion.driving.v7_r1r6_local import reconstruct_image
            from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina
            from fly_emotion.driving.v7_visual_corridor_goal import obstacle_distance_profile

            teacher_config = yaml.safe_load(
                (root / "configs/driving-v7-r1r6-local.yaml").read_text()
            )
            retina = build_mass_balanced_retina(root)
            local_cache = {}
            for seed in seeds:
                environment = DrivingEnvironment()
                image = environment.reset(seed)
                local_features.reset()
                command = 0.0
                rows, targets = [], []
                while not environment.done:
                    even, odd = local_features.step(image)
                    rows.append(_current_mean_features(local_features, even, odd))
                    profile = obstacle_distance_profile(
                        reconstruct_image(retina, retina.encode(image)), teacher_config
                    )
                    image, command, _channels = _teacher_step(
                        environment, image, command, retina, teacher_config, teacher
                    )
                    targets.append(
                        np.mean(
                            profile.reshape(6, 8),
                            axis=1,
                        )
                    )
                local_cache[seed] = {
                    "features": np.asarray(rows),
                    "targets": np.asarray(targets),
                }
        fold_metrics = []
        for held_out in pairs:
            train = [seed for seed in seeds if seed not in held_out]
            model = _fit(local_cache, train, float(config["readout"]["ridge_alpha"]))
            fold_metrics.append(
                {
                    "training_seeds": train,
                    "held_out_seeds": held_out,
                    "model": model,
                    **_safety_metrics(model, local_cache, held_out),
                }
            )
        result = {
            "backend": backend,
            "held_out_safety_mae": float(
                np.mean([item["mean_absolute_error"] for item in fold_metrics])
            ),
            "held_out_safety_mean_r2": float(
                np.mean([item["mean_r2"] for item in fold_metrics])
            ),
            "folds": fold_metrics,
        }
        results.append(result)
        caches[backend] = local_cache
        feature_views[backend] = local_features
    selected = min(results, key=_key)
    backend = selected["backend"]
    modified_base = deepcopy(base)
    modified_base["input"]["visual_graph"] = backend
    global_features, global_cache = _collect_teacher_features(root, modified_base, seeds, teacher)
    global_variant = {
        "name": "current_mean",
        "statistics": ["mean"],
        "temporal_terms": ["current"],
    }
    global_mask = _feature_mask(modified_base, global_features, global_variant)
    closed_loop = []
    for held_out in pairs:
        train = [seed for seed in seeds if seed not in held_out]
        safety_model = _fit(
            caches[backend], train, float(config["readout"]["ridge_alpha"])
        )
        global_model = _fit_from_cache(
            global_features,
            global_cache,
            train,
            1.0,
            feature_mask=global_mask,
            feature_variant=global_variant["name"],
        )
        episodes = [
            _episode(
                feature_views[backend],
                global_model,
                safety_model,
                teacher,
                heading,
                corridor,
                seed,
                global_features=StructuredNeuralFeatures(root, modified_base),
            )
            for seed in held_out
        ]
        closed_loop.append(
            {
                "held_out_seeds": held_out,
                "success_count": sum(item["success"] for item in episodes),
                "total_obstacles_passed": sum(
                    item["obstacles_passed"] for item in episodes
                ),
                "episodes": episodes,
            }
        )
    successes = sum(item["success_count"] for item in closed_loop)
    obstacles = sum(item["total_obstacles_passed"] for item in closed_loop)
    passed = successes == len(seeds) and obstacles == 9 * len(seeds)
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
            },
            "tuning_only": True,
            "diagnostic_two_state_views": True,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "backend_results": results,
        "selected_backend": backend,
        "selected_backend_result": selected,
        "closed_loop_folds": closed_loop,
        "closed_loop_success_count": successes,
        "closed_loop_obstacles_passed": obstacles,
        "cross_validation_passed": bool(passed),
        "advance_to_single_state_refactor": bool(passed),
        "advance_to_calibration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
