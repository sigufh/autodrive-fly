from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.driving.sensory import AnatomySensoryProjection
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_neural_channels import StructuredNeuralFeatures
from fly_emotion.driving.v7_neural_corridor import (
    _episode,
    _fit,
)
from fly_emotion.driving.v7_neural_dynamics_local import _safety_metrics
from fly_emotion.driving.v7_neural_episode_cv import (
    _collect_teacher_features,
    _feature_mask,
    _fit_from_cache,
)

CONFIG = Path("configs/driving-v7-visual-layer-locality.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_visual_layer_locality.py")


class VisualLayerFeatures(StructuredNeuralFeatures):
    def __init__(self, root: Path, base: dict, family: str, cell_types: list[str], bins: int):
        super().__init__(root, base)
        table = feather.read_table(
            root / "data/raw/malecns-v1.0/body-annotations.feather",
            columns=["bodyId", "type", "somaSide", "assignedOlHex1"],
            memory_map=True,
        ).to_pandas()
        rows = table[table["type"].isin(cell_types)].copy()
        declared_ids = rows["bodyId"].to_numpy(dtype=np.int64)
        declared_nodes = np.searchsorted(self.probe.graph.body_ids, declared_ids)
        valid_declared = declared_nodes < self.probe.graph.node_count
        valid_declared[valid_declared] &= (
            self.probe.graph.body_ids[declared_nodes[valid_declared]]
            == declared_ids[valid_declared]
        )
        self.declared_target_count = int(len(declared_ids))
        if family == "T4_T5_output":
            projection = AnatomySensoryProjection.from_annotations(
                self.probe.graph.body_ids,
                root / "data/raw/malecns-v1.0/body-annotations.feather",
                adjacency=self.probe.graph.adjacency,
                flow_regions=bins,
            )
            self.groups = [
                np.unique(np.concatenate((positive, negative))).astype(np.int32)
                for positive, negative in zip(
                    projection.flow_positive_bins, projection.flow_negative_bins, strict=True
                )
            ]
            self.unmapped_target_count = int(len(projection.flow_unmapped))
        else:
            located = rows.dropna(subset=["assignedOlHex1"]).copy()
            ids = located["bodyId"].to_numpy(dtype=np.int64)
            nodes = np.searchsorted(self.probe.graph.body_ids, ids)
            valid = nodes < self.probe.graph.node_count
            valid[valid] &= self.probe.graph.body_ids[nodes[valid]] == ids[valid]
            located = located.iloc[np.flatnonzero(valid)]
            nodes = nodes[valid].astype(np.int32)
            sides = located["somaSide"].fillna("").to_numpy(dtype=str)
            coordinate = located["assignedOlHex1"].to_numpy(dtype=np.float64)
            global_u = np.full(len(nodes), np.nan, dtype=np.float64)
            for side in ("L", "R"):
                mask = sides == side
                values = coordinate[mask]
                span = float(values.max() - values.min())
                local = (values - values.min()) / span if span > 0 else np.full(len(values), 0.5)
                global_u[mask] = (1.0 - local) * 0.5 if side == "L" else 0.5 + local * 0.5
            bin_index = np.clip((global_u * bins).astype(np.int32), 0, bins - 1)
            self.groups = [nodes[bin_index == index] for index in range(bins)]
            self.unmapped_target_count = self.declared_target_count - int(len(nodes))
        self.group_names = [f"{family}_{index}" for index in range(bins)]
        self.group_mirror = np.arange(bins - 1, -1, -1, dtype=np.int32)
        self.stat_count = 5
        self.stat_mirror = np.concatenate(
            [np.arange(self.stat_count) + self.stat_count * value for value in self.group_mirror]
        )
        self.bin_counts = [len(group) for group in self.groups]
        if min(self.bin_counts) == 0:
            raise ValueError(f"visual layer {family} has an empty spatial bin")


def _collect_family(
    root: Path, base: dict, seeds: list[int], teacher: dict, family_config: dict, family: str
):
    class FamilyFeatureFactory(VisualLayerFeatures):
        def __init__(self, root_path: Path, _config: dict):
            super().__init__(
                root_path,
                base,
                family,
                list(family_config["families"][family]),
                int(family_config["spatial_bins"]),
            )

    # _collect accepts a file-configured factory, so reproduce its short loop by
    # temporarily using a local class through an explicit collection below.
    from fly_emotion.driving.environment import DrivingEnvironment
    from fly_emotion.driving.v7_neural_channels import _teacher_step
    from fly_emotion.driving.v7_neural_corridor import (
        _current_mean_features,
        _safety_targets,
    )
    from fly_emotion.driving.v7_r1r6_local import reconstruct_image
    from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina
    from fly_emotion.driving.v7_visual_corridor_goal import obstacle_distance_profile

    features = FamilyFeatureFactory(root, base)
    retina = build_mass_balanced_retina(root)
    teacher_config = yaml.safe_load((root / "configs/driving-v7-r1r6-local.yaml").read_text())
    cache = {}
    for seed in seeds:
        environment = DrivingEnvironment()
        image = environment.reset(seed)
        features.reset()
        command = 0.0
        rows, targets = [], []
        while not environment.done:
            even, odd = features.step(image)
            rows.append(_current_mean_features(features, even, odd))
            reconstructed = reconstruct_image(retina, retina.encode(image))
            targets.append(
                _safety_targets(
                    obstacle_distance_profile(reconstructed, teacher_config), 6
                )
            )
            image, command, _channels = _teacher_step(
                environment, image, command, retina, teacher_config, teacher
            )
        cache[seed] = {"features": np.asarray(rows), "targets": np.asarray(targets)}
    return features, cache


def _key(result: dict) -> tuple:
    return (
        result["held_out_safety_mae"],
        -result["held_out_safety_mean_r2"],
        result["unmapped_fraction"],
        result["family"],
    )


def evaluate_v7_visual_layer_locality(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested_path = Path(config["nested_protocol"])
    base_path = Path(config["base_neural_config"])
    corridor_path = Path(config["neural_corridor_config"])
    nested = yaml.safe_load((root / nested_path).read_text())
    base = yaml.safe_load((root / base_path).read_text())
    base = deepcopy(base)
    base["input"]["visual_graph"] = config["dynamics_backend"]
    corridor = yaml.safe_load((root / corridor_path).read_text())
    heading = yaml.safe_load((root / corridor["heading_config"]).read_text())
    teacher = json.loads((root / base["adapter_source"]).read_text())["selected_candidate"]
    pairs = [[int(seed) for seed in pair] for pair in config["tuning_pairs"]]
    protocol_pairs = [
        [int(seed) for seed in item["mirror_pair_seeds"]]
        for item in nested["conditions"]
        if item["role"] == "tuning"
    ]
    if pairs != protocol_pairs:
        raise ValueError("visual layer locality pairs differ from nested protocol")
    seeds = [seed for pair in pairs for seed in pair]
    results, caches, feature_views = [], {}, {}
    for family in config["families"]:
        features, cache = _collect_family(root, base, seeds, teacher, config, family)
        folds = []
        for held_out in pairs:
            train = [seed for seed in seeds if seed not in held_out]
            model = _fit(cache, train, float(config["readout"]["ridge_alpha"]))
            folds.append(
                {
                    "training_seeds": train,
                    "held_out_seeds": held_out,
                    "model": model,
                    **_safety_metrics(model, cache, held_out),
                }
            )
        result = {
            "family": family,
            "declared_target_count": features.declared_target_count,
            "unmapped_target_count": features.unmapped_target_count,
            "unmapped_fraction": features.unmapped_target_count
            / features.declared_target_count,
            "minimum_bin_count": min(features.bin_counts),
            "maximum_bin_count": max(features.bin_counts),
            "held_out_safety_mae": float(
                np.mean([item["mean_absolute_error"] for item in folds])
            ),
            "held_out_safety_mean_r2": float(
                np.mean([item["mean_r2"] for item in folds])
            ),
            "folds": folds,
        }
        results.append(result)
        caches[family] = cache
        feature_views[family] = features
    selected = min(results, key=_key)
    family = selected["family"]
    global_features, global_cache = _collect_teacher_features(root, base, seeds, teacher)
    global_variant = {
        "name": "current_mean",
        "statistics": ["mean"],
        "temporal_terms": ["current"],
    }
    global_mask = _feature_mask(base, global_features, global_variant)
    closed_loop = []
    for held_out in pairs:
        train = [seed for seed in seeds if seed not in held_out]
        safety_model = _fit(
            caches[family], train, float(config["readout"]["ridge_alpha"])
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
                feature_views[family],
                global_model,
                safety_model,
                teacher,
                heading,
                corridor,
                seed,
                global_features=StructuredNeuralFeatures(root, base),
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
                str(corridor_path): _sha256(root / corridor_path),
            },
            "tuning_only": True,
            "direct_R1_R6_readout_used": False,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "family_results": results,
        "selected_family": family,
        "selected_family_result": selected,
        "closed_loop_folds": closed_loop,
        "closed_loop_success_count": successes,
        "closed_loop_obstacles_passed": obstacles,
        "cross_validation_passed": bool(passed),
        "advance_to_full_tuning": bool(passed),
        "advance_to_calibration": False,
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
