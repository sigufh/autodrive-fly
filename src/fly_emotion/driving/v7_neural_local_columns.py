from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.sensory import AnatomySensoryProjection
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_heading_ring import _heading_episode
from fly_emotion.driving.v7_neural_channels import (
    StructuredNeuralFeatures,
    _mirror_checks,
    _model_digest,
    _neural_episode,
    _teacher_step,
)
from fly_emotion.driving.v7_r1r6_local import _episode as _r1r6_local_episode
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina

CONFIG = Path("configs/driving-v7-neural-local-columns.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_local_columns.py")


class LocalColumnNeuralFeatures(StructuredNeuralFeatures):
    def __init__(self, root: Path, config: dict):
        base_path = Path(config["base_neural_config"])
        base = yaml.safe_load((root / base_path).read_text())
        super().__init__(root, base)
        bins = int(config["features"]["T4_T5_spatial_bins"])
        raw = root / "data/raw/malecns-v1.0/body-annotations.feather"
        projection = AnatomySensoryProjection.from_annotations(
            self.probe.graph.body_ids,
            raw,
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
        for name in config["features"]["looming_populations"]:
            self.groups.append(self.probe.populations[name])
            self.group_names.append(name)
        self.unmapped_flow_count = int(len(projection.flow_unmapped))
        self.motion_bin_counts = [len(group) for group in self.groups[: 2 * bins]]
        minimum = int(config["features"]["minimum_cells_per_motion_direction_bin"])
        if min(self.motion_bin_counts) < minimum:
            raise ValueError("local neural motion bin lacks minimum target coverage")
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
        self.stat_count = len(config["features"]["per_group_statistics"])
        self.stat_mirror = np.concatenate(
            [np.arange(self.stat_count) + self.stat_count * value for value in self.group_mirror]
        )
        self.feature_group_names = [
            name
            for _temporal in config["features"]["temporal_terms"]
            for name in self.group_names
            for _statistic in config["features"]["per_group_statistics"]
        ]
        self.config = {
            **base,
            "features": {
                **base["features"],
                "exponential_mean_update": config["features"][
                    "exponential_mean_update"
                ],
            },
        }


def _fit_local_model(root: Path, config: dict) -> tuple[LocalColumnNeuralFeatures, dict]:
    teacher_report = json.loads((root / config["teacher_tuning_evidence"]).read_text())
    teacher = teacher_report["selected_candidate"]
    teacher_config = yaml.safe_load((root / "configs/driving-v7-r1r6-local.yaml").read_text())
    retina = build_mass_balanced_retina(root)
    features = LocalColumnNeuralFeatures(root, config)
    even_rows, odd_rows, targets = [], [], []
    for seed in config["tuning"]["mirror_pair_seeds"]:
        environment = DrivingEnvironment()
        image = environment.reset(int(seed))
        features.reset()
        command = 0.0
        while not environment.done:
            even, odd = features.step(image)
            image, command, channels = _teacher_step(
                environment, image, command, retina, teacher_config, teacher
            )
            even_rows.append(even)
            odd_rows.append(odd)
            targets.append(
                [channels["danger"], channels["obstacle_asymmetry"], channels["road_centroid"]]
            )
    even_matrix = np.asarray(even_rows)
    odd_matrix = np.asarray(odd_rows)
    target_matrix = np.asarray(targets)
    alpha = float(config["readout"]["alpha"])
    coefficients, scales, fit_r2 = [], [], []
    for output_index, matrix in enumerate((even_matrix, odd_matrix, odd_matrix)):
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale[scale < 1e-7] = 1.0
        normalized = (matrix - mean) / scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        penalty = np.diag([0.0] + [alpha] * normalized.shape[1])
        coefficient = np.linalg.solve(
            design.T @ design + penalty, design.T @ target_matrix[:, output_index]
        )
        prediction = design @ coefficient
        denominator = np.sum(
            (target_matrix[:, output_index] - np.mean(target_matrix[:, output_index])) ** 2
        )
        fit_r2.append(
            float(
                1.0
                - np.sum((target_matrix[:, output_index] - prediction) ** 2)
                / max(float(denominator), 1e-12)
            )
        )
        coefficients.append(coefficient.tolist())
        scales.append({"mean": mean.tolist(), "scale": scale.tolist()})
    return features, {
        "output_names": config["readout"]["outputs"],
        "alpha": alpha,
        "coefficients": coefficients,
        "scales": scales,
        "fit_r2": fit_r2,
        "group_names": features.group_names,
        "group_mirror": features.group_mirror.tolist(),
        "feature_dimension_per_parity": int(even_matrix.shape[1]),
        "training_samples": int(len(target_matrix)),
    }


def evaluate_v7_neural_local_columns(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    heading_path = Path(config["heading_config"])
    heading_config = yaml.safe_load((root / heading_path).read_text())
    features, model = _fit_local_model(root, config)
    teacher = json.loads((root / config["teacher_tuning_evidence"]).read_text())[
        "selected_candidate"
    ]
    tuning = [
        _heading_episode(features, model, teacher, heading_config, int(seed), "neural_heading")
        for seed in config["tuning"]["mirror_pair_seeds"]
    ]
    tuning_passed = all(item["success"] and item["obstacles_passed"] == 9 for item in tuning)
    calibration = (
        [
            _heading_episode(
                features, model, teacher, heading_config, int(seed), "neural_heading"
            )
            for seed in config["calibration"]["mirror_pair_seeds"]
        ]
        if tuning_passed
        else []
    )
    calibration_passed = bool(calibration) and all(
        item["success"] and item["obstacles_passed"] == 9 for item in calibration
    )
    calibration_seeds = [int(seed) for seed in config["calibration"]["mirror_pair_seeds"]]
    base_path = Path(config["base_neural_config"])
    base_config = yaml.safe_load((root / base_path).read_text())
    base_tuning = json.loads((root / "artifacts/v7-neural-channels-tuning.json").read_text())
    base_features = StructuredNeuralFeatures(root, base_config)
    base_episodes = [
        _neural_episode(base_features, base_tuning["model"], teacher, seed)
        for seed in calibration_seeds
    ]
    r1r6_config = yaml.safe_load((root / "configs/driving-v7-r1r6-local.yaml").read_text())
    r1r6_episodes = [
        _r1r6_local_episode(
            build_mass_balanced_retina(root), r1r6_config, teacher, seed
        )
        for seed in calibration_seeds
    ]
    attribution = {
        "role": "post_failure_attribution_only_not_parameter_selection",
        "six_bin_reference": [
            {key: value for key, value in item.items() if key != "trace"}
            for item in base_episodes
        ],
        "r1r6_local_upper_bound": [
            {key: value for key, value in item.items() if key != "trace"}
            for item in r1r6_episodes
        ],
        "six_bin_reference_passes_both": all(
            item["success"] and item["obstacles_passed"] == 9 for item in base_episodes
        ),
        "r1r6_local_upper_bound_passes_both": all(
            item["success"] and item["obstacles_passed"] == 9 for item in r1r6_episodes
        ),
        "diagnosis": "24_bin_tuning_overfit",
    }
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(heading_path): _sha256(root / heading_path),
                config["teacher_tuning_evidence"]: _sha256(
                    root / config["teacher_tuning_evidence"]
                ),
            },
            "visual_input_only_via_R1_R6": True,
            "target_activity_injection": False,
            "raw_heading_read_by_controller": False,
            "runtime_modified": False,
            "external_final_evaluated": False,
        },
        "structure": {
            "motion_spatial_bins": int(config["features"]["T4_T5_spatial_bins"]),
            "motion_bin_counts": features.motion_bin_counts,
            "minimum_motion_bin_count": min(features.motion_bin_counts),
            "unmapped_flow_target_count": features.unmapped_flow_count,
            "looming_populations": config["features"]["looming_populations"],
        },
        "model": model,
        "model_sha256": _model_digest(model),
        "tuning_episodes": [
            {key: value for key, value in item.items() if key != "trace"} for item in tuning
        ],
        "tuning_mirror_checks": _mirror_checks(tuning),
        "tuning_passed": bool(tuning_passed),
        "calibration_episodes": [
            {key: value for key, value in item.items() if key != "trace"}
            for item in calibration
        ],
        "calibration_mirror_checks": _mirror_checks(calibration) if calibration else [],
        "calibration_passed": bool(calibration_passed),
        "calibration_failure_attribution": attribution,
        "advance_to_topology_controls": bool(calibration_passed),
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
