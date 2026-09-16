from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_heading_ring import _heading_episode
from fly_emotion.driving.v7_neural_channels import StructuredNeuralFeatures, _model_digest

CONFIG = Path("configs/driving-v7-neural-episode-cv.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_neural_episode_controls.py")
CANDIDATE_ARTIFACT = Path("artifacts/v7-neural-episode-cv.json")
CONTROL_SEED = 20260916


def _summary(episodes: list[dict]) -> dict:
    return {
        "success_count": sum(item["success"] for item in episodes),
        "total_obstacles_passed": sum(item["obstacles_passed"] for item in episodes),
        "terminal_counts": {
            reason: sum(item["terminal_reason"] == reason for item in episodes)
            for reason in ("success", "obstacle", "road_boundary", "timeout")
        },
        "episodes": [
            {key: value for key, value in item.items() if key != "trace"}
            for item in episodes
        ],
    }


def evaluate_v7_neural_episode_controls(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    base_path = Path(config["base_neural_config"])
    heading_path = Path(config["heading_config"])
    teacher_path = Path(config["teacher_tuning_evidence"])
    base = yaml.safe_load((root / base_path).read_text())
    heading = yaml.safe_load((root / heading_path).read_text())
    candidate = json.loads((root / CANDIDATE_ARTIFACT).read_text())
    teacher = json.loads((root / teacher_path).read_text())["selected_candidate"]
    if not candidate["cross_validation_passed"] or not candidate["calibration_passed"]:
        raise ValueError("controls require a frozen CV and calibration passed candidate")
    if _model_digest(candidate["model"]) != candidate["model_sha256"]:
        raise ValueError("frozen episode-CV model changed")
    if candidate["selected_feature_variant"] != "current_mean":
        raise ValueError("unexpected episode-CV feature candidate")
    seeds = [seed for pair in config["tuning_mirror_pairs"] for seed in pair]
    spatial = {
        name
        for name in candidate["model"]["group_names"]
        if name.startswith(("front_to_back", "back_to_front"))
    }
    arms = {
        "full": {"groups": set(), "heading_arm": "neural_heading"},
        "no_T4_T5_spatial": {"groups": spatial, "heading_arm": "neural_heading"},
        "no_LPLC1": {
            "groups": {"LPLC1_L", "LPLC1_R"},
            "heading_arm": "neural_heading",
        },
        "no_LPLC2": {
            "groups": {"LPLC2_L", "LPLC2_R"},
            "heading_arm": "neural_heading",
        },
        "no_LC4": {
            "groups": {"LC4_L", "LC4_R"},
            "heading_arm": "neural_heading",
        },
        "no_all_LPLC_LC": {
            "groups": {
                "LPLC1_L",
                "LPLC1_R",
                "LPLC2_L",
                "LPLC2_R",
                "LC4_L",
                "LC4_R",
            },
            "heading_arm": "neural_heading",
        },
        "frozen_heading": {"groups": set(), "heading_arm": "frozen_heading"},
        "reversed_yaw_update": {
            "groups": set(),
            "heading_arm": "reversed_yaw_update",
        },
    }
    ablations = {}
    for name, arm in arms.items():
        features = StructuredNeuralFeatures(root, base)
        episodes = [
            _heading_episode(
                features,
                candidate["model"],
                teacher,
                heading,
                seed,
                arm["heading_arm"],
                ablated_groups=arm["groups"],
            )
            for seed in seeds
        ]
        ablations[name] = _summary(episodes)
    topology_controls = {}
    for control in (
        "real_malecns",
        "shuffled_retina_coordinates",
        "source_preserving_target_shuffle",
        "shuffled_transmitter_signs",
    ):
        features = StructuredNeuralFeatures(
            root, base, topology_control=control, control_seed=CONTROL_SEED
        )
        episodes = [
            _heading_episode(
                features,
                candidate["model"],
                teacher,
                heading,
                seed,
                "neural_heading",
            )
            for seed in seeds
        ]
        topology_controls[control] = _summary(episodes)
    full = ablations["full"]
    real = topology_controls["real_malecns"]
    topology_gates = {
        "real_replays_all_successes": real["success_count"] == len(seeds),
        "retina_shuffle_reduces_success": topology_controls[
            "shuffled_retina_coordinates"
        ]["success_count"]
        < real["success_count"],
        "target_shuffle_reduces_success": topology_controls[
            "source_preserving_target_shuffle"
        ]["success_count"]
        < real["success_count"],
        "transmitter_shuffle_reduces_success": topology_controls[
            "shuffled_transmitter_signs"
        ]["success_count"]
        < real["success_count"],
    }
    causal_gates = {
        "T4_T5_spatial_is_causal": ablations["no_T4_T5_spatial"]["success_count"]
        < full["success_count"],
        "all_LPLC_LC_is_causal": ablations["no_all_LPLC_LC"]["success_count"]
        < full["success_count"],
        "heading_is_causal": ablations["frozen_heading"]["success_count"]
        < full["success_count"],
        "yaw_sign_is_causal": ablations["reversed_yaw_update"]["success_count"]
        < full["success_count"],
    }
    calibration_seeds = [int(seed) for seed in config["calibration"]["mirror_pair_seeds"]]
    calibration_ablations = {}
    for name, arm in arms.items():
        features = StructuredNeuralFeatures(root, base)
        episodes = [
            _heading_episode(
                features,
                candidate["model"],
                teacher,
                heading,
                seed,
                arm["heading_arm"],
                ablated_groups=arm["groups"],
            )
            for seed in calibration_seeds
        ]
        calibration_ablations[name] = _summary(episodes)
    calibration_topology_controls = {}
    for control in topology_controls:
        features = StructuredNeuralFeatures(
            root, base, topology_control=control, control_seed=CONTROL_SEED
        )
        episodes = [
            _heading_episode(
                features,
                candidate["model"],
                teacher,
                heading,
                seed,
                "neural_heading",
            )
            for seed in calibration_seeds
        ]
        calibration_topology_controls[control] = _summary(episodes)
    calibration_full = calibration_ablations["full"]
    all_lplc_lc_collectively_redundant = (
        ablations["no_all_LPLC_LC"]["success_count"] == full["success_count"]
        and calibration_ablations["no_all_LPLC_LC"]["success_count"]
        == calibration_full["success_count"]
    )
    causal_gates["all_LPLC_LC_is_causal"] = (
        ablations["no_all_LPLC_LC"]["success_count"] < full["success_count"]
        or calibration_ablations["no_all_LPLC_LC"]["success_count"]
        < calibration_full["success_count"]
    )
    for control in (
        "shuffled_retina_coordinates",
        "source_preserving_target_shuffle",
        "shuffled_transmitter_signs",
    ):
        topology_gates[f"{control}_reduces_calibration_success"] = (
            calibration_topology_controls[control]["success_count"]
            < calibration_topology_controls["real_malecns"]["success_count"]
        )
    return {
        "protocol": {
            "name": "v7-neural-current-mean-causal-controls",
            "phase": "post_calibration_frozen_controls",
            "control_seed": CONTROL_SEED,
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(CANDIDATE_ARTIFACT): _sha256(root / CANDIDATE_ARTIFACT),
                str(base_path): _sha256(root / base_path),
                str(heading_path): _sha256(root / heading_path),
            },
            "model_sha256": candidate["model_sha256"],
            "model_changed_after_calibration": False,
            "controls_used_for_selection": False,
            "runtime_modified": False,
            "external_final_evaluated": False,
        },
        "seeds": seeds,
        "ablations": ablations,
        "ablation_interpretation": {
            name: (
                "necessary_or_synergistic"
                if result["success_count"] < full["success_count"]
                else "not_necessary_under_this_tuning_screen"
            )
            for name, result in ablations.items()
            if name != "full"
        },
        "topology_controls": topology_controls,
        "calibration_ablations": calibration_ablations,
        "calibration_topology_controls": calibration_topology_controls,
        "all_LPLC_LC_collectively_redundant_under_frozen_candidate": bool(
            all_lplc_lc_collectively_redundant
        ),
        "causal_gates": causal_gates,
        "topology_gates": topology_gates,
        "causal_controls_passed": all(causal_gates.values()),
        "real_topology_advantage_passed": all(topology_gates.values()),
        "advance_to_fc2_pfl_comparison": all(causal_gates.values())
        and all(topology_gates.values()),
        "advance_to_navigation_release": False,
    }
