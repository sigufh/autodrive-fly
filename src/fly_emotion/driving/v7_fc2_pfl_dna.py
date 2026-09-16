from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.environment import DrivingEnvironment
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_heading_ring import EPGPENPEGHeadingRing, _heading_episode
from fly_emotion.driving.v7_neural_channels import StructuredNeuralFeatures, _model_digest, _predict
from fly_emotion.driving.v7_r1r6_local import _episode as _r1r6_local_episode
from fly_emotion.driving.v7_r1r6_multi import build_mass_balanced_retina

CONFIG = Path("configs/driving-v7-fc2-pfl-dna.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_fc2_pfl_dna.py")


def _intent(
    visual_target_signal: float,
    epg_heading: float,
    heading_gain: float,
    readout: str,
) -> dict:
    fc2_goal_heading = float(visual_target_signal / heading_gain)
    heading_error = fc2_goal_heading - epg_heading
    pfl3_left = float(max(heading_error, 0.0))
    pfl3_right = float(max(-heading_error, 0.0))
    pfl2_magnitude = float(abs(heading_error))
    dna02_left = pfl3_right
    dna02_right = pfl3_left
    if readout == "hierarchical_fc2_pfl3_dna02":
        motor_intent = np.tanh(heading_gain * (dna02_right - dna02_left))
    elif readout in {"direct_dna01_dna02", "direct_dnp20"}:
        motor_intent = np.tanh(visual_target_signal - heading_gain * epg_heading)
    else:
        raise ValueError(f"unknown descending readout: {readout}")
    return {
        "fc2_goal_heading": fc2_goal_heading,
        "heading_error": heading_error,
        "pfl3_left": pfl3_left,
        "pfl3_right": pfl3_right,
        "pfl2_magnitude": pfl2_magnitude,
        "dna02_left": dna02_left,
        "dna02_right": dna02_right,
        "motor_intent": float(motor_intent),
    }


def _episode(features, model, teacher: dict, heading_config: dict, seed: int, arm: str) -> dict:
    environment = DrivingEnvironment()
    image = environment.reset(seed)
    features.reset()
    ring = EPGPENPEGHeadingRing(heading_config)
    decoded_heading = ring.decode()
    command = 0.0
    trace = []
    for _step in range(environment.max_steps):
        if environment.done:
            break
        even, odd = features.step(image)
        danger, asymmetry, road = _predict(model, even, odd)
        visual_signal = (
            float(teacher["obstacle_gain"]) * danger * asymmetry
            + float(teacher["road_gain"]) * road
        )
        heading_value = decoded_heading
        if arm == "no_fc2_goal":
            visual_signal = 0.0
        elif arm == "no_epg_heading":
            heading_value = 0.0
        readout = (
            arm
            if arm in {
                "hierarchical_fc2_pfl3_dna02",
                "direct_dna01_dna02",
                "direct_dnp20",
            }
            else "hierarchical_fc2_pfl3_dna02"
        )
        values = _intent(visual_signal, heading_value, float(teacher["heading_gain"]), readout)
        if arm in {"no_pfl3", "no_dna02"}:
            values["motor_intent"] = 0.0
        command = 0.4 * command + 0.6 * values["motor_intent"]
        image, _, _ = environment.step(command, float(teacher["throttle"]), 0.0)
        decoded_heading = ring.step(
            environment.last_yaw_rate, environment.dt, cue_visible=False
        )
        trace.append(
            {
                **values,
                "danger": danger,
                "obstacle_asymmetry": asymmetry,
                "road_center": road,
                "decoded_heading": decoded_heading,
                "steering_command": command,
                "vehicle_x": environment.x,
                "vehicle_y": environment.y,
            }
        )
    return {
        "seed": seed,
        "terminal_reason": environment.terminal_reason,
        "success": environment.terminal_reason == "success",
        "obstacles_passed": environment.obstacles_passed,
        "distance": environment.y,
        "steps": environment.steps,
        "maximum_absolute_lateral_position": float(
            max(abs(item["vehicle_x"]) for item in trace)
        ),
        "trace_sha256": hashlib.sha256(
            np.asarray(
                [
                    [
                        item["motor_intent"],
                        item["steering_command"],
                        item["vehicle_x"],
                        item["vehicle_y"],
                    ]
                    for item in trace
                ],
                dtype="<f8",
            ).tobytes()
        ).hexdigest(),
    }


def _turn_termination(config: dict, heading_config: dict) -> dict:
    assay = config["turn_termination"]
    records = []
    mirror_errors = []
    for initial in assay["initial_heading_radians"]:
        for target in assay["target_heading_radians"]:
            ring = EPGPENPEGHeadingRing(heading_config)
            ring.reset(float(initial))
            previous_magnitude = abs(float(target) - float(initial))
            monotone = True
            steps_taken = 0
            for _step in range(int(assay["maximum_steps"])):
                steps_taken += 1
                error = float(target) - ring.decode()
                pfl3_left = max(error, 0.0)
                pfl3_right = max(-error, 0.0)
                pfl2 = abs(error)
                if pfl2 <= float(assay["error_threshold"]):
                    break
                yaw = np.tanh(pfl3_left - pfl3_right)
                ring.step(float(yaw), float(assay["update_gain"]))
                magnitude = abs(float(target) - ring.decode())
                monotone &= magnitude <= previous_magnitude + 1e-12
                previous_magnitude = magnitude
            final_error = abs(float(target) - ring.decode())
            records.append(
                {
                    "initial_heading": float(initial),
                    "target_heading": float(target),
                    "steps": steps_taken,
                    "final_error": final_error,
                    "monotone_error_reduction": bool(monotone),
                    "pfl2_termination": final_error
                    <= float(assay["error_threshold"]),
                }
            )
            mirrored = EPGPENPEGHeadingRing(heading_config)
            mirrored.reset(float(-initial))
            for _ in range(steps_taken):
                mirrored_error = float(-target) - mirrored.decode()
                if abs(mirrored_error) <= float(assay["error_threshold"]):
                    break
                mirrored.step(
                    float(np.tanh(mirrored_error)), float(assay["update_gain"])
                )
            mirror_errors.append(abs(ring.decode() + mirrored.decode()))
    passed = all(
        item["monotone_error_reduction"]
        and item["final_error"] <= float(assay["maximum_final_error"])
        and item["pfl2_termination"]
        for item in records
    )
    return {
        "records": records,
        "maximum_final_error": max(item["final_error"] for item in records),
        "maximum_mirror_error": max(mirror_errors),
        "passed": bool(passed and max(mirror_errors) < 1e-12),
    }


def _structure(root: Path) -> dict:
    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
    table = feather.read_table(
        root / "data/raw/malecns-v1.0/body-annotations.feather",
        columns=["bodyId", "type", "somaSide"],
        memory_map=True,
    ).to_pandas()
    ids = table["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(graph.body_ids, ids)
    valid = nodes < graph.node_count
    valid[valid] &= graph.body_ids[nodes[valid]] == ids[valid]
    table = table.iloc[np.flatnonzero(valid)].copy()
    table["node"] = nodes[valid]
    groups = {
        "FC2": ("FC2A", "FC2B", "FC2C"),
        "EPG": ("EPG",),
        "PFL3": ("PFL3",),
        "PFL2": ("PFL2",),
        "DNa01": ("DNa01",),
        "DNa02": ("DNa02",),
        "DNp20": ("DNp20",),
    }
    group_nodes = {
        name: table.loc[table["type"].isin(types), "node"].to_numpy(dtype=np.int32)
        for name, types in groups.items()
    }
    edges = {}
    for source in groups:
        for target in groups:
            matrix = graph.adjacency[group_nodes[target], :][:, group_nodes[source]]
            edges[f"{source}->{target}"] = {
                "edge_count": int(matrix.nnz),
                "synapse_weight": float(np.sum(matrix.data)),
            }
    side_edges = {}
    for source_side, target_side in (("L", "R"), ("R", "L")):
        source = table.loc[
            table["type"].eq("PFL3") & table["somaSide"].eq(source_side), "node"
        ].to_numpy(dtype=np.int32)
        target = table.loc[
            table["type"].eq("DNa02") & table["somaSide"].eq(target_side), "node"
        ].to_numpy(dtype=np.int32)
        matrix = graph.adjacency[target, :][:, source]
        side_edges[f"PFL3_{source_side}->DNa02_{target_side}"] = {
            "edge_count": int(matrix.nnz),
            "synapse_weight": float(np.sum(matrix.data)),
        }
    return {
        "population_counts": {name: len(nodes) for name, nodes in group_nodes.items()},
        "edges": edges,
        "cross_side_pfl3_dna02_edges": side_edges,
        "supported_hierarchy": ["FC2->PFL3", "EPG->PFL3", "PFL3->DNa02"],
        "unsupported_direct_claims": ["PFL3->DNa01", "PFL3->DNp20"],
        "structure_is_not_functional_validation": True,
    }


def _summary(episodes: list[dict]) -> dict:
    return {
        "success_count": sum(item["success"] for item in episodes),
        "total_obstacles_passed": sum(item["obstacles_passed"] for item in episodes),
        "episodes": episodes,
    }


def evaluate_v7_fc2_pfl_dna(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    candidate_path = Path(config["candidate_evidence"])
    control_path = Path(config["control_evidence"])
    base_path = Path(config["base_neural_config"])
    heading_path = Path(config["heading_config"])
    candidate = json.loads((root / candidate_path).read_text())
    controls = json.loads((root / control_path).read_text())
    base = yaml.safe_load((root / base_path).read_text())
    heading = yaml.safe_load((root / heading_path).read_text())
    if not candidate["calibration_passed"] or not controls["advance_to_fc2_pfl_comparison"]:
        raise ValueError("FC2/PFL comparison requires passed frozen visual and causal gates")
    if _model_digest(candidate["model"]) != candidate["model_sha256"]:
        raise ValueError("frozen visual model changed")
    teacher = json.loads((root / base["adapter_source"]).read_text())["selected_candidate"]
    arms = {}
    for arm in [*config["readouts"], *config["controls"]]:
        features = StructuredNeuralFeatures(root, base)
        tuning = [
            _episode(features, candidate["model"], teacher, heading, int(seed), arm)
            for seed in config["tuning_seeds"]
        ]
        arms[arm] = {"tuning": _summary(tuning)}
    hierarchy = arms["hierarchical_fc2_pfl3_dna02"]["tuning"]
    tuning_passed = hierarchy["success_count"] == 6 and hierarchy["total_obstacles_passed"] == 54
    readout_equivalence = all(
        [item["trace_sha256"] for item in arms[name]["tuning"]["episodes"]]
        == [item["trace_sha256"] for item in hierarchy["episodes"]]
        for name in ("direct_dna01_dna02", "direct_dnp20")
    )
    control_passed = all(
        arms[name]["tuning"]["success_count"] < hierarchy["success_count"]
        for name in config["controls"]
    )
    calibration = []
    if tuning_passed and readout_equivalence and control_passed:
        features = StructuredNeuralFeatures(root, base)
        calibration = [
            _episode(
                features,
                candidate["model"],
                teacher,
                heading,
                int(seed),
                "hierarchical_fc2_pfl3_dna02",
            )
            for seed in config["calibration_seeds"]
        ]
    calibration_passed = bool(calibration) and all(
        item["success"] and item["obstacles_passed"] == 9 for item in calibration
    )
    attribution = {
        "role": "post_failure_attribution_only_not_parameter_selection",
        "direct_18_feature_reference": [],
        "r1r6_local_upper_bound": [],
    }
    if calibration:
        direct_features = StructuredNeuralFeatures(root, base)
        direct = [
            _heading_episode(
                direct_features,
                candidate["model"],
                teacher,
                heading,
                int(seed),
                "neural_heading",
            )
            for seed in config["calibration_seeds"]
        ]
        r1r6_config = yaml.safe_load(
            (root / "configs/driving-v7-r1r6-local.yaml").read_text()
        )
        retina = build_mass_balanced_retina(root)
        upper = [
            _r1r6_local_episode(retina, r1r6_config, teacher, int(seed))
            for seed in config["calibration_seeds"]
        ]
        direct_matches = all(
            direct_item["terminal_reason"] == hierarchy_item["terminal_reason"]
            and direct_item["obstacles_passed"] == hierarchy_item["obstacles_passed"]
            and direct_item["steps"] == hierarchy_item["steps"]
            for direct_item, hierarchy_item in zip(direct, calibration, strict=True)
        )
        upper_passes = all(
            item["success"] and item["obstacles_passed"] == 9 for item in upper
        )
        attribution = {
            "role": "post_failure_attribution_only_not_parameter_selection",
            "direct_18_feature_reference": [
                {key: value for key, value in item.items() if key != "trace"}
                for item in direct
            ],
            "r1r6_local_upper_bound": [
                {key: value for key, value in item.items() if key != "trace"}
                for item in upper
            ],
            "direct_reference_matches_hierarchical_failure": bool(direct_matches),
            "r1r6_local_upper_bound_passes_both": bool(upper_passes),
            "diagnosis": (
                "frozen_neural_visual_readout_generalization_gap"
                if direct_matches and upper_passes
                else "attribution_inconclusive"
            ),
        }
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(candidate_path): _sha256(root / candidate_path),
                str(control_path): _sha256(root / control_path),
                str(base_path): _sha256(root / base_path),
                str(heading_path): _sha256(root / heading_path),
            },
            "model_sha256": candidate["model_sha256"],
            "transparent_fixed_mapping": True,
            "environment_geometry_read_by_controller": False,
            "reward_read_by_controller": False,
            "raw_heading_read_by_controller": False,
            "calibration_run_once_after_candidate_freeze": True,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "malecns_structure": _structure(root),
        "turn_termination_assay": _turn_termination(config, heading),
        "readout_arms": arms,
        "readout_action_equivalence": bool(readout_equivalence),
        "causal_controls_passed": bool(control_passed),
        "tuning_passed": bool(tuning_passed),
        "calibration_episodes": calibration,
        "calibration_passed": bool(calibration_passed),
        "calibration_failure_attribution": attribution,
        "advance_to_navigation_release": False,
        "advance_to_mushroom_body": bool(
            tuning_passed
            and calibration_passed
            and control_passed
            and _turn_termination(config, heading)["passed"]
        ),
        "boundary": config["boundary"],
    }
