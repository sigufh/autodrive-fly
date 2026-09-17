from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7_descending_path_audit import (
    ANNOTATIONS,
    BODY_IDS,
    ENGINE_IMPLEMENTATION,
    NEUROTRANSMITTERS,
    _node_metadata,
)
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-descending-propagation-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_descending_propagation_precheck.py")
NORMALIZED_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_target_norm.npz")


def _state_digest(state: np.ndarray) -> str:
    return hashlib.sha256(state.astype("<f4", copy=False).tobytes()).hexdigest()


def _stimulus_trace(
    adjacency,
    metadata: dict[str, np.ndarray],
    body_ids: np.ndarray,
    source_type: str,
    source_side: str,
    target_types: list[str],
    maximum_hops: int,
    amplitude: float,
) -> dict:
    source_nodes = np.flatnonzero(
        (metadata["type"] == source_type) & (metadata["side"] == source_side)
    )
    state = np.zeros(adjacency.shape[0], dtype=np.float32)
    state[source_nodes] = np.float32(amplitude)
    hops = {
        "0": {
            "state_sha256": _state_digest(state),
            "nonzero_node_count": int(np.count_nonzero(state)),
            "l2_norm": float(np.linalg.norm(state)),
            "targets": {},
        }
    }
    source_sign = metadata["sign"].astype(np.float32, copy=False)
    for hop in range(1, maximum_hops + 1):
        state = np.asarray(adjacency @ (source_sign * state), dtype=np.float32)
        targets = {}
        for target_type in target_types:
            by_side = {}
            for side in ("L", "R"):
                nodes = np.flatnonzero(
                    (metadata["type"] == target_type) & (metadata["side"] == side)
                )
                values = state[nodes]
                by_side[side] = {
                    "body_ids": [int(body_ids[node]) for node in nodes],
                    "fixed_denominator": int(nodes.size),
                    "finite_count": int(np.count_nonzero(np.isfinite(values))),
                    "nonzero_count": int(np.count_nonzero(values)),
                    "values": [float(value) for value in values],
                    "mean_state": float(np.mean(values)) if values.size else None,
                }
            targets[target_type] = by_side
        hops[str(hop)] = {
            "state_sha256": _state_digest(state),
            "nonzero_node_count": int(np.count_nonzero(state)),
            "l2_norm": float(np.linalg.norm(state)),
            "targets": targets,
        }
    return {
        "source_type": source_type,
        "source_side": source_side,
        "source_body_ids": [int(body_ids[node]) for node in source_nodes],
        "source_count": int(source_nodes.size),
        "input_amplitude_per_source": amplitude,
        "hops": hops,
    }


def evaluate_v7_descending_propagation_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    path_evidence_path = Path(config["path_evidence"])
    threshold_source_path = Path(config["threshold_source"])
    path_evidence = json.loads((root / path_evidence_path).read_text(encoding="utf-8"))
    threshold_source = yaml.safe_load((root / threshold_source_path).read_text(encoding="utf-8"))
    if not path_evidence["authorize_fixed_input_functional_propagation_precheck"]:
        raise ValueError("bilateral structural paths are required before propagation")

    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=True)
    expected_thresholds = {
        "minimum_absolute_readout": threshold_source["thresholds"][
            "minimum_valid_denominator"
        ],
        "maximum_normalized_mirror_error": threshold_source["thresholds"][
            "maximum_energy_weighted_mirror_error"
        ],
    }
    if config["thresholds"] != expected_thresholds:
        raise ValueError("descending precheck thresholds differ from frozen stage-1 scoring")
    metadata = _node_metadata(root, graph.body_ids)
    target_types = list(config["target_types"])
    traces = {
        side: _stimulus_trace(
            graph.adjacency,
            metadata,
            graph.body_ids,
            str(config["source_type"]),
            side,
            target_types,
            int(config["maximum_hops"]),
            float(config["input_amplitude_per_source"]),
        )
        for side in config["source_sides"]
    }

    minimum = float(config["thresholds"]["minimum_absolute_readout"] )
    maximum_mirror_error = float(config["thresholds"]["maximum_normalized_mirror_error"] )
    results = {}
    for target_type in target_types:
        structural_records = path_evidence["target_shortest_paths"][target_type]
        selected_hop = max(int(item["shortest_hops"]) for item in structural_records)
        readouts = {}
        denominators = {}
        finite = True
        for stimulus_side in config["source_sides"]:
            targets = traces[stimulus_side]["hops"][str(selected_hop)]["targets"][
                target_type
            ]
            left = float(targets["L"]["mean_state"])
            right = float(targets["R"]["mean_state"])
            readouts[stimulus_side] = right - left
            denominators[stimulus_side] = {
                side: targets[side]["fixed_denominator"] for side in ("L", "R")
            }
            finite &= all(
                targets[side]["finite_count"] == targets[side]["fixed_denominator"]
                for side in ("L", "R")
            )
        scale = max(abs(value) for value in readouts.values())
        mirror_error = (
            abs(readouts["L"] + readouts["R"]) / scale if scale > 0.0 else None
        )
        magnitude_passed = all(abs(value) >= minimum for value in readouts.values())
        sign_passed = readouts["L"] > 0.0 and readouts["R"] < 0.0
        mirror_passed = mirror_error is not None and mirror_error <= maximum_mirror_error
        results[target_type] = {
            "selected_hop_from_maximum_bilateral_shortest_depth": selected_hop,
            "fixed_readout": "right_mean_state_minus_left_mean_state",
            "fixed_target_denominators": denominators,
            "readout_by_PFL3_source_side": readouts,
            "minimum_absolute_readout": min(abs(value) for value in readouts.values()),
            "normalized_mirror_error": mirror_error,
            "finite_denominator_passed": bool(finite),
            "magnitude_gate_passed": bool(magnitude_passed),
            "opponent_sign_gate_passed": bool(sign_passed),
            "mirror_gate_passed": bool(mirror_passed),
            "precheck_passed": bool(finite and magnitude_passed and sign_passed and mirror_passed),
        }
    all_passed = all(result["precheck_passed"] for result in results.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(ANNOTATIONS): _sha256(root / ANNOTATIONS),
                str(NEUROTRANSMITTERS): _sha256(root / NEUROTRANSMITTERS),
                str(BODY_IDS): _sha256(root / BODY_IDS),
                str(NORMALIZED_ADJACENCY): _sha256(root / NORMALIZED_ADJACENCY),
                str(ENGINE_IMPLEMENTATION): _sha256(root / ENGINE_IMPLEMENTATION),
                str(path_evidence_path): _sha256(root / path_evidence_path),
                str(threshold_source_path): _sha256(root / threshold_source_path),
            },
            "state_update": (
                "x[h+1] = target_normalized_adjacency @ "
                "(presynaptic_consensus_sign * x[h])"
            ),
            "activation": "identity",
            "source_stimulus": "fixed_unilateral_population_pulse",
            "stimulus_generated_without_environment": True,
            "vehicle_not_run": True,
            "parameter_fit": False,
            "threshold_search": False,
            "calibration_consumed": False,
            "final_consumed": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "thresholds": config["thresholds"],
        "threshold_provenance": config["threshold_provenance"],
        "stimulus_traces": traces,
        "target_readout_results": results,
        "passing_target_populations": [
            name for name, result in results.items() if result["precheck_passed"]
        ],
        "failed_target_populations": [
            name for name, result in results.items() if not result["precheck_passed"]
        ],
        "all_descending_readouts_precheck_passed": all_passed,
        "real_graph_propagated_target_states_were_read": True,
        "biological_neural_states_measured": False,
        "physiological_neural_states_claimed": False,
        "algebraic_action_equivalence_used_as_evidence": False,
        "functional_neural_readout_validated": False,
        "advance_to_vehicle_assay": False,
        "advance_to_navigation_release": False,
        "advance_to_mushroom_body": False,
        "stop_reason": (
            None
            if all_passed
            else "one_or_more_real_descending_state_readouts_failed_fixed_precheck"
        ),
        "boundary": config["boundary"],
    }
