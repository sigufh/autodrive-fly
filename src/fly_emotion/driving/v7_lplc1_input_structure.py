"""Structure-only audit for LPLC1 object, motion, and inhibitory inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-lplc1-input-structure.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_lplc1_input_structure.py")
ANNOTATIONS = Path("data/raw/malecns-v1.0/body-annotations.feather")
NEUROTRANSMITTERS = Path("data/raw/malecns-v1.0/body-neurotransmitters.feather")
RAW_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_raw.npz")
BODY_IDS = Path("data/processed/malecns-v1.0/body_ids.npy")


def _node_metadata(root: Path, body_ids: np.ndarray) -> tuple[np.ndarray, ...]:
    annotations = feather.read_table(
        root / ANNOTATIONS,
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    transmitters = feather.read_table(
        root / NEUROTRANSMITTERS, columns=["body", "consensus_nt"], memory_map=True
    ).to_pandas()
    annotations = annotations.merge(
        transmitters.rename(columns={"body": "bodyId"}), on="bodyId", how="left"
    )
    ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(body_ids, ids)
    valid = nodes < len(body_ids)
    valid[valid] &= body_ids[nodes[valid]] == ids[valid]
    rows = annotations.iloc[np.flatnonzero(valid)]
    nodes = nodes[valid]
    node_types = np.full(len(body_ids), "", dtype=object)
    sides = np.full(len(body_ids), "", dtype=object)
    neurotransmitters_by_node = np.full(len(body_ids), "missing", dtype=object)
    coordinates = np.full((len(body_ids), 2), np.nan, dtype=np.float64)
    node_types[nodes] = rows["type"].fillna("").to_numpy(dtype=object)
    sides[nodes] = rows["somaSide"].fillna("").to_numpy(dtype=object)
    neurotransmitters_by_node[nodes] = rows["consensus_nt"].fillna("missing").to_numpy(
        dtype=object
    )
    coordinates[nodes] = rows[["assignedOlHex1", "assignedOlHex2"]].to_numpy(
        dtype=np.float64
    )
    return node_types, sides, neurotransmitters_by_node, coordinates


def _one_hop_positions(
    graph, coordinates: np.ndarray, source_nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = coordinates.copy()
    native = np.all(np.isfinite(coordinates), axis=1)
    inferred = np.zeros(len(coordinates), dtype=bool)
    for source in source_nodes:
        if native[source]:
            continue
        row = graph.adjacency.getrow(int(source))
        upstream = row.indices
        weights = np.abs(row.data).astype(np.float64)
        known = native[upstream]
        if np.any(known):
            positions[source] = np.average(
                coordinates[upstream[known]], axis=0, weights=weights[known]
            )
            inferred[source] = True
    return positions, inferred


def _group_summary(
    records: list[dict], group: str, target_count: int, thresholds: dict
) -> dict:
    with_input = np.asarray([record[group]["weight"] > 0.0 for record in records])
    total_weight = sum(record[group]["weight"] for record in records)
    located_weight = sum(record[group]["located_weight"] for record in records)
    target_fraction = float(np.mean(with_input))
    located_fraction = float(located_weight / total_weight) if total_weight > 0.0 else 0.0
    gates = {
        "target_coverage": target_fraction >= float(thresholds["minimum_target_coverage"]),
        "located_source_weight_fraction": located_fraction
        >= float(thresholds["minimum_located_source_weight_fraction"]),
    }
    return {
        "target_count": target_count,
        "targets_with_input_count": int(np.count_nonzero(with_input)),
        "targets_with_input_fraction": target_fraction,
        "source_cell_edge_count": int(sum(record[group]["source_count"] for record in records)),
        "total_synapse_weight": int(total_weight),
        "native_coordinate_source_cell_edge_count": int(
            sum(record[group]["native_coordinate_source_count"] for record in records)
        ),
        "native_coordinate_synapse_weight": int(
            sum(record[group]["native_coordinate_weight"] for record in records)
        ),
        "one_hop_inferred_source_cell_edge_count": int(
            sum(record[group]["one_hop_inferred_source_count"] for record in records)
        ),
        "one_hop_inferred_synapse_weight": int(
            sum(record[group]["one_hop_inferred_weight"] for record in records)
        ),
        "located_source_cell_edge_count": int(
            sum(record[group]["located_source_count"] for record in records)
        ),
        "located_synapse_weight": int(located_weight),
        "located_source_weight_fraction": located_fraction,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def evaluate_v7_lplc1_input_structure(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    mechanism_path = Path(config["mechanism_evidence"])
    mechanism = json.loads((root / mechanism_path).read_text())
    near_path = Path(config["near_collision_evidence"])
    near = json.loads((root / near_path).read_text())
    if mechanism["target_contracts"]["LPLC1"]["output_role"] != (
        "near_collision_specific_slowing_signal"
    ):
        raise ValueError("LPLC1 mechanism boundary changed")
    if not near["stimulus_geometry_gate_passed"]:
        raise ValueError("LPLC1 input audit requires valid dedicated stimulus geometry")
    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
    node_types, sides, neurotransmitters, coordinates = _node_metadata(root, graph.body_ids)
    lplc1_targets = np.flatnonzero(node_types == "LPLC1")
    source_nodes = np.unique(graph.adjacency[lplc1_targets, :].indices)
    positions, inferred = _one_hop_positions(graph, coordinates, source_nodes)
    native = np.all(np.isfinite(coordinates), axis=1)
    located = np.all(np.isfinite(positions), axis=1)
    groups = {
        "object_detectors": np.isin(node_types, config["source_groups"]["object_detectors"]),
        "motion_detectors": np.isin(node_types, config["source_groups"]["motion_detectors"]),
        "glutamatergic": np.isin(
            neurotransmitters, config["inhibition_candidates"]["glutamatergic"]
        ),
        "GABAergic": np.isin(
            neurotransmitters,
            config["inhibition_candidates"]["GABAergic_reported_separately"],
        ),
    }
    side_results = {}
    all_records = []
    for side in "LR":
        targets = np.flatnonzero((node_types == "LPLC1") & (sides == side))
        records = []
        type_totals = {}
        for target in targets:
            row = graph.adjacency.getrow(int(target))
            sources = row.indices
            weights = np.abs(row.data).astype(np.float64)
            group_values = {}
            for name, node_mask in groups.items():
                selected = node_mask[sources]
                group_values[name] = {
                    "source_count": int(np.count_nonzero(selected)),
                    "weight": float(np.sum(weights[selected])),
                    "native_coordinate_source_count": int(
                        np.count_nonzero(selected & native[sources])
                    ),
                    "native_coordinate_weight": float(
                        np.sum(weights[selected & native[sources]])
                    ),
                    "one_hop_inferred_source_count": int(
                        np.count_nonzero(selected & inferred[sources])
                    ),
                    "one_hop_inferred_weight": float(
                        np.sum(weights[selected & inferred[sources]])
                    ),
                    "located_source_count": int(np.count_nonzero(selected & located[sources])),
                    "located_weight": float(np.sum(weights[selected & located[sources]])),
                }
            for source_type in np.unique(node_types[sources]):
                if not source_type:
                    continue
                selected = node_types[sources] == source_type
                entry = type_totals.setdefault(
                    str(source_type),
                    {
                        "target_count": 0,
                        "source_cell_edge_count": 0,
                        "synapse_weight": 0.0,
                        "located_synapse_weight": 0.0,
                        "native_coordinate_synapse_weight": 0.0,
                        "one_hop_inferred_synapse_weight": 0.0,
                        "neurotransmitter_weights": {},
                    },
                )
                entry["target_count"] += 1
                entry["source_cell_edge_count"] += int(np.count_nonzero(selected))
                entry["synapse_weight"] += float(np.sum(weights[selected]))
                entry["located_synapse_weight"] += float(
                    np.sum(weights[selected & located[sources]])
                )
                entry["native_coordinate_synapse_weight"] += float(
                    np.sum(weights[selected & native[sources]])
                )
                entry["one_hop_inferred_synapse_weight"] += float(
                    np.sum(weights[selected & inferred[sources]])
                )
                for transmitter in np.unique(neurotransmitters[sources][selected]):
                    selected_transmitter = selected & (neurotransmitters[sources] == transmitter)
                    entry["neurotransmitter_weights"][str(transmitter)] = int(
                        entry["neurotransmitter_weights"].get(str(transmitter), 0)
                        + np.sum(weights[selected_transmitter])
                    )
            record = {
                "body_id": int(graph.body_ids[target]),
                "side": side,
                **group_values,
            }
            records.append(record)
            all_records.append(record)
        summaries = {
            name: _group_summary(records, name, len(targets), config["gate"]) for name in groups
        }
        side_results[side] = {
            "target_count": int(len(targets)),
            "groups": summaries,
            "source_types_by_synapse_weight": [
                {
                    "type": name,
                    **{
                        key: (int(value) if key != "neurotransmitter_weights" else value)
                        for key, value in item.items()
                    },
                    "located_synapse_weight_fraction": (
                        float(item["located_synapse_weight"] / item["synapse_weight"])
                        if item["synapse_weight"] > 0.0
                        else 0.0
                    ),
                }
                for name, item in sorted(
                    type_totals.items(), key=lambda pair: pair[1]["synapse_weight"], reverse=True
                )
            ],
        }
    required = ("object_detectors", "motion_detectors", "glutamatergic")
    strict = all(side_results[side]["groups"][name]["passed"] for side in "LR" for name in required)
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(mechanism_path): _sha256(root / mechanism_path),
                str(near_path): _sha256(root / near_path),
                str(ANNOTATIONS): _sha256(root / ANNOTATIONS),
                str(NEUROTRANSMITTERS): _sha256(root / NEUROTRANSMITTERS),
                str(RAW_ADJACENCY): _sha256(root / RAW_ADJACENCY),
                str(BODY_IDS): _sha256(root / BODY_IDS),
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
            "source_position_rule": (
                "native optic-hex coordinate or one-hop raw-edge-weighted centroid of "
                "native-coordinate direct inputs"
            ),
        },
        "source_group_definitions": {
            "object_detectors": config["source_groups"]["object_detectors"],
            "motion_detectors": config["source_groups"]["motion_detectors"],
            "glutamatergic": config["inhibition_candidates"]["glutamatergic"],
            "GABAergic": config["inhibition_candidates"][
                "GABAergic_reported_separately"
            ],
        },
        "per_side": side_results,
        "target_records": all_records,
        "strict_input_structure_gate_passed": bool(strict),
        "authorize_spatial_inhibition_mechanism": bool(strict),
        "stop_reason": (
            None
            if strict
            else "glutamatergic spatial-coordinate coverage is below the frozen gate"
        ),
        "boundary": config["boundary"],
    }
