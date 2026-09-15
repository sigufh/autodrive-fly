from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-visual-target-input-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_visual_target_input_audit.py")


def _verify_evidence_dependencies(root: Path, evidence: dict, label: str) -> None:
    for path, expected in evidence.get("protocol", {}).get("dependencies_sha256", {}).items():
        if _sha256(root / path) != expected:
            raise ValueError(f"{label} has stale dependency: {path}")


def _load_metadata(root: Path) -> tuple:
    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
    annotations_path = root / "data/raw/malecns-v1.0/body-annotations.feather"
    transmitters_path = root / "data/raw/malecns-v1.0/body-neurotransmitters.feather"
    annotations = feather.read_table(
        annotations_path,
        columns=[
            "bodyId",
            "type",
            "somaSide",
            "rootSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
        memory_map=True,
    ).to_pandas()
    ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(graph.body_ids, ids)
    valid = nodes < graph.node_count
    valid[valid] &= graph.body_ids[nodes[valid]] == ids[valid]
    annotations = annotations.iloc[np.flatnonzero(valid)].copy()
    annotations["node"] = nodes[valid]
    transmitters = feather.read_table(
        transmitters_path, columns=["body", "consensus_nt"], memory_map=True
    ).to_pandas()
    return graph, annotations, transmitters, annotations_path, transmitters_path


def _node_metadata(graph, annotations, transmitters) -> tuple[np.ndarray, ...]:
    node_type = np.full(graph.node_count, "", dtype=object)
    node_side = np.full(graph.node_count, "", dtype=object)
    coordinates = np.full((graph.node_count, 2), np.nan, dtype=np.float64)
    nodes = annotations["node"].to_numpy(dtype=np.int32)
    node_type[nodes] = annotations["type"].fillna("").to_numpy(dtype=object)
    node_side[nodes] = annotations["somaSide"].fillna("").to_numpy(dtype=object)
    coordinates[nodes] = annotations[["assignedOlHex1", "assignedOlHex2"]].to_numpy(dtype=float)
    node_nt = np.full(graph.node_count, "missing", dtype=object)
    ids = transmitters["body"].to_numpy(dtype=np.int64)
    nt_nodes = np.searchsorted(graph.body_ids, ids)
    valid = nt_nodes < graph.node_count
    valid[valid] &= graph.body_ids[nt_nodes[valid]] == ids[valid]
    node_nt[nt_nodes[valid]] = (
        transmitters.iloc[np.flatnonzero(valid)]["consensus_nt"]
        .fillna("missing")
        .to_numpy(dtype=object)
    )
    return node_type, node_side, coordinates, node_nt


def _source_summary(
    graph, node_type, node_side, coordinates, node_nt, source_type: str, target_nodes: np.ndarray
) -> dict:
    source_nodes = np.flatnonzero(node_type == source_type)
    matrix = graph.adjacency[target_nodes, :][:, source_nodes]
    source_weight = np.asarray(np.abs(matrix).sum(axis=0)).ravel()
    weighted_nt: Counter[str] = Counter()
    for node, weight in zip(source_nodes, source_weight, strict=True):
        weighted_nt[str(node_nt[node])] += float(weight)
    located = np.all(np.isfinite(coordinates[source_nodes]), axis=1)
    return {
        "source_cells": int(source_nodes.size),
        "coordinate_cells": int(np.count_nonzero(located)),
        "coordinate_fraction": float(np.mean(located)) if source_nodes.size else 0.0,
        "soma_side_counts": {
            value: int(np.count_nonzero(node_side[source_nodes] == value))
            for value in ("L", "R", "")
        },
        "consensus_neurotransmitter_cells": dict(
            sorted(Counter(map(str, node_nt[source_nodes])).items())
        ),
        "target_input_weight_by_source_neurotransmitter": dict(sorted(weighted_nt.items())),
        "total_target_input_weight": float(source_weight.sum()),
    }


def _target_records(
    graph,
    annotations,
    node_type,
    node_side,
    coordinates,
    source_types: tuple[str, ...],
    target_types: tuple[str, ...],
    group_kind: str,
    config: dict,
) -> tuple[list[dict], dict]:
    records = []
    summary = {}
    fast = tuple(config["t5_groups"]["fast_candidate_sources"])
    delayed = tuple(config["t5_groups"]["delayed_candidate_sources"])
    t4_sources = tuple(config["looming_groups"]["T4_sources"])
    t5_sources = tuple(config["looming_groups"]["T5_sources"])
    for target_type in target_types:
        for side in ("L", "R"):
            rows = annotations.loc[
                annotations["type"].eq(target_type) & annotations["somaSide"].eq(side),
                ["bodyId", "node"],
            ].sort_values("bodyId")
            group_records = []
            for body_id, target in rows.itertuples(index=False, name=None):
                row = graph.adjacency.getrow(int(target))
                source_nodes = row.indices
                weights = np.abs(row.data).astype(np.float64)
                by_type = {}
                for source_type in source_types:
                    selected = node_type[source_nodes] == source_type
                    weight = float(weights[selected].sum())
                    same_side = selected & (node_side[source_nodes] == side)
                    located = selected & np.all(np.isfinite(coordinates[source_nodes]), axis=1)
                    by_type[source_type] = {
                        "weight": weight,
                        "edge_count": int(np.count_nonzero(selected)),
                        "same_side_weight_fraction": (
                            float(weights[same_side].sum() / weight) if weight > 0 else None
                        ),
                        "located_weight_fraction": (
                            float(weights[located].sum() / weight) if weight > 0 else None
                        ),
                    }
                if group_kind == "T5":
                    rule = {
                        "every_fast_source_present": all(
                            by_type[name]["weight"] > 0 for name in fast
                        ),
                        "any_delayed_source_present": any(
                            by_type[name]["weight"] > 0 for name in delayed
                        ),
                        "all_five_candidate_sources_present": all(
                            by_type[name]["weight"] > 0 for name in (*fast, *delayed)
                        ),
                    }
                else:
                    rule = {
                        "any_T4_input_present": any(
                            by_type[name]["weight"] > 0 for name in t4_sources
                        ),
                        "any_T5_input_present": any(
                            by_type[name]["weight"] > 0 for name in t5_sources
                        ),
                    }
                record = {
                    "body_id": int(body_id),
                    "target_type": target_type,
                    "side": side,
                    "source_types": by_type,
                    "coverage_rules": rule,
                }
                records.append(record)
                group_records.append(record)
            summary[f"{target_type}_{side}"] = {
                "cells": len(group_records),
                "coverage_counts": {
                    key: sum(record["coverage_rules"][key] for record in group_records)
                    for key in group_records[0]["coverage_rules"]
                }
                if group_records
                else {},
                "source_type_present_counts": {
                    source_type: sum(
                        record["source_types"][source_type]["weight"] > 0
                        for record in group_records
                    )
                    for source_type in source_types
                },
                "source_type_total_weights": {
                    source_type: float(
                        sum(
                            record["source_types"][source_type]["weight"]
                            for record in group_records
                        )
                    )
                    for source_type in source_types
                },
            }
    return records, summary


def evaluate_v7_visual_target_input_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    development_path = Path(config["development_evidence"])
    geometry_path = Path(config["geometry_ab_evidence"])
    development = json.loads((root / development_path).read_text(encoding="utf-8"))
    geometry = json.loads((root / geometry_path).read_text(encoding="utf-8"))
    _verify_evidence_dependencies(root, development, "development evidence")
    _verify_evidence_dependencies(root, geometry, "geometry evidence")
    if (
        development["development_response_gates_pass"]
        or geometry["development_response_gates_pass"]
    ):
        raise ValueError("structure audit is only defined after failed development responses")
    boundary = config["boundary"]
    if not boundary["anatomy_only"] or boundary["infer_function_from_connectivity"]:
        raise ValueError("visual target input audit must remain anatomy-only")
    graph, annotations, transmitters, annotations_path, transmitters_path = _load_metadata(root)
    node_type, node_side, coordinates, node_nt = _node_metadata(graph, annotations, transmitters)
    t5_sources = tuple(
        [
            *config["t5_groups"]["fast_candidate_sources"],
            *config["t5_groups"]["delayed_candidate_sources"],
        ]
    )
    looming_sources = tuple(
        [*config["looming_groups"]["T4_sources"], *config["looming_groups"]["T5_sources"]]
    )
    t5_targets = annotations.loc[
        annotations["type"].isin(config["t5_groups"]["target_types"]), "node"
    ].to_numpy(dtype=np.int32)
    looming_targets = annotations.loc[
        annotations["type"].isin(config["looming_groups"]["target_types"]), "node"
    ].to_numpy(dtype=np.int32)
    t5_records, t5_summary = _target_records(
        graph,
        annotations,
        node_type,
        node_side,
        coordinates,
        t5_sources,
        tuple(config["t5_groups"]["target_types"]),
        "T5",
        config,
    )
    looming_records, looming_summary = _target_records(
        graph,
        annotations,
        node_type,
        node_side,
        coordinates,
        looming_sources,
        tuple(config["looming_groups"]["target_types"]),
        "looming",
        config,
    )
    t5_source_summary = {
        source_type: _source_summary(
            graph, node_type, node_side, coordinates, node_nt, source_type, t5_targets
        )
        for source_type in t5_sources
    }
    looming_source_summary = {
        source_type: _source_summary(
            graph, node_type, node_side, coordinates, node_nt, source_type, looming_targets
        )
        for source_type in looming_sources
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(development_path): _sha256(root / development_path),
                str(geometry_path): _sha256(root / geometry_path),
                str(annotations_path.relative_to(root)): _sha256(annotations_path),
                str(transmitters_path.relative_to(root)): _sha256(transmitters_path),
                "data/processed/malecns-v1.0/adjacency_raw.npz": _sha256(
                    root / "data/processed/malecns-v1.0/adjacency_raw.npz"
                ),
            },
            "anatomy_only": True,
            "neural_evaluation_performed": False,
            "parameter_fitting": False,
            "runtime_modified": False,
        },
        "T5": {
            "source_types": t5_source_summary,
            "population_summary": t5_summary,
            "targets": t5_records,
        },
        "looming_targets": {
            "source_types": looming_source_summary,
            "population_summary": looming_summary,
            "targets": looming_records,
        },
        "structural_findings": {
            "T5_targets_with_every_fast_and_any_delayed_source": int(
                sum(
                    item["coverage_rules"]["every_fast_source_present"]
                    and item["coverage_rules"]["any_delayed_source_present"]
                    for item in t5_records
                )
            ),
            "T5_target_count": len(t5_records),
            "T5_targets_with_all_five_candidate_sources": int(
                sum(
                    item["coverage_rules"]["all_five_candidate_sources_present"]
                    for item in t5_records
                )
            ),
            "looming_targets_with_any_T4_and_any_T5": int(
                sum(
                    item["coverage_rules"]["any_T4_input_present"]
                    and item["coverage_rules"]["any_T5_input_present"]
                    for item in looming_records
                )
            ),
            "looming_target_count": len(looming_records),
            "LC4_requires_direct_T4_input": False,
        },
        "coverage_rules": config["coverage_rules"],
        "boundary": boundary,
        "limitations": [
            "Structural connectivity and transmitter annotations do not prove functional sign.",
            "T4/T5 and looming target coordinates are absent from the annotation table.",
            "CT1 has two large-field cells and no optic-hex coordinate in this table.",
            "LC4 is predominantly T5-driven and is not required to have direct T4 input.",
            "No cell is removed and no missing input is treated as a zero biological response.",
        ],
        "advance_to_model_change": False,
        "advance_to_validation": False,
        "advance_to_central_complex": False,
    }
