"""Blindly replay the MaleCNS one-hop optic-hex fallback by source type."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc1_input_structure import (
    IMPLEMENTATION as ONE_HOP_IMPLEMENTATION,
)
from fly_emotion.driving.v7_lplc1_input_structure import _one_hop_positions

CONFIG = Path(
    "configs/driving-v7-malecns-one-hop-coordinate-validation-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_malecns_one_hop_coordinate_validation_audit.py"
)


def _summary(errors: np.ndarray, exact: np.ndarray) -> dict:
    return {
        "evaluated_count": int(errors.size),
        "rounded_exact_count": int(np.count_nonzero(exact)),
        "rounded_exact_fraction": float(np.mean(exact)),
        "median_euclidean_hex_error": float(np.median(errors)),
        "p90_euclidean_hex_error": float(np.quantile(errors, 0.90)),
        "p95_euclidean_hex_error": float(np.quantile(errors, 0.95)),
        "maximum_euclidean_hex_error": float(np.max(errors)),
    }


def evaluate_v7_malecns_one_hop_coordinate_validation_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    mapping_protocol_path = Path(config["mapping_protocol"])
    mapping_protocol = yaml.safe_load(
        (root / mapping_protocol_path).read_text(encoding="utf-8")
    )
    annotation_path = Path(mapping_protocol["annotations"])
    adjacency_path = Path(mapping_protocol["adjacency_raw"])
    body_ids_path = Path(mapping_protocol["body_ids"])
    graph_metadata_path = Path(mapping_protocol["graph_metadata"])
    graph = load_graph(root / graph_metadata_path.parent, normalized=False)
    annotations = feather.read_table(
        root / annotation_path,
        columns=[
            "bodyId",
            "type",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
        memory_map=True,
    ).to_pandas()
    annotation_ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(graph.body_ids, annotation_ids)
    valid = nodes < graph.node_count
    valid[valid] &= graph.body_ids[nodes[valid]] == annotation_ids[valid]
    rows = annotations.iloc[np.flatnonzero(valid)]
    nodes = nodes[valid]
    node_types = np.full(graph.node_count, "", dtype=object)
    node_sides = np.full(graph.node_count, "", dtype=object)
    coordinates = np.full((graph.node_count, 2), np.nan, dtype=np.float64)
    node_types[nodes] = rows["type"].fillna("").to_numpy(dtype=object)
    node_sides[nodes] = rows["somaSide"].fillna("").to_numpy(dtype=object)
    coordinates[nodes] = rows[["assignedOlHex1", "assignedOlHex2"]].to_numpy(
        dtype=np.float64
    )

    source_results = {}
    for source_type in config["source_types"]:
        selected = np.flatnonzero(node_types == source_type)
        native_selected = selected[
            np.all(np.isfinite(coordinates[selected]), axis=1)
        ]
        expected_native = int(config["expected_native_counts"][source_type])
        if len(native_selected) != expected_native:
            raise ValueError(f"{source_type} native coordinate count changed")
        blind_coordinates = coordinates.copy()
        blind_coordinates[native_selected] = np.nan
        blind_positions, inferred = _one_hop_positions(
            graph, blind_coordinates, selected
        )
        evaluable = native_selected[
            np.all(np.isfinite(blind_positions[native_selected]), axis=1)
        ]
        overall = None
        per_side = {}
        if len(evaluable):
            delta = blind_positions[evaluable] - coordinates[evaluable]
            errors = np.linalg.norm(delta, axis=1)
            exact = np.all(
                np.rint(blind_positions[evaluable]) == coordinates[evaluable],
                axis=1,
            )
            overall = _summary(errors, exact)
            for side in ("L", "R"):
                side_mask = node_sides[evaluable] == side
                side_errors = errors[side_mask]
                side_exact = exact[side_mask]
                per_side[side] = (
                    _summary(side_errors, side_exact)
                    if len(side_errors)
                    else {
                        "evaluated_count": 0,
                        "rounded_exact_count": 0,
                        "rounded_exact_fraction": None,
                        "median_euclidean_hex_error": None,
                        "p90_euclidean_hex_error": None,
                        "p95_euclidean_hex_error": None,
                        "maximum_euclidean_hex_error": None,
                    }
                )
        source_results[source_type] = {
            "body_count": int(len(selected)),
            "native_reference_count": int(len(native_selected)),
            "blind_replay_evaluable_count": int(len(evaluable)),
            "blind_replay_coverage_fraction": (
                float(len(evaluable) / len(native_selected))
                if len(native_selected)
                else None
            ),
            "same_type_native_coordinates_hidden": True,
            "one_hop_inferred_count_during_blind_replay": int(
                np.count_nonzero(inferred[selected])
            ),
            "overall": overall,
            "per_side": per_side,
            "same_type_native_validation_available": bool(len(evaluable)),
            "current_mapping_native_count": expected_native,
            "current_mapping_one_hop_inferred_count": int(
                config["expected_one_hop_inferred_counts"][source_type]
            ),
        }
    inference_dependent = [
        source
        for source, result in source_results.items()
        if result["current_mapping_one_hop_inferred_count"] > 0
    ]
    no_native_reference = [
        source
        for source, result in source_results.items()
        if not result["same_type_native_validation_available"]
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(mapping_protocol_path): _sha256(root / mapping_protocol_path),
                str(annotation_path): _sha256(root / annotation_path),
                str(adjacency_path): _sha256(root / adjacency_path),
                str(body_ids_path): _sha256(root / body_ids_path),
                str(graph_metadata_path): _sha256(root / graph_metadata_path),
                str(ONE_HOP_IMPLEMENTATION): _sha256(root / ONE_HOP_IMPLEMENTATION),
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "validation_design": {
            "method": (
                "leave_entire_source_type_native_coordinates_out_then_replay_"
                "one_hop_weighted_input_centroid"
            ),
            "inference_dependent_source_types": inference_dependent,
            "source_types_without_same_type_native_reference": no_native_reference,
            "acceptance_threshold_defined": False,
        },
        "source_results": source_results,
        "every_inference_dependent_source_has_same_type_native_validation": all(
            source_results[source]["same_type_native_validation_available"]
            for source in inference_dependent
        ),
        "Tm3_same_type_native_validation_available": source_results["Tm3"][
            "same_type_native_validation_available"
        ],
        "Tm4_blind_replay_rounded_exact_fraction": source_results["Tm4"][
            "overall"
        ]["rounded_exact_fraction"],
        "one_hop_coordinates_are_native_equivalent": False,
        "authorize_coordinate_rule_as_experimental_mapping": False,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            "Tm3_has_no_same_type_native_reference_and_Tm4_blind_replay_is_"
            "not_native_equivalent"
        ),
        "boundary": config["boundary"],
    }
