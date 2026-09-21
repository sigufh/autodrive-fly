"""Decompose direct inputs to Tm states used by measured-kernel diagnostics."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe

CONFIG = Path(
    "configs/driving-v7-t5-measured-kernel-source-input-decomposition-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_measured_kernel_source_input_decomposition_audit.py"
)


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.unparse(function)


def evaluate_v7_t5_measured_kernel_source_input_decomposition_audit(
    root: Path,
) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "cascade_evidence",
            "measured_kernel_config",
            "lamina_split_config",
            "lamina_split_implementation",
            "nested_probe_implementation",
            "v7_implementation",
        )
    }
    cascade = json.loads((root / paths["cascade_evidence"]).read_text())
    measured = yaml.safe_load((root / paths["measured_kernel_config"]).read_text())
    lamina = yaml.safe_load((root / paths["lamina_split_config"]).read_text())
    if not cascade["typed_recurrent_source_state_then_measured_kernel_cascade_verified"]:
        raise ValueError("measured-kernel cascade must be verified first")
    if measured["source_types"] != config["source_types"]:
        raise ValueError("Tm source order changed")
    if lamina["lamina_split"]["ON_targets"] != ["L1"] or lamina[
        "lamina_split"
    ]["OFF_targets"] != ["L2", "L3"]:
        raise ValueError("lamina overwrite set changed")

    advance_source = _function_source(root / paths["v7_implementation"], "_advance")
    lamina_advance_source = _function_source(
        root / paths["lamina_split_implementation"], "_advance"
    )
    ordering = {
        "base_recurrent_update_before_lamina_overwrite": (
            lamina_advance_source.index("super()._advance")
            < lamina_advance_source.index("updated[nodes] =")
        ),
        "base_recurrent_update_reads_prior_state": (
            "transmitted = state.copy()" in advance_source
            and "self.adjacency @ (transmitted * self.source_sign)" in advance_source
        ),
        "lamina_overwrite_uses_current_retinal_drive": (
            "matrix @ (positive if is_on else negative)" in lamina_advance_source
        ),
        "Tm_current_update_cannot_read_same_update_lamina_overwrite": True,
    }
    if not all(ordering.values()):
        raise ValueError("lamina/source update ordering changed")

    probe = LaminaSplitProbe(root, lamina)
    node_types = probe.node_types
    retina_nodes = np.zeros(probe.graph.node_count, dtype=bool)
    retina_nodes[probe.retina.node_indices] = True
    categories = config["mutually_exclusive_input_categories"]
    expected_categories = [
        "R1-R6",
        "overwritten_lamina_L1_L2_L3",
        "other_lamina_L4_L5",
        "same_Tm_type",
        "other_Tm_type",
        "CT1",
        "T4",
        "T5_feedback",
        "other_visual_subgraph",
    ]
    if categories != expected_categories:
        raise ValueError("Tm input decomposition categories changed")

    results = {}
    for target_type in config["source_types"]:
        targets = np.flatnonzero(node_types == target_type)
        edges = probe.adjacency[targets].tocoo()
        sources = edges.col
        weights = np.abs(edges.data.astype(np.float64))
        names = node_types[sources]
        is_tm = np.fromiter(
            (str(name).startswith("Tm") for name in names), bool, len(names)
        )
        masks = {
            "R1-R6": retina_nodes[sources],
            "overwritten_lamina_L1_L2_L3": np.isin(names, ("L1", "L2", "L3")),
            "other_lamina_L4_L5": np.isin(names, ("L4", "L5")),
            "same_Tm_type": names == target_type,
            "other_Tm_type": is_tm & (names != target_type),
            "CT1": names == "CT1",
            "T4": np.fromiter(
                (str(name).startswith("T4") for name in names), bool, len(names)
            ),
            "T5_feedback": np.fromiter(
                (str(name).startswith("T5") for name in names), bool, len(names)
            ),
        }
        assigned = np.logical_or.reduce(list(masks.values()))
        masks["other_visual_subgraph"] = ~assigned
        membership = sum(mask.astype(np.int8) for mask in masks.values())
        if not np.all(membership == 1):
            raise ValueError(f"input categories overlap or omit edges: {target_type}")
        total = float(np.sum(weights))
        category_results = {}
        for name, mask in masks.items():
            source_mask = np.zeros(probe.graph.node_count, dtype=np.float32)
            source_mask[np.unique(sources[mask])] = 1.0
            category_matrix = probe.adjacency[targets].multiply(source_mask).tocsr()
            category_matrix.eliminate_zeros()
            category_results[name] = {
                "edge_count": int(np.count_nonzero(mask)),
                "normalized_absolute_input_mass": float(np.sum(weights[mask])),
                "normalized_absolute_input_fraction": float(
                    np.sum(weights[mask]) / total
                ),
                "target_count_with_any_direct_input": int(
                    np.count_nonzero(np.diff(category_matrix.indptr))
                ),
            }
        fractions_sum = sum(
            item["normalized_absolute_input_fraction"]
            for item in category_results.values()
        )
        recurrent_fraction = sum(
            category_results[name]["normalized_absolute_input_fraction"]
            for name in ("same_Tm_type", "other_Tm_type", "T4", "T5_feedback")
        )
        results[target_type] = {
            "target_count": len(targets),
            "direct_input_edge_count": len(weights),
            "normalized_absolute_input_mass": total,
            "category_results": category_results,
            "category_fraction_sum": fractions_sum,
            "recurrent_or_target_feedback_input_fraction": recurrent_fraction,
            "has_recurrent_or_target_feedback_input": recurrent_fraction > 0.0,
        }
    all_partitioned = all(
        np.isclose(item["category_fraction_sum"], 1.0) for item in results.values()
    )
    every_source_recurrent = all(
        item["has_recurrent_or_target_feedback_input"] for item in results.values()
    )
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        **{str(path): _sha256(root / path) for path in paths.values()},
        "data/processed/malecns-v1.0/adjacency_target_norm.npz": _sha256(
            root / "data/processed/malecns-v1.0/adjacency_target_norm.npz"
        ),
        "data/processed/malecns-v1.0/body_ids.npy": _sha256(
            root / "data/processed/malecns-v1.0/body_ids.npy"
        ),
        "data/processed/malecns-v1.0/graph.json": _sha256(
            root / "data/processed/malecns-v1.0/graph.json"
        ),
        "data/raw/malecns-v1.0/body-annotations.feather": _sha256(
            root / "data/raw/malecns-v1.0/body-annotations.feather"
        ),
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "parameter_fit": False,
            "functional_stimulus_evaluated": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "update_ordering": ordering,
        "source_results": results,
        "all_direct_input_edges_partitioned_exactly_once": all_partitioned,
        "every_source_has_recurrent_or_target_feedback_input": every_source_recurrent,
        "feedforward_only_source_drive_available_from_current_trace": False,
        "measured_kernel_replacement_of_source_dynamics_evaluated": False,
        "source_dynamics_replacement_requires_explicit_intervention": True,
        "authorize_source_dynamics_replacement": False,
        "authorize_T5_functional_precheck": False,
        "direction_scoring_authorized": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "current_Tm_trace_contains_recurrent_and_target_feedback_before_measured_FIR"
        ),
        "boundary": config["boundary"],
    }
