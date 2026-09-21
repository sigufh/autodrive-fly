"""Compare T4 target-synapse axes with source optic-hex RF axes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import _node_annotations
from fly_emotion.driving.v7_t4_synapse_crossfit_precheck import _crossfit_moments

CONFIG = Path("configs/driving-v7-t4-synapse-rf-axis-correspondence-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_synapse_rf_axis_correspondence_audit.py"
)
RAW_ADJACENCY = Path("data/processed/malecns-v1.0/adjacency_raw.npz")


def _summary(angles: np.ndarray, cosines: np.ndarray, cardinal_match: np.ndarray) -> dict:
    return {
        "count": int(len(angles)),
        "median_angle_degrees": float(np.median(angles)),
        "angle_q025_degrees": float(np.quantile(angles, 0.025)),
        "angle_q25_degrees": float(np.quantile(angles, 0.25)),
        "angle_q75_degrees": float(np.quantile(angles, 0.75)),
        "angle_q975_degrees": float(np.quantile(angles, 0.975)),
        "positive_cosine_fraction": float(np.mean(cosines > 0.0)),
        "same_cardinal_direction_fraction": float(np.mean(cardinal_match)),
    }


def _compare(reference: np.ndarray, candidate: np.ndarray) -> tuple[dict, np.ndarray]:
    cosines = np.sum(reference * candidate, axis=1)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    cardinal = np.asarray(((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)))
    cardinal_match = np.argmax(reference @ cardinal.T, axis=1) == np.argmax(
        candidate @ cardinal.T, axis=1
    )
    return _summary(angles, cosines, cardinal_match), angles


def evaluate_v7_t4_synapse_rf_axis_correspondence_audit(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(audit[name])
        for name in (
            "crossfit_precheck_config",
            "crossfit_axis_evidence",
            "synapse_spatial_protocol",
            "conductance_protocol",
        )
    }
    precheck = yaml.safe_load((root / paths["crossfit_precheck_config"]).read_text())
    crossfit = json.loads((root / paths["crossfit_axis_evidence"]).read_text())
    if not crossfit["T4_synapse_crossfit_axis_passed"]:
        raise ValueError("cross-fit target-synapse axis must be available")
    axis_protocol = yaml.safe_load((root / precheck["crossfit_axis_protocol"]).read_text())
    precheck["crossfit_split_seed"] = axis_protocol["split_seed"]
    conductance = yaml.safe_load((root / paths["conductance_protocol"]).read_text())
    normalization = _collect_source_normalization(
        root,
        retinal_backend="linear_luminance",
        retinal_geometry=conductance["primary_retinal_geometry"],
        config=conductance,
    )
    probe = MotionSignConductanceProbe(
        root, retinal_backend="linear_luminance", normalization=normalization
    )
    moments = _crossfit_moments(root, probe, precheck, crossfit)
    raw = load_graph(root / RAW_ADJACENCY.parent, normalized=False).adjacency
    sides, source_coordinates = _node_annotations(root, probe)
    source_coordinates = source_coordinates.copy()
    source_coordinates[sides == "L", 0] *= -1.0
    source_masks = {
        channel: np.isin(probe.node_types, types)
        for channel, types in audit["source_groups"].items()
    }
    source_axes = np.full_like(moments["axes"], np.nan)
    source_coverage = {"fast": [], "delayed": []}
    for row_index, target in enumerate(probe.conductance["targets"]):
        edge = raw.getrow(int(target))
        sources = edge.indices
        weights = np.abs(edge.data.astype(np.float64))
        centroids = {}
        for channel in ("fast", "delayed"):
            keep = source_masks[channel][sources]
            keep &= np.all(np.isfinite(source_coordinates[sources]), axis=1)
            source_coverage[channel].append(bool(np.any(keep)))
            if np.any(keep):
                centroids[channel] = np.average(
                    source_coordinates[sources[keep]],
                    axis=0,
                    weights=weights[keep],
                )
        if set(centroids) == {"fast", "delayed"}:
            axis = centroids["fast"] - centroids["delayed"]
            norm = float(np.linalg.norm(axis))
            if np.isfinite(norm) and norm > 1e-12:
                source_axes[row_index] = axis / norm

    valid = moments["valid"] & np.all(np.isfinite(moments["axes"]), axis=1)
    valid &= np.all(np.isfinite(source_axes), axis=1)
    reference = moments["axes"][valid]
    source = source_axes[valid]
    target_bodies = moments["target_bodies"]
    node_by_body = {int(body): index for index, body in enumerate(probe.graph.body_ids)}
    target_nodes = np.asarray(
        [node_by_body[int(body)] for body in target_bodies], dtype=np.int32
    )
    comparisons = {}
    per_population = {}
    for name, values in audit["comparison_transforms"].items():
        transformed = source @ np.asarray(values, dtype=np.float64)
        comparison, angles = _compare(reference, transformed)
        comparisons[name] = comparison
        population_results = {}
        for subtype in "abcd":
            for side in "LR":
                group = valid & (probe.node_types[target_nodes] == f"T4{subtype}")
                group &= sides[target_nodes] == side
                local_reference = moments["axes"][group]
                local_source = source_axes[group] @ np.asarray(values, dtype=np.float64)
                result, _ = _compare(local_reference, local_source)
                population_results[f"T4{subtype}_{side}"] = result
        per_population[name] = population_results
    best_name = min(
        comparisons, key=lambda name: comparisons[name]["median_angle_degrees"]
    )
    fixed_denominator = sum(
        len(probe.populations[f"T4{subtype}_{side}"])
        for subtype in "abcd"
        for side in "LR"
    )
    dependency_paths = {
        str(CONFIG): CONFIG,
        str(IMPLEMENTATION): IMPLEMENTATION,
        str(RAW_ADJACENCY): RAW_ADJACENCY,
        **{str(path): path for path in paths.values()},
        precheck["crossfit_axis_protocol"]: Path(precheck["crossfit_axis_protocol"]),
        precheck["synapse_spatial_protocol"]: Path(precheck["synapse_spatial_protocol"]),
    }
    spatial = yaml.safe_load((root / paths["synapse_spatial_protocol"]).read_text())
    for name in ("annotations", "synapse_partners"):
        dependency_paths[spatial[name]] = Path(spatial[name])
    return {
        "protocol": {
            "name": audit["name"],
            "observed_on": audit["observed_on"],
            "dependencies_sha256": {
                name: _sha256(root / path) for name, path in dependency_paths.items()
            },
            "parameter_fit_to_neural_response": False,
            "functional_stimulus_evaluated": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "axis_definitions": audit["axis_definitions"],
        "fixed_T4_population_denominator": int(fixed_denominator),
        "conductance_target_count": int(len(target_bodies)),
        "joint_valid_target_count": int(np.count_nonzero(valid)),
        "joint_valid_fraction_of_fixed_denominator": float(
            np.count_nonzero(valid) / fixed_denominator
        ),
        "source_coordinate_coverage": {
            channel: {
                "covered_target_count": int(np.count_nonzero(values)),
                "covered_target_fraction_of_conductance_targets": float(np.mean(values)),
            }
            for channel, values in source_coverage.items()
        },
        "comparison_results": comparisons,
        "population_comparison_results": per_population,
        "descriptive_best_signed_permutation": best_name,
        "raw_source_RF_axis_exactly_matches_target_synapse_axis": bool(
            np.array_equal(source, reference)
        ),
        "source_RF_axis_interchangeability_verified": False,
        "authorize_RF_axis_replacement": False,
        "authorize_new_T4_functional_candidate": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "source_RF_and_target_synapse_axes_are_not_interchangeable",
        "boundary": audit["boundary"],
    }
