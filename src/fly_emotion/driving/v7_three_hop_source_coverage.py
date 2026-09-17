"""Structural reachability audit for independent three-hop T4/T5 channels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe

CONFIG = Path("configs/driving-v7-three-hop-source-coverage.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_three_hop_source_coverage.py")


def evaluate_v7_three_hop_source_coverage(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    probe_config = yaml.safe_load(
        (root / "configs/driving-v7-t4t5-local-edge-precheck.yaml").read_text()
    )
    probe = MassBalancedVisualProbe(root, probe_config)
    adjacency = probe.adjacency
    receptor_sets = {
        "balanced_1914": probe.retina.node_indices,
        "all_mapped_3377": np.flatnonzero(probe.node_types == "R1-R6"),
    }
    results = {}
    minimum = float(config["gate"]["minimum_population_coverage"])
    independent = ("T4_center", "T4_proximal", "T4_distal", "T5_fast", "T5_delayed")
    cumulative_results = {}
    for mode, receptors in receptor_sets.items():
        reached = np.zeros(probe.graph.node_count, dtype=bool)
        reached[receptors] = True
        frontier = reached.copy()
        first_depth = np.full(probe.graph.node_count, -1, dtype=np.int16)
        first_depth[frontier] = 0
        for depth in range(1, max(config["cumulative_path_lengths_edges"])):
            next_frontier = np.asarray(adjacency @ frontier.astype(np.float32)).ravel() != 0
            next_frontier &= ~reached
            first_depth[next_frontier] = depth
            reached |= next_frontier
            frontier = next_frontier
        mode_results = {}
        cumulative_mode = {}
        for group_name, types in config["source_groups"].items():
            family = group_name[:2]
            source_nodes = np.flatnonzero(np.isin(probe.node_types, types))
            upstream = adjacency[source_nodes, :] @ adjacency[:, receptors]
            populations = {}
            for subtype in "abcd":
                for side in "LR":
                    population = f"{family}{subtype}_{side}"
                    targets = probe.populations[population]
                    paths = (adjacency[targets, :][:, source_nodes] @ upstream).tocsr()
                    reached = np.diff(paths.indptr) > 0
                    fraction = float(np.mean(reached))
                    populations[population] = {
                        "target_count": int(len(targets)),
                        "reachable_target_count": int(np.count_nonzero(reached)),
                        "reachable_target_fraction": fraction,
                        "passed": fraction >= minimum,
                    }
            mode_results[group_name] = {
                "source_types": list(types),
                "source_cell_count": int(len(source_nodes)),
                "populations": populations,
                "minimum_population_coverage": min(
                    value["reachable_target_fraction"] for value in populations.values()
                ),
                "all_populations_passed": all(value["passed"] for value in populations.values()),
            }
            by_length = {}
            for total_length in config["cumulative_path_lengths_edges"]:
                eligible = source_nodes[
                    (first_depth[source_nodes] >= 0)
                    & (first_depth[source_nodes] <= int(total_length) - 1)
                ]
                length_populations = {}
                for subtype in "abcd":
                    for side in "LR":
                        population = f"{family}{subtype}_{side}"
                        targets = probe.populations[population]
                        paths = adjacency[targets, :][:, eligible].tocsr()
                        target_reached = np.diff(paths.indptr) > 0
                        fraction = float(np.mean(target_reached))
                        length_populations[population] = {
                            "reachable_target_count": int(np.count_nonzero(target_reached)),
                            "reachable_target_fraction": fraction,
                            "passed": fraction >= minimum,
                        }
                by_length[str(total_length)] = {
                    "eligible_source_count": int(len(eligible)),
                    "minimum_population_coverage": min(
                        value["reachable_target_fraction"] for value in length_populations.values()
                    ),
                    "all_populations_passed": all(
                        value["passed"] for value in length_populations.values()
                    ),
                    "populations": length_populations,
                }
            cumulative_mode[group_name] = {
                "source_first_depth_counts": {
                    str(depth): int(np.count_nonzero(first_depth[source_nodes] == depth))
                    for depth in range(1, max(config["cumulative_path_lengths_edges"]))
                },
                "unreachable_source_count": int(np.count_nonzero(first_depth[source_nodes] < 0)),
                "by_total_path_length": by_length,
            }
        results[mode] = {
            "receptor_count": int(len(receptors)),
            "groups": mode_results,
        }
        cumulative_results[mode] = cumulative_mode
    independent_gate = all(
        results[mode]["groups"][group]["all_populations_passed"]
        for mode in results
        for group in independent
    )
    minimum_sufficient = next(
        (
            int(length)
            for length in config["cumulative_path_lengths_edges"]
            if all(
                cumulative_results["balanced_1914"][group]["by_total_path_length"][str(length)][
                    "all_populations_passed"
                ]
                for group in independent
            )
        ),
        None,
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "path_length_edges": int(config["path_length_edges"]),
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "results": results,
        "cumulative_path_coverage": cumulative_results,
        "minimum_sufficient_total_path_length_balanced": minimum_sufficient,
        "independent_fast_delayed_coverage_gate_passed": bool(independent_gate),
        "authorize_four_hop_scalar_reichardt_precheck": minimum_sufficient == 4,
        "authorize_scalar_reichardt": False,
        "advance_to_calibration": False,
        "boundary": config["boundary"],
    }
