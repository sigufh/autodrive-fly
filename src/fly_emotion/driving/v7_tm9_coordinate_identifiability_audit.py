"""Explain why MaleCNS Tm9 body 532266 lacks an identifiable optic hex."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as dataset
import pyarrow.feather as feather
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc1_input_structure import (
    IMPLEMENTATION as ONE_HOP_IMPLEMENTATION,
)
from fly_emotion.driving.v7_lplc1_input_structure import _one_hop_positions

CONFIG = Path("configs/driving-v7-tm9-coordinate-identifiability-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_tm9_coordinate_identifiability_audit.py"
)


def _summary(values: list[float], exact: list[bool], support: list[tuple[int, float]]) -> dict:
    errors = np.asarray(values, dtype=np.float64)
    supports = np.asarray(support, dtype=np.float64)
    return {
        "evaluated_count": int(errors.size),
        "median_error": float(np.median(errors)),
        "p90_error": float(np.quantile(errors, 0.90)),
        "p95_error": float(np.quantile(errors, 0.95)),
        "maximum_error": float(np.max(errors)),
        "rounded_exact_fraction": float(np.mean(exact)),
        "median_support_count": float(np.median(supports[:, 0])),
        "median_support_weight": float(np.median(supports[:, 1])),
    }


def _candidate(
    graph,
    matrix,
    node: int,
    positions: np.ndarray,
    mask_by_node: np.ndarray,
) -> tuple[np.ndarray | None, list[int], list[int]]:
    row = matrix.getrow(node)
    selected = mask_by_node[row.indices]
    if not np.any(selected):
        return None, [], []
    sources = row.indices[selected]
    weights = np.abs(row.data[selected]).astype(np.float64)
    coordinate = np.average(positions[sources], axis=0, weights=weights)
    return coordinate, sources.astype(int).tolist(), weights.astype(int).tolist()


def _swc_summary(path: Path) -> dict:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        points.append([float(fields[2]), float(fields[3]), float(fields[4])])
    values = np.asarray(points, dtype=np.float64)
    return {
        "node_count": int(values.shape[0]),
        "minimum_xyz": values.min(axis=0).tolist(),
        "maximum_xyz": values.max(axis=0).tolist(),
        "centroid_xyz": values.mean(axis=0).tolist(),
        "root_xyz": values[0].tolist(),
        "coordinate_semantics": "MaleCNS_tissue_xyz_not_optic_hex",
    }


def evaluate_v7_tm9_coordinate_identifiability_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    manifest_path = Path(config["data_manifest"])
    manifest = yaml.safe_load((root / manifest_path).read_text(encoding="utf-8"))
    annotation_path = Path(config["annotations"])
    raw_connections_path = Path(config["raw_connections"])
    mapping_path = Path(config["mapping_evidence"])
    mapping_config_path = Path(config["mapping_config"])
    mapping = json.loads((root / mapping_path).read_text(encoding="utf-8"))
    mapping_config = yaml.safe_load((root / mapping_config_path).read_text(encoding="utf-8"))
    synapse_column_path = Path(config["official_synapse_column_evidence"])
    synapse_column = json.loads((root / synapse_column_path).read_text(encoding="utf-8"))
    if mapping["source_mapping"]["Tm9"]["one_hop_unlocated_body_ids"] != [
        int(config["target_body_id"])
    ]:
        raise ValueError("existing Tm9 mapping evidence unlocated set changed")

    release_spec = config["current_release_registry"]
    release_path = root / release_spec["path"]
    if (
        release_path.stat().st_size != int(release_spec["bytes"])
        or _sha256(release_path) != release_spec["sha256"]
    ):
        raise ValueError("MaleCNS public release registry snapshot changed")
    release_registry = json.loads(release_path.read_text(encoding="utf-8"))
    public_releases = {
        name: details
        for name, details in release_registry.items()
        if name.startswith("male-cns:") and not details["hidden"]
    }
    if (
        sorted(public_releases) != ["male-cns:v0.9", "male-cns:v1.0"]
        or release_spec["latest_public_release"] not in public_releases
        or public_releases[release_spec["latest_public_release"]]["uuid"]
        != release_spec["latest_public_release_uuid"]
    ):
        raise ValueError("MaleCNS latest public release boundary changed")

    object_spec = config["current_annotation_object"]
    object_path = root / object_spec["path"]
    if (
        object_path.stat().st_size != int(object_spec["bytes"])
        or _sha256(object_path) != object_spec["sha256"]
    ):
        raise ValueError("MaleCNS annotation object observation changed")
    object_observation = json.loads(object_path.read_text(encoding="utf-8"))
    if (
        object_observation["response_status"] != int(object_spec["response_status"])
        or object_observation["etag_md5"] != object_spec["etag_md5"]
        or not object_observation["downloaded_object_matches_frozen_local_file"]
    ):
        raise ValueError("MaleCNS current annotation object identity changed")

    files = manifest["datasets"]["malecns"]["files"]
    for path, spec in (
        (annotation_path, files["annotations"]),
        (raw_connections_path, files["connections"]),
    ):
        if (root / path).stat().st_size != int(spec["bytes"]):
            raise ValueError(f"MaleCNS source size changed: {path}")
        if _sha256(root / path) != spec["sha256"]:
            raise ValueError(f"MaleCNS source SHA-256 changed: {path}")

    skeleton_spec = config["target_skeleton"]
    skeleton_path = root / skeleton_spec["path"]
    if skeleton_path.stat().st_size != int(skeleton_spec["bytes"]):
        raise ValueError("Tm9 target skeleton size changed")
    if _sha256(skeleton_path) != skeleton_spec["sha256"]:
        raise ValueError("Tm9 target skeleton SHA-256 changed")
    skeleton = _swc_summary(skeleton_path)
    if skeleton["node_count"] != int(skeleton_spec["node_count"]):
        raise ValueError("Tm9 target skeleton node count changed")

    partner_spec = config["synapse_partners"]
    partner_path = root / partner_spec["path"]
    if (
        partner_path.stat().st_size != int(partner_spec["bytes"])
        or _sha256(partner_path) != partner_spec["sha256"]
    ):
        raise ValueError("MaleCNS synapse-partner file changed")
    partners = dataset.dataset(root / partner_path, format="feather")
    if partners.schema.names != partner_spec["expected_schema"]:
        raise ValueError("MaleCNS synapse-partner schema changed")

    columns = [
        "bodyId",
        "group",
        "type",
        "flywireType",
        "instance",
        "somaSide",
        "status",
        "statusLabel",
        "superclass",
        "assignedOlHex1",
        "assignedOlHex2",
    ]
    annotations = feather.read_table(
        root / annotation_path, columns=columns, memory_map=True
    ).to_pandas()
    body_id = int(config["target_body_id"])
    incoming_synapses = partners.to_table(
        filter=dataset.field("body_post") == body_id
    ).to_pandas()
    outgoing_synapses = partners.to_table(
        filter=dataset.field("body_pre") == body_id
    ).to_pandas()
    incoming_roi_counts = {
        str(key): int(value)
        for key, value in incoming_synapses["primary_post"]
        .astype(object)
        .value_counts()
        .sort_index()
        .items()
    }
    outgoing_roi_counts = {
        str(key): int(value)
        for key, value in outgoing_synapses["primary_post"]
        .astype(object)
        .value_counts()
        .sort_index()
        .items()
    }
    partner_checks = {
        "target_incoming_rows": len(incoming_synapses),
        "target_outgoing_rows": len(outgoing_synapses),
        "target_incoming_unique_bodies": int(incoming_synapses["body_pre"].nunique()),
        "target_outgoing_unique_bodies": int(outgoing_synapses["body_post"].nunique()),
    }
    for name, value in partner_checks.items():
        if value != int(partner_spec[f"expected_{name}"]):
            raise ValueError(f"Tm9 synapse-partner count changed: {name}")
    if (
        incoming_roi_counts != partner_spec["expected_incoming_primary_post"]
        or outgoing_roi_counts != partner_spec["expected_outgoing_primary_post"]
    ):
        raise ValueError("Tm9 synapse-partner ROI distribution changed")
    target_rows = annotations.loc[annotations["bodyId"].eq(body_id)]
    if len(target_rows) != 1:
        raise ValueError("Tm9 target annotation is not unique")
    target = target_rows.iloc[0]
    for key, expected in config["expected_annotation"].items():
        actual = int(target[key]) if key == "group" else str(target[key])
        if actual != expected:
            raise ValueError(f"Tm9 target annotation changed: {key}")
    native_coordinate = bool(
        np.isfinite(target["assignedOlHex1"])
        and np.isfinite(target["assignedOlHex2"])
    )
    if native_coordinate:
        raise ValueError("Tm9 target gained a native optic-hex coordinate")

    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
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
    node = int(np.searchsorted(graph.body_ids, body_id))
    if node >= graph.node_count or int(graph.body_ids[node]) != body_id:
        raise ValueError("Tm9 target missing from canonical graph")

    native = np.all(np.isfinite(coordinates), axis=1)
    required_types = set(mapping_config["required_sources"])
    source_nodes = np.flatnonzero(np.isin(node_types, sorted(required_types)))
    positions, inferred = _one_hop_positions(graph, coordinates, source_nodes)
    located = np.all(np.isfinite(positions), axis=1)
    transpose = graph.adjacency.T.tocsr()
    incoming = graph.adjacency.getrow(node)
    outgoing = transpose.getrow(node)

    expected_graph = config["expected_graph"]
    graph_counts = {
        "canonical_incoming_edge_count": incoming.nnz,
        "canonical_incoming_synapse_weight": int(incoming.data.sum(dtype=np.int64)),
        "canonical_outgoing_edge_count": outgoing.nnz,
        "canonical_outgoing_synapse_weight": int(outgoing.data.sum(dtype=np.int64)),
        "native_coordinate_incoming_edge_count": int(
            np.count_nonzero(native[incoming.indices])
        ),
        "native_coordinate_incoming_synapse_weight": int(
            incoming.data[native[incoming.indices]].sum(dtype=np.int64)
        ),
        "native_coordinate_outgoing_edge_count": int(
            np.count_nonzero(native[outgoing.indices])
        ),
        "native_coordinate_outgoing_synapse_weight": int(
            outgoing.data[native[outgoing.indices]].sum(dtype=np.int64)
        ),
    }
    for key, actual in graph_counts.items():
        if actual != int(expected_graph[key]):
            raise ValueError(f"Tm9 target graph count changed: {key}")

    raw = dataset.dataset(root / raw_connections_path, format="feather")
    raw_incoming = raw.to_table(filter=dataset.field("body_post") == body_id).to_pandas()
    raw_counts = {
        "raw_incoming_edge_count": len(raw_incoming),
        "raw_incoming_synapse_weight": int(raw_incoming["weight"].sum()),
    }
    for key, actual in raw_counts.items():
        if actual != int(expected_graph[key]):
            raise ValueError(f"Tm9 target raw graph count changed: {key}")
    canonical_ids = set(graph.body_ids.astype(int).tolist())
    raw_noncanonical = sorted(
        int(value)
        for value in raw_incoming["body_pre"]
        if int(value) not in canonical_ids
    )
    if raw_noncanonical != expected_graph["raw_noncanonical_incoming_body_ids"]:
        raise ValueError("Tm9 raw noncanonical input set changed")
    annotated_ids = set(annotations["bodyId"].astype(int).tolist())
    raw_noncanonical_annotated = [value for value in raw_noncanonical if value in annotated_ids]

    tm9_nodes = np.flatnonzero(node_types == "Tm9")
    tm9_native = tm9_nodes[native[tm9_nodes]]
    tm9_missing_native = tm9_nodes[~native[tm9_nodes]]
    population = {
        "body_count": int(tm9_nodes.size),
        "native_coordinate_count": int(tm9_native.size),
        "missing_native_coordinate_count": int(tm9_missing_native.size),
        "one_hop_inferred_count": int(np.count_nonzero(inferred[tm9_nodes])),
        "one_hop_unlocated_count": int(np.count_nonzero(~located[tm9_nodes])),
    }
    for key, actual in population.items():
        if actual != int(config["expected_Tm9_population"][key]):
            raise ValueError(f"Tm9 population coordinate count changed: {key}")
    unlocated_ids = graph.body_ids[tm9_nodes[~located[tm9_nodes]]].astype(int).tolist()
    if unlocated_ids != [body_id]:
        raise ValueError("Tm9 one-hop unlocated body set changed")

    same_side_mask = located & (node_sides == node_sides[node])
    same_side_coordinate, same_side_sources, same_side_weights = _candidate(
        graph, graph.adjacency, node, positions, same_side_mask
    )
    native_out_coordinate, native_out_sources, native_out_weights = _candidate(
        graph, transpose, node, positions, native
    )
    if same_side_coordinate is None or native_out_coordinate is None:
        raise ValueError("Tm9 diagnostic candidate support disappeared")

    occupancy = {}
    candidate_rows = {}
    for name, coordinate, sources, weights in (
        (
            "same_side_recursive_incoming",
            same_side_coordinate,
            same_side_sources,
            same_side_weights,
        ),
        ("native_outgoing", native_out_coordinate, native_out_sources, native_out_weights),
    ):
        rounded = np.rint(coordinate).astype(int)
        occupied_nodes = tm9_nodes[
            native[tm9_nodes]
            & (node_sides[tm9_nodes] == node_sides[node])
            & np.all(coordinates[tm9_nodes] == rounded.astype(float), axis=1)
        ]
        occupied_ids = graph.body_ids[occupied_nodes].astype(int).tolist()
        expected = config["candidate_diagnostics"][name]
        if not np.allclose(coordinate, expected["coordinate"], rtol=0.0, atol=1e-12):
            raise ValueError(f"Tm9 diagnostic candidate changed: {name}")
        if rounded.tolist() != expected["rounded_coordinate"]:
            raise ValueError(f"Tm9 rounded candidate changed: {name}")
        if occupied_ids != expected["occupied_by_body_ids"]:
            raise ValueError(f"Tm9 candidate occupancy changed: {name}")
        if len(sources) != int(expected["support_count"]) or sum(weights) != int(
            expected["support_weight"]
        ):
            raise ValueError(f"Tm9 candidate support changed: {name}")
        candidate_rows[name] = {
            "coordinate": coordinate.tolist(),
            "rounded_coordinate": rounded.tolist(),
            "support_count": len(sources),
            "support_synapse_weight": sum(weights),
            "support": [
                {
                    "body_id": int(graph.body_ids[source]),
                    "type": str(node_types[source]),
                    "side": str(node_sides[source]),
                    "weight": weight,
                    "coordinate": positions[source].tolist(),
                    "coordinate_is_native": bool(native[source]),
                    "coordinate_is_one_hop_inferred": bool(inferred[source]),
                }
                for source, weight in zip(sources, weights, strict=True)
            ],
            "occupied_by_Tm9_body_ids": occupied_ids,
            "unoccupied_coordinate": not occupied_ids,
            "authorized_for_mapping": False,
        }
        occupancy[name] = not occupied_ids

    validation = {}
    for mode in ("native_input", "same_side_recursive_input", "native_output"):
        errors = []
        exact = []
        supports = []
        matrix = transpose if mode == "native_output" else graph.adjacency
        for current in tm9_native:
            if mode == "native_input":
                mask = native
            elif mode == "same_side_recursive_input":
                mask = located & (node_sides == node_sides[current])
            else:
                mask = native
            prediction, sources, weights = _candidate(
                graph, matrix, int(current), positions, mask
            )
            if prediction is None:
                continue
            errors.append(float(np.linalg.norm(prediction - coordinates[current])))
            exact.append(bool(np.all(np.rint(prediction) == coordinates[current])))
            supports.append((len(sources), float(sum(weights))))
        validation[mode] = {
            "native_Tm9_count": int(tm9_native.size),
            "coverage_fraction": len(errors) / tm9_native.size,
            **_summary(errors, exact, supports),
        }
    recursive = validation["same_side_recursive_input"]
    validation_gates = {
        "same_side_recursive_replay_accuracy": recursive["rounded_exact_fraction"]
        >= float(
            config["validation_gates"][
                "minimum_same_side_recursive_rounded_exact_fraction"
            ]
        ),
        "same_side_recursive_p95_error": recursive["p95_error"]
        <= float(config["validation_gates"]["maximum_same_side_recursive_p95_error"]),
        "same_side_recursive_candidate_unoccupied": occupancy[
            "same_side_recursive_incoming"
        ],
        "native_outgoing_candidate_unoccupied": occupancy["native_outgoing"],
        "candidate_methods_agree": bool(
            np.array_equal(np.rint(same_side_coordinate), np.rint(native_out_coordinate))
        ),
    }
    post_hoc_mapping_authorized = all(validation_gates.values())
    official_synapse_coordinate = synapse_column[
        "Tm9_532266_official_synapse_column_coordinate_identifiable"
    ]
    recovered_coordinate = synapse_column["Tm9_532266_recovered_coordinate"]
    if synapse_column["target"]["body_id"] != body_id:
        raise ValueError("official synapse-column target changed")
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(manifest_path): _sha256(root / manifest_path),
                str(annotation_path): _sha256(root / annotation_path),
                str(raw_connections_path): _sha256(root / raw_connections_path),
                str(partner_spec["path"]): _sha256(partner_path),
                str(Path(config["adjacency_raw"])): _sha256(
                    root / config["adjacency_raw"]
                ),
                str(Path(config["body_ids"])): _sha256(root / config["body_ids"]),
                str(Path(config["graph_metadata"])): _sha256(
                    root / config["graph_metadata"]
                ),
                str(mapping_path): _sha256(root / mapping_path),
                str(mapping_config_path): _sha256(root / mapping_config_path),
                str(ONE_HOP_IMPLEMENTATION): _sha256(root / ONE_HOP_IMPLEMENTATION),
                str(release_spec["path"]): _sha256(release_path),
                str(object_spec["path"]): _sha256(object_path),
                str(synapse_column_path): _sha256(root / synapse_column_path),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "target": {
            "body_id": body_id,
            "annotation": {
                key: (int(target[key]) if key == "group" else str(target[key]))
                for key in config["expected_annotation"]
            },
            "native_optic_hex_available": native_coordinate,
            "existing_one_hop_optic_hex_available": bool(located[node]),
            "skeleton": {
                **skeleton_spec,
                "actual_sha256": _sha256(skeleton_path),
                **skeleton,
            },
        },
        "latest_public_release_check": {
            "observed_on": config["observed_on"],
            "registry_url": release_spec["source_url"],
            "public_male_cns_releases": sorted(public_releases),
            "latest_public_release": release_spec["latest_public_release"],
            "latest_public_release_uuid": release_spec["latest_public_release_uuid"],
            "newer_public_release_present": False,
            "current_annotation_object": object_observation,
            "current_object_matches_frozen_local_annotation": (
                object_observation["sha256"] == _sha256(root / annotation_path)
                and object_observation["bytes"] == (root / annotation_path).stat().st_size
            ),
            "target_native_optic_hex_still_missing_in_latest_public_release": (
                not native_coordinate
            ),
        },
        "graph_evidence": {
            **graph_counts,
            **raw_counts,
            "raw_noncanonical_incoming_body_ids": raw_noncanonical,
            "raw_noncanonical_incoming_bodies_with_annotations": (
                raw_noncanonical_annotated
            ),
            "canonical_filter_removed_coordinate_support": False,
        },
        "synapse_level_evidence": {
            "schema": partners.schema.names,
            **partner_checks,
            "incoming_primary_post_counts": incoming_roi_counts,
            "outgoing_primary_post_counts": outgoing_roi_counts,
            "optic_column_ROI_field_present": False,
            "coordinate_fields_are_tissue_xyz_not_optic_hex": True,
            "native_optic_hex_recovered": False,
        },
        "Tm9_population": population,
        "candidate_diagnostics": candidate_rows,
        "blind_replay_on_native_Tm9": validation,
        "candidate_validation_gates": validation_gates,
        "official_synapse_column_evidence": {
            "body_annotation_native_fields_remain_missing": (not native_coordinate),
            "official_rule_native_Tm9_replay": synapse_column["native_Tm9_replay"],
            "input_optic_column_counts": synapse_column["target"]["input_optic_column_counts"],
            "output_optic_column_counts": synapse_column["target"]["output_optic_column_counts"],
            "consensus_column_id": synapse_column["target"]["consensus_column_id"],
            "recovered_coordinate": recovered_coordinate,
            "coordinate_source": "official_MaleCNS_v1_0_synapse_and_column_pin_annotations",
        },
        "Tm9_532266_coordinate_identifiable_under_existing_rule": bool(
            located[node]
        ),
        "Tm9_532266_coordinate_identifiable_under_official_synapse_rule": bool(
            official_synapse_coordinate
        ),
        "Tm9_532266_coordinate_repair_authorized": post_hoc_mapping_authorized,
        "complete_Tm9_columnar_retinotopy_available": bool(
            located[node] or official_synapse_coordinate
        ),
        "authorize_source_mapping_update": bool(official_synapse_coordinate),
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            None
            if official_synapse_coordinate
            else (
                "existing_rule_has_no_native_input_support_and_post_hoc_candidates_"
                "conflict_or_are_occupied"
            )
        ),
        "boundary": config["boundary"],
    }
