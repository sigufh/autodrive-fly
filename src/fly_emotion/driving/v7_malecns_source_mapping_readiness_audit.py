"""Audit MaleCNS type/body and retinotopic mapping for the nine source types."""

from __future__ import annotations

import hashlib
import json
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

CONFIG = Path("configs/driving-v7-malecns-source-mapping-readiness-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_malecns_source_mapping_readiness_audit.py")


def _body_set_sha256(body_ids: np.ndarray) -> str:
    return hashlib.sha256(np.sort(body_ids).astype("<i8", copy=False).tobytes()).hexdigest()


def evaluate_v7_malecns_source_mapping_readiness_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    manifest_path = Path(config["data_manifest"])
    manifest = yaml.safe_load((root / manifest_path).read_text(encoding="utf-8"))
    annotation_path = Path(config["annotations"])
    annotation_spec = manifest["datasets"]["malecns"]["files"]["annotations"]
    if (root / annotation_path).stat().st_size != int(annotation_spec["bytes"]):
        raise ValueError("MaleCNS annotation size mismatch")
    if _sha256(root / annotation_path) != annotation_spec["sha256"]:
        raise ValueError("MaleCNS annotation SHA-256 mismatch")
    contract_path = Path(config["external_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    synapse_column_path = Path(config["official_synapse_column_evidence"])
    synapse_column = json.loads((root / synapse_column_path).read_text(encoding="utf-8"))
    expected_types = {
        source
        for family in contract["required_families"].values()
        for source in family["source_types"]
    }
    if set(config["required_sources"]) != expected_types:
        raise ValueError("MaleCNS mapping audit source types differ from external contract")
    annotations = feather.read_table(
        root / annotation_path,
        columns=[
            "bodyId",
            "type",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
            "instance",
            "flywireType",
            "vfbId",
        ],
        memory_map=True,
    ).to_pandas()
    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
    annotation_ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(graph.body_ids, annotation_ids)
    valid = nodes < graph.node_count
    valid[valid] &= graph.body_ids[nodes[valid]] == annotation_ids[valid]
    rows = annotations.iloc[np.flatnonzero(valid)].copy()
    nodes = nodes[valid]
    node_types = np.full(graph.node_count, "", dtype=object)
    node_sides = np.full(graph.node_count, "", dtype=object)
    coordinates = np.full((graph.node_count, 2), np.nan, dtype=np.float64)
    node_types[nodes] = rows["type"].fillna("").to_numpy(dtype=object)
    node_sides[nodes] = rows["somaSide"].fillna("").to_numpy(dtype=object)
    coordinates[nodes] = rows[["assignedOlHex1", "assignedOlHex2"]].to_numpy(dtype=np.float64)
    source_nodes = np.flatnonzero(np.isin(node_types, sorted(expected_types)))
    positions, inferred = _one_hop_positions(graph, coordinates, source_nodes)
    native = np.all(np.isfinite(coordinates), axis=1)
    located = np.all(np.isfinite(positions), axis=1)
    source_reports = {}
    for source_type, expected in config["required_sources"].items():
        selected = np.flatnonzero(node_types == source_type)
        body_ids = graph.body_ids[selected]
        side_counts = {
            side: int(np.count_nonzero(node_sides[selected] == side)) for side in ("L", "R")
        }
        actual = {
            "count": int(selected.size),
            "left": side_counts["L"],
            "right": side_counts["R"],
            "native": int(np.count_nonzero(native[selected])),
            "inferred": int(np.count_nonzero(inferred[selected])),
            "unlocated": int(np.count_nonzero(~located[selected])),
        }
        for name, value in actual.items():
            if value != int(expected[name]):
                raise ValueError(f"MaleCNS {source_type} {name} changed")
        annotated = rows.loc[rows["type"].fillna("").eq(source_type)]
        external_ids_complete = all(
            annotated[column].fillna("").ne("").all()
            for column in ("instance", "flywireType", "vfbId")
        )
        one_hop_unlocated_body_ids = graph.body_ids[selected[~located[selected]]].tolist()
        officially_recovered_body_ids = (
            [int(synapse_column["target"]["body_id"])]
            if source_type == "Tm9"
            and synapse_column[
                "Tm9_532266_official_synapse_column_coordinate_identifiable"
            ]
            else []
        )
        unlocated_body_ids = [
            body_id
            for body_id in one_hop_unlocated_body_ids
            if body_id not in officially_recovered_body_ids
        ]
        source_reports[source_type] = {
            "family": expected["family"],
            "body_count": actual["count"],
            "body_ids_sha256": _body_set_sha256(body_ids),
            "all_bodies_in_canonical_graph": True,
            "soma_side_counts": side_counts,
            "soma_side_complete": sum(side_counts.values()) == actual["count"],
            "external_annotation_ids_complete": bool(external_ids_complete),
            "native_optic_hex_count": actual["native"],
            "one_hop_inferred_coordinate_count": actual["inferred"],
            "official_synapse_column_recovered_count": len(officially_recovered_body_ids),
            "official_synapse_column_recovered_body_ids": officially_recovered_body_ids,
            "official_synapse_column_recovered_coordinates": (
                [synapse_column["Tm9_532266_recovered_coordinate"]]
                if officially_recovered_body_ids
                else []
            ),
            "one_hop_unlocated_body_ids": one_hop_unlocated_body_ids,
            "located_count": actual["count"] - len(unlocated_body_ids),
            "located_fraction": (actual["count"] - len(unlocated_body_ids)) / actual["count"],
            "unlocated_body_ids": unlocated_body_ids,
            "columnar_retinotopy_available": bool(
                source_type != "CT1" and len(unlocated_body_ids) == 0
            ),
        }
    ct1_ids = sorted(graph.body_ids[np.flatnonzero(node_types == "CT1")].astype(int).tolist())
    if ct1_ids != config["CT1_expected_body_ids"]:
        raise ValueError("MaleCNS CT1 body IDs changed")
    exact_type_sets = all(
        item["body_count"] > 0 and item["all_bodies_in_canonical_graph"]
        for item in source_reports.values()
    )
    sides_complete = all(item["soma_side_complete"] for item in source_reports.values())
    coordinate_complete = all(
        item["columnar_retinotopy_available"] for item in source_reports.values()
    )
    gates = {
        "every_source_type_has_exact_MaleCNS_body_set": exact_type_sets,
        "every_source_body_has_soma_side": sides_complete,
        "every_source_body_has_columnar_retinotopic_coordinate": coordinate_complete,
        "external_recording_declares_explicit_type_average_or_body_mapping": False,
        "external_recording_to_specific_MaleCNS_body_identified": False,
    }
    ready = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(manifest_path): _sha256(root / manifest_path),
                str(annotation_path): _sha256(root / annotation_path),
                str(Path(config["adjacency_raw"])): _sha256(root / config["adjacency_raw"]),
                str(Path(config["body_ids"])): _sha256(root / config["body_ids"]),
                str(Path(config["graph_metadata"])): _sha256(root / config["graph_metadata"]),
                str(ONE_HOP_IMPLEMENTATION): _sha256(root / ONE_HOP_IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
                str(synapse_column_path): _sha256(root / synapse_column_path),
            },
            "MaleCNS_release": manifest["datasets"]["malecns"]["release"],
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "coordinate_rule": config["coordinate_rule"],
        "source_mapping": source_reports,
        "CT1": {
            "body_ids": ct1_ids,
            "body_count": len(ct1_ids),
            "sides": ["L", "R"],
            "one_body_per_side": True,
            "one_hop_global_centroid_available": True,
            "columnar_Lo1_retinotopy_available": False,
        },
        "mapping_gates": gates,
        "MaleCNS_type_average_body_sets_available": exact_type_sets,
        "external_source_mapping_contract_satisfied": ready,
        "authorize_external_source_payload": ready,
        "advance_to_T4_T5_functional_precheck": ready,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if ready
            else (
                "MaleCNS_type_body_sets_exist_but_external_recording_identity_and_"
                "columnar_mapping_do_not"
            )
        ),
        "boundary": config["boundary"],
    }
