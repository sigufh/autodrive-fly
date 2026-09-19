"""Audit native MaleCNS CT1 synapse-level Lo1 columnar structure."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

import pyarrow.feather as feather
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_neuroglancer_sharded import (
    ShardingSpec,
    decode_column_pin_annotations,
    decode_synapse_annotations,
    read_all_shard_values,
    read_sharded_value,
)

CONFIG = Path("configs/driving-v7-malecns-ct1-columnar-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_malecns_ct1_columnar_audit.py"
)
SHARDED_IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_neuroglancer_sharded.py"
)


def _verified_path(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"frozen MaleCNS object changed: {spec['path']}")
    return path


def _local_reader(directory: Path):
    def read(shard_name: str, start: int, end: int) -> bytes:
        with (directory / shard_name).open("rb") as stream:
            stream.seek(start)
            payload = stream.read(end - start)
        if len(payload) != end - start:
            raise ValueError(f"short shard read: {directory / shard_name}")
        return payload

    return read


def _id_set_sha256(values: set[int]) -> str:
    payload = b"".join(struct.pack("<Q", value) for value in sorted(values))
    return hashlib.sha256(payload).hexdigest()


def _pin_column_universe(path: Path, relation: dict, expected_roi: int) -> set[int]:
    values = read_all_shard_values(
        path.read_bytes(), ShardingSpec.from_json(relation["sharding"])
    )
    columns = set(values) - {0}
    for column_id in columns:
        pins = decode_column_pin_annotations(values[column_id])
        hexes = {(row["hex1"], row["hex2"]) for row in pins}
        if len(hexes) != 1 or any(row["roi"] != expected_roi for row in pins):
            raise ValueError(f"LO column pin mapping is not unique: {column_id}")
    return columns


def _relation_summary(rows: list[dict], roi_id: int, official_columns: set[int]) -> dict:
    roi_counts = Counter(row["primary_roi"] for row in rows)
    layer_counts = Counter(
        (row["primary_roi"], row["optic_layer"]) for row in rows
    )
    lo1_rows = [
        row for row in rows if row["primary_roi"] == roi_id and row["optic_layer"] == 1
    ]
    column_counts = Counter(row["optic_column"] for row in lo1_rows)
    zero_column_count = column_counts.pop(0, 0)
    columns = set(column_counts)
    if not columns <= official_columns:
        raise ValueError("CT1 Lo1 synapses contain unknown official column IDs")
    return {
        "total_synapse_count": len(rows),
        "primary_roi_counts": {str(key): value for key, value in sorted(roi_counts.items())},
        "primary_roi_layer_counts": {
            f"{roi}:{layer}": value
            for (roi, layer), value in sorted(layer_counts.items())
        },
        "Lo1_synapse_count": len(lo1_rows),
        "Lo1_zero_column_synapse_count": zero_column_count,
        "Lo1_all_synapses_have_native_column": zero_column_count == 0,
        "Lo1_column_count": len(columns),
        "Lo1_column_ids_sha256": _id_set_sha256(columns),
        "Lo1_column_ids": sorted(columns),
    }


def evaluate_v7_malecns_ct1_columnar_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    dependencies = [CONFIG, IMPLEMENTATION, SHARDED_IMPLEMENTATION]
    annotation_path = Path(config["annotations"])
    dependencies.append(annotation_path)
    for group in (
        "synapse_annotation_info",
        "optic_column_pin_info",
    ):
        path = _verified_path(root, config[group])
        dependencies.append(path.relative_to(root))
    for group in ("relationship_shards", "LO_column_shards"):
        for spec in config[group].values():
            path = _verified_path(root, spec)
            dependencies.append(path.relative_to(root))
    official_query = _verified_path(
        root, config["official_layer_semantics"]["query_source"]
    )
    dependencies.append(official_query.relative_to(root))

    annotations = feather.read_table(
        root / annotation_path,
        columns=["bodyId", "type", "instance", "somaSide"],
    ).to_pandas()
    synapse_info = json.loads(
        (root / config["synapse_annotation_info"]["path"]).read_text()
    )
    pin_info = json.loads(
        (root / config["optic_column_pin_info"]["path"]).read_text()
    )
    if [item["id"] for item in synapse_info["properties"]][7:9] != [
        "optic_column",
        "optic_layer",
    ]:
        raise ValueError("MaleCNS synapse optic-column/layer schema changed")
    relations = {item["id"]: item for item in synapse_info["relationships"]}
    pin_relations = {item["id"]: item for item in pin_info["relationships"]}
    roi_ids = {"L": 43, "R": 44}
    official_columns = {}
    for side in "LR":
        relation = pin_relations[f"LO({side})_column_segment"]
        path = root / config["LO_column_shards"][side]["path"]
        official_columns[side] = _pin_column_universe(
            path, relation, roi_ids[side]
        )

    bodies = {}
    unions = {}
    expected = config["expected"]
    for raw_body_id, body_spec in config["CT1_bodies"].items():
        body_id = int(raw_body_id)
        selected = annotations.loc[annotations["bodyId"].eq(body_id)]
        if len(selected) != 1 or selected.iloc[0]["type"] != "CT1":
            raise ValueError(f"CT1 annotation changed: {body_id}")
        annotation = selected.iloc[0]
        if (
            annotation["instance"] != body_spec["instance"]
            or annotation["somaSide"] != body_spec["soma_side"]
        ):
            raise ValueError(f"CT1 identity fields changed: {body_id}")
        optic_side = body_spec["optic_lobe_side"]
        relation_summaries = {}
        provenance = {}
        for relationship in ("body_pre", "body_post"):
            relation = relations[relationship]
            shard_path = root / config["relationship_shards"][relationship]["path"]
            payload, provenance[relationship] = read_sharded_value(
                body_id,
                ShardingSpec.from_json(relation["sharding"]),
                _local_reader(shard_path.parent),
            )
            synapses = decode_synapse_annotations(payload or b"")
            body_field = "body_pre_u32" if relationship == "body_pre" else "body_post_u32"
            if any(row[body_field] != body_id for row in synapses):
                raise ValueError(
                    f"relationship index returned wrong body ID: {body_id} {relationship}"
                )
            relation_summaries[relationship] = _relation_summary(
                synapses,
                roi_ids[optic_side],
                official_columns[optic_side],
            )
        pre_columns = set(relation_summaries["body_pre"].pop("Lo1_column_ids"))
        post_columns = set(relation_summaries["body_post"].pop("Lo1_column_ids"))
        union = pre_columns | post_columns
        intersection = pre_columns & post_columns
        unions[optic_side] = union
        expected_body = expected["bodies"][body_id]
        for relationship in ("body_pre", "body_post"):
            summary = relation_summaries[relationship]
            if summary["total_synapse_count"] != expected_body["total_synapses"][relationship]:
                raise ValueError(f"CT1 total synapse count changed: {body_id} {relationship}")
            if summary["Lo1_synapse_count"] != expected_body["Lo1_synapses"][relationship]:
                raise ValueError(f"CT1 Lo1 synapse count changed: {body_id} {relationship}")
            if summary["Lo1_column_count"] != expected_body["Lo1_columns"][relationship]:
                raise ValueError(f"CT1 Lo1 column count changed: {body_id} {relationship}")
        if (
            len(union) != expected_body["Lo1_columns"]["union"]
            or len(intersection) != expected_body["Lo1_columns"]["intersection"]
        ):
            raise ValueError(f"CT1 Lo1 union/intersection changed: {body_id}")
        missing = official_columns[optic_side] - union
        if len(missing) != expected_body["missing_official_LO_columns"]:
            raise ValueError(f"CT1 missing Lo1 column count changed: {body_id}")
        bodies[str(body_id)] = {
            "annotation": {
                "type": "CT1",
                "instance": str(annotation["instance"]),
                "soma_side": str(annotation["somaSide"]),
                "optic_lobe_side_from_synapse_ROI": optic_side,
            },
            "relationship_index_provenance": provenance,
            "synapse_directions": relation_summaries,
            "Lo1_column_union": {
                "column_count": len(union),
                "official_LO_column_count": len(official_columns[optic_side]),
                "official_LO_column_fraction": len(union)
                / len(official_columns[optic_side]),
                "column_ids_sha256": _id_set_sha256(union),
                "missing_column_count": len(missing),
                "missing_column_ids": sorted(missing),
                "missing_column_ids_sha256": _id_set_sha256(missing),
            },
            "Lo1_column_intersection": {
                "column_count": len(intersection),
                "official_LO_column_fraction": len(intersection)
                / len(official_columns[optic_side]),
                "column_ids_sha256": _id_set_sha256(intersection),
            },
        }

    observed_intersection = unions["L"] & unions["R"]
    observed_union = unions["L"] | unions["R"]
    official_intersection = official_columns["L"] & official_columns["R"]
    official_union = official_columns["L"] | official_columns["R"]
    bilateral = {
        "observed_column_intersection_count": len(observed_intersection),
        "observed_column_union_count": len(observed_union),
        "observed_jaccard": len(observed_intersection) / len(observed_union),
        "official_bilateral_column_intersection_count": len(official_intersection),
        "official_bilateral_column_union_count": len(official_union),
        "both_CT1_cover_fraction_of_official_common": len(
            observed_intersection & official_intersection
        )
        / len(official_intersection),
        "observed_column_intersection_sha256": _id_set_sha256(observed_intersection),
        "observed_column_union_sha256": _id_set_sha256(observed_union),
    }
    for key, value in expected["bilateral"].items():
        if bilateral[key] != value:
            raise ValueError(f"CT1 bilateral column count changed: {key}")
    per_synapse_available = all(
        direction["Lo1_synapse_count"] > 0
        and direction["Lo1_all_synapses_have_native_column"]
        for body in bodies.values()
        for direction in body["synapse_directions"].values()
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(path): _sha256(root / path) for path in dependencies
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "official_layer_semantics": config["official_layer_semantics"],
        "official_LO_column_universe": {
            side: {
                "column_count": len(columns),
                "column_ids_sha256": _id_set_sha256(columns),
            }
            for side, columns in official_columns.items()
        },
        "CT1_bodies": bodies,
        "bilateral": bilateral,
        "CT1_per_synapse_Lo1_columnar_retinotopy_available": per_synapse_available,
        "CT1_complete_official_LO_column_coverage": all(
            not body["Lo1_column_union"]["missing_column_count"]
            for body in bodies.values()
        ),
        "authorize_CT1_experimental_voltage_transfer": False,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "boundary": config["boundary"],
    }
