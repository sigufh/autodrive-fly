"""Audit an official MaleCNS synapse-column route for Tm9 body 532266."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.feather as feather
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_neuroglancer_sharded import (
    ShardingSpec,
    decode_column_pin_annotations,
    decode_synapse_annotations,
    read_sharded_value,
)

CONFIG = Path("configs/driving-v7-malecns-synapse-column-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_malecns_synapse_column_audit.py"
)
SHARDED_IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_neuroglancer_sharded.py"
)
RETRIEVAL_IMPLEMENTATION = Path("scripts/inspect_malecns_optic_columns.py")


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


def _column_hex(pin_map: dict[int, list[dict]], column_id: int, side: str) -> list[int] | None:
    hexes = {
        tuple(hex_)
        for match in pin_map[column_id]
        if f"({side})" in match["relationship"]
        for hex_ in match["hexes"]
    }
    return list(next(iter(hexes))) if len(hexes) == 1 else None


def _official_synapse_count_candidate(
    row: dict, pin_map: dict[int, list[dict]], side: str
) -> dict:
    roi_id = 47 if side == "L" else 48
    counts = Counter()
    for relationship in ("body_pre", "body_post"):
        for pair, count in row[f"{relationship}_roi_column"].items():
            raw_roi, raw_column = pair.split(":")
            column_id = int(raw_column)
            if int(raw_roi) != roi_id or not column_id:
                continue
            coordinate = _column_hex(pin_map, column_id, side)
            if coordinate is not None:
                counts[tuple(coordinate)] += int(count)
    maximum = max(counts.values()) if counts else 0
    modes = [coordinate for coordinate, count in counts.items() if count == maximum]
    return {
        "ROI": f"ME({side})",
        "hex_counts": {f"{a},{b}": count for (a, b), count in sorted(counts.items())},
        "unique_mode": len(modes) == 1,
        "recovered_hex": list(modes[0]) if len(modes) == 1 else None,
    }


def evaluate_v7_malecns_synapse_column_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    dependencies = [
        CONFIG,
        IMPLEMENTATION,
        SHARDED_IMPLEMENTATION,
        RETRIEVAL_IMPLEMENTATION,
    ]
    for spec in config["official_algorithm"].values():
        if isinstance(spec, dict) and "path" in spec:
            path = _verified_path(root, spec)
            dependencies.append(path.relative_to(root))
    annotation_path = Path(config["annotations"])
    synapse_info_path = _verified_path(root, config["synapse_annotation_info"])
    pin_info_path = _verified_path(root, config["optic_column_pin_info"])
    dependencies.extend(
        (
            annotation_path,
            synapse_info_path.relative_to(root),
            pin_info_path.relative_to(root),
        )
    )
    for group in ("target_shards", "pin_shards", "validation_payloads"):
        for spec in config[group].values():
            if "bytes" in spec and "sha256" in spec:
                path = _verified_path(root, spec)
                dependencies.append(path.relative_to(root))

    annotations = feather.read_table(
        root / annotation_path,
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
    ).to_pandas()
    target_body_id = int(config["target_body_id"])
    target_rows = annotations.loc[annotations["bodyId"].eq(target_body_id)]
    if len(target_rows) != 1 or target_rows.iloc[0]["type"] != "Tm9":
        raise ValueError("target is not one unique Tm9 annotation")
    target_annotation = target_rows.iloc[0]
    native_missing = bool(
        target_annotation[["assignedOlHex1", "assignedOlHex2"]].isna().all()
    )
    if not native_missing:
        raise ValueError("target body annotation no longer lacks native hex fields")

    synapse_info = json.loads(synapse_info_path.read_text())
    relations = {item["id"]: item for item in synapse_info["relationships"]}
    target_synapses = {}
    target_provenance = {}
    for relationship in ("body_pre", "body_post"):
        relation = relations[relationship]
        shard_path = root / config["target_shards"][relationship]["path"]
        payload, provenance = read_sharded_value(
            target_body_id,
            ShardingSpec.from_json(relation["sharding"]),
            _local_reader(shard_path.parent),
        )
        target_synapses[relationship] = decode_synapse_annotations(payload or b"")
        target_provenance[relationship] = provenance

    target_counts = {
        relationship: Counter(row["optic_column"] for row in rows)
        for relationship, rows in target_synapses.items()
    }
    expected = config["expected"]
    if len(target_synapses["body_post"]) != expected["target_incoming_synapse_count"]:
        raise ValueError("target incoming synapse count changed")
    if len(target_synapses["body_pre"]) != expected["target_outgoing_synapse_count"]:
        raise ValueError("target outgoing synapse count changed")
    for relation, expected_name in (
        ("body_post", "target_input_column_counts"),
        ("body_pre", "target_output_column_counts"),
    ):
        frozen = Counter({int(key): value for key, value in expected[expected_name].items()})
        if target_counts[relation] != frozen:
            raise ValueError(f"target {relation} optic-column counts changed")

    validation_path = root / config["validation_payloads"]["synapse_counts"]["path"]
    validation_rows = json.loads(validation_path.read_text())
    pin_map_path = root / config["validation_payloads"]["pin_map"]["path"]
    pin_map = {int(key): value for key, value in json.loads(pin_map_path.read_text()).items()}
    official_validation = []
    for row in validation_rows:
        annotation = annotations.loc[annotations["bodyId"].eq(row["body_id"])].iloc[0]
        candidate = _official_synapse_count_candidate(
            row, pin_map, str(annotation["somaSide"])
        )
        if candidate["unique_mode"]:
            native_hex = [int(annotation["assignedOlHex1"]), int(annotation["assignedOlHex2"])]
            official_validation.append(candidate["recovered_hex"] == native_hex)
    if len(validation_rows) != expected["native_validation_Tm9_count"]:
        raise ValueError("native Tm9 validation denominator changed")
    if (len(official_validation), sum(official_validation)) != (
        expected["official_rule_evaluable_count"],
        expected["official_rule_exact_count"],
    ):
        raise ValueError("official synapse-count replay changed")
    exact_fraction = sum(official_validation) / len(official_validation)

    target_pin_map = {}
    pin_info = json.loads(pin_info_path.read_text())
    pin_relations = {item["id"]: item for item in pin_info["relationships"]}
    for column_id in sorted(
        (set(target_counts["body_pre"]) | set(target_counts["body_post"])) - {0}
    ):
        matches = []
        for neuropil in ("ME", "LO", "LOP"):
            name = f"{neuropil}(L)_column_segment"
            relation = pin_relations[name]
            local_name = name.replace("(L)", "_L")
            directory = (root / config["pin_shards"][local_name.split("_column")[0]]["path"]).parent
            payload, _ = read_sharded_value(
                column_id,
                ShardingSpec.from_json(relation["sharding"]),
                _local_reader(directory),
            )
            if payload is not None:
                pins = decode_column_pin_annotations(payload)
                matches.append(
                    {
                        "relationship": name,
                        "annotation_count": len(pins),
                        "hexes": [
                            list(value)
                            for value in sorted(
                                {(p["hex1"], p["hex2"]) for p in pins}
                            )
                        ],
                    }
                )
        target_pin_map[column_id] = matches
    target_row = {
        "body_pre": dict(target_counts["body_pre"]),
        "body_post": dict(target_counts["body_post"]),
        "body_pre_roi_column": dict(
            Counter(
                f"{row['primary_roi']}:{row['optic_column']}"
                for row in target_synapses["body_pre"]
            )
        ),
        "body_post_roi_column": dict(
            Counter(
                f"{row['primary_roi']}:{row['optic_column']}"
                for row in target_synapses["body_post"]
            )
        ),
    }
    target_candidate = _official_synapse_count_candidate(target_row, target_pin_map, "L")
    recovered = bool(
        target_candidate["unique_mode"]
        and target_candidate["recovered_hex"] == expected["target_recovered_hex"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {str(path): _sha256(root / path) for path in dependencies},
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "official_sources": {
            "synapse_annotation_schema": synapse_info,
            "optic_column_pin_schema": pin_info,
        },
        "official_algorithm": config["official_algorithm"],
        "native_Tm9_replay": {
            "native_Tm9_count": len(validation_rows),
            "evaluable_count": len(official_validation),
            "exact_count": sum(official_validation),
            "exact_fraction": exact_fraction,
            "ties_or_missing_ME_column_count": len(validation_rows) - len(official_validation),
            "replay_is_diagnostic_not_acceptance_threshold": True,
        },
        "target": {
            "body_id": target_body_id,
            "side": str(target_annotation["somaSide"]),
            "body_annotation_native_hex_missing": native_missing,
            "incoming_synapse_count": len(target_synapses["body_post"]),
            "outgoing_synapse_count": len(target_synapses["body_pre"]),
            "input_optic_column_counts": dict(sorted(target_counts["body_post"].items())),
            "output_optic_column_counts": dict(sorted(target_counts["body_pre"].items())),
            "relationship_index_provenance": target_provenance,
            "column_pin_mappings": target_pin_map,
            "consensus_column_id": expected["target_consensus_column_id"],
            **target_candidate,
        },
        "Tm9_532266_official_synapse_column_coordinate_identifiable": recovered,
        "Tm9_532266_recovered_coordinate": target_candidate["recovered_hex"] if recovered else None,
        "authorize_source_mapping_update": recovered,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "boundary": config["boundary"],
    }
