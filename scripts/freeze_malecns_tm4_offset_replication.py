"""Freeze the preregistered MaleCNS Tm4 offset-replication payload."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fly_emotion.driving.v7_neuroglancer_sharded import (
    ShardingSpec,
    decode_column_pin_annotations,
    decode_synapse_annotations,
    read_sharded_value,
)

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data/raw/malecns-v1.0"
SYNAPSE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "male-cns-v1.0-synapses-precomputed"
)


def _local_reader(directory: Path):
    def read(shard_name: str, start: int, end: int) -> bytes:
        with (directory / shard_name).open("rb") as stream:
            stream.seek(start)
            value = stream.read(end - start)
        if len(value) != end - start:
            raise ValueError(f"short local read from {directory / shard_name}")
        return value

    return read


def _http_reader(directory_url: str):
    cache = {}

    def read(shard_name: str, start: int, end: int) -> bytes:
        cache_key = (shard_name, start, end)
        if cache_key in cache:
            return cache[cache_key]
        for attempt in range(3):
            try:
                result = subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--silent",
                        "--show-error",
                        "--location",
                        "--connect-timeout",
                        "5",
                        "--max-time",
                        "20",
                        "--range",
                        f"{start}-{end - 1}",
                        f"{directory_url}/{shard_name}",
                    ],
                    check=True,
                    capture_output=True,
                    timeout=25,
                )
                payload = result.stdout
                if len(payload) != end - start:
                    raise ValueError(f"short HTTP range read for {shard_name}")
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1.0 + attempt)
        cache[cache_key] = payload
        return payload

    return read


def freeze(output: Path, preregistration_path: Path) -> None:
    preregistration_sha256 = hashlib.sha256(
        preregistration_path.read_bytes()
    ).hexdigest()
    retrieval_implementation_sha256 = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if not preregistration["replication_protocol_frozen"]:
        raise ValueError("Tm4 offset replication is not preregistered")
    if preregistration["protocol"]["replication_outputs_observed"]:
        raise ValueError("Tm4 offset replication outputs were already observed")
    body_ids = [int(value) for value in preregistration["replication_sample_body_ids"]]
    if len(body_ids) != 96 or len(set(body_ids)) != 96:
        raise ValueError("unexpected Tm4 replication sample identities")

    synapse_info = json.loads((RAW / "optic-column-pins/synapses-info.json").read_text())
    relations = {item["id"]: item for item in synapse_info["relationships"]}
    readers = {
        relationship: _http_reader(
            f"{SYNAPSE_URL}/{relations[relationship]['key']}"
        )
        for relationship in ("body_pre", "body_post")
    }
    parts_directory = output.parent / f"{output.name}.parts"
    parts_directory.mkdir(parents=True, exist_ok=True)

    def inspect(body_id: int) -> dict:
        part_path = parts_directory / f"{body_id}.json"
        if part_path.exists():
            cached = json.loads(part_path.read_text(encoding="utf-8"))
            if int(cached["body_id"]) != body_id:
                raise ValueError(f"Tm4 replication part identity mismatch: {part_path}")
            if cached["preregistration_sha256"] != preregistration_sha256:
                raise ValueError(
                    f"Tm4 replication part preregistration mismatch: {part_path}"
                )
            if (
                cached["retrieval_implementation_sha256"]
                != retrieval_implementation_sha256
            ):
                raise ValueError(
                    f"Tm4 replication part implementation mismatch: {part_path}"
                )
            return cached
        print(f"fetching Tm4 replication body {body_id}", file=sys.stderr, flush=True)
        row = {
            "body_id": body_id,
            "side": "R",
            "preregistration_sha256": preregistration_sha256,
            "retrieval_implementation_sha256": retrieval_implementation_sha256,
        }
        for relationship in ("body_pre", "body_post"):
            relation = relations[relationship]
            payload, provenance = read_sharded_value(
                body_id,
                ShardingSpec.from_json(relation["sharding"]),
                readers[relationship],
            )
            synapses = decode_synapse_annotations(payload or b"")
            row[f"{relationship}_roi_column"] = dict(
                sorted(
                    Counter(
                        f"{item['primary_roi']}:{item['optic_column']}"
                        for item in synapses
                    ).items()
                )
            )
            row[f"{relationship}_payload_sha256"] = hashlib.sha256(
                payload or b""
            ).hexdigest()
            row[f"{relationship}_provenance"] = provenance
        temporary = part_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        temporary.replace(part_path)
        return row

    rows = []
    batch_size = 4
    for offset in range(0, len(body_ids), batch_size):
        batch = body_ids[offset : offset + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            rows.extend(executor.map(inspect, batch))
        print(
            f"cached {len(rows)}/{len(body_ids)} Tm4 replication bodies",
            file=sys.stderr,
            flush=True,
        )
    if [int(row["body_id"]) for row in rows] != body_ids:
        raise ValueError("Tm4 replication output order or completeness changed")

    column_ids = sorted(
        {
            int(pair.split(":")[1])
            for row in rows
            for relationship in ("body_pre", "body_post")
            for pair in row[f"{relationship}_roi_column"]
            if int(pair.split(":")[1])
        }
    )
    pin_info = json.loads((RAW / "optic-column-pins/info.json").read_text())
    pin_relations = {item["id"]: item for item in pin_info["relationships"]}
    pin_map = {}
    for column_id in column_ids:
        matches = []
        for side in ("L", "R"):
            for neuropil in ("ME", "LO", "LOP"):
                name = f"{neuropil}({side})_column_segment"
                relation = pin_relations[name]
                local_key = relation["key"].replace(f"({side})", f"_{side}")
                payload, _ = read_sharded_value(
                    column_id,
                    ShardingSpec.from_json(relation["sharding"]),
                    _local_reader(RAW / "optic-column-pins" / local_key),
                )
                if payload is not None:
                    pins = decode_column_pin_annotations(payload)
                    matches.append(
                        {
                            "relationship": name,
                            "hexes": [
                                list(value)
                                for value in sorted(
                                    {(pin["hex1"], pin["hex2"]) for pin in pins}
                                )
                            ],
                        }
                    )
        pin_map[str(column_id)] = matches

    payload = {
        "protocol": {
            "name": "Tm4-offset-replication-synapse-snapshot",
            "preregistration_sha256": preregistration_sha256,
            "body_count": len(body_ids),
            "synapse_source": SYNAPSE_URL,
            "retrieval_implementation_sha256": retrieval_implementation_sha256,
        },
        "rows": rows,
        "column_pin_map": pin_map,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(f"{output.suffix}.part")
    temporary_output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    temporary_output.replace(output)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: freeze_malecns_tm4_offset_replication.py OUTPUT PREREGISTRATION"
        )
    freeze(Path(sys.argv[1]), Path(sys.argv[2]))
