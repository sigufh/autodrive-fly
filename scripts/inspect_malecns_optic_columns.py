"""Inspect frozen MaleCNS Neuroglancer annotation shards (not a pipeline CLI)."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow.feather as feather

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


def local_reader(directory: Path):
    def read(shard_name: str, start: int, end: int) -> bytes:
        with (directory / shard_name).open("rb") as stream:
            stream.seek(start)
            value = stream.read(end - start)
        if len(value) != end - start:
            raise ValueError(f"short local read from {directory / shard_name}")
        return value

    return read


def http_reader(directory_url: str):
    cache = {}

    def read(shard_name: str, start: int, end: int) -> bytes:
        cache_key = (shard_name, start, end)
        if cache_key in cache:
            return cache[cache_key]
        request = Request(
            f"{directory_url}/{shard_name}",
            headers={"Range": f"bytes={start}-{end - 1}"},
        )
        with urlopen(request, timeout=60) as response:  # noqa: S310
            payload = response.read()
        if len(payload) != end - start:
            raise ValueError(f"short HTTP range read for {shard_name}")
        cache[cache_key] = payload
        return cache[cache_key]

    return read


def inspect_body(body_id: int) -> dict:
    info = json.loads((RAW / "optic-column-pins/synapses-info.json").read_text())
    relations = {item["id"]: item for item in info["relationships"]}
    synapses = {}
    provenance = {}
    for relationship in ("body_pre", "body_post"):
        relation = relations[relationship]
        directory = RAW / "synapses-precomputed" / relation["key"]
        payload, provenance[relationship] = read_sharded_value(
            body_id,
            ShardingSpec.from_json(relation["sharding"]),
            local_reader(directory),
        )
        synapses[relationship] = decode_synapse_annotations(payload or b"")

    pin_info = json.loads((RAW / "optic-column-pins/info.json").read_text())
    pin_relations = {item["id"]: item for item in pin_info["relationships"]}
    column_ids = sorted(
        {
            item["optic_column"]
            for rows in synapses.values()
            for item in rows
            if item["optic_column"]
        }
    )
    pin_rows = {}
    sides = ("L", "R")
    for column_id in column_ids:
        matches = []
        for side in sides:
            for neuropil in ("ME", "LO", "LOP"):
                relationship = f"{neuropil}({side})_column_segment"
                relation = pin_relations[relationship]
                local_key = relation["key"].replace(f"({side})", f"_{side}")
                payload, prov = read_sharded_value(
                    column_id,
                    ShardingSpec.from_json(relation["sharding"]),
                    local_reader(RAW / "optic-column-pins" / local_key),
                )
                if payload is not None:
                    matches.append(
                        {
                            "relationship": relationship,
                            "annotations": decode_column_pin_annotations(payload),
                            "provenance": prov,
                        }
                    )
        pin_rows[str(column_id)] = matches

    return {
        "body_id": body_id,
        "relationship_provenance": provenance,
        "synapse_counts": {k: len(v) for k, v in synapses.items()},
        "optic_column_counts": {
            k: dict(sorted(Counter(x["optic_column"] for x in v).items()))
            for k, v in synapses.items()
        },
        "roi_column_counts": {
            relationship: dict(
                sorted(
                    Counter(
                        f"{item['primary_roi']}:{item['optic_column']}"
                        for item in rows
                    ).items()
                )
            )
            for relationship, rows in synapses.items()
        },
        "primary_roi_counts": {
            k: dict(sorted(Counter(x["primary_roi"] for x in v).items()))
            for k, v in synapses.items()
        },
        "pin_rows": pin_rows,
    }


def main() -> None:
    body_ids = [int(value) for value in sys.argv[1:]] or [532266]
    print(json.dumps([inspect_body(body_id) for body_id in body_ids], indent=2))


def shard_peers() -> None:
    info = json.loads((RAW / "optic-column-pins/synapses-info.json").read_text())
    relations = {item["id"]: item for item in info["relationships"]}
    annotations = feather.read_table(
        RAW / "body-annotations.feather",
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
    ).to_pandas()
    native = annotations.loc[
        annotations["type"].eq("Tm9")
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
    ]
    result = {}
    for relationship in ("body_pre", "body_post"):
        spec = ShardingSpec.from_json(relations[relationship]["sharding"])
        target_shard, _ = spec.locate(532266)
        result[relationship] = [
            {
                "body_id": int(row.bodyId),
                "side": str(row.somaSide),
                "hex1": int(row.assignedOlHex1),
                "hex2": int(row.assignedOlHex2),
                "minishard": spec.locate(int(row.bodyId))[1],
            }
            for row in native.itertuples(index=False)
            if spec.locate(int(row.bodyId))[0] == target_shard
        ]
    print(json.dumps(result, indent=2))


def validate_native_tm9() -> None:
    info = json.loads((RAW / "optic-column-pins/synapses-info.json").read_text())
    relations = {item["id"]: item for item in info["relationships"]}
    annotations = feather.read_table(
        RAW / "body-annotations.feather",
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
    ).to_pandas()
    native = annotations.loc[
        annotations["type"].eq("Tm9")
        & annotations["assignedOlHex1"].notna()
        & annotations["assignedOlHex2"].notna()
    ].sort_values(["assignedOlHex1", "assignedOlHex2", "bodyId"])
    # A deterministic, coordinate-ordered sample spans the full native Tm9 map
    # while keeping this exploratory network retrieval bounded.
    full = "--full" in sys.argv[1:]
    if not full:
        sample_count = min(96, len(native))
        sample_indices = [
            round(i * (len(native) - 1) / (sample_count - 1))
            for i in range(sample_count)
        ]
        native = native.iloc[sample_indices]

    def inspect(row):
        body = {
            "body_id": int(row.bodyId),
            "side": str(row.somaSide),
            "native_hex": [int(row.assignedOlHex1), int(row.assignedOlHex2)],
        }
        for relationship in ("body_pre", "body_post"):
            relation = relations[relationship]
            payload, provenance = read_sharded_value(
                body["body_id"],
                ShardingSpec.from_json(relation["sharding"]),
                http_reader(f"{SYNAPSE_URL}/{relation['key']}"),
            )
            rows = decode_synapse_annotations(payload or b"")
            body[relationship] = dict(
                sorted(Counter(x["optic_column"] for x in rows).items())
            )
            body[f"{relationship}_roi_column"] = dict(
                sorted(
                    Counter(
                        f"{x['primary_roi']}:{x['optic_column']}" for x in rows
                    ).items()
                )
            )
            body[f"{relationship}_payload_sha256"] = hashlib.sha256(
                payload or b""
            ).hexdigest()
            body[f"{relationship}_provenance"] = provenance
        return body

    if full:
        with ThreadPoolExecutor(max_workers=16) as executor:
            outputs = list(executor.map(inspect, native.itertuples(index=False)))
    else:
        outputs = [inspect(row) for row in native.itertuples(index=False)]
    print(json.dumps(outputs, indent=2))


def summarize_validation(path: Path, column_map_path: Path | None = None) -> None:
    rows = json.loads(path.read_text())
    column_map = (
        {int(key): value for key, value in json.loads(column_map_path.read_text()).items()}
        if column_map_path is not None
        else None
    )
    summary = {}
    for method in ("body_pre", "body_post", "combined"):
        exact = 0
        errors = []
        no_nonzero = 0
        tied = 0
        examples = []
        top_fractions = []
        for row in rows:
            counts = Counter()
            for relationship in ("body_pre", "body_post"):
                if method == "combined" or method == relationship:
                    counts.update(
                        {int(key): value for key, value in row[relationship].items() if int(key)}
                    )
            if not counts:
                no_nonzero += 1
                continue
            maximum = max(counts.values())
            modes = [key for key, value in counts.items() if value == maximum]
            if len(modes) != 1:
                tied += 1
                continue
            if column_map is None:
                predicted = [modes[0] // 100, modes[0] % 100]
            else:
                mapped = {
                    tuple(hex_)
                    for match in column_map[modes[0]]
                    for hex_ in match["hexes"]
                }
                if len(mapped) != 1:
                    raise ValueError(f"column {modes[0]} does not have one native hex")
                predicted = list(next(iter(mapped)))
            is_exact = predicted == row["native_hex"]
            exact += is_exact
            errors.append(
                max(
                    abs(a - b)
                    for a, b in zip(predicted, row["native_hex"], strict=True)
                )
            )
            top_fractions.append(maximum / sum(counts.values()))
            if not is_exact and len(examples) < 6:
                examples.append(
                    {
                        "body_id": row["body_id"],
                        "native_hex": row["native_hex"],
                        "predicted_hex": predicted,
                        "counts": dict(counts),
                    }
                )
        summary[method] = {
            "evaluated": len(errors),
            "exact": exact,
            "exact_fraction": exact / len(errors),
            "no_nonzero": no_nonzero,
            "tied": tied,
            "maximum_chebyshev_error": max(errors),
            "minimum_top_fraction": min(top_fractions),
            "examples": examples,
        }
    consensus_exact = 0
    consensus_evaluated = 0
    consensus_disagreement = 0
    consensus_no_mode = 0
    for row in rows:
        modes = []
        for relationship in ("body_pre", "body_post"):
            counts = {
                int(key): value
                for key, value in row[relationship].items()
                if int(key)
            }
            if not counts:
                modes.append(None)
                continue
            maximum = max(counts.values())
            top = [key for key, value in counts.items() if value == maximum]
            modes.append(top[0] if len(top) == 1 else None)
        if None in modes:
            consensus_no_mode += 1
        elif modes[0] != modes[1]:
            consensus_disagreement += 1
        else:
            consensus_evaluated += 1
            if column_map is None:
                predicted = [modes[0] // 100, modes[0] % 100]
            else:
                mapped = {
                    tuple(hex_)
                    for match in column_map[modes[0]]
                    for hex_ in match["hexes"]
                }
                if len(mapped) != 1:
                    raise ValueError(f"column {modes[0]} does not have one native hex")
                predicted = list(next(iter(mapped)))
            consensus_exact += predicted == row["native_hex"]
    summary["independent_pre_post_mode_consensus"] = {
        "evaluated": consensus_evaluated,
        "exact": consensus_exact,
        "exact_fraction": consensus_exact / consensus_evaluated,
        "no_unique_mode": consensus_no_mode,
        "pre_post_disagreement": consensus_disagreement,
    }
    strict_rows = []
    for row in rows:
        pre = Counter(
            {int(key): value for key, value in row["body_pre"].items() if int(key)}
        )
        post = Counter(
            {int(key): value for key, value in row["body_post"].items() if int(key)}
        )
        if len(pre) != 1 or not post:
            continue
        column_id = next(iter(pre))
        post_maximum = max(post.values())
        post_modes = [key for key, value in post.items() if value == post_maximum]
        if post_modes != [column_id]:
            continue
        mapped = {
            tuple(hex_)
            for match in column_map[column_id]
            for hex_ in match["hexes"]
        }
        if len(mapped) != 1:
            continue
        predicted = list(next(iter(mapped)))
        strict_rows.append(
            {
                "body_id": row["body_id"],
                "exact": predicted == row["native_hex"],
                "post_mode_fraction": post_maximum / sum(post.values()),
            }
        )
    summary["unanimous_output_and_matching_input_mode"] = {
        "evaluated": len(strict_rows),
        "exact": sum(row["exact"] for row in strict_rows),
        "exact_fraction": sum(row["exact"] for row in strict_rows) / len(strict_rows),
        "errors": [row for row in strict_rows if not row["exact"]],
    }
    official = []
    if column_map is None:
        raise ValueError("official synapse-count summary requires column pin map")
    for row in rows:
        counts_by_roi = {}
        for relationship in ("body_pre", "body_post"):
            for pair, count in row[f"{relationship}_roi_column"].items():
                raw_roi, raw_column_id = pair.split(":")
                roi_id, column_id = int(raw_roi), int(raw_column_id)
                if not column_id:
                    continue
                matches = [
                    match
                    for match in column_map[column_id]
                    if any(pin_roi == roi_id for pin_roi in (
                        [47] if match["relationship"].startswith("ME(L)") else
                        [48] if match["relationship"].startswith("ME(R)") else
                        [43] if match["relationship"].startswith("LO(L)") else
                        [44] if match["relationship"].startswith("LO(R)") else
                        [45] if match["relationship"].startswith("LOP(L)") else
                        [46]
                    ))
                ]
                for match in matches:
                    roi = match["relationship"].split("_column_segment")[0]
                    for hex_ in match["hexes"]:
                        counts_by_roi.setdefault(roi, Counter())[tuple(hex_)] += count
        me_counts = counts_by_roi.get(f"ME({row['side']})", Counter())
        maximum = max(me_counts.values()) if me_counts else 0
        modes = [hex_ for hex_, count in me_counts.items() if count == maximum]
        official.append(
            {
                "body_id": row["body_id"],
                "unique": len(modes) == 1,
                "exact": len(modes) == 1 and list(modes[0]) == row["native_hex"],
            }
        )
    eligible = [row for row in official if row["unique"]]
    summary["official_per_roi_synapse_count"] = {
        "evaluated": len(eligible),
        "exact": sum(row["exact"] for row in eligible),
        "exact_fraction": sum(row["exact"] for row in eligible) / len(eligible),
        "ties_or_no_ME_column": len(official) - len(eligible),
        "example_errors": [row for row in eligible if not row["exact"]][:12],
    }
    print(json.dumps(summary, indent=2))


def validation_column_ids(path: Path) -> None:
    rows = json.loads(path.read_text())
    values = sorted(
        {
            int(column_id)
            for row in rows
            for relationship in ("body_pre", "body_post")
            for column_id in row[relationship]
            if int(column_id)
        }
    )
    print(" ".join(map(str, values)))


def map_column_ids(path: Path) -> None:
    rows = json.loads(path.read_text())
    column_ids = sorted(
        {
            int(column_id)
            for row in rows
            for relationship in ("body_pre", "body_post")
            for column_id in row[relationship]
            if int(column_id)
        }
    )
    pin_info = json.loads((RAW / "optic-column-pins/info.json").read_text())
    relations = {item["id"]: item for item in pin_info["relationships"]}
    output = {}
    for column_id in column_ids:
        matches = []
        for side in ("L", "R"):
            for neuropil in ("ME", "LO", "LOP"):
                name = f"{neuropil}({side})_column_segment"
                relation = relations[name]
                local_key = relation["key"].replace(f"({side})", f"_{side}")
                payload, _ = read_sharded_value(
                    column_id,
                    ShardingSpec.from_json(relation["sharding"]),
                    local_reader(RAW / "optic-column-pins" / local_key),
                )
                if payload is not None:
                    pins = decode_column_pin_annotations(payload)
                    matches.append(
                        {
                            "relationship": name,
                            "annotation_count": len(pins),
                            "hexes": sorted(
                                {
                                    (pin["hex1"], pin["hex2"])
                                    for pin in pins
                                }
                            ),
                        }
                    )
        output[column_id] = matches
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--shard-peers"]:
        shard_peers()
    elif sys.argv[1:] in (["--validate-native-tm9"], ["--validate-native-tm9", "--full"]):
        validate_native_tm9()
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "--summarize-validation":
        summarize_validation(
            Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) == 4 else None
        )
    elif len(sys.argv) == 3 and sys.argv[1] == "--validation-column-ids":
        validation_column_ids(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--map-column-ids":
        map_column_ids(Path(sys.argv[2]))
    else:
        main()
