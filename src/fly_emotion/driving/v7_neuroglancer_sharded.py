"""Minimal read-only decoder for Neuroglancer uint64 sharded data.

The implementation is intentionally small: it supports the exact v1 sharding
operations needed to audit immutable MaleCNS annotation indexes without adding
TensorStore or CloudVolume as a runtime dependency.
"""

from __future__ import annotations

import gzip
import struct
from collections.abc import Callable
from dataclasses import dataclass

UINT32_MASK = (1 << 32) - 1
UINT64_MASK = (1 << 64) - 1


def _u32(value: int) -> int:
    return value & UINT32_MASK


def _rotl32(value: int, bits: int) -> int:
    value = _u32(value)
    return _u32((value << bits) | (value >> (32 - bits)))


def _mix32(value: int) -> int:
    value = _u32(value ^ (value >> 16))
    value = _u32(value * 0x85EBCA6B)
    value = _u32(value ^ (value >> 13))
    value = _u32(value * 0xC2B2AE35)
    return _u32(value ^ (value >> 16))


def murmurhash3_x86_128_low64_uint64(value: int, seed: int = 0) -> int:
    """Hash one uint64le value as Neuroglancer does, returning low 64 bits."""

    if value < 0 or value > UINT64_MASK:
        raise ValueError("value must be a uint64")
    h1 = h2 = h3 = h4 = _u32(seed)
    c1, c2, c3 = 0x239B961B, 0xAB0E9789, 0x38B34AE5

    k2 = _u32((value >> 32) * c2)
    k2 = _rotl32(k2, 16)
    k2 = _u32(k2 * c3)
    h2 = _u32(h2 ^ k2)

    k1 = _u32((value & UINT32_MASK) * c1)
    k1 = _rotl32(k1, 15)
    k1 = _u32(k1 * c2)
    h1 = _u32(h1 ^ k1)

    h1 = _u32(h1 ^ 8)
    h2 = _u32(h2 ^ 8)
    h3 = _u32(h3 ^ 8)
    h4 = _u32(h4 ^ 8)
    h1 = _u32(h1 + h2)
    h1 = _u32(h1 + h3)
    h1 = _u32(h1 + h4)
    h2 = _u32(h2 + h1)
    h3 = _u32(h3 + h1)
    h4 = _u32(h4 + h1)
    h1, h2, h3, h4 = map(_mix32, (h1, h2, h3, h4))
    h1 = _u32(h1 + h2)
    h1 = _u32(h1 + h3)
    h1 = _u32(h1 + h4)
    h2 = _u32(h2 + h1)
    return h1 | (h2 << 32)


@dataclass(frozen=True)
class ShardingSpec:
    preshift_bits: int
    shard_bits: int
    minishard_bits: int
    hash: str = "murmurhash3_x86_128"
    minishard_index_encoding: str = "raw"
    data_encoding: str = "raw"

    @classmethod
    def from_json(cls, value: dict) -> ShardingSpec:
        if value.get("@type") != "neuroglancer_uint64_sharded_v1":
            raise ValueError("unsupported Neuroglancer sharding type")
        return cls(
            preshift_bits=int(value["preshift_bits"]),
            shard_bits=int(value["shard_bits"]),
            minishard_bits=int(value["minishard_bits"]),
            hash=str(value["hash"]),
            minishard_index_encoding=str(
                value.get("minishard_index_encoding", "raw")
            ),
            data_encoding=str(value.get("data_encoding", "raw")),
        )

    @property
    def shard_index_size(self) -> int:
        return 16 << self.minishard_bits

    def locate(self, key: int) -> tuple[int, int]:
        shifted = key >> self.preshift_bits
        if self.hash == "identity":
            hashed = shifted
        elif self.hash == "murmurhash3_x86_128":
            hashed = murmurhash3_x86_128_low64_uint64(shifted)
        else:
            raise ValueError(f"unsupported sharding hash: {self.hash}")
        minishard_mask = (1 << self.minishard_bits) - 1
        shard_mask = (1 << self.shard_bits) - 1
        return (hashed >> self.minishard_bits) & shard_mask, hashed & minishard_mask

    def shard_name(self, shard: int) -> str:
        width = (self.shard_bits + 3) // 4
        return f"{shard:0{width}x}.shard"


def _decode(payload: bytes, encoding: str) -> bytes:
    if encoding == "raw":
        return payload
    if encoding == "gzip":
        return gzip.decompress(payload)
    raise ValueError(f"unsupported encoding: {encoding}")


def decode_minishard_index(
    payload: bytes, *, shard_index_size: int, encoding: str
) -> dict[int, tuple[int, int]]:
    """Return key -> (absolute start, size) from a minishard index."""

    decoded = _decode(payload, encoding)
    if len(decoded) % 24:
        raise ValueError("decoded minishard index length is not divisible by 24")
    count = len(decoded) // 24
    values = struct.unpack(f"<{count * 3}Q", decoded) if count else ()
    key_deltas = values[:count]
    offset_deltas = values[count : 2 * count]
    sizes = values[2 * count :]
    result: dict[int, tuple[int, int]] = {}
    key = 0
    start = shard_index_size
    for key_delta, offset_delta, size in zip(
        key_deltas, offset_deltas, sizes, strict=True
    ):
        key += key_delta
        start += offset_delta
        result[key] = (start, size)
        start += size
    return result


def list_shard_keys(payload: bytes, spec: ShardingSpec) -> set[int]:
    """List every uint64 key stored in one complete shard payload."""

    if len(payload) < spec.shard_index_size:
        raise ValueError("shard payload is shorter than its fixed index")
    keys: set[int] = set()
    for minishard in range(1 << spec.minishard_bits):
        relative_start, relative_end = struct.unpack_from(
            "<QQ", payload, minishard * 16
        )
        if relative_start == relative_end:
            continue
        start = spec.shard_index_size + relative_start
        end = spec.shard_index_size + relative_end
        if end > len(payload):
            raise ValueError("minishard index extends beyond shard payload")
        entries = decode_minishard_index(
            payload[start:end],
            shard_index_size=spec.shard_index_size,
            encoding=spec.minishard_index_encoding,
        )
        overlap = keys.intersection(entries)
        if overlap:
            raise ValueError(f"duplicate keys across minishards: {sorted(overlap)}")
        keys.update(entries)
    return keys


def read_all_shard_values(payload: bytes, spec: ShardingSpec) -> dict[int, bytes]:
    """Decode every keyed value from one complete shard payload."""

    if len(payload) < spec.shard_index_size:
        raise ValueError("shard payload is shorter than its fixed index")
    result: dict[int, bytes] = {}
    for minishard in range(1 << spec.minishard_bits):
        relative_start, relative_end = struct.unpack_from(
            "<QQ", payload, minishard * 16
        )
        if relative_start == relative_end:
            continue
        start = spec.shard_index_size + relative_start
        end = spec.shard_index_size + relative_end
        if end > len(payload):
            raise ValueError("minishard index extends beyond shard payload")
        entries = decode_minishard_index(
            payload[start:end],
            shard_index_size=spec.shard_index_size,
            encoding=spec.minishard_index_encoding,
        )
        overlap = result.keys() & entries.keys()
        if overlap:
            raise ValueError(f"duplicate keys across minishards: {sorted(overlap)}")
        for key, (data_start, data_size) in entries.items():
            data_end = data_start + data_size
            if data_end > len(payload):
                raise ValueError("data value extends beyond shard payload")
            result[key] = _decode(payload[data_start:data_end], spec.data_encoding)
    return result


def read_sharded_value(
    key: int,
    spec: ShardingSpec,
    read_range: Callable[[str, int, int], bytes],
) -> tuple[bytes | None, dict]:
    """Read one key using a callback accepting shard, start, exclusive end."""

    shard, minishard = spec.locate(key)
    shard_name = spec.shard_name(shard)
    index_entry = read_range(shard_name, minishard * 16, minishard * 16 + 16)
    if len(index_entry) != 16:
        raise ValueError("short shard-index read")
    relative_start, relative_end = struct.unpack("<QQ", index_entry)
    provenance = {
        "key": key,
        "shard": shard,
        "minishard": minishard,
        "shard_name": shard_name,
        "minishard_index_relative_range": [relative_start, relative_end],
    }
    if relative_start == relative_end:
        return None, provenance
    index_start = spec.shard_index_size + relative_start
    index_end = spec.shard_index_size + relative_end
    index_payload = read_range(shard_name, index_start, index_end)
    entries = decode_minishard_index(
        index_payload,
        shard_index_size=spec.shard_index_size,
        encoding=spec.minishard_index_encoding,
    )
    entry = entries.get(key)
    provenance["minishard_entry_count"] = len(entries)
    if entry is None:
        return None, provenance
    data_start, data_size = entry
    data_payload = read_range(shard_name, data_start, data_start + data_size)
    provenance["compressed_data_range"] = [data_start, data_start + data_size]
    provenance["compressed_data_bytes"] = data_size
    return _decode(data_payload, spec.data_encoding), provenance


def _multiple_annotation_records(payload: bytes, record_size: int):
    if len(payload) < 8:
        raise ValueError("multiple-annotation payload lacks count")
    count = struct.unpack_from("<Q", payload)[0]
    expected = 8 + count * (record_size + 8)
    if len(payload) != expected:
        raise ValueError(
            f"multiple-annotation payload has {len(payload)} bytes, expected {expected}"
        )
    records_start = 8
    ids_start = records_start + count * record_size
    for index in range(count):
        record_start = records_start + index * record_size
        annotation_id = struct.unpack_from("<Q", payload, ids_start + index * 8)[0]
        yield annotation_id, memoryview(payload)[
            record_start : record_start + record_size
        ]


def decode_synapse_annotations(payload: bytes) -> list[dict]:
    """Decode MaleCNS v1.0 synapse relationship-index annotations."""

    names = (
        "optic_layer",
        "compartment_pre",
        "compartment_post",
        "predicted_nt",
        "nt_neuron_prediction",
        "is_traced_pre",
        "is_traced_post",
        "tbar_fanout",
        "traced_fanout",
    )
    result = []
    for annotation_id, record in _multiple_annotation_records(payload, 64):
        geometry = struct.unpack_from("<6f", record, 0)
        floats = struct.unpack_from("<4f", record, 24)
        body_pre, body_post = struct.unpack_from("<2I", record, 40)
        primary_roi, optic_column = struct.unpack_from("<hH", record, 48)
        bytes_ = struct.unpack_from("<9B", record, 52)
        result.append(
            {
                "annotation_id": annotation_id,
                "point_a": list(geometry[:3]),
                "point_b": list(geometry[3:]),
                "conf_pre": floats[0],
                "conf_post": floats[1],
                "predicted_nt_prob": floats[2],
                "nt_tbar_confidence_score": floats[3],
                "body_pre_u32": body_pre,
                "body_post_u32": body_post,
                "primary_roi": primary_roi,
                "optic_column": optic_column,
                **dict(zip(names, bytes_, strict=True)),
            }
        )
    return result


def decode_column_pin_annotations(payload: bytes) -> list[dict]:
    """Decode MaleCNS v1.0 optic-column-pin relationship annotations."""

    result = []
    for annotation_id, record in _multiple_annotation_records(payload, 32):
        geometry = struct.unpack_from("<6f", record, 0)
        depth, roi, hex1, hex2, layer = struct.unpack_from("<hBbbb", record, 24)
        result.append(
            {
                "annotation_id": annotation_id,
                "point_a": list(geometry[:3]),
                "point_b": list(geometry[3:]),
                "depth": depth,
                "roi": roi,
                "hex1": hex1,
                "hex2": hex2,
                "layer": layer,
            }
        )
    return result
