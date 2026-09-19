import gzip
import struct

from fly_emotion.driving.v7_neuroglancer_sharded import (
    ShardingSpec,
    decode_column_pin_annotations,
    decode_minishard_index,
    decode_synapse_annotations,
    list_shard_keys,
    murmurhash3_x86_128_low64_uint64,
    read_all_shard_values,
)


def test_murmurhash_matches_neuroglancer_reference_for_tm9_body() -> None:
    assert murmurhash3_x86_128_low64_uint64(532266 >> 10) == 0x7C91BB3B173FCA8A


def test_malecns_relationship_shard_locations() -> None:
    pre = ShardingSpec(10, 9, 2, data_encoding="gzip")
    post = ShardingSpec(10, 9, 8, data_encoding="gzip")
    assert (*pre.locate(532266), pre.shard_name(pre.locate(532266)[0])) == (
        162,
        2,
        "0a2.shard",
    )
    assert (*post.locate(532266), post.shard_name(post.locate(532266)[0])) == (
        458,
        138,
        "1ca.shard",
    )


def test_gzip_minishard_delta_decoding() -> None:
    # Keys 10, 15; starts 16 + 7 and then prior end + 3; sizes 5, 11.
    raw = struct.pack("<6Q", 10, 5, 7, 3, 5, 11)
    assert decode_minishard_index(
        gzip.compress(raw), shard_index_size=16, encoding="gzip"
    ) == {10: (23, 5), 15: (31, 11)}


def test_complete_shard_key_listing() -> None:
    spec = ShardingSpec(0, 0, 1, minishard_index_encoding="gzip")
    first = gzip.compress(struct.pack("<3Q", 10, 7, 5))
    second = gzip.compress(struct.pack("<6Q", 3, 6, 1, 2, 4, 8))
    index = struct.pack("<4Q", 0, len(first), len(first), len(first) + len(second))
    assert list_shard_keys(index + first + second, spec) == {3, 9, 10}


def test_complete_shard_value_decoding() -> None:
    spec = ShardingSpec(
        0, 0, 1, minishard_index_encoding="gzip", data_encoding="gzip"
    )
    values = {3: gzip.compress(b"three"), 9: gzip.compress(b"nine")}
    data = b"".join(values.values())
    first_index = gzip.compress(struct.pack("<3Q", 3, 0, len(values[3])))
    second_offset = len(data) + len(first_index)
    second_index = gzip.compress(
        struct.pack("<3Q", 9, len(values[3]), len(values[9]))
    )
    index = struct.pack(
        "<4Q",
        len(data),
        len(data) + len(first_index),
        second_offset,
        second_offset + len(second_index),
    )
    assert read_all_shard_values(
        index + data + first_index + second_index, spec
    ) == {3: b"three", 9: b"nine"}


def test_multiple_annotation_decoders_preserve_ids_and_native_fields() -> None:
    synapse_record = bytearray(64)
    struct.pack_into("<6f", synapse_record, 0, 1, 2, 3, 4, 5, 6)
    struct.pack_into("<4f", synapse_record, 24, 0.1, 0.2, 0.3, 0.4)
    struct.pack_into("<2IhH9B", synapse_record, 40, 7, 8, 47, 91, *range(9))
    synapse = decode_synapse_annotations(
        struct.pack("<Q", 1) + synapse_record + struct.pack("<Q", 123)
    )[0]
    assert synapse["annotation_id"] == 123
    assert synapse["body_pre_u32"] == 7
    assert synapse["body_post_u32"] == 8
    assert synapse["primary_roi"] == 47
    assert synapse["optic_column"] == 91
    assert synapse["optic_layer"] == 0
    assert synapse["traced_fanout"] == 8

    pin_record = bytearray(32)
    struct.pack_into("<6fhBbbb", pin_record, 0, 1, 2, 3, 4, 5, 6, 12, 47, -4, 9, 3)
    pin = decode_column_pin_annotations(
        struct.pack("<Q", 1) + pin_record + struct.pack("<Q", 456)
    )[0]
    assert pin["annotation_id"] == 456
    assert (pin["roi"], pin["hex1"], pin["hex2"], pin["layer"]) == (
        47,
        -4,
        9,
        3,
    )
