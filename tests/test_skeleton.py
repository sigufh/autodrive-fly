import struct

import pytest

from fly_emotion.connectome.skeleton import parse_precomputed_skeleton, skeleton_segments


def test_parse_neuroglancer_skeleton() -> None:
    payload = struct.pack("<IIffffffII", 2, 1, 1000, 2000, 3000, 4000, 5000, 6000, 0, 1)
    skeleton = parse_precomputed_skeleton(payload)
    assert skeleton.vertices.shape == (2, 3)
    assert skeleton.edges.tolist() == [[0, 1]]
    assert skeleton_segments(skeleton).tolist() == [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]


def test_parse_rejects_invalid_edge() -> None:
    payload = struct.pack("<IIfffII", 1, 1, 0, 0, 0, 0, 2)
    with pytest.raises(ValueError, match="invalid vertex"):
        parse_precomputed_skeleton(payload)
