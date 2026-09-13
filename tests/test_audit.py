from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

from fly_emotion.data.audit import audit_annotations, audit_connections, audit_neurotransmitters


def test_audit_uses_explicit_neuron_superclass_rule(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.feather"
    feather.write_feather(
        pa.table(
            {
                "bodyId": [1, 2, 3],
                "status": ["Traced", "Anchor", "Traced"],
                "statusLabel": ["Reviewed", "Anchor", "Roughly traced"],
                "superclass": ["cb_intrinsic", None, "descending_neuron"],
                "type": ["A", None, "B"],
            }
        ),
        annotations,
    )
    report, ids = audit_annotations(annotations)
    assert ids == {1, 3}
    assert report["canonical_nodes"] == 2
    assert report["strictly_traced_nodes"] == 2


def test_neurotransmitter_coverage_is_scoped_to_canonical_nodes(tmp_path: Path) -> None:
    path = tmp_path / "nt.feather"
    feather.write_feather(
        pa.table(
            {
                "body": [1, 2, 99],
                "predicted_nt": ["gaba", "acetylcholine", "gaba"],
                "predicted_nt_confidence": [0.9, 0.8, 0.7],
                "consensus_nt": ["gaba", "acetylcholine", None],
            }
        ),
        path,
    )
    report = audit_neurotransmitters(path, {1, 2})
    assert report["canonical_rows"] == 2
    assert report["canonical_coverage"] == 1.0


def test_connections_keep_all_edges_with_two_canonical_endpoints(tmp_path: Path) -> None:
    path = tmp_path / "edges.feather"
    feather.write_feather(
        pa.table(
            {
                "body_pre": [1, 1, 2, 3, 99],
                "body_post": [2, 1, 3, 1, 1],
                "weight": [5, 2, 7, 11, 13],
            }
        ),
        path,
    )
    report = audit_connections(path, {1, 2})
    assert report["source_rows"] == 5
    assert report["canonical_edges"] == 2
    assert report["canonical_synapse_weight_sum"] == 7
    assert report["self_edges"] == 1
