import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-malecns-ct1-columnar-audit.json"


def test_CT1_columnar_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_both_CT1_bodies_have_native_per_synapse_Lo1_columns() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "10009": ("R", 105140, 39643, 868, 875),
        "10157": ("L", 91721, 34126, 802, 867),
    }
    for body_id, (side, pre, post, union, universe) in expected.items():
        body = report["CT1_bodies"][body_id]
        assert body["annotation"]["optic_lobe_side_from_synapse_ROI"] == side
        assert body["synapse_directions"]["body_pre"]["Lo1_synapse_count"] == pre
        assert body["synapse_directions"]["body_post"]["Lo1_synapse_count"] == post
        assert all(
            value["Lo1_all_synapses_have_native_column"]
            for value in body["synapse_directions"].values()
        )
        assert all(
            value["Lo1_zero_column_synapse_count"] == 0
            for value in body["synapse_directions"].values()
        )
        assert body["Lo1_column_union"]["column_count"] == union
        assert body["Lo1_column_union"]["official_LO_column_count"] == universe
    assert report["CT1_per_synapse_Lo1_columnar_retinotopy_available"] is True
    assert report["official_layer_semantics"]["optic_layer_1_label"] == "LO1"
    assert report["official_layer_semantics"]["tag"] == "v1.0"


def test_CT1_coverage_is_high_but_not_falsely_declared_complete() -> None:
    report = json.loads(REPORT.read_text())
    assert report["CT1_bodies"]["10009"]["Lo1_column_union"]["missing_column_count"] == 7
    assert report["CT1_bodies"]["10157"]["Lo1_column_union"]["missing_column_count"] == 65
    assert report["bilateral"]["observed_column_intersection_count"] == 792
    assert report["bilateral"]["observed_column_union_count"] == 878
    assert report["bilateral"]["observed_jaccard"] > 0.90
    assert report["CT1_complete_official_LO_column_coverage"] is False


def test_connectome_structure_does_not_authorize_voltage_or_downstream_work() -> None:
    report = json.loads(REPORT.read_text())
    assert report["authorize_CT1_experimental_voltage_transfer"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
