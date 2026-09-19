import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-malecns-synapse-column-audit.json"


def test_synapse_column_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_official_rule_is_replayed_on_all_native_tm9() -> None:
    report = json.loads(REPORT.read_text())
    replay = report["native_Tm9_replay"]
    assert replay["native_Tm9_count"] == 1743
    assert replay["evaluable_count"] == 1742
    assert replay["exact_count"] == 1717
    assert replay["exact_fraction"] > 0.985


def test_target_is_recovered_by_official_per_roi_synapse_count_method() -> None:
    report = json.loads(REPORT.read_text())
    target = report["target"]
    assert target["body_id"] == 532266
    assert target["body_annotation_native_hex_missing"] is True
    assert target["incoming_synapse_count"] == 49
    assert target["outgoing_synapse_count"] == 140
    assert target["input_optic_column_counts"] == {"0": 7, "1502": 31, "1603": 11}
    assert target["output_optic_column_counts"] == {"0": 119, "1502": 21}
    assert target["ROI"] == "ME(L)"
    assert target["hex_counts"] == {"15,2": 52, "16,3": 10}
    assert target["unique_mode"] is True
    assert target["consensus_column_id"] == 1502
    assert target["recovered_hex"] == [15, 2]
    assert report["Tm9_532266_official_synapse_column_coordinate_identifiable"] is True
    assert report["authorize_source_mapping_update"] is True


def test_recovery_does_not_open_downstream_experimental_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"]["no_nearest_neighbor_or_tissue_xyz_conversion"] is True
