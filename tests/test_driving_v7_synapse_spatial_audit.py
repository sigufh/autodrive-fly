import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-synapse-spatial-audit.json"


def test_synapse_spatial_audit_is_structure_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    source = report["protocol"]["source_file"]
    assert source["bytes"] == 6_777_179_098
    assert source["md5"] == "58efcf712f8c4d4de5f2ad51e97def76"
    assert source["full_row_count"] == 311_833_243
    assert source["selected_row_count"] == 1_535_378
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_synapse_clusters_cover_targets_and_match_aggregate_weights() -> None:
    report = json.loads(REPORT.read_text())
    expected = {"T4": (6861, 6860), "T5": (6719, 6718)}
    for family, (targets, complete) in expected.items():
        result = report["families"][family]
        assert result["target_count"] == targets
        assert result["complete_target_count"] == complete
        assert result["synapse_rows_match_aggregate_graph_weight"] is True
        assert result["median_normalized_cluster_separation"] > 0.60
        assert result["separated_target_fraction"] > 0.97
        assert result["all_population_structure_gates_passed"] is True
        assert result["all_population_mirror_gates_passed"] is True
        assert result["strict_structure_gate_passed"] is True
        assert len(result["target_body_ids_sha256"]) == 64
        assert len(result["missing_target_body_ids"]) == 1
        assert result["stored_example_count"] == 8
        assert result["all_targets_in_aggregate_statistics"] is True
    assert report["strict_synapse_spatial_structure_gate_passed"] is True
    assert report["authorize_single_condition_synapse_spatial_precheck"] is True


def test_synapse_structure_does_not_claim_function_or_release() -> None:
    report = json.loads(REPORT.read_text())
    boundary = report["boundary"]
    assert boundary["structure_only"] is True
    assert boundary["no_camera_direction_inferred_from_volume_axes"] is True
    assert boundary["source_file_is_optional_and_not_committed"] is True
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
