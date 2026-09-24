import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-malecns-tm4-synapse-column-boundary-audit.json"


def test_Tm4_synapse_column_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_Tm4_native_coordinates_are_strictly_one_sided() -> None:
    coverage = json.loads(REPORT.read_text())["Tm4_annotation_coverage"]
    assert coverage["body_count_by_side"] == {"L": 837, "R": 833}
    assert coverage["native_count_by_side"] == {"L": 0, "R": 833}
    assert coverage["strictly_one_sided_native_coverage"] is True


def test_right_validation_blocks_left_candidate_writeback() -> None:
    report = json.loads(REPORT.read_text())
    assert report["sample_results"]["R"]["unique_candidate_count"] == 48
    assert report["sample_results"]["R"]["exact_native_match_count"] == 28
    assert report["right_native_validation_exact_fraction"] == 28 / 48
    assert report["right_native_candidate_hex_distance_counts"] == {
        "0": 28,
        "1": 5,
        "2": 14,
        "3": 1,
    }
    assert report["right_native_candidate_within_one_hex_fraction"] == 33 / 48
    assert report["sample_results"]["L"]["unique_candidate_count"] == 48
    assert report["left_missing_unique_candidate_fraction"] == 1.0
    assert report["left_missing_distinct_candidate_hex_count"] == 48
    assert report["authorize_post_hoc_hex_distance_tolerance"] is False
    assert report["official_synapse_count_candidate_is_native_equivalent_for_Tm4"] is False
    assert report["authorize_left_Tm4_coordinate_writeback"] is False
    assert report["authorize_source_mapping_gate_change"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
