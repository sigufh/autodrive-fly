import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-gou-dandi-stimulus-metadata-audit.json"


def test_gou_dandi_stimulus_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_complete_lindi_index_has_no_failed_assets_or_stimulus_payloads() -> None:
    report = json.loads(REPORT.read_text())
    index = report["complete_index"]
    assert index["asset_count"] == index["successful_index_count"] == 282
    assert index["failed_index_count"] == 0
    assert index["all_asset_IDs_match_DANDI_index"] is True
    assert set(index["empty_group_match_counts"].values()) == {282}
    assert index["location_counts"]["Mi1"] == 144
    assert index["location_counts"]["Tm3"] == 20


def test_Mi1_Tm3_assets_have_identity_but_no_stimulus_fields() -> None:
    report = json.loads(REPORT.read_text())
    for source, expected in {"Mi1": 144, "Tm3": 20}.items():
        item = report["source_results"][source]
        assert item["asset_count"] == item["unique_subject_count"] == expected
        assert item["all_subjects_unique"] is True
        assert item["all_stimulus_groups_empty"] is True
        assert item["all_intervals_groups_empty"] is True
        assert item["stimulus_direction_field_count"] == 0
        assert item["stimulus_angular_position_field_count"] == 0
        assert item["stimulus_timestamp_field_count"] == 0


def test_range_cross_checks_and_transfer_boundary_remain_closed() -> None:
    report = json.loads(REPORT.read_text())
    for source, item in report["range_cross_checks"].items():
        assert item["location"] == source
        assert item["asset_index_matches"] is True
        assert item["LINDI_index_matches"] is True
        assert item["is_smallest_indexed_asset_for_source"] is True
        assert item["location_matches"] is True
        assert item["stimulus_presentation_keys"] == []
        assert item["stimulus_template_keys"] == []
        assert item["interval_keys"] == []
        assert item["processing_keys"] == []
        assert item["scratch_keys"] == []
    assert report["Mi1_Tm3_NWB_stimulus_metadata_available"] is False
    assert report["Mi1_Tm3_NWB_direction_or_position_fields_available"] is False
    assert report["Dryad_processed_rows_to_DANDI_subject_crosswalk_available"] is False
    assert report["authorize_direction_invariance_audit"] is False
    assert report["authorize_T4_source_kernel_transfer"] is False
