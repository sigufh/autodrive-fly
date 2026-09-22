import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-c3-directional-edge-audit.json"


def test_c3_directional_edge_audit_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    source = report["protocol"]["source_data"]
    assert source["git_blob_sha1"] == "4d21bbfbe5d0c800f035029b3022ad03f37bed04"
    assert source["sha256"] == (
        "f159c40368875e4e1ec4a5922429b517897997b40ee1a6d5f271170913243264"
    )


def test_c3_directional_payload_fields_and_caption_discrepancy_are_disclosed() -> None:
    report = json.loads(REPORT.read_text())
    measurement = report["measurement"]
    assert measurement["modality"] == "two_photon_GCaMP6f_calcium"
    assert measurement["response_unit"] == "deltaF_over_F"
    assert measurement["record_count"] == measurement["ROI_count"] == 77
    assert measurement["fly_count_in_public_payload"] == 6
    assert measurement["fly_count_in_version_of_record_caption"] == 8
    assert measurement["aggregate_shape"] == [6, 8, 80]
    assert measurement["direction_degrees"] == [90, 45, 0, 315, 270, 225, 180, 135]
    assert measurement["edge_velocity_degrees_per_second"] == 20.0
    assert measurement["interpolated_sample_interval_seconds"] == 0.1
    assert measurement["paper_reports_direction_preference"] is False


def test_c3_directional_data_is_not_independent_or_transferable() -> None:
    report = json.loads(REPORT.read_text())
    cohort = report["cohort_relationship"]
    assert cohort["flash_overlap_count"] == 6
    assert cohort["STRF_overlap_count"] == 6
    assert cohort["every_directional_fly_in_flash_cohort"] is True
    assert cohort["every_directional_fly_in_STRF_cohort"] is True
    assert cohort["independent_study"] is False
    assert cohort["independent_cohort"] is False
    assert report["experimental_membrane_voltage"] is False
    assert report["direction_specific_C3_source_kernel_verified"] is False
    assert report["independent_dynamic_validation_available"] is False
    assert report["authorize_C3_source_dynamics_transfer"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["advance_to_T4_functional_precheck"] is False
