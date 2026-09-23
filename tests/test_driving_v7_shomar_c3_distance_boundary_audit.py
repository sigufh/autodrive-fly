import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-shomar-c3-distance-boundary-audit.json"


def test_shomar_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_c3_behavior_is_not_combined_with_lc15_imaging() -> None:
    report = json.loads(REPORT.read_text())
    assert report["full_text_term_hits"] == {
        "Mi4": 0,
        "C3": 9,
        "GCaMP6f": 3,
        "LC15": 65,
    }
    scope = report["experimental_scope"]
    assert scope["C3_manipulation"] == "shibire_ts_silencing_in_gap_crossing_behavior"
    assert scope["direct_neural_imaging_cell_types"] == ["LC15"]
    assert scope["C3_direct_neural_recording_verified"] is False
    assert report["classification"] == "C3_behavioral_silencing_with_LC15_calcium_imaging"


def test_public_indexes_bound_data_scope_without_claiming_absence() -> None:
    report = json.loads(REPORT.read_text())
    github = report["GitHub"]
    assert github["head_commit"] == "7d550627e67899ed135a5aa8ce366ba823aebef0"
    assert github["tracked_file_count"] == 116
    assert github["filename_term_hits"]["C3"] == 0
    assert github["LC15_imaging_script_count"] == 18
    dryad = report["Dryad"]
    assert dryad["version_number"] == 4
    assert dryad["file_count"] == 6
    assert dryad["total_declared_bytes"] == 214_370_930_562
    assert dryad["imaging_archives"] == [
        "LC15_Imaging_Data.zip",
        "LC15_Imaging_and_Behavior_Data.zip",
    ]
    assert dryad["C3_named_archive_found"] is False
    assert dryad["README_download_status"] == 401
    assert dryad["bulk_download_performed"] is False
    assert report["boundary"]["HTTP_401_is_not_payload_absence"] is True
    assert report["C3_public_numeric_source_dynamics_verified"] is False
    assert report["authorize_C3_source_dynamics_transfer"] is False
