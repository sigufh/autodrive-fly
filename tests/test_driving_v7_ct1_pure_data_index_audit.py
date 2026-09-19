import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-ct1-pure-data-index-audit.json"


def test_CT1_PuRe_index_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_two_official_archives_are_recovered_through_persistent_handles() -> None:
    report = json.loads(REPORT.read_text())
    index = report["official_index"]
    assert index["archive_candidate_count"] == 2
    assert index["unresolved_archive_count"] == 0
    assert {
        item["file_id"] for item in index["components"].values() if item["link_class"] == "zip"
    } == {"file_3247642", "file_3247643"}
    assert report["archive_contents_verified"] is True
    assert report["retrieval_observation"]["numerical_payload_present"] is False
    assert (
        report["boundary"]["inaccessible_component_endpoints_are_not_zero_result_evidence"] is True
    )


def test_verified_duplicate_PDF_archives_do_not_open_any_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["new_CT1_numerical_payload_verified"] is False
    assert report["new_CT1_Lo1_time_series_verified"] is False
    assert report["new_stable_biological_individual_ids_verified"] is False
    assert report["new_complete_stimulus_and_baseline_fields_verified"] is False
    assert report["authorize_CT1_source_dynamics_transfer"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    for archive in report["archive_inspection"].values():
        assert archive["zip_integrity_passed"] is True
        assert archive["member_count"] == 2
        assert archive["contains_only_already_audited_PDFs"] is True
        assert archive["new_numeric_or_code_member_count"] == 0


def test_all_public_author_repository_history_contains_only_model_assets() -> None:
    history = json.loads(REPORT.read_text())["author_repository_history"]
    assert history["commit_count"] == 2
    assert history["all_paths_across_history"] == [
        "BigCT1.swc",
        "CT1CompModeling.py",
        "CT1Stitcher.py",
        "CompModeling.py",
    ]
    assert history["experimental_data_file_count"] == 0
    assert history["only_morphology_and_model_code_present"] is True
