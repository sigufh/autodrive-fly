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


def test_two_official_archives_are_preserved_as_unresolved_candidates() -> None:
    report = json.loads(REPORT.read_text())
    index = report["official_index"]
    assert index["archive_candidate_count"] == 2
    assert index["unresolved_archive_count"] == 2
    assert {
        item["file_id"] for item in index["components"].values() if item["link_class"] == "zip"
    } == {"file_3247642", "file_3247643"}
    assert report["archive_contents_verified"] is False
    assert report["retrieval_observation"]["numerical_payload_present"] is None
    assert (
        report["boundary"]["inaccessible_component_endpoints_are_not_zero_result_evidence"] is True
    )


def test_unresolved_archives_do_not_open_any_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["new_CT1_numerical_payload_verified"] is False
    assert report["new_CT1_Lo1_time_series_verified"] is False
    assert report["new_stable_biological_individual_ids_verified"] is False
    assert report["new_complete_stimulus_and_baseline_fields_verified"] is False
    assert report["authorize_CT1_source_dynamics_transfer"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
