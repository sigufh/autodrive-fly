import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-motyxia2-public-history-audit.json"


def test_public_history_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_all_frozen_public_refs_and_history_are_bounded() -> None:
    repository = json.loads(REPORT.read_text())["repository"]
    assert repository["public_branch_count"] == 22
    assert repository["public_tag_count"] == 0
    assert repository["reachable_commit_count"] == 447
    assert repository["historical_unique_path_count"] == 280
    assert repository["unique_blob_count"] == 1015


def test_public_history_has_no_exact_record_or_stimulus_join() -> None:
    report = json.loads(REPORT.read_text())
    assert report["search"]["recording_id_count"] > 20
    assert report["search"]["exact_content_hits"] == []
    candidates = report["candidate_artifacts"]
    assert len(candidates["public_log_paths"]) == 10
    assert candidates["public_log_scope"] == "fischerfritz_test_only"
    assert candidates["stimulus_cache_historical_path_count"] == 48
    assert candidates["stimulus_cache_unique_blob_version_count"] == 52
    assert candidates["stimulus_cache_recording_identity_keys_present"] is False


def test_history_absence_claim_is_bounded_and_gates_stay_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report[
        "Kohn_Portes_record_specific_stimulus_log_found_in_audited_public_history"
    ] is False
    assert report["authorize_recording_field_recovery"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["boundary"][
        "audited_public_GitLab_refs_and_reachable_history_only"
    ] is True
    assert report["boundary"]["no_claim_of_global_nonexistence"] is True
