import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-legacy-c3-candidate-audit.json"


def test_legacy_c3_candidate_audit_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_legacy_candidates_preserve_modality_and_access_boundaries() -> None:
    report = json.loads(REPORT.read_text())
    candidates = report["candidates"]
    assert candidates["Tuthill_2013"]["classification"] == (
        "independent_C3_intervention_only"
    )
    assert candidates["Tuthill_2013"]["direct_C3_neural_recording_verified"] is False
    assert candidates["Maisak_2018"]["classification"] == (
        "anatomical_candidate_mentions_only"
    )
    assert candidates["Maisak_2018"]["direct_C3_or_Mi4_recording_verified"] is False
    ramos = candidates["Ramos_Traslosheros_2020"]
    assert ramos["fulltext_probe_status"] == 403
    assert ramos["fulltext_retrieved"] is False
    assert ramos["complete_fulltext_scope_resolved"] is False
    assert ramos["classification"] == "unresolved_beyond_abstract_scope"
    assert report["direct_C3_or_Mi4_source_dynamics_candidates"] == []
    assert report["authorize_Mi4_C3_source_dynamics_transfer"] is False
