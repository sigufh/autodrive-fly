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
    assert ramos["initial_fulltext_probe_status"] == 403
    assert ramos["fulltext_retrieved_after_official_proof_of_work"] is True
    assert ramos["page_count"] == 153
    assert ramos["embedded_attachment_count"] == 0
    assert ramos["Mi4_exact_term_pages"] == [24, 26, 60]
    assert ramos["C3_exact_term_pages"] == [24, 60, 66, 67, 130]
    assert ramos["direct_neural_recording_targets_in_relevant_C3_experiment"] == [
        "Tm9"
    ]
    assert ramos["C3_direct_neural_recording_verified"] is False
    assert ramos["Mi4_direct_neural_recording_verified"] is False
    assert ramos["related_2021_public_workbook_source_types"] == [
        "Tm4",
        "Tm9",
        "CT1",
    ]
    assert ramos["related_2021_public_workbook_sheet_count"] == 32
    assert ramos["related_2021_workbook_Mi4_C3_exact_cell_hits"] == {
        "Mi4": [],
        "C3": [],
    }
    assert ramos["related_2021_workbook_contains_Mi4_or_C3_source_block"] is False
    assert ramos["complete_fulltext_scope_resolved"] is True
    assert ramos["classification"] == "C3_perturbation_with_Tm9_calcium_readout"
    assert report["unresolved_fulltext_candidates"] == []
    assert report["direct_C3_or_Mi4_source_dynamics_candidates"] == []
    assert report["authorize_Mi4_C3_source_dynamics_transfer"] is False
