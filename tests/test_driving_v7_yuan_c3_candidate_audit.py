import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-yuan-c3-candidate-audit.json"


def test_yuan_C3_candidate_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["paper"]["doi"] == "10.1111/jnc.15036"


def test_yuan_C3_is_an_intervention_not_direct_recording_candidate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["C3_intervention_candidate_verified"] is True
    assert report["abstract_evidence"]["directly_recorded_neural_activity_sources"] == [
        "L1",
        "L2",
    ]
    assert report["C3_direct_recording_candidate_verified"] is False
    assert report["C3_public_numeric_source_dynamics_payload_verified"] is False
    assert report["C3_experimental_membrane_voltage_verified"] is False


def test_inaccessible_material_is_not_absence_and_gates_stay_closed() -> None:
    report = json.loads(REPORT.read_text())
    access = report["open_access_status"]
    assert access["OpenAlex_status"] == "closed"
    assert access["publisher_fulltext_and_supplement_retrieved"] is False
    assert access["publisher_responses_are_access_challenges"] is True
    assert report["authorize_C3_source_dynamics_transfer"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["boundary"][
        "inaccessible_fulltext_and_supplement_are_not_classified_as_absent"
    ] is True
