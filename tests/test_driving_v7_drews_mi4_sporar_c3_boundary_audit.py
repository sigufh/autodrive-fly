import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-drews-mi4-sporar-c3-boundary-audit.json"


def test_drews_sporar_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["Drews_2020"]["pickle_security"]["restricted_unpickler_used"] is True
    assert report["Drews_2020"]["pickle_security"]["arbitrary_repository_code_executed"] is False


def test_drews_public_payload_reconstructs_individual_mi4_calcium() -> None:
    report = json.loads(REPORT.read_text())
    drews = report["Drews_2020"]
    payload = drews["Mi4_payload"]
    assert drews["repository"]["head_commit"] == "f3d29fefd532eb01be18dbf6d4e181ea5c3c4d92"
    assert drews["dissertation_term_evidence"]["Mi4"] == {
        "exact_term_pages": [28, 55, 56, 57, 68, 69, 72, 78, 98, 99],
        "exact_term_count": 15,
    }
    assert payload["shape"] == [210600, 2]
    assert payload["ROI_count"] == 20
    assert payload["pseudonymous_fly_count"] == 13
    assert payload["condition_count"] == 42
    assert payload["trial_labels"] == [0, 1, 2]
    assert payload["time_sample_count"] == 84
    assert abs(payload["sample_interval_seconds"] - 0.08448) < 1e-14
    assert payload["missing_signal_count"] == 5964
    assert payload["finite_signal_count"] == 204636
    assert payload["response_unit"] == "deltaF_over_F"
    assert payload["individual_ROI_trial_numeric_time_series_verified"] is True
    assert payload["paper_reported_20_cells_13_flies_reconstructed"] is True
    assert payload["recording_to_MaleCNS_body_crosswalk_found"] is False
    assert drews["Mi4_direct_neural_recording_verified"] is True
    assert drews["C3_direct_neural_recording_verified"] is False
    assert drews["experimental_membrane_voltage_verified"] is False


def test_sporar_c3_mentions_do_not_become_direct_recording_evidence() -> None:
    report = json.loads(REPORT.read_text())
    sporar = report["Sporar_2020"]
    assert sporar["term_evidence"]["Mi4"] == {
        "exact_term_pages": [],
        "exact_term_count": 0,
    }
    assert sporar["term_evidence"]["C3"] == {
        "exact_term_pages": [15, 20, 101, 114],
        "exact_term_count": 9,
    }
    assert sporar["directly_recorded_neural_activity_sources_in_C3_context"] == []
    assert sporar["C3_direct_neural_recording_verified"] is False
    assert sporar["classification"] == (
        "C3_anatomy_prior_intervention_and_future_experiment_context_only"
    )
    assert report["independent_Mi4_numeric_calcium_dynamics_verified"] is True
    assert report["independent_C3_numeric_dynamics_verified"] is False
    assert report["independent_Mi4_C3_numeric_membrane_voltage_verified"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["advance_to_T4_functional_precheck"] is False
