import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-typed-spatial-pair-precheck.json"


def test_t5_typed_spatial_pair_is_hash_bound_and_tuning_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False


def test_t5_typed_spatial_pair_preserves_source_roles() -> None:
    report = json.loads(REPORT.read_text())
    mechanism = report["mechanism"]
    assert mechanism["fast_sources"] == ["Tm1", "Tm2", "Tm4"]
    assert mechanism["delayed_excitation"] == "Tm9"
    assert mechanism["inhibitory_modulator_kept_separate"] == "CT1"
    assert mechanism["horizontal_pair"] == ["Tm2", "Tm9"]
    assert mechanism["vertical_pair"] == ["Tm9", "Tm1"]
    assert report["source_coverage"]["target_count"] == 6719


def test_t5_typed_spatial_pair_obeys_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["ordered_precheck_passed"] is False
    assert report["control_eligible_candidates"] == []
    assert report["controls_evaluated"] is False
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["boundary"]["labels_changed"] is False
    assert report["boundary"]["thresholds_changed"] is False
    assert report["boundary"]["Tm9_and_CT1_not_merged"] is True
