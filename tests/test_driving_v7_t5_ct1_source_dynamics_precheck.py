import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-ct1-source-dynamics-precheck.json"


def test_t5_ct1_dynamics_is_hash_bound_and_tuning_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["source_roles"]["fast_excitation"] == ["Tm1", "Tm2", "Tm4"]
    assert report["source_roles"]["enhancer_kept_separate"] == ["Tm9"]
    assert report["source_roles"]["slow_inhibition"] == ["CT1"]


def test_t5_ct1_dynamics_obeys_single_condition_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["source_coverage"]["target_count"] == 6719
    assert report["ordered_precheck_passed"] is False
    assert report["control_eligible_gains"] == [0.125, 0.5, 2.0, 8.0]
    assert report["controls_evaluated"] is True
    assert set(report["control_results"]) == {"temporal_shuffle", "static_sham"}
    assert report["control_passing_strict_gains"] == []
    assert report["three_condition_evaluation_performed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["boundary"]["Tm9_and_CT1_not_merged"] is True
    assert report["boundary"]["no_T4_mapping_reused"] is True
