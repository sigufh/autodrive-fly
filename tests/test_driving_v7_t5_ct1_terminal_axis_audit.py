import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-ct1-terminal-axis-audit.json"


def test_t5_ct1_terminal_axis_audit_is_hash_bound_and_structure_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit_to_neural_response"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["axis_contract"]["fast_sources"] == ["Tm1", "Tm2", "Tm4"]
    assert report["axis_contract"]["suppressor"] == "CT1"
    assert report["axis_contract"]["enhancer_kept_separate"] == "Tm9"


def test_t5_ct1_mapping_and_axis_obey_strict_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["mapping_holdout"]) == {"L", "R"}
    assert report["target_record_count"] == 6719
    assert report["strict_CT1_terminal_axis_gate_passed"] is False
    assert report["authorize_CT1_spatial_dynamics_candidate"] is False
    assert report["advance_to_T5_functional_precheck"] is False
    assert report["boundary"]["Tm9_and_CT1_not_merged"] is True
    assert report["boundary"]["LPLC_and_vehicle_experiments_remain_frozen"] is True
