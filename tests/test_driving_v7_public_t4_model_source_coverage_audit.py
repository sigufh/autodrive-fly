import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-public-t4-model-source-coverage-audit.json"


def test_public_model_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["required_sources"] == ["Mi1", "Tm3", "Mi4", "C3"]
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_Clark_model_has_three_abstract_source_arms_but_not_full_contract() -> None:
    model = json.loads(REPORT.read_text())["models"]["Clark_SynapticModel"]
    assert model["version"] == "18b9db5f0ac09033d7d9b297ca706e4dcb6f626c"
    assert model["independently_parameterized_source_types"] == ["Mi9", "Mi1", "Mi4"]
    assert model["required_source_coverage_fraction"] == 0.5
    assert model["missing_required_sources"] == ["C3", "Tm3"]
    assert model["time_step_seconds"] == 1 / 240
    assert model["time_parameters_seconds"] == {"low_pass": 0.15, "high_pass": 0.15}


def test_ModelDB_and_Clark_models_do_not_authorize_C3_transfer() -> None:
    report = json.loads(REPORT.read_text())
    modeldb = report["models"]["ModelDB_239435"]
    assert modeldb["source_arm_annotations"] == ["excitatory", "inhibitory"]
    assert modeldb["independently_parameterized_source_types"] == []
    assert modeldb["solver_step_milliseconds"] == 0.1
    assert modeldb["moving_bar_durations_milliseconds"] == [20, 40, 80, 160]
    assert set(report["transfer_gates"].values()) == {False}
    assert report["complete_source_dynamics_transfer_authorized"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["boundary"]["Mi4_must_not_substitute_for_C3"] is True
    assert report["boundary"]["pooled_inhibitory_arm_must_not_substitute_for_Mi4_and_C3"] is True
