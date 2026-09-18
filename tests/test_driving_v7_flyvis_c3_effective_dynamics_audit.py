import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-flyvis-c3-effective-dynamics-audit.json"


def test_effective_dynamics_audit_is_hash_bound_and_complete() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    raw = report["protocol"]["raw_response_payload"]
    assert raw["bytes"] == 3_959_520
    assert raw["sha256"] == (
        "bd44e339ae09a7b7dc899cffefa20fd857482cff2c56ec72cd62ac6378db3eff"
    )
    assert report["protocol"]["model_count"] == 50
    assert report["protocol"]["fixed_denominator"] == 50
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["direction_label_used"] is False


def test_effective_response_uses_only_permitted_receptors() -> None:
    report = json.loads(REPORT.read_text())
    inputs = report["input_contract"]
    assert inputs["native_input_cell_types"] == [f"R{index}" for index in range(1, 9)]
    assert inputs["permitted_input_cell_types"] == [f"R{index}" for index in range(1, 7)]
    assert inputs["excluded_input_cell_types"] == ["R7", "R8"]
    assert inputs["R7_R8_external_activity"] == 0.0
    assert inputs["target_recorded"] == "C3_u0_v0"
    assert inputs["target_activity_injected"] is False


def test_effective_C3_dynamics_fail_preregistered_stability_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["cross_dt_summary"]["fixed_denominator"] == 50
    assert report["cross_dt_summary"]["minimum"] < 0.67
    assert report["transfer_gates"]["every_model_cross_dt_correlation"] is False
    for item in report["ensemble_stability"].values():
        assert item["unit_shape_leave_one_model_out_summary"]["minimum"] < -0.79
        assert item["every_model_passed"] is False
    assert report["transfer_gates"]["every_model_leave_one_model_out_at_each_dt"] is False


def test_effective_C3_dynamics_fail_external_flash_gates() -> None:
    report = json.loads(REPORT.read_text())
    for by_tau in report["external_C3_flash_consistency"].values():
        for item in by_tau.values():
            assert item["ensemble_equal_shape_correlation"] < 0.72
            assert item["ensemble_correlation_passed"] is False
            assert item["ensemble_features_passed"] is False
            assert item["individual_model_correlation_summary"]["fixed_denominator"] == 50
    assert report["transfer_gates"][
        "every_calcium_assumption_ensemble_external_correlation"
    ] is False
    assert report["transfer_gates"][
        "every_calcium_assumption_ensemble_features_in_empirical_intervals"
    ] is False
    assert report["FlyVis_C3_effective_dynamics_transfer_authorized"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["boundary"]["T5_and_LPLC_remain_frozen"] is True
