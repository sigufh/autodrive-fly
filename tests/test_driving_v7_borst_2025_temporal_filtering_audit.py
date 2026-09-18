import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-borst-2025-temporal-filtering-audit.json"


def test_borst_2025_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["repository"]["commit"] == (
        "2e277fef43139230fafc598c93301bf7b27645dd"
    )
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_numeric_target_is_verified_as_parameterized_calcium_derived_data() -> None:
    report = json.loads(REPORT.read_text())
    inventory = report["repository_inventory"]
    assert inventory["archive_member_count"] == 49
    assert inventory["file_count"] == 39
    assert inventory["data_copies_identical"] is True
    assert len(inventory["data_copies"]) == 2
    assert all(item["shape"] == [13, 9, 200] for item in inventory["data_copies"])
    target = report["target_provenance"]
    assert target["source_measurement"] == "calcium_imaging_white_noise_reverse_correlation"
    assert target["source_contains_membrane_voltage_information"] is False
    assert target["numeric_target_unit_semantics"] == "arbitrary_20mV_model_comparison_scale"
    assert target["measured_membrane_voltage"] is False
    assert target["time_step_milliseconds"] == 10


def test_borst_target_does_not_complete_nine_source_contract() -> None:
    report = json.loads(REPORT.read_text())
    assert report["v7_source_coverage"]["covered_sources"] == [
        "Mi1", "Mi4", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9"
    ]
    assert report["v7_source_coverage"]["missing_sources"] == ["C3", "CT1"]
    assert report["transfer_gates"]["experimental_membrane_voltage_payload"] is False
    assert report["transfer_gates"]["stable_biological_individual_ids_available"] is False
    assert report["source_dynamics_transfer_authorized"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
