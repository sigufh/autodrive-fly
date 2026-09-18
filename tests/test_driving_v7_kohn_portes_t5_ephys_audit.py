import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-t5-ephys-audit.json"


def test_kohn_portes_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["repository"]["commit"] == (
        "ecc7703e506d7491d01532e9e339f2246648447f"
    )
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_four_Tm_sources_have_numeric_voltage_and_physical_time() -> None:
    report = json.loads(REPORT.read_text())
    contract = report["T5_source_contract"]
    assert contract["sources_with_local_numeric_membrane_voltage"] == [
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
    ]
    assert contract["missing_numeric_membrane_voltage_sources"] == ["CT1"]
    assert contract["coverage_fraction"] == 0.8
    assert contract["fast_sources"] == ["Tm1", "Tm2", "Tm4"]
    assert contract["delayed_source_kept_separate"] == "Tm9"
    for source in contract["sources_with_local_numeric_membrane_voltage"]:
        payload = report["source_payloads"][source]
        assert payload["condition_count"] == 8
        assert payload["all_conditions_have_physical_time_axis"] is True
        assert payload["all_conditions_have_numerical_membrane_voltage"] is True
        for condition in payload["conditions"].values():
            assert condition["sample_count"] == 50_000
            assert condition["sample_interval_seconds"] == 0.0002
            assert condition["duration_seconds"] == 10.0
            assert condition["stored_response_unit"] == "volts"
            assert condition["contract_response_unit_after_exact_scale"] == "millivolts"


def test_anonymous_preprocessing_does_not_authorize_transfer() -> None:
    report = json.loads(REPORT.read_text())
    assert any(
        not condition["reported_n_equals_trace_count"]
        for payload in report["source_payloads"].values()
        for condition in payload["conditions"].values()
    )
    assert report["transfer_gates"]["stable_recording_ids_retained"] is False
    assert report["transfer_gates"]["biological_individual_ids_retained"] is False
    assert report["transfer_gates"]["required_angular_stimulus_fields_retained"] is False
    assert report["transfer_gates"]["baseline_window_seconds_retained"] is False
    assert report["transfer_gates"]["every_T5_source_covered"] is False
    assert report["T5_source_dynamics_transfer_authorized"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
