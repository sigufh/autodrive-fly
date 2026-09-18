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
    assert report["transfer_gates"]["stable_recording_id_field_retained"] is True
    assert report["transfer_gates"]["biological_individual_ids_retained"] is False
    assert report["transfer_gates"]["required_angular_stimulus_fields_retained"] is False
    assert report["transfer_gates"]["baseline_window_seconds_retained"] is False
    assert report["transfer_gates"]["every_T5_source_covered"] is False
    assert report["T5_source_dynamics_transfer_authorized"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False


def test_white_noise_payload_preserves_recording_metadata_but_not_fly_identity() -> None:
    report = json.loads(REPORT.read_text())
    expected = {"Tm1": (8, 7), "Tm2": (5, 5), "Tm4": (6, 6), "Tm9": (6, 6)}
    for source, (records, unique_ids) in expected.items():
        payload = report["white_noise_payloads"][source]
        assert payload["record_count"] == records
        assert payload["unique_recording_id_count"] == unique_ids
        assert payload["stable_recording_id_field_retained"] is True
        assert payload["explicit_biological_individual_id_field_retained"] is False
        assert payload["raw_numerical_membrane_voltage_retained"] is True
        assert payload["physical_timestamps_retained"] is True
        for record in payload["records"]:
            assert record["raw_voltage_sample_count"] == record["timestamp_sample_count"]
            assert record["sample_interval_seconds"] == 0.0002
            assert record["bar_width_degrees"] == 5
            assert record["white_noise_temporal_frequency_hz"] == 20
    assert report["white_noise_payloads"]["Tm1"]["recording_ids_unique"] is False
    contract = report["T5_source_contract"]
    assert contract["minimum_unique_recording_ids_for_training_and_validation"] == 8
    assert contract["recording_id_upper_bound_enough_for_training_and_validation_by_source"] == {
        "Tm1": False,
        "Tm2": False,
        "Tm4": False,
        "Tm9": False,
    }
    combined = {
        source: item["combined_unique_recording_id_count"]
        for source, item in report["white_noise_OA_payloads"].items()
    }
    assert combined == {"Tm1": 7, "Tm2": 7, "Tm4": 7, "Tm9": 6}
    assert report["white_noise_OA_payloads"]["Tm1"]["overlapping_saline_recording_ids"] == [
        "190701",
        "190708",
        "190715",
        "190722",
    ]
