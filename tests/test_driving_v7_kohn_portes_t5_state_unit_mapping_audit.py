import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-t5-state-unit-mapping-audit.json"


def test_state_unit_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_voltage_and_prediction_units_are_explicit() -> None:
    report = json.loads(REPORT.read_text())
    assert report["record_inventory"]["record_count"] == 41
    assert report["record_inventory"]["unique_recording_id_counts_by_source"] == {
        "Tm1": 7,
        "Tm2": 7,
        "Tm4": 7,
        "Tm9": 6,
    }
    units = report["unit_evidence"]
    assert units["stored_raw_voltage_unit"] == "volts"
    assert units["display_conversion"] == "millivolts = volts * 1000"
    assert units["maximum_stored_softplus_reconstruction_error_volts"] < 1.5e-7
    assert report["gates"]["exact_volts_to_millivolts_scale_verified"] is True


def test_voltage_unit_conversion_is_not_v7_state_mapping() -> None:
    report = json.loads(REPORT.read_text())
    boundary = report["mapping_boundary"]
    assert boundary["v7_state_activation"] == "signed_tanh"
    assert boundary["v7_runtime_reported_unit"] == "simulated_activation_not_millivolts"
    assert boundary["author_declares_raw_temporal_filter_correct_scale"] is False
    assert boundary["author_reports_cross_stimulus_flash_mismatch"] is True
    assert boundary["candidate_normalization_formula"] is None
    assert boundary["candidate_clipping_rule"] is None
    assert report["volts_to_millivolts_mapping_available"] is True
    assert report["millivolts_or_filter_output_to_v7_state_mapping_available"] is False
    assert report["authorize_source_kernel_gain_transfer"] is False
    assert report["authorize_source_dynamics_transfer"] is False
