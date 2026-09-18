import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-yang-t5-voltage-evidence-audit.json"


def test_yang_voltage_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["paper"]["doi"] == "10.1016/j.cell.2016.05.031"
    assert report["protocol"]["paper"]["pii"] == "S0092867416305827"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert all(
        item["sha256"] == item["actual_sha256"] and item["embedded_attachment_count"] == 0
        for item in report["protocol"]["supplements"]
    )


def test_Tm1_Tm2_optical_voltage_phenotypes_are_physical_but_partial() -> None:
    report = json.loads(REPORT.read_text())
    assert report["measurement_protocol"]["acquisition_rate_hz"] == 38.9
    assert report["measurement_protocol"]["posthoc_resample_rate_hz"] == 120
    assert report["source_evidence"]["Tm1"]["compartment"] == "lobula_Lo1"
    assert report["source_evidence"]["Tm1"]["voltage_response_fly_count"] == 3
    assert report["source_evidence"]["Tm2"]["compartment"] == "medulla_M2"
    assert report["source_evidence"]["Tm2"]["fly_count_report_consistent"] is False
    assert report["T5_source_contract"]["sources_with_optical_voltage_phenotype"] == [
        "Tm1",
        "Tm2",
    ]
    assert report["T5_source_contract"]["sources_without_optical_voltage_phenotype"] == [
        "Tm4",
        "Tm9",
        "CT1",
    ]


def test_plotted_voltage_means_do_not_open_T5_or_downstream_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["T5_partial_optical_voltage_phenotype_verified"] is True
    assert report["T5_numerical_voltage_payload_verified"] is False
    assert report["transfer_gates"]["numerical_per_fly_time_series_available"] is False
    assert report["transfer_gates"]["Tm2_fly_count_report_consistent"] is False
    assert report["T5_source_dynamics_transfer_authorized"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"]["no_curve_digitization"] is True
