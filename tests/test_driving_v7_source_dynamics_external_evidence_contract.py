import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-source-dynamics-external-evidence-contract.json"


def test_source_dynamics_external_contract_is_hash_bound_and_complete() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["required_families"]["T4"]["source_types"] == [
        "Mi1",
        "Tm3",
        "Mi4",
        "C3",
    ]
    assert report["required_families"]["T5"]["source_types"] == [
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
        "CT1",
    ]
    assert set(report["manifest_template"]["recording_fields"]) == set(
        report["required_recording_fields"]
    )
    assert set(report["manifest_template"]["mapping_fields"]) == set(
        report["required_mapping_fields"]
    )


def test_missing_external_payload_keeps_every_advancement_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["external_payload_present"] is False
    assert report["contract_satisfied"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"]["contract_only_no_external_payload_present"] is True
