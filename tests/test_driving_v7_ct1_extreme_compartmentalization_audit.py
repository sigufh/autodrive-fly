import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-ct1-extreme-compartmentalization-audit.json"


def test_CT1_extreme_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert all(
        not item["embedded_attachment_names"] for item in report["official_supplements"].values()
    )


def test_Zenodo_contains_only_model_code_and_morphology() -> None:
    report = json.loads(REPORT.read_text())
    release = report["zenodo_model_release"]
    assert release["doi"] == "10.5281/zenodo.2636606"
    assert len(release["actual_members"]) == 3
    assert all(name.endswith((".py", ".swc")) for name in release["actual_members"])
    assert release["model_generated_output_names"] == [
        "RecFields",
        "VmDistribution",
        "Vm_Ratio",
    ]
    assert report["model_evidence"]["output_unit"] == "millivolts"
    assert report["model_evidence"]["output_is_simulated"] is True


def test_simulated_voltage_does_not_authorize_CT1_transfer() -> None:
    report = json.loads(REPORT.read_text())
    assert report["experimental_evidence"]["response_unit"] == "deltaF_over_F0"
    assert report["experimental_evidence"]["experimental_membrane_voltage_measured"] is False
    assert report["transfer_gates"]["experimental_CT1_calcium_phenotype_verified"] is True
    assert report["transfer_gates"]["experimental_CT1_allowed_response_unit_available"] is False
    assert report["CT1_experimental_source_dynamics_transfer_authorized"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
