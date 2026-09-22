import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-hao-asap7y-candidate-audit.json"


def test_ASAP7y_candidate_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["paper"]["doi"] == "10.64898/2026.05.27.728040"


def test_ASAP7y_is_real_fly_voltage_evidence_but_cell_types_are_unresolved() -> None:
    report = json.loads(REPORT.read_text())
    scope = report["verified_scope"]
    assert scope["Drosophila_in_vivo_voltage_imaging"] is True
    assert scope["measurement_modality"] == "two_photon_ASAP7y_voltage_imaging"
    assert scope["millisecond_subcellular_subthreshold_resolution"] is True
    assert scope["visual_system_model_cell_type_count"] == 717
    assert scope["cites_Groschner_2022"] is True
    cell_types = report["cell_type_resolution"]
    assert cell_types["full_text_retrieved"] is False
    assert cell_types["experimental_Drosophila_cell_types_named_in_accessible_sources"] == []
    assert cell_types["Mi4_direct_recording_verified"] is False
    assert cell_types["C3_direct_recording_verified"] is False
    assert cell_types["candidate_classification"] == "unresolved_high_value_candidate"


def test_unresolved_candidate_does_not_change_transfer_or_downstream_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["public_numeric_payload_verified"] is False
    assert report["authorize_Mi4_C3_candidate_classification"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["boundary"][
        "high_value_candidate_kept_unresolved_until_cell_types_are_verified"
    ] is True
