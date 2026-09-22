import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-afterimages-mi4-candidate-audit.json"


def test_afterimages_mi4_audit_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_afterimages_mi4_measurement_is_calcium_phenotype_only() -> None:
    report = json.loads(REPORT.read_text())
    measurement = report["measurement"]
    assert measurement["Mi4_measurement_modality"] == "two_photon_GCaMP6f_calcium"
    assert measurement["Mi4_response_unit"] == "deltaF_over_F"
    assert measurement["Mi4_fly_count"] == 6
    assert measurement["Mi4_ROI_count"] == 113
    assert measurement["Mi4_and_Mi9_afterimage_like_response_strong"] is False
    assert report["supplement"]["PDF_pages"] == 6
    assert report["supplement"]["numeric_attachment_count"] == 0
    assert report["public_repository_links_in_body"] == []
    assert report["public_numeric_Mi4_payload_verified"] is False
    assert report["experimental_membrane_voltage"] is False


def test_afterimages_candidate_keeps_transfer_and_downstream_gates_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["C3_direct_recording_found"] is False
    assert report["recording_to_MaleCNS_body_crosswalk_found"] is False
    assert report["global_payload_absence_claimed"] is False
    assert report["authorize_Mi4_source_dynamics_transfer"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
