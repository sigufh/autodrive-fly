import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-c2c3-version-of-record-data-audit.json"


def test_version_of_record_audit_is_revision_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["local_checked_out_revision"] == (
        "745c6114b72a61dabc9389a1f024dbe22f679edf"
    )
    assert report["protocol"]["remote_head_observed"] == (
        "745c6114b72a61dabc9389a1f024dbe22f679edf"
    )
    assert report["version_of_record"]["doi"] == "10.7554/eLife.108529.3"
    assert report["version_of_record"]["published_on"] == "2026-05-26"
    assert report["supplementary_relation"]["archived_revision"] == (
        "745c6114b72a61dabc9389a1f024dbe22f679edf"
    )


def test_new_strf_blobs_have_stable_fly_names_but_no_voltage_fields() -> None:
    report = json.loads(REPORT.read_text())
    assert report["repository_release_count"] == 0
    assert report["repository_tag_count"] == 0
    assert report["revision_commit_count"] == 4
    assert report["declared_new_numerical_blobs_present"] is True
    assert report["C3_unique_Flyname_count"] == 8
    assert report["Mi1_control_unique_Flyname_count"] == 7
    assert report["new_payload_source_types"] == ["C3", "Mi1"]
    assert report["new_payload_measurement_modality"] == "calcium_fluorescence_STRF"
    for payload in report["new_numerical_payloads"].values():
        inventory = payload["inventory"]
        assert inventory["axis_fields"] == [
            "AV",
            "Cluster",
            "STRFs",
            "Zdepth",
            "corrcoefs",
            "pred",
            "roi_ind",
        ]
        assert inventory["voltage_field_found"] is False
        assert inventory["current_clamp_field_found"] is False
        assert inventory["MaleCNS_body_field_found"] is False
    code = report["source_code_evidence"][
        "PythonCode_for_STRF_analysis/rf_tools.py"
    ]
    assert code["mentions_fluorescence"] is True
    assert code["defines_delta_F_over_F"] is True


def test_version_of_record_does_not_change_transfer_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["new_C3_or_Mi4_membrane_voltage_payload_found"] is False
    assert report["new_Mi4_numerical_payload_found"] is False
    assert report["new_recording_to_MaleCNS_body_crosswalk_found"] is False
    assert report["source_dynamics_transfer_gate_changed"] is False
    assert report["authorize_new_T4_functional_candidate"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["boundary"][
        "final_article_XML_endpoint_timed_out_and_was_not_treated_as_absence"
    ] is True
