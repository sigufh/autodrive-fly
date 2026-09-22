import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-inhibitory-source-external-audit.json"


def test_inhibitory_source_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_Mi4_independent_evidence_is_calcium_and_not_public_numeric_payload() -> None:
    mi4 = json.loads(REPORT.read_text())["source_evidence"]["Mi4"]
    assert mi4["measurement"]["indicators"] == ["GCaMP6f", "GCaMP6s"]
    assert mi4["measurement"]["data_availability"] == "upon_request"
    assert mi4["public_numeric_payload_links_found"] == []
    assert mi4["local_numeric_payload_verified"] is False
    assert mi4["allowed_response_unit"] is False


def test_C3_numeric_external_evidence_still_fails_fixed_gate_and_unit() -> None:
    c3 = json.loads(REPORT.read_text())["source_evidence"]["C3"]
    assert c3["local_numeric_payload_verified"] is True
    assert c3["measurement"]["STRF_fly_count"] == 8
    assert c3["measurement"]["external_flash_fly_count"] == 22
    assert c3["measurement"]["source_and_external_fly_ids_disjoint"] is True
    assert c3["measurement"]["mean_waveform_correlation"] > 0.97
    assert c3["measurement"]["bootstrap_correlation_p05"] < 0.11
    assert c3["allowed_response_unit"] is False
    assert c3["independent_fixed_robustness_passed"] is False


def test_official_2019_attachments_do_not_supply_Mi4_C3_source_dynamics() -> None:
    evidence = json.loads(REPORT.read_text())["official_source_data_attachments"]
    assert evidence["paper"]["doi"] == "10.7554/eLife.49373"
    assert evidence["attachment_count"] == 11
    assert evidence["all_mean_plus_minus_sem_tables"] is True
    assert evidence["Mi1_Tm3_GCaMP_summary_evidence_found"] is True
    assert evidence["Mi4_or_C3_attachment_payload_found"] is False
    assert evidence["individual_source_dynamics_payload_found"] is False
    assert evidence["experimental_membrane_voltage_payload_found"] is False


def test_gonzalez_suarez_Mi4_filter_is_independent_calcium_type_average() -> None:
    evidence = json.loads(REPORT.read_text())["additional_Mi4_evidence"][
        "Gonzalez_Suarez_2022"
    ]
    assert evidence["paper"]["doi"] == "10.1016/j.cub.2022.06.075"
    assert evidence["Mi4_GCaMP6f_fly_count"] == 15
    assert evidence["Mi4_type_average_filter_available"] is True
    assert evidence["filter_sample_interval_seconds"] == 1 / 30
    assert evidence["individual_cell_axis_available"] is False
    assert evidence["Mi4_experimental_membrane_voltage_available"] is False
    assert evidence["C3_source_dynamics_available"] is False
    assert evidence["bioRxiv_supplement_retrieved"] is True
    assert evidence["individual_flies_are_statistical_units"] is True
    assert evidence["public_individual_fly_numeric_payload_attached"] is False
    assert evidence["data_availability"] == "upon_request"
    indexes = json.loads(REPORT.read_text())["additional_Mi4_evidence"][
        "Strother_2018_public_indexes"
    ]
    assert indexes["PMC_attachment_inventory"] == {
        "supplementary_PDF_count": 1,
        "video_count": 1,
        "numeric_data_attachment_count": 0,
    }
    assert indexes["local_numeric_Mi4_trace_payload_found"] is False
    assert indexes["global_absence_claimed"] is False


def test_yuan_C3_is_intervention_evidence_not_direct_source_recording() -> None:
    evidence = json.loads(REPORT.read_text())["additional_C3_evidence"]["Yuan_2020"]
    assert evidence["paper"]["doi"] == "10.1111/jnc.15036"
    assert evidence["C3_intervention_candidate_verified"] is True
    assert evidence["directly_recorded_neural_activity_sources"] == ["L1", "L2"]
    assert evidence["C3_direct_recording_candidate_verified"] is False
    assert evidence["C3_public_numeric_source_dynamics_payload_verified"] is False
    assert evidence["C3_experimental_membrane_voltage_verified"] is False
    assert evidence["publisher_supplement_retrieved"] is False


def test_inhibitory_source_transfer_and_downstream_remain_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["both_inhibitory_sources_have_transferable_external_validation"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
