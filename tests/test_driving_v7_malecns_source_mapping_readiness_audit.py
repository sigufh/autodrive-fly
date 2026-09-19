import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-malecns-source-mapping-readiness-audit.json"


def test_MaleCNS_mapping_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["MaleCNS_release"] == "v1.0"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_all_nine_exact_type_body_sets_and_sides_exist() -> None:
    report = json.loads(REPORT.read_text())
    expected_counts = {
        "Mi1": 1773,
        "Tm3": 2054,
        "Mi4": 1772,
        "C3": 1779,
        "Tm1": 1777,
        "Tm2": 1766,
        "Tm4": 1670,
        "Tm9": 1771,
        "CT1": 2,
    }
    assert {name: item["body_count"] for name, item in report["source_mapping"].items()} == (
        expected_counts
    )
    assert all(
        item["all_bodies_in_canonical_graph"] and item["soma_side_complete"]
        for item in report["source_mapping"].values()
    )
    assert report["MaleCNS_type_average_body_sets_available"] is True


def test_one_hop_coordinates_close_columnar_gaps_without_fabrication() -> None:
    report = json.loads(REPORT.read_text())
    mapping = report["source_mapping"]
    assert mapping["Tm3"]["native_optic_hex_count"] == 0
    assert mapping["Tm3"]["one_hop_inferred_coordinate_count"] == 2054
    assert mapping["Tm4"]["one_hop_inferred_coordinate_count"] == 837
    assert mapping["Tm9"]["one_hop_unlocated_body_ids"] == [532266]
    assert mapping["Tm9"]["official_synapse_column_recovered_body_ids"] == [532266]
    assert mapping["Tm9"]["official_synapse_column_recovered_coordinates"] == [[15, 2]]
    assert mapping["Tm9"]["unlocated_body_ids"] == []
    assert report["CT1"]["body_ids"] == [10009, 10157]
    assert report["CT1"]["columnar_Lo1_retinotopy_available"] is False
    assert report["boundary"]["incomplete_Tm9_coordinate_is_not_imputed"] is True


def test_type_body_availability_does_not_fabricate_recording_mapping() -> None:
    report = json.loads(REPORT.read_text())
    gates = report["mapping_gates"]
    mapping = report["source_mapping"]
    assert gates["every_source_type_has_exact_MaleCNS_body_set"] is True
    assert gates["every_source_body_has_soma_side"] is True
    assert mapping["Tm9"]["columnar_retinotopy_available"] is True
    # CT1 remains global/non-columnar, so the all-source gate stays closed.
    assert gates["every_source_body_has_columnar_retinotopic_coordinate"] is False
    assert gates["external_recording_declares_explicit_type_average_or_body_mapping"] is False
    assert gates["external_recording_to_specific_MaleCNS_body_identified"] is False
    assert report["external_source_mapping_contract_satisfied"] is False
    assert report["authorize_external_source_payload"] is False
    assert report["advance_to_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
