import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-tm9-coordinate-identifiability-audit.json"


def test_Tm9_coordinate_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    skeleton = report["target"]["skeleton"]
    assert hashlib.sha256((ROOT / skeleton["path"]).read_bytes()).hexdigest() == (
        skeleton["actual_sha256"]
    )
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_existing_one_hop_rule_has_no_coordinate_support() -> None:
    report = json.loads(REPORT.read_text())
    assert report["target"]["body_id"] == 532266
    assert report["target"]["annotation"]["type"] == "Tm9"
    assert report["target"]["annotation"]["somaSide"] == "L"
    assert report["target"]["native_optic_hex_available"] is False
    assert report["target"]["existing_one_hop_optic_hex_available"] is False
    assert report["graph_evidence"]["native_coordinate_incoming_edge_count"] == 0
    assert report["graph_evidence"]["native_coordinate_incoming_synapse_weight"] == 0
    assert report["graph_evidence"]["canonical_filter_removed_coordinate_support"] is False
    assert report["graph_evidence"]["raw_noncanonical_incoming_bodies_with_annotations"] == []


def test_latest_public_release_retains_the_same_missing_native_coordinate() -> None:
    report = json.loads(REPORT.read_text())
    latest = report["latest_public_release_check"]
    assert latest["public_male_cns_releases"] == ["male-cns:v0.9", "male-cns:v1.0"]
    assert latest["latest_public_release"] == "male-cns:v1.0"
    assert latest["latest_public_release_uuid"] == (
        "4b2087c0fbe046bfaf0d60bc970e3e5d"
    )
    assert latest["newer_public_release_present"] is False
    assert latest["current_annotation_object"]["response_status"] == 200
    assert latest["current_object_matches_frozen_local_annotation"] is True
    assert latest["target_native_optic_hex_still_missing_in_latest_public_release"] is True


def test_official_synapse_column_chain_recovers_coordinate_without_rewriting_native_field() -> None:
    report = json.loads(REPORT.read_text())
    evidence = report["official_synapse_column_evidence"]
    assert evidence["body_annotation_native_fields_remain_missing"] is True
    assert evidence["official_rule_native_Tm9_replay"]["exact_fraction"] > 0.985
    assert evidence["consensus_column_id"] == 1502
    assert evidence["recovered_coordinate"] == [15, 2]
    assert report["Tm9_532266_coordinate_identifiable_under_official_synapse_rule"] is True


def test_post_hoc_candidates_conflict_and_are_occupied() -> None:
    report = json.loads(REPORT.read_text())
    recursive = report["candidate_diagnostics"]["same_side_recursive_incoming"]
    outgoing = report["candidate_diagnostics"]["native_outgoing"]
    assert recursive["rounded_coordinate"] == [16, 3]
    assert recursive["occupied_by_Tm9_body_ids"] == [514902]
    assert outgoing["rounded_coordinate"] == [15, 2]
    assert outgoing["occupied_by_Tm9_body_ids"] == [141921]
    assert report["candidate_validation_gates"]["same_side_recursive_replay_accuracy"] is True
    assert report["candidate_validation_gates"]["candidate_methods_agree"] is False
    assert report["candidate_validation_gates"]["same_side_recursive_candidate_unoccupied"] is False
    assert report["candidate_validation_gates"]["native_outgoing_candidate_unoccupied"] is False


def test_synapse_level_table_has_only_coarse_ROI_and_tissue_coordinates() -> None:
    report = json.loads(REPORT.read_text())
    synapses = report["synapse_level_evidence"]
    assert synapses["target_incoming_rows"] == 49
    assert synapses["target_outgoing_rows"] == 140
    assert synapses["target_incoming_unique_bodies"] == 22
    assert synapses["target_outgoing_unique_bodies"] == 75
    assert synapses["incoming_primary_post_counts"] == {
        "LOP(L)": 1,
        "ME(L)": 41,
        "Optic-unspecified(L)": 7,
    }
    assert synapses["outgoing_primary_post_counts"] == {
        "ME(L)": 21,
        "Optic-unspecified(L)": 119,
    }
    assert synapses["optic_column_ROI_field_present"] is False
    assert synapses["coordinate_fields_are_tissue_xyz_not_optic_hex"] is True
    assert synapses["native_optic_hex_recovered"] is False


def test_Tm9_coordinate_remains_unidentified_and_all_downstream_gates_stay_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["Tm9_population"]["one_hop_unlocated_count"] == 1
    assert report["Tm9_532266_coordinate_identifiable_under_existing_rule"] is False
    assert report["Tm9_532266_coordinate_repair_authorized"] is False
    assert report["complete_Tm9_columnar_retinotopy_available"] is True
    assert report["authorize_source_mapping_update"] is True
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
