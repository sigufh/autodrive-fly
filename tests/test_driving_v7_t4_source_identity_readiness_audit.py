import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-identity-readiness-audit.json"


def test_T4_identity_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    workbook = report["workbook_identity_audit"]
    assert workbook["sheet_count"] == 6
    assert workbook["all_sheets_visible"] is True
    assert workbook["custom_document_property_count"] == 0
    assert workbook["identity_fields"] == []
    assert workbook["header_semantics"] == "anonymous_source_type_cell_ordinal_only"
    assert all(workbook["ON_OFF_ordinals_pair_by_source"].values())
    paper = report["paper_identity_provenance"]
    assert paper["actual_pages"] == 22
    assert paper["embedded_attachment_names"] == []
    assert paper["one_electrophysiology_cell_per_different_animal_declared"] is True
    assert paper["source_cell_ordinals_are_pseudonymous_individual_keys"] is True


def test_verified_Edmond_subset_has_no_identity_sidecar() -> None:
    subset = json.loads(REPORT.read_text())["verified_Edmond_subset_identity_audit"]
    assert subset["file_count"] == 13
    assert subset["identity_sidecars"] == []
    assert subset["complete_dataset_manifest_rechecked"] is False
    assert subset["reason"] == "Edmond_API_connection_timeout_during_this_audit"


def test_no_T4_source_passes_both_fixed_robustness_conditions() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["source_summary"]) == {"Tm3", "Mi1", "Mi4", "C3"}
    assert all(
        item["all_robustness_conditions_ready"] is False
        for item in report["source_summary"].values()
    )
    assert (
        report["prior_group_ordering"]["every_delayed_source_later_than_every_fast_source"] is False
    )


def test_T4_individual_validation_and_downstream_gates_remain_closed() -> None:
    report = json.loads(REPORT.read_text())
    gates = report["transfer_gates"]
    assert gates["all_four_T4_source_types_have_millivolt_traces"] is True
    assert gates["physical_source_time_axis_available"] is True
    assert gates["stable_pseudonymous_biological_individual_ID_available"] is True
    assert gates["individual_disjoint_training_validation_split_constructible"] is True
    assert report["minimum_unique_individuals_for_training_and_validation"] == 8
    assert all(report["training_validation_capacity_by_source"].values())
    assert gates["every_source_passes_both_robustness_conditions"] is False
    assert report["T4_individual_level_source_validation_ready"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
