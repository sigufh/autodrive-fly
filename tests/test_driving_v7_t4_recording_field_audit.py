import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-recording-field-audit.json"


def test_T4_field_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_verified_payload_supports_ten_of_fifteen_fields() -> None:
    report = json.loads(REPORT.read_text())
    assert report["required_field_count"] == 15
    assert report["available_field_count"] == 10
    assert report["missing_fields"] == [
        "cohort_role",
        "stimulus_id",
        "stimulus_direction",
        "stimulus_angular_position_degrees",
        "baseline_window_seconds",
    ]
    assert report["all_required_recording_fields_available"] is False


def test_PD_ND_synthesis_and_analysis_baseline_do_not_count_as_source_fields() -> None:
    report = json.loads(REPORT.read_text())
    assert report["array_has_direction_axis"] is False
    assert (
        report["notebook_evidence"][
            "PD_ND_labels_apply_after_source_average_and_synthetic_shift"
        ]
        is True
    )
    assert report["field_status"]["stimulus_direction"]["available"] is False
    assert report["field_status"]["baseline_window_seconds"]["available"] is False
    assert report["boundary"]["baseline_window_is_post_hoc_analysis_protocol"] is True
    assert (
        report["paper_evidence"][
            "one_second_prestimulus_baseline_applies_to_PD_grating_protocol"
        ]
        is True
    )
    assert (
        report["paper_evidence"]["Fig3_edge_recording_baseline_window_declared"]
        is False
    )


def test_complete_dataset_and_workbook_internals_do_not_recover_missing_fields() -> (
    None
):
    report = json.loads(REPORT.read_text())
    index = report["complete_public_dataset_index"]
    assert index["dataset_file_count"] == 74
    assert index["Fig3_directory_file_count"] == 9
    assert index["Fig3_identity_or_metadata_sidecars"] == []
    versions = report["public_dataset_version_history"]
    assert versions["public_version_count"] == 1
    assert versions["release_id"] == 398
    assert versions["version"] == "1.0"
    assert versions["internal_version_number"] == 4
    assert versions["version_state"] == "RELEASED"
    assert versions["file_count"] == 74
    assert versions["older_public_payload_version_available"] is False
    assert versions["internal_version_number_interpreted_as_public_history"] is False
    workbook = report["source_workbook_package"]
    assert workbook["hidden_sheets"] == []
    assert workbook["connection_count"] == 38
    assert workbook["all_connections_deleted"] is True
    assert len(workbook["raw_input_connection_names"]) == 10
    assert workbook["derived_connection_count"] == 28
    assert workbook["package_metadata_members"] == []
    assert workbook["table_members"] == []
    assert workbook["defined_name_count"] == 38
    assert workbook["all_defined_names_sheet_local"] is True
    assert workbook["hidden_defined_names"] == []
    assert workbook["raw_source_headers_are_only_time_and_type_ordinals"] is True
    assert workbook["PD_ND_headers_are_only_on_derived_sheets"] is True
    assert workbook["candidate_recording_metadata_strings"] == []
    assert all(
        item["hidden_row_count"] == 0
        and item["hidden_column_range_count"] == 0
        and item["formulas_restricted_to_time_column_A"] is True
        for item in workbook["worksheet_structures"].values()
    )
    assert workbook["recording_metadata_recovered_from_package"] is False
    notebooks = report["cross_directory_notebook_audit"]
    assert notebooks["verified_file_count"] == 16
    assert notebooks["verified_total_bytes"] == 5352395
    assert notebooks["Fig3_source_consumer_notebook_count"] == 4
    assert sorted(notebooks["Fig3_source_consumer_notebooks"]) == [
        "edfig7.ipynb",
        "edfig8.ipynb",
        "fig3.ipynb",
        "fig5.ipynb",
    ]
    assert all(
        item["direct_numpy_load_references"]
        == item["fig3_source_array_references"]
        and item["identity_or_recording_sidecar_references"] == []
        and item["averages_source_arrays_over_cell_axis"] is True
        for item in notebooks["Fig3_source_consumer_notebooks"].values()
    )
    assert notebooks["identity_or_recording_sidecar_reference_found"] is False
    assert notebooks["recording_metadata_recovered_from_other_notebooks"] is False
    cross_figure = report["cross_figure_identity_boundary"]
    assert cross_figure["Fig1_individual_spatial_RF_counts"] == {
        "Mi1": 22,
        "Tm3": 11,
        "Mi4": 10,
        "C3": 16,
    }
    assert cross_figure["Fig3_voltage_cell_counts"] == {
        "Mi1": 24,
        "Tm3": 12,
        "Mi4": 19,
        "C3": 16,
    }
    assert cross_figure["shared_stable_individual_identifier_present"] is False
    assert cross_figure["cohort_overlap_or_disjointness_identifiable"] is False
    assert (
        report["paper_evidence"][
            "Fig3_input_classes_aligned_from_template_RF_relative_distances"
        ]
        is True
    )
    assert (
        report["paper_evidence"][
            "per_recording_RF_centres_or_angular_positions_published"
        ]
        is False
    )
    assert (
        report["notebook_evidence"][
            "source_cell_axis_averaged_before_direction_synthesis"
        ]
        is True
    )
    assert report["notebook_evidence"]["synthetic_PD_ND_shift_samples"] == 160


def test_incomplete_fields_keep_all_downstream_gates_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["complete_stimulus_and_baseline_fields_on_allowed_payload"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
