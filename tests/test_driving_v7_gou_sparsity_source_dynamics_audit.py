import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-gou-sparsity-source-dynamics-audit.json"


def test_gou_audit_is_hash_bound_read_only_and_repository_specific() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["repositories"]["Dryad"]["version_id"] == 402168
    assert report["repositories"]["DANDI"]["version"] == "0.250602.0251"
    assert report["repositories"]["DANDI"]["asset_count"] == 282
    assert report["protocol"]["external_payload_evaluated"] is True
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_gou_data_cover_only_four_of_nine_required_sources() -> None:
    report = json.loads(REPORT.read_text())
    assert report["covered_required_sources_by_family"] == {
        "T4": ["Mi1", "Tm3"],
        "T5": ["Tm1", "Tm2"],
    }
    assert report["missing_required_sources_by_family"] == {
        "T4": ["Mi4", "C3"],
        "T5": ["Tm4", "Tm9", "CT1"],
    }
    assert report["covered_required_source_count"] == 4
    assert report["required_source_count"] == 9
    assert report["coverage_fraction"] == 4 / 9
    assert report["partial_source_dynamics_evidence_present"] is True


def test_partial_fluorescence_evidence_does_not_open_any_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["source_response_unit_gates"].values()) == {False}
    assert set(report["source_moving_bar_direction_label_gates"].values()) == {False}
    assert (
        report["transfer_gates"]["every_covered_source_has_record_level_direction_labels"]
        is False
    )
    assert report["complete_external_source_dynamics_evidence"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["advance_to_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["download_disposition"]["Dryad_archive_download_authorized"] is False
    assert report["download_disposition"]["DANDI_bulk_download_authorized"] is False
    assert (
        report["boundary"]["partial_source_coverage_does_not_satisfy_the_unified_contract"] is True
    )


def test_local_archive_and_processed_calcium_arrays_are_verified() -> None:
    report = json.loads(REPORT.read_text())
    dryad = report["repositories"]["Dryad"]
    inventory = report["local_payload_inventory"]
    assert dryad["payload_downloaded"] is True
    assert dryad["payload_sha256_locally_verified"] is True
    assert dryad["publisher_archive_hash_matches_file_record"] is True
    assert inventory["archive"]["publisher_archive"]["zip_integrity_passed"] is True
    assert inventory["archive"]["publisher_archive"]["real_file_count"] == 55
    assert inventory["archive"]["publisher_archive"]["real_mat_file_count"] == 25
    expected_flash = {"Mi1": 16, "Tm3": 11, "Tm1": 9, "Tm2": 8}
    expected_moving = {"Mi1": 10, "Tm3": 6, "Tm1": 8, "Tm2": 12}
    for source, fly_count in expected_flash.items():
        item = inventory["flash"][source]
        assert item["identity"]["fly_axis_size"] == fly_count
        assert item["arrays"]["indFilters"]["shape"] == [fly_count, 70]
        for array in item["arrays"].values():
            assert array["finite_count"] == array["element_count"]
            assert array["nan_count"] == 0
            assert array["infinite_count"] == 0
    for source, fly_count in expected_moving.items():
        item = inventory["moving_bar"][source]
        assert item["identity"]["fly_axis_size"] == fly_count
        assert item["arrays"]["kernels"]["shape"] == [141, 6, fly_count]
        assert item["arrays"]["LightImpulseResps"]["shape"] == [141, 6, fly_count]
        assert item["arrays"]["DarkImpulseResps"]["shape"] == [141, 6, fly_count]
        assert item["arrays"]["whiteKernels"]["shape"] == [141, 12, fly_count]
        assert item["arrays"]["blackKernels"]["shape"] == [141, 12, fly_count]
        assert item["field_semantics"]["direction_named_field_names"] == []


def test_script_time_axes_baselines_and_identity_boundaries_are_explicit() -> None:
    report = json.loads(REPORT.read_text())
    inventory = report["local_payload_inventory"]
    scripts = inventory["script_semantics"]
    assert scripts["flash"]["analysis_rate_hz"] == 30
    assert scripts["flash"]["filter_point_count"] == 70
    assert scripts["flash"]["impulse_point_count"] == 71
    assert scripts["moving_bar"]["analysis_rate_hz"] == 60
    assert scripts["moving_bar"]["point_count"] == 141
    assert scripts["moving_bar"]["condition_order"] == [
        "1/12",
        "2/12",
        "4/12",
        "6/12",
        "8/12",
        "12/12",
    ]
    assert scripts["analysis_axes_are_not_exact_acquisition_timestamps"] is True
    assert inventory["identity_conclusion"]["stable_biological_individual_ids_verified"] is False
    assert inventory["measurement_boundary"]["experimental_membrane_voltage"] is False
    assert report["stimulus_and_recording"]["baseline_windows_verified_from_payload"] is False
    assert report["stimulus_and_recording"]["reported_moving_bar_direction_labels"] == []
    assert (
        report["stimulus_and_recording"]["processed_moving_bar_direction_axis_present"]
        is False
    )
    assert (
        report["stimulus_and_recording"][
            "processed_moving_bar_direction_order_verified_from_payload_or_Fig6_script"
        ]
        is False
    )
    assert all(
        item["moving_bar_stimulus_present"]
        and not item["processed_moving_bar_direction_labels_available"]
        for item in report["source_evidence"].values()
    )


def test_dandi_has_asset_subjects_but_no_dryad_row_crosswalk() -> None:
    report = json.loads(REPORT.read_text())
    identity = report["DANDI_identity_audit"]
    assert identity["asset_index"]["asset_count"] == 282
    assert identity["asset_index"]["unique_subject_id_count"] == 282
    assert identity["asset_index"]["unique_session_start_count"] == 282
    assert identity["identity_conclusions"][
        "DANDI_asset_level_stable_participant_IDs_available"
    ] is True
    assert identity["Dryad_identity_labels"]["distinct_fliesUsed_label_count"] == 66
    assert identity["Dryad_identity_labels"]["exact_match_count"] == 0
    assert identity["identity_conclusions"][
        "Dryad_fliesUsed_to_DANDI_subject_crosswalk_verified"
    ] is False
    assert report["transfer_gates"]["processed_rows_linked_to_stable_subject_ids"] is False
