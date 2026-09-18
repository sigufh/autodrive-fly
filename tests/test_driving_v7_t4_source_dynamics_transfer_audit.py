import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-dynamics-transfer-audit.json"


def test_T4_source_dynamics_transfer_audit_is_read_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["required_sources"] == ["Mi1", "Tm3", "Mi4", "C3"]
    assert report["protocol"]["read_only"] is True
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_T4_source_recordings_do_not_supply_label_blind_transfer_contract() -> None:
    report = json.loads(REPORT.read_text())
    assert report["available_source_recordings"]["axes"] == [
        "stimulus_on_off",
        "cell",
        "time_milliseconds",
    ]
    assert all(
        item["stable_MaleCNS_body_ids_available"] is False
        for item in report["available_source_recordings"]["sources"].values()
    )
    synthesis = report["paper_direction_synthesis"]
    assert synthesis["shift_samples"] == 160
    assert synthesis["target_PD_ND_conditioned"] is True
    assert synthesis["may_be_used_as_label_blind_v7_source_kernel"] is False
    assert set(report["required_transfer_fields_available"].values()) == {False}
    assert set(report["transfer_gates"].values()) == {False}
    package = report["official_unified_model_package"]
    assert package["doi"] == "10.25378/janelia.16663486"
    assert package["article_id"] == 16663486
    assert package["size_bytes"] == 3_630_019
    assert package["description_names_optTables"] is True
    assert package["prior_live_audit_file_manifest_retrieved"] is False
    assert package["file_manifest_retrieved"] is True
    assert package["recovered_manifest_consistent"] is True
    assert package["recovered_file_manifest"] == {
        "recovery_source": "internet_archive_official_page_snapshot",
        "snapshot_timestamp": "20241127113612",
        "file_id": 31067761,
        "file_name": "modelFigure.zip",
        "size_bytes": 3_630_019,
        "md5": "14ba0fa761a513d55cacc41610881e80",
        "mime_type": "application/zip",
        "official_download_url": (
            "https://janelia.figshare.com/ndownloader/files/31067761"
        ),
    }
    assert package["payload_retrieved"] is True
    assert package["prior_live_audit_files_verified"] is False
    assert package["files_verified"] is True
    semantics = report["verified_unified_model_semantics"]
    assert semantics["T4_target_model_parameters_available"] is True
    assert semantics["target_components"] == ["E", "I", "E2", "I2"]
    assert semantics["source_type_mapping_available"] is False
    assert semantics["cardinal_diagonal_to_subtype_mapping_available"] is False
    assert semantics["independent_cell_holdout_available"] is False
    assert semantics["MaleCNS_source_kernel_transfer_authorized"] is False
    fig3 = report["verified_fig3_source_kernel_readiness"]
    assert fig3["source_workbook_verified"] is True
    assert fig3["ON_source_specific_kernels_ready"] is False
    assert fig3["prior_two_pool_kernel_authorized"] is False
    assert fig3["source_specific_kernel_candidate_authorized"] is False
    assert fig3["cross_cell_robustness_passed"] is False
    arenz = report["verified_Arenz_source_filter_readiness"]
    assert arenz["filter_parameters_verified"] is True
    assert arenz["current_source_coverage_fraction"] == 0.75
    assert arenz["missing_current_sources"] == ["C3"]
    assert arenz["physical_time_transfer_authorized"] is False
    assert arenz["source_filter_candidate_authorized"] is False
    c3 = report["verified_C3_STRF_readiness"]
    assert c3["numerical_data_verified"] is True
    assert c3["direct_temporal_measurement_available"] is True
    assert c3["all_required_sources_have_some_direct_temporal_evidence"] is True
    assert c3["all_required_sources_share_one_transferable_parameterization"] is False
    assert c3["membrane_voltage_or_validated_deconvolved_kernel_available"] is False
    assert c3["source_filter_candidate_authorized"] is False
    fig1 = report["verified_Fig1_source_temporal_readiness"]
    assert fig1["workbooks_verified"] is True
    assert fig1["source_average_tables_have_time_axis"] is False
    assert fig1["individual_source_tables_have_time_axis"] is False
    assert fig1["object_payload_hash_verified"] is False
    assert fig1["object_payload_safely_inspected"] is False
    assert fig1["source_temporal_kernel_transfer_authorized"] is False
    c3_filter = report["verified_C3_analytic_filter_precheck"]
    assert c3_filter["band_pass_BIC_below_low_pass_BIC"] is True
    assert c3_filter["minimum_leave_one_fly_out_correlation"] < 0.67
    assert c3_filter["median_leave_one_fly_out_correlation"] > 0.84
    assert c3_filter["filter_family_precheck_passed"] is False
    assert c3_filter["source_filter_transfer_authorized"] is False
    timing = report["verified_TimingModels_source_filter_readiness"]
    assert timing["repository_commit"] == "100bb2f52cb9628477c3883e4b17774b0b244e67"
    assert timing["covered_source_filter_stability_verified"] is True
    assert timing["current_source_coverage_fraction"] == 0.75
    assert timing["missing_current_sources"] == ["C3"]
    assert timing["complete_source_filter_candidate_authorized"] is False
    flyvis = report["verified_FlyVis_C3_time_constant_readiness"]
    assert flyvis["repository_commit"] == "92b3845cc426dd309a1a0e1b3890156c42e14021"
    assert flyvis["pretrained_model_count"] == 50
    assert flyvis["solver_dt_seconds"] == 0.02
    assert 0.066 < flyvis["C3_median_time_constant_seconds"] < 0.068
    assert flyvis["C3_models_at_or_below_solver_dt"] == 21
    assert flyvis["C3_time_constant_transfer_authorized"] is False
    measured = report["verified_C3_measured_filter_robustness"]
    assert measured["cross_deconvolution_stability_passed"] is True
    assert set(measured["minimum_held_out_correlation_by_assumption"]) == {
        "200ms",
        "250ms",
        "300ms",
        "350ms",
    }
    assert max(measured["minimum_held_out_correlation_by_assumption"].values()) < 0.58
    assert measured["every_deconvolution_assumption_passed"] is False
    assert measured["type_shared_kernel_authorized"] is False
    public_models = report["verified_public_T4_model_source_coverage"]
    assert public_models["Clark_required_source_coverage_fraction"] == 0.5
    assert public_models["Clark_missing_required_sources"] == ["C3", "Tm3"]
    assert public_models["ModelDB_required_source_coverage_fraction"] == 0.0
    assert public_models["C3_specific_parameters_available"] is False
    assert public_models["complete_source_dynamics_transfer_authorized"] is False


def test_T4_source_dynamics_transfer_stop_rule_preserves_boundaries() -> None:
    report = json.loads(REPORT.read_text())
    assert report["source_dynamics_transfer_authorized"] is False
    assert report["next_candidate_authorized"] is False
    assert report["stop_reason"] == (
        "published_source_dynamics_not_transferable_to_label_blind_v7"
    )
    assert report["boundary"]["infer_kernel_from_T4_target_labels"] is False
    assert report["boundary"]["use_160ms_shift_as_v7_lag"] is False
    assert report["boundary"]["archived_manifest_does_not_substitute_for_payload"] is True
    assert report["boundary"]["T5_and_LPLC_remain_frozen"] is True
