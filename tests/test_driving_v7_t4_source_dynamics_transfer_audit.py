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
