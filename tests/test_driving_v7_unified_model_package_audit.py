import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-unified-model-package-audit.json"


def test_unified_model_packages_and_members_are_verified() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    model = report["packages"]["model"]
    assert model["file_id"] == 31067761
    assert model["bytes"] == 3_630_019
    assert model["md5"] == "14ba0fa761a513d55cacc41610881e80"
    assert model["zip_integrity_passed"] is True
    supporting = report["packages"]["supporting"]
    assert supporting["file_id"] == 30862813
    assert supporting["bytes"] == 17_374
    assert supporting["zip_integrity_passed"] is True
    assert report["files_verified"] is True


def test_unified_model_tables_decode_and_selection_is_reproducible() -> None:
    report = json.loads(REPORT.read_text())
    tables = report["tables"]
    assert tables["names"] == [
        "cardT4OptTab",
        "cardT5OptTab",
        "diagT4OptTab",
        "diagT5OptTab",
    ]
    assert set(map(tuple, tables["all_shapes"].values())) == {(1000, 131)}
    assert set(tables["all_parameter_column_counts"].values()) == {29}
    assert tables["author_selection_reproduced"] is True
    assert tables["author_selected_cardT5_row_one_based"] == 345
    assert tables["T4_candidates"]["cardT4OptTab"]["selected_row_one_based"] == 69
    assert tables["T4_candidates"]["diagT4OptTab"]["selected_row_one_based"] == 596
    assert all(
        item["minimum_is_unique"] for item in tables["T4_candidates"].values()
    )


def test_verified_target_RF_parameters_do_not_authorize_MaleCNS_transfer() -> None:
    report = json.loads(REPORT.read_text())
    semantics = report["model_semantics"]
    assert semantics["target_component_model_verified"] is True
    assert semantics["components"] == ["E", "I", "E2", "I2"]
    assert set(semantics["source_type_mentions_in_t4_model"].values()) == {False}
    scan = semantics["package_text_scan"]
    assert scan["member_count"] == 9
    assert scan["any_source_type_token_found"] is False
    assert all(not matches for matches in scan["source_type_token_hits"].values())
    assert all(scan["abstract_component_token_hits"][name] for name in ("E2", "I2"))
    assert semantics["manual_photoreceptor_to_target_delay_milliseconds"] == 30.0
    assert semantics["ON_OFF_parameter_block_reordering_present"] is True
    gates = report["transfer_gates"]
    assert gates["model_package_verified"] is True
    assert gates["supporting_package_verified"] is True
    assert gates["author_preselected_T4_rows_available"] is False
    assert gates["E_I_E2_I2_to_MaleCNS_source_mapping_available"] is False
    assert gates["cardinal_diagonal_to_T4_subtype_mapping_available"] is False
    assert gates["independent_cell_holdout_available"] is False
    assert report["T4_target_model_parameters_available"] is True
    assert report["MaleCNS_source_kernel_transfer_authorized"] is False
    assert report["T4_functional_candidate_authorized"] is False
    assert report["boundary"]["raw_packages_ignored_and_not_committed"] is True
