import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-source-mapping-scope-audit.json"


def test_mapping_scope_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_four_voltage_sources_have_exact_type_average_mapping() -> None:
    report = json.loads(REPORT.read_text())
    assert report["Tm1_Tm2_Tm4_Tm9_exact_type_average_mapping_complete"] is True
    for source in ("Tm1", "Tm2", "Tm4", "Tm9"):
        row = report["source_mappings"][source]
        assert row["mapping_mode"] == "exact_type_average"
        assert row["recording_level_body_assignment"] is False
        assert row["mapping_scope_complete"] is True
        assert all(row["gates"].values())


def test_CT1_keeps_five_source_mapping_and_downstream_gates_closed() -> None:
    report = json.loads(REPORT.read_text())
    ct1 = report["source_mappings"]["CT1"]
    assert ct1["mapping_mode"] == "unavailable"
    assert ct1["gates"]["voltage_derived_source_kernel_available"] is False
    assert ct1["gates"]["explicit_exact_type_average_declared"] is False
    assert ct1["gates"]["complete_columnar_retinotopy_available"] is False
    boundary = report["CT1_mapping_boundary"]
    assert boundary["body_ids"] == [10009, 10157]
    assert boundary["per_synapse_Lo1_columnar_retinotopy_available"] is True
    assert boundary["Lo1_column_count_by_body"] == {"10009": 868, "10157": 802}
    assert boundary["complete_official_LO_column_coverage"] is False
    assert boundary["single_body_column_coordinate_available"] is False
    assert report["T5_all_five_source_mapping_complete"] is False
    assert report["stable_source_or_target_cell_to_MaleCNS_mapping"] is False
    assert report["authorize_source_dynamics_transfer"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
