import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-source-type-average-mapping-contract.json"


def test_type_average_mapping_contract_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_eight_voltage_sources_have_explicit_exact_type_average_mapping() -> None:
    report = json.loads(REPORT.read_text())
    assert report["sources_with_explicit_type_average_mapping"] == [
        "Mi1",
        "Tm3",
        "Mi4",
        "C3",
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
    ]
    for source in report["sources_with_explicit_type_average_mapping"]:
        row = report["source_mappings"][source]
        assert row["recording_level_body_assignment"] is False
        assert row["broadcast_rule"] == ("one_population_mean_kernel_to_all_exact_same_type_bodies")


def test_incomplete_retinotopy_and_CT1_payload_keep_mapping_contract_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert (
        report["source_mappings"]["Tm9"]["gates"]["complete_columnar_retinotopy_available"] is False
    )
    ct1 = report["source_mappings"]["CT1"]
    assert ct1["gates"]["local_allowed_unit_payload_available"] is False
    assert ct1["gates"]["explicit_exact_type_average_declared"] is False
    assert ct1["gates"]["complete_columnar_retinotopy_available"] is False
    assert report["all_nine_source_mapping_contracts_complete"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
