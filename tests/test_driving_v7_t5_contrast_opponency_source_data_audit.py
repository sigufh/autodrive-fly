import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-contrast-opponency-source-data-audit.json"


def test_contrast_opponency_source_workbook_is_verified_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    source = report["protocol"]["source_data"]
    assert source["bytes"] == 4_459_008
    assert source["sha256"] == source["actual_sha256"]
    assert source["actual_sheet_count"] == 32
    assert source["xlrd_version"] == "2.0.1"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_Tm4_and_Tm9_temporal_blocks_are_distinct_and_physical() -> None:
    blocks = json.loads(REPORT.read_text())["verified_dynamic_blocks"]
    assert blocks["Tm4_full_field"]["roi_count"] == 131
    assert blocks["Tm9_full_field"]["roi_count"] == 119
    assert math.isclose(blocks["Tm4_full_field"]["median_time_step_seconds"], 0.1)
    assert math.isclose(blocks["Tm9_full_field"]["median_time_step_seconds"], 0.1)
    assert blocks["Tm4_full_field"]["indicator"] == "jRGECO1a"
    assert blocks["Tm9_full_field"]["indicator"] == "GCaMP6f"


def test_CT1_is_spatial_only_and_transfer_remains_closed() -> None:
    report = json.loads(REPORT.read_text())
    contract = report["T5_source_contract"]
    assert contract["sources_with_verified_temporal_blocks"] == ["Tm4", "Tm9"]
    assert contract["sources_with_verified_spatial_but_not_temporal_blocks"] == ["CT1"]
    assert contract["missing_temporal_sources"] == ["Tm1", "Tm2", "CT1"]
    assert contract["temporal_coverage_fraction"] == 0.4
    for block in report["verified_spatial_blocks"].values():
        assert block["position_samples"] == 25
        assert block["roi_count"] == 162
        assert block["median_position_step_degrees"] == 2.0
    assert report["complete_T5_source_dynamics_transfer_authorized"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"]["CT1_spatial_response_is_not_CT1_temporal_dynamics"] is True
