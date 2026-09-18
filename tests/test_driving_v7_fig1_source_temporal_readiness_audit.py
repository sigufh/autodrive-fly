import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-fig1-source-temporal-readiness-audit.json"


def test_fig1_source_temporal_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    files = report["protocol"]["source_files"]
    assert files["figure_1"]["sha256"] == (
        "218eaca70ddbf2d073695c376e19eea4a75a15f0fd3a44e219b33f56cef9b421"
    )
    assert files["extended_figure_1"]["sha256"] == (
        "418da8fc20565af66fc66ba8ebce4c6f61a11b11e46a4788c5323da17ec05da4"
    )
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_fig1_xlsx_exports_source_spatial_but_not_temporal_fields() -> None:
    report = json.loads(REPORT.read_text())
    source = report["figure_1_source_tables"]
    assert set(source) == {"Mi9", "Tm3", "Mi1", "Mi4", "C3"}
    assert {tuple(item["shape"]) for item in source.values()} == {(131, 131)}
    assert all(item["first_axis_label"] == "Elevation\\Azimuth (°)" for item in source.values())
    assert all(item["contains_time_axis"] is False for item in source.values())
    extended = report["extended_figure_1"]
    assert extended["individual_source_spatial_RF_counts"] == {
        "Mi9": 22,
        "Tm3": 11,
        "Mi1": 22,
        "Mi4": 10,
        "C3": 16,
    }
    assert extended["individual_T4_spatial_RF_count"] == 46
    assert extended["contains_source_time_axis"] is False


def test_safely_loaded_object_payload_keeps_non_voltage_transfer_closed() -> None:
    report = json.loads(REPORT.read_text())
    payload = report["edmond_object_payload"]
    assert payload["file_id"] == 104768
    assert payload["bytes"] == 35_520_207
    assert payload["sha256"] == (
        "0dd4a309e3ddec79b5898b4764e0dec12d707c14bb0865c4aae45f5f0d381d82"
    )
    assert payload["header"]["contains_python_objects"] is True
    assert payload["locally_available"] is True
    assert payload["hash_verified"] is True
    assert payload["safely_inspected"] is True
    assert payload["pickle_globals"] == [
        "_codecs encode",
        "numpy dtype",
        "numpy ndarray",
        "numpy.core.multiarray _reconstruct",
    ]
    assert set(payload["source_arrays"]) == {"Mi9", "Tm3", "Mi1", "Mi4", "C3"}
    assert all(item["all_finite"] for item in payload["source_arrays"].values())
    assert all(
        item["all_ordinals_match"]
        for item in report["individual_spatial_workbook_cross_check"].values()
    )
    assert report["observations"]["individual_source_temporal_RF_arrays_available"] is True
    assert report["source_filter_fit_gates"]["allowed_membrane_voltage_response_unit"] is False
    assert report["source_filter_fit_gates"]["independent_from_Fig3_training_cohort"] is False
    assert report["source_temporal_kernel_transfer_authorized"] is False
    assert report["advance_to_source_filter_fit"] is False
    assert report["boundary"]["moving_edge_trace_conflates_spatial_and_temporal_filtering"] is True
    assert report["boundary"]["T4_spatiotemporal_RF_must_not_define_source_dynamics"] is True
