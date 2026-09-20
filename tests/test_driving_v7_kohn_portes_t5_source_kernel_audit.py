import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-t5-source-kernel-audit.json"


def test_source_kernel_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_all_four_voltage_derived_kernel_sets_have_physical_time() -> None:
    report = json.loads(REPORT.read_text())
    assert list(report["source_results"]) == ["Tm1", "Tm2", "Tm4", "Tm9"]
    assert report["aggregate"]["saline_record_count"] == 25
    assert report["aggregate"]["saline_unique_recording_id_count_sum"] == 24
    assert report["aggregate"]["kernel_lengths"] == [499, 500]
    assert report["aggregate"]["orientation_counts"] == {
        "null": 24,
        "vertical": 1,
    }
    assert report["aggregate"]["maximum_temporal_formula_error"] < 1e-15
    assert report["aggregate"]["maximum_complete_formula_error"] < 1e-15
    assert report["gates"]["all_four_voltage_derived_temporal_kernel_sets_verified"]
    assert report["gates"]["physical_kernel_time_axis_verified"]
    assert report["Tm1_Tm2_Tm4_Tm9_voltage_derived_temporal_kernels_verified"] is True


def test_kernel_existence_does_not_imply_gain_or_transfer_validity() -> None:
    report = json.loads(REPORT.read_text())
    assert report["author_method"]["raw_temporal_filter_declared_correct_scale"] is False
    assert (
        report["author_method"]["author_reported_LN_score_independent_validation_verified"] is False
    )
    assert report["author_method"]["paper_reports_cross_stimulus_flash_mismatch"] is True
    assert report["gates"]["raw_temporal_filter_absolute_gain_transferable"] is False
    assert report["gates"]["stimulus_invariant_source_kernel_verified"] is False
    assert report["gates"]["biological_individual_identity_verified"] is False
    assert report["gates"]["independent_dynamic_validation_available"] is False
    assert report["gates"]["CT1_voltage_derived_temporal_kernel_available"] is False
    assert report["gates"]["v7_normalized_state_mapping_available"] is False
    assert report["source_kernel_transfer_authorized"] is False
    assert report["authorize_T5_functional_precheck"] is False


def test_leave_one_recording_id_out_is_descriptive_only() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Tm1", "Tm2", "Tm4", "Tm9"):
        diagnostic = report["source_results"][source][
            "descriptive_leave_one_recording_id_out_shape_correlation"
        ]
        assert diagnostic["used_as_transfer_gate"] is False
        assert (
            diagnostic["summary"]["count"]
            == report["source_results"][source]["unique_recording_id_count"]
        )
        assert all(item["correlation"] < 1 for item in diagnostic["folds"])
