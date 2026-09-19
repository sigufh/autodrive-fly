import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-gou-moving-bar-direction-audit.json"


def test_gou_direction_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_fig6_consumed_condition_axis_is_sparsity_not_direction() -> None:
    report = json.loads(REPORT.read_text())
    semantics = report["script_semantics"]
    assert semantics["Fig6_condition_order"] == [
        "1/12",
        "2/12",
        "4/12",
        "6/12",
        "8/12",
        "12/12",
    ]
    assert semantics["Fig6_condition_axis_is_sparsity"] is True
    assert semantics["Fig6_reads_unconsumed_twelve_condition_arrays"] is False
    for source, count in {"Mi1": 10, "Tm3": 6}.items():
        item = report["source_results"][source]
        assert item["fly_axis_size"] == count
        assert item["array_shapes"]["kernels"] == [141, 6, count]
        assert item["direction_named_fields"] == []
        assert item["six_condition_axis_identified_as_sparsity"] is True


def test_unlabelled_arrays_and_target_rules_do_not_create_source_directions() -> None:
    report = json.loads(REPORT.read_text())
    semantics = report["script_semantics"]
    assert semantics["Fig7_target_direction_rule_explicit"] is True
    assert semantics["Fig7_target_direction_rule_applied_to_Fig6_source_arrays"] is False
    for item in report["source_results"].values():
        assert item["array_shapes"]["whiteKernels"][1] == 12
        assert item["array_shapes"]["blackKernels"][1] == 12
        assert item["twelve_condition_array_column_labels_available"] is False
        assert item["record_level_direction_axis_identifiable"] is False
    assert report["Gou_Mi1_Tm3_record_level_direction_axis_identifiable"] is False
    assert report["authorize_direction_invariance_audit"] is False
    assert report["direction_invariance_evaluated"] is False
    assert report["authorize_T4_source_kernel_transfer"] is False
