import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-lplc1-input-structure.json"


def test_lplc1_input_structure_is_hash_bound_and_non_interventional() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["boundary"]["structure_only"] is True
    assert report["boundary"]["all_LPLC1_cells_in_denominator"] is True
    assert report["boundary"]["universal_physiological_inhibition_claimed"] is False


def test_lplc1_object_and_motion_coverage_pass_but_spatial_inhibition_does_not() -> None:
    report = json.loads(REPORT.read_text())
    left, right = report["per_side"]["L"], report["per_side"]["R"]
    assert (left["target_count"], right["target_count"]) == (68, 66)
    for side in (left, right):
        assert side["groups"]["object_detectors"]["passed"] is True
        assert side["groups"]["motion_detectors"]["passed"] is True
        assert side["groups"]["glutamatergic"]["targets_with_input_fraction"] == 1.0
    assert left["groups"]["glutamatergic"]["located_source_weight_fraction"] < 0.61
    assert right["groups"]["glutamatergic"]["located_source_weight_fraction"] < 0.78
    assert left["groups"]["GABAergic"]["located_source_weight_fraction"] < 0.78
    assert 0.829 < right["groups"]["GABAergic"]["located_source_weight_fraction"] < 0.83
    assert report["strict_input_structure_gate_passed"] is False
    assert report["authorize_spatial_inhibition_mechanism"] is False
    assert "glutamatergic spatial-coordinate coverage" in report["stop_reason"]
