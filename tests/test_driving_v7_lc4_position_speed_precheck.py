import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-lc4-position-speed-precheck.json"


def test_lc4_position_speed_precheck_is_tuning_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_id"] == "LPLC-T01"
    assert protocol["position_count"] == 15
    assert protocol["duration_frames"] == [32, 16, 8]
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["runtime_modified"] is False
    assert report["primary_precheck_readout"] == "peak_positive_state_derivative"


def test_lc4_position_speed_has_negative_not_positive_slope() -> None:
    report = json.loads(REPORT.read_text())
    assert report["per_side"]["L"]["cell_count"] == 71
    assert report["per_side"]["R"]["cell_count"] == 55
    assert report["readouts"]["peak_positive_state_derivative"] == report["per_side"]
    derivative = report["readouts"]["peak_positive_state_derivative"]
    assert derivative["L"]["valid_cell_fraction"] < 0.16
    assert derivative["R"]["valid_cell_fraction"] < 0.24
    assert derivative["L"]["positive_slope_fraction"] == 0.0
    assert derivative["R"]["positive_slope_fraction"] < 0.08
    for result in derivative.values():
        assert result["median_r_squared"] > 0.90
        assert result["passed"] is False
    assert report["LC4_position_speed_precheck_passed"] is False
    assert report["expand_to_three_conditions"] is False
    assert report["advance_to_calibration"] is False
