import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-pool-local.json"


def test_source_pool_local_is_read_only_tuning_diagnostic() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_id"] == "LPLC-T01"
    assert protocol["target_count"] == 6861
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["runtime_modified"] is False


def test_source_pool_has_only_unilateral_signal_and_does_not_authorize_formula() -> None:
    report = json.loads(REPORT.read_text())
    assert report["direction_pass_counts"] == {
        "center": 0,
        "proximal": 1,
        "distal": 1,
        "center_proximal_correlation": 0,
    }
    assert all(not values for values in report["bilateral_passing_subtypes"].values())
    assert report["maximum_bilateral_direction_pair_count"] == 0
    assert report["authorize_new_target_formula"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
