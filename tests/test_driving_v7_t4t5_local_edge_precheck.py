import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4t5-local-edge-precheck.json"


def test_local_edge_precheck_is_single_condition_and_non_advancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_id"] == "LPLC-T01"
    assert protocol["position_count"] == 15
    assert protocol["stimulus_count"] == 120
    assert protocol["noise_standard_deviation"] == 0.0
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["runtime_modified"] is False
    assert report["assignment"]["target_response_used_for_assignment"] is False
    assert report["assignment"]["unmapped_cells_retained_as_invalid"] is True


def test_local_edge_precheck_stops_after_zero_direction_populations_pass() -> None:
    report = json.loads(REPORT.read_text())
    assert len(report["population_scores"]) == 16
    assert report["summary"] == {
        "direction_pass_count": 0,
        "direction_population_count": 16,
        "polarity_pass_count": 0,
        "polarity_population_count": 16,
        "minimum_direction_populations_to_expand": 1,
        "expand_to_three_conditions": False,
    }
    assert all(
        not score[kind]["passed"]
        for score in report["population_scores"].values()
        for kind in ("direction", "polarity")
    )
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
