import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-four-hop-scalar-precheck.json"


def test_four_hop_scalar_precheck_is_frozen_and_non_advancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_id"] == "S1-T01"
    assert protocol["candidate_count"] == 3
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["runtime_modified"] is False


def test_four_hop_coverage_is_sufficient_but_scalar_direction_gate_fails() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["candidates"]) == {"T4_proximal", "T4_distal", "T5_tm9"}
    for result in report["candidates"].values():
        assert result["fast_path_summary"]["reachable_target_fraction"] > 0.99
        assert result["delayed_path_summary"]["reachable_target_fraction"] > 0.99
        assert result["modes"]["ordered"]["bilateral_direction_subtypes"] == []
        assert result["passed"] is False
    assert report["candidates"]["T4_distal"]["modes"]["temporal_shuffle"][
        "bilateral_direction_subtypes"
    ] == ["a", "c"]
    assert report["candidates"]["T5_tm9"]["modes"]["temporal_shuffle"][
        "bilateral_direction_subtypes"
    ] == ["a"]
    assert report["advancing_candidates"] == []
    assert report["four_hop_scalar_gate_passed"] is False
    assert report["expand_to_three_conditions"] is False
    assert report["advance_to_calibration"] is False
