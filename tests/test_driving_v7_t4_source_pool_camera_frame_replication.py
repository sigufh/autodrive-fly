import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-pool-camera-frame-replication.json"


def test_camera_frame_replication_is_preregistered_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["preregistration_commit"] == (
        "7e4ef1a88579137bac722527f88ffc0d3af5620b"
    )
    assert report["protocol"]["preregistration_sha256"] == (
        "cf7f311138fb029b19c4bf155b73eb551ec665994ebc0d18de53be72de2a0c4a"
    )
    assert report["protocol"]["discovery_condition_excluded"] is True
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_ordered_replication_fails_and_controls_obey_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["fixed_whole_T4_population_denominator"] == 6861
    assert report["replication_evaluated"] is True
    assert report["candidate_target_population_denominator"] == 1709
    expected = {
        "LPLC-T02": {
            "T4d_L": (-0.8703534267888534, 0.17355371900826447),
            "T4d_R": (-0.7715781038598406, 0.25844004656577413),
        },
        "LPLC-T03": {
            "T4d_L": (-0.8737953557321949, 0.17473435655253838),
            "T4d_R": (-0.7844614740115432, 0.26426076833527357),
        },
    }
    for condition, populations in expected.items():
        result = report["ordered_results"][condition]
        assert result["mode"] == "ordered"
        assert result["bilateral_T4d_passed"] is False
        for population, values in populations.items():
            score = result["population_scores"][population]
            assert score["median_signed_contrast"] == values[0]
            assert score["positive_cell_fraction"] == values[1]
            assert score["passed"] is False
    assert report["ordered_replication_passed"] is False
    assert report["controls_evaluated"] is False
    assert report["control_results"] == {}
    assert report["controls_passed"] is False
    assert report["replication_gate_passed"] is False
    assert report["authorize_new_target_formula"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["stop_reason"] == (
        "preregistered_ordered_replication_failed_controls_not_run"
    )
