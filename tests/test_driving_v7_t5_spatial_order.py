import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-spatial-order.json"


def test_t5_spatial_order_is_tuning_only_and_nonadvancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_ids"] == ["S1-T01", "S1-T02", "S1-T03"]
    assert report["protocol"]["moving_edge_stimulus_count"] == 24
    assert report["protocol"]["target_sample_count"] == 128
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False
    assert report["advance_to_full_population"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["advance_to_navigation"] is False


def test_t5_spatial_order_preserves_separate_source_semantics() -> None:
    mechanism = json.loads(REPORT.read_text())["mechanism"]
    assert mechanism["fast_sources"] == ["Tm1", "Tm2", "Tm4"]
    assert mechanism["spatial_delayed_sources"] == ["Tm9"]
    assert mechanism["global_modulator_sources"] == ["CT1"]
    assert mechanism["ct1_has_optic_hex_coordinates"] is False
    assert mechanism["ct1_used_in_spatial_pool"] is False
    assert mechanism["labels_frozen"] is True
    assert mechanism["thresholds_frozen"] is True


def test_t5_spatial_order_rejects_b_d_expansion() -> None:
    report = json.loads(REPORT.read_text())
    result = report["b_d_reachability"]
    assert result == {
        "reachable_count": 3,
        "target_count": 64,
        "reachable_fraction": 3 / 64,
        "minimum_required_fraction": 0.25,
        "passed": False,
    }
    populations = report["reachable_by_population"]
    assert populations["T5b_R"]["all_six_comparisons_reachable_count"] == 0
    assert populations["T5d_R"]["all_six_comparisons_reachable_count"] == 0
    body_ids = [item["body_id"] for item in report["target_results"]]
    assert len(body_ids) == len(set(body_ids)) == 128
