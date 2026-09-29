import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-pair-failure-boundary-audit.json"


def test_source_pair_failure_boundary_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_ordered_and_control_failures_are_population_orthogonal() -> None:
    report = json.loads(REPORT.read_text())
    assert report["focus_pair"] == "Tm3_x_C3"
    assert report["focus_reduction"] == "positive_peak"
    assert report["ordered_direction_missing_populations"] == ["T4d_L"]
    assert report["ordered_polarity_missing_populations"] == ["T4d_L"]
    assert report["ordered_missing_population_metrics"] == {
        "T4d_L": {
            "direction_median_signed_contrast": 0.04787008727666105,
            "direction_positive_cell_fraction": 0.5482673267326733,
        }
    }
    assert report["frozen_direction_thresholds"] == {
        "minimum_median_signed_contrast": 0.1,
        "minimum_positive_cell_fraction": 0.6,
    }
    assert report["control_bilateral_direction_subtypes"] == ["a"]
    assert report["control_bilateral_populations"] == ["T4a_L", "T4a_R"]
    assert report["ordered_missing_and_control_bilateral_populations_overlap"] == []
    assert report["ordered_failure_and_control_failure_are_population_orthogonal"] is True


def test_orthogonal_failure_does_not_authorize_post_hoc_repair() -> None:
    report = json.loads(REPORT.read_text())
    assert report["authorize_post_hoc_T4d_L_repair"] is False
    assert report["authorize_ignore_temporal_shuffle_T4a"] is False
    assert report["authorize_source_pair_candidate"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
