import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-three-hop-temporal-consistency.json"


def test_temporal_consistency_audit_is_tuning_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["metric_count"] == 6
    assert report["protocol"]["stimulus_count_per_family"] == 60
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_all_label_free_temporal_metrics_fail_to_attenuate_shuffle() -> None:
    report = json.loads(REPORT.read_text())
    assert report["maximum_shuffle_to_ordered_energy_ratio"] == 0.50
    assert set(report["families"]) == {"T4", "T5"}
    for family in report["families"].values():
        assert set(family["metrics"]) == {
            "normalized_mean_vector_coherence",
            "vector_sign_consistency",
            "mean_vector_magnitude",
            "coherence_weighted_magnitude",
            "squared_coherence_weighted_magnitude",
            "sign_persistence_weighted_magnitude",
        }
        assert all(item["passed"] is False for item in family["metrics"].values())
        assert all(
            item["attenuated_population_count"] == 0
            for item in family["metrics"].values()
        )
        assert all(
            item["minimum_shuffle_to_ordered_ratio"] > 1.0
            for item in family["metrics"].values()
        )
        for name in (
            "mean_vector_magnitude",
            "coherence_weighted_magnitude",
            "squared_coherence_weighted_magnitude",
            "sign_persistence_weighted_magnitude",
        ):
            assert family["metrics"][name]["minimum_shuffle_to_ordered_ratio"] > 2.0
    assert report["passing_family_metrics"] == []
    assert report["temporal_consistency_gate_passed"] is False


def test_temporal_consistency_stop_rule_prevents_expansion() -> None:
    report = json.loads(REPORT.read_text())
    assert report["authorize_new_motion_field_candidate"] is False
    assert report["advance_to_three_tuning_conditions"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["stop_reason"] == (
        "all_label_free_consistency_metrics_amplified_temporal_shuffle"
    )
    assert report["boundary"]["label_free_temporal_metrics"] is True
    assert report["boundary"]["stimulus_direction_used_by_metrics"] is False
    assert report["boundary"]["may_validate_direction_selectivity"] is False
