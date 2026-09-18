import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-individual-split-1khz-audit.json"


def test_1khz_split_is_hash_bound_and_reuses_frozen_rules() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["sample_interval_milliseconds"] == 1.0
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["frozen_split_contract"]["validation_individual_count"] == 3
    assert report["frozen_gates"] == {
        "minimum_validation_fold_correlation": 0.8,
        "minimum_median_individual_correlation": 0.8,
        "minimum_individual_correlation": 0.0,
        "require_every_source_type_and_condition": True,
    }


def test_all_payloads_and_ordinals_are_verified() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["verified_payloads"]) == {
        "fig3_Tm3.npy",
        "fig3_Mi1.npy",
        "fig3_Mi4.npy",
        "fig3_C3.npy",
    }
    assert all(item["shape"][0] == 2 for item in report["verified_payloads"].values())
    assert report["identity_evidence"]["all_four_sources_both_conditions_verified"] is True


def test_full_resolution_does_not_change_the_pass_fail_pattern() -> None:
    report = json.loads(REPORT.read_text())
    comparison = report["comparison_to_10ms"]
    assert comparison["prior_passing_source_conditions"] == ["on:Tm3"]
    assert comparison["full_resolution_passing_source_conditions"] == ["on:Tm3"]
    assert comparison["pass_fail_pattern_changed"] is False
    assert all(
        not values["pass_changed"]
        for condition in comparison["metric_deltas_1khz_minus_10ms"].values()
        for values in condition.values()
    )


def test_every_individual_is_held_out_under_the_same_fixed_denominator() -> None:
    report = json.loads(REPORT.read_text())
    expected = {"Tm3": 12, "Mi1": 24, "Mi4": 19, "C3": 16}
    for condition in report["conditions"].values():
        for source, count in expected.items():
            result = condition["sources"][source]
            assert result["individual_count"] == count
            assert result["validation_coverage_count"] == count
            assert result["response_sample_count"] == 1501
            assert result["gates"]["every_individual_validated"] is True
            assert result["gates"]["every_fold_training_validation_disjoint"] is True


def test_seven_failures_keep_every_downstream_gate_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["all_source_condition_individual_split_gates_passed"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"][
        "T5_LPLC_vehicle_MB_OOD_final_and_deployment_remain_frozen"
    ] is True
