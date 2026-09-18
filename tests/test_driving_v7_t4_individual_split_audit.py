import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-individual-split-audit.json"


def test_individual_split_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert (
        report["boundary"]["post_hoc_diagnostic_after_full_population_metrics_were_observed"]
        is True
    )
    assert report["boundary"]["not_external_final"] is True


def test_every_individual_is_validated_without_train_overlap() -> None:
    report = json.loads(REPORT.read_text())
    expected = {"Tm3": 12, "Mi1": 24, "Mi4": 19, "C3": 16}
    for condition in report["conditions"].values():
        for source, count in expected.items():
            result = condition["sources"][source]
            assert result["individual_count"] == count
            assert result["validation_coverage_count"] == count
            assert result["gates"]["every_individual_validated"] is True
            assert result["gates"]["every_fold_training_validation_disjoint"] is True
            for fold in result["folds"]:
                assert set(fold["training_individual_ids"]).isdisjoint(
                    fold["validation_individual_ids"]
                )
                assert len(fold["validation_individual_ids"]) == 3


def test_only_Tm3_ON_passes_fixed_individual_split_gates() -> None:
    report = json.loads(REPORT.read_text())
    passing = [
        f"{condition}:{source}"
        for condition, details in report["conditions"].items()
        for source, result in details["sources"].items()
        if result["passed"]
    ]
    assert passing == ["on:Tm3"]
    assert report["conditions"]["on"]["sources"]["Mi1"]["minimum_individual_correlation"] < 0.0
    assert report["conditions"]["off"]["sources"]["Mi4"]["minimum_individual_correlation"] < 0.0


def test_split_diagnostic_does_not_authorize_downstream_work() -> None:
    report = json.loads(REPORT.read_text())
    assert report["all_source_condition_individual_split_gates_passed"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
