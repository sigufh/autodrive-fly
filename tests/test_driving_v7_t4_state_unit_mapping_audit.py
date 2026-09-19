import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-state-unit-mapping-audit.json"


def test_T4_state_unit_mapping_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_author_minmax_formula_is_reproduced_but_not_a_v7_state_mapping() -> None:
    report = json.loads(REPORT.read_text())
    author = report["author_mapping"]
    assert author["formula_reproduced_on_full_cohort"] is True
    assert author["uses_both_ON_and_OFF_conditions_to_define_constants"] is True
    assert author["held_out_clipping_rule_declared"] is False
    state = report["v7_state_semantics"]
    assert state["activation_function"] == "signed_tanh"
    assert state["reported_unit"] == "simulated_activation_not_millivolts"
    assert state["author_zero_to_one_voltage_semantics_match_v7_state"] is False


def test_training_only_constants_extrapolate_outside_author_interval() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "Mi1": (16_584, 128_000),
        "Tm3": (11_186, 64_000),
        "Mi4": (32_149, 112_000),
        "C3": (18_094, 96_000),
    }
    for source, (outside, total) in expected.items():
        item = report["source_results"][source]
        assert item["held_out_samples_outside_author_interval"] == outside
        assert item["held_out_sample_count"] == total
        assert item["held_out_outside_interval_fraction"] == outside / total
        assert item["all_held_out_samples_in_author_interval"] is False
        assert item["distinct_training_constant_pair_count"] == item["held_out_fold_count"]
        assert item["training_constants_identical_across_folds"] is False
    assert report["gates"]["author_formula_reproduced"] is True
    assert report["gates"]["every_held_out_sample_in_author_interval"] is False
    assert report["gates"]["single_training_fitted_mapping_per_fold"] is True
    assert report["gates"]["deployment_training_cohort_declared"] is False


def test_state_mapping_and_all_downstream_authorizations_remain_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["millivolts_to_v7_normalized_state_mapping_available"] is False
    assert report["authorize_source_dynamics_transfer"] is False
    assert report["authorize_runtime_integration"] is False
    assert report["boundary"]["no_arbitrary_extrapolation_or_clipping_added"] is True
    assert report["boundary"]["no_target_activity_injection"] is True
