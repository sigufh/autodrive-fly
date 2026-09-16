import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-lplc2-radial-opponency.json"


def _report() -> dict:
    return json.loads(REPORT.read_text())


def test_radial_opponency_is_source_resolved_tuning_only_and_non_authorizing() -> None:
    report = _report()
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_ids"] == ["LPLC-T01", "LPLC-T02", "LPLC-T03"]
    assert protocol["stimulus_count"] == 12
    assert protocol["fixed_target_denominators"] == {"L": 94, "R": 91}
    assert protocol["direction_labels_used_by_mechanism"] is True
    assert protocol["may_authorize_strict_visual_gate"] is False
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["external_final_evaluated"] is False
    assert protocol["runtime_modified"] is False
    mechanism = report["mechanism"]
    assert mechanism["direct_edge_weights_preserved"] is True
    assert mechanism["ct1_or_other_sources_in_spatial_pool"] is False
    assert mechanism["unmapped_sources_dropped_from_denominator"] is False
    assert mechanism["biological_direction_gate_claimed"] is False


def test_radial_opponency_has_full_structural_coverage_but_fails_functional_gates() -> None:
    report = _report()
    sources = report["anatomy"]["source_positions"]["families"]
    assert sources["T4"]["mapped_source_count"] == sources["T4"]["source_count"]
    assert sources["T5"]["source_count"] - sources["T5"]["mapped_source_count"] == 1
    for side, expected in (("L", 94), ("R", 91)):
        anatomy = report["anatomy"]["populations"][side]
        assert anatomy["target_count"] == expected
        assert anatomy["targets_with_any_direct_t4_t5"] == expected
        assert anatomy["targets_with_mapped_direct_source"] == expected
        assert anatomy["targets_with_both_radial_pools"] == expected
    consistency = report["population_consistency"]
    assert consistency["R"]["outward_vs_translation"]["passing_condition_count"] == 3
    assert (
        consistency["R"]["outward_vs_translation"]["coverage"]["all_condition_success_fraction"]
        < 0.60
    )
    assert all(
        not result["passed"]
        for mechanisms in consistency.values()
        for result in mechanisms.values()
    )
    assert all(result["passed"] for result in report["mirror_summary"].values())
    assert report["radial_opponency_mechanism_gates_passed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["advance_to_navigation_release"] is False
