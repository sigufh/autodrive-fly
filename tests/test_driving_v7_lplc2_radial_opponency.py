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
    temporal = report["temporal_identifiability"]
    assert temporal["post_failure_localization_only"] is True
    assert temporal["parameter_search"] is False
    assert temporal["passed"] is False
    assert temporal["population_consistency"]["L"]["coverage"]["joint_valid_fraction"] == 1.0
    assert temporal["population_consistency"]["R"]["coverage"]["joint_valid_fraction"] == 1.0
    assert temporal["population_consistency"]["L"]["coverage"]["all_condition_separated_count"] == 4
    assert temporal["population_consistency"]["R"]["coverage"]["all_condition_separated_count"] == 9
    for condition in temporal["per_condition"].values():
        for side in condition.values():
            assert side["median_normalized_forward_trace_error"] < 0.03
            assert (
                side["median_normalized_time_reversed_trace_error"]
                > side["median_normalized_forward_trace_error"]
            )
    localization = report["layer_localization"]
    assert localization["post_failure_localization_only"] is True
    assert localization["parameter_search"] is False
    assert localization["selected_receptors_only"] is True
    assert localization["passing_populations"] == []
    receptor = localization["population_consistency"]["mapped_R1-R6"]
    assert receptor["cell_count"] == 1914
    assert receptor["passing_condition_count"] == 0
    assert receptor["all_condition_separated_count"] == 168
    assert receptor["all_condition_separated_fraction"] < 0.10
    for population in ("L1", "L2", "L3", "L5", "T4a", "T4b", "T5a", "T5b"):
        assert localization["population_consistency"][population]["passed"] is False
    edge_audit = localization["selected_receptor_edge_audit"]
    assert edge_audit["mapped_receptor_count"] == 1914
    assert edge_audit["all_graph_R1_R6_count"] == 3377
    assert edge_audit["unselected_graph_R1_R6_count"] == 1463
    assert edge_audit["visual_graph_keeps_unselected_R1_R6_outputs"] is True
    assert edge_audit["renormalization_AB_parameter_search"] is False
    assert edge_audit["renormalization_AB_gate_restored"] is False
    full = report["input_projection_ab"]
    assert full["post_failure_localization_only"] is True
    assert full["parameter_fit"] is False
    assert full["runtime_modified"] is False
    assert full["calibration_evaluated"] is False
    assert full["full_mapped_receptor_count"] == 3344
    assert full["full_mapping_is_exact_mirror"] is False
    assert full["same_noise_realization_within_pair"] is True
    assert full["passing_populations"] == []
    assert full["population_consistency"]["mapped_R1-R6"]["all_condition_separated_count"] == 574
    for population in ("L1", "L2", "L3", "Mi1", "Tm1", "T4a", "T5a"):
        assert full["population_consistency"][population]["passed"] is False
    assert localization["interpretation_allowed"] is True
    noise = report["paired_noise_control"]
    assert noise["post_failure_correction"] is True
    assert noise["parameter_fit"] is False
    assert noise["runtime_modified"] is False
    assert noise["calibration_evaluated"] is False
    assert noise["primary_interpretation_source"] == "matched_noise"
    assert noise["unmatched_noise_layer_localization_confounded"] is True
    for control in noise["controls"].values():
        assert control["same_noise_realization_within_pair"] is True
        assert control["passing_populations"] == []
    matched = noise["controls"]["matched_noise"]["population_consistency"]
    assert matched["mapped_R1-R6"]["all_condition_separated_count"] == 168
    assert matched["mapped_R1-R6"]["passed"] is False
    assert matched["L1"]["all_condition_separated_count"] == 115
    assert matched["T4a"]["all_condition_separated_count"] == 48
    assert matched["T5a"]["all_condition_separated_count"] == 55
    radial = noise["controls"]["matched_noise"]["radial_population_consistency"]
    assert radial["R"]["outward_vs_translation"]["passing_condition_count"] == 3
    assert radial["R"]["outward_vs_translation"]["passed"] is False
    assert report["radial_opponency_mechanism_gates_passed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["advance_to_navigation_release"] is False
