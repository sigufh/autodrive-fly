import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-resolved.json"


def test_source_resolved_t4_is_tuning_only_and_nonadvancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_ids"] == ["S1-T01", "S1-T02", "S1-T03"]
    assert report["protocol"]["moving_edge_stimulus_count"] == 12
    assert report["protocol"]["variant_count"] == 10
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["labels_used_by_dynamics"] is False
    assert report["protocol"]["subtype_conditioned_dynamics"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False
    assert report["passing_variants"] == []
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["advance_to_navigation"] is False


def test_source_resolved_t4_keeps_branches_and_transmitter_signs_separate() -> None:
    source = json.loads(REPORT.read_text())["source_contract"]
    assert source["groups"] == {
        "center": ["Mi1", "Tm3"],
        "proximal": ["Mi4", "C3"],
        "wide_field": ["CT1"],
        "distal": ["Mi9"],
    }
    assert source["transmitter_signs_from_malecns"] is True
    assert source["single_union_normalization_preserves_relative_source_mass"] is True
    assert source["ct1_kept_separate"] is True
    assert source["geometry_ab_holds_receptor_ids_mass_and_dynamics_fixed"] is True
    assert source["branch_axis_used_for_diagnosis_only"] is True


def test_source_resolution_does_not_rescue_t4a_or_authorize_calibration() -> None:
    report = json.loads(REPORT.read_text())
    variants = report["variants"]
    for variant in variants.values():
        assert variant["strict_t4ab_tuning_passed"] is False
        assert any(
            not score["passed"]
            for condition in variant["per_condition_scores"].values()
            for head in condition.values()
            for score in head.values()
        )
    declared = variants["source_resolved_declared_delay"]
    assert declared["per_condition_scores"]["S1-T03"]["polarity"]["T4b_R"][
        "median_signed_contrast"
    ] < 0.10
    no_ct1 = variants["no_wide_field_ct1"]
    assert abs(
        no_ct1["per_condition_scores"]["S1-T01"]["direction"]["T4b_L"][
            "median_signed_contrast"
        ]
    ) < 0.10
    conductance = variants["published_conductance_reversed_branch_axis_native_timebase"]
    for condition in ("S1-T01", "S1-T02", "S1-T03"):
        for population in ("T4a_L", "T4a_R", "T4b_L", "T4b_R"):
            assert conductance["per_condition_scores"][condition]["polarity"][population][
                "passed"
            ] is True
    assert conductance["strict_t4ab_tuning_passed"] is False


def test_t4_spatial_pilot_is_informed_only_and_fails_to_expand() -> None:
    pilot = json.loads(REPORT.read_text())["spatial_reachability_pilot"]
    assert pilot["status"] == "pilot_informed_reachability_envelope_only"
    assert pilot["candidate_count"] == 32
    assert pilot["targetwise_label_based_candidate_selection"] is True
    assert pilot["may_authorize_candidate"] is False
    assert pilot["ct1_used_in_spatial_pool"] is False
    populations = pilot["population_results"]
    assert {name: item["target_count"] for name, item in populations.items()} == {
        "T4a_L": 16,
        "T4a_R": 16,
        "T4b_L": 16,
        "T4b_R": 16,
    }
    assert sum(
        item["all_six_comparisons_reachable_count"] for item in populations.values()
    ) == 1
    assert pilot["advance_to_full_population"] is False
    assert pilot["advance_to_calibration"] is False


def test_hybrid_conductance_ordering_has_signal_but_no_global_candidate() -> None:
    pilot = json.loads(REPORT.read_text())["hybrid_conductance_anatomical_order_pilot"]
    assert pilot["status"] == "pilot_informed_reachability_envelope_only"
    assert pilot["target_count"] == 64
    assert pilot["candidate_count"] == 24
    assert pilot["targetwise_label_based_candidate_selection"] is True
    assert pilot["targetwise_selection_may_authorize_candidate"] is False
    populations = pilot["population_results"]
    assert {
        name: item["all_six_comparisons_reachable_count"]
        for name, item in populations.items()
    } == {"T4a_L": 10, "T4a_R": 2, "T4b_L": 5, "T4b_R": 11}
    global_candidate = pilot["best_single_global_candidate"]
    assert global_candidate == {
        "lag_substeps": 3,
        "delayed_source": "proximal",
        "combination_mode": "positive_product",
        "passed_population_condition_head_gates": 17,
        "total_population_condition_head_gates": 24,
    }
    assert pilot["advance_to_full_population"] is False
    assert pilot["advance_to_calibration"] is False
