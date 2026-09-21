import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-author-row-weighted-full-support-sensitivity.json"


def test_row_weighted_sensitivity_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["stimulus_count_per_mode"] == 120
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_author_row_weighting_changes_only_tm1() -> None:
    contract = json.loads(REPORT.read_text())["aggregation_contract"]
    assert contract["author_default_baseline_none_preserved"] is True
    assert contract["changed_sources"] == ["Tm1"]
    tm1 = contract["kernel_comparison"]["Tm1"]
    assert tm1["payload_row_count"] == 8
    assert tm1["recording_id_count"] == 7
    assert tm1["kernels_exactly_equal"] is False
    assert np.isclose(tm1["correlation"], 0.9995598341098436)
    assert np.isclose(tm1["maximum_absolute_difference"], 0.001330646526801052)
    for source in ("Tm2", "Tm4", "Tm9"):
        assert contract["kernel_comparison"][source]["kernels_exactly_equal"] is True


def test_author_row_weighted_full_support_obeys_stop_gate() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "summed_filtered_source_centroid_projection": {
            "1": (2.251889505723192, 0.9998560630757318),
            "4": (2.215410098512955, 0.9994754390409343),
        },
        "fast_pool_vs_Tm9_centroid_difference": {
            "1": (1.4274448805129158, 0.7920155888613984),
            "4": (1.4994825851503417, 0.7573270858024239),
        },
        "temporal_difference_filtered_Tm_pair_reichardt": {
            "1": (0.9864673924746497, 0.5871518058928907),
            "4": (1.203525385385167, 0.658050086521343),
        },
    }
    assert set(report["candidate_results"]) == set(expected)
    for name, by_update in expected.items():
        result = report["candidate_results"][name]
        for update, ratios in by_update.items():
            scored = result["by_brain_updates_per_frame"][update]
            assert np.isclose(scored["shuffle_to_ordered_residual_energy_ratio"], ratios[0])
            assert np.isclose(scored["static_to_ordered_energy_ratio"], ratios[1])
            assert set(scored["gates"].values()) == {False}
            assert scored["passed"] is False
        assert result["passed_every_update_count"] is False
    assert report["all_candidates_failed_every_update_count"] is True
    assert report["cross_substep_temporal_identifiability_passed"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["direction_scoring_performed"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["authorize_physical_source_dynamics_transfer"] is False


def test_row_weighted_boundary_keeps_downstream_stages_frozen() -> None:
    report = json.loads(REPORT.read_text())
    boundary = report["boundary"]
    assert boundary["only_population_aggregation_weighting_changed"] is True
    assert boundary["author_default_baseline_none_preserved"] is True
    assert boundary["source_activity_driven_only_from_R1_R6_propagation"] is True
    assert boundary["subtype_and_direction_labels_not_read_by_dynamics"] is True
    assert boundary["target_activity_injection"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
