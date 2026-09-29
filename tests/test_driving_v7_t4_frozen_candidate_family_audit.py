import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-frozen-candidate-family-audit.json"


def test_frozen_candidate_family_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["candidate_outputs_reused_without_recomputation"] is True
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_all_frozen_families_share_the_same_unilateral_pass_pattern() -> None:
    report = json.loads(REPORT.read_text())
    assert report["candidate_family_count"] == 4
    assert report["candidate_count"] == 32
    assert report["candidate_grid"] == {
        "temporal_reductions": ["positive_peak", "signed_mean"],
        "additive_gains": [0.125, 0.5, 2.0, 8.0],
    }
    assert report["direction_pass_set_frequency"] == {
        "T4a_R": 3,
        "T4a_R|T4c_R": 29,
    }
    assert report["direction_passing_population_frequency"] == {
        "T4a_L": 0,
        "T4a_R": 32,
        "T4b_L": 0,
        "T4b_R": 0,
        "T4c_L": 0,
        "T4c_R": 29,
        "T4d_L": 0,
        "T4d_R": 0,
    }
    assert report["maximum_direction_pass_count"] == 2
    assert report["all_candidates_pass_all_polarity_populations"] is True
    assert report["all_candidates_lack_bilateral_direction_subtypes"] is True
    assert report["mirror_gate_passing_candidate_count"] == 3


def test_structure_passes_but_exact_mirror_input_does_not_rescue_function() -> None:
    report = json.loads(REPORT.read_text())
    assert report["crossfit_anatomical_axis_gate_passed"] is True
    assert report["crossfit_axis_minimum_population_accuracy"] > 0.93
    control = report["exact_mirror_retinal_control"]
    assert control == {
        "exact_mirror_drive": True,
        "reference_maximum_ordered_direction_pass_count": 2,
        "balanced_maximum_ordered_direction_pass_count": 0,
        "retinal_sampling_imbalance_explains_direction_failure": False,
        "did_not_rescue_ordered_direction": True,
    }
    assert report["frozen_candidate_grid_failed"] is True
    assert report["authorize_post_hoc_additive_gain_expansion"] is False
    assert report["authorize_direction_or_subtype_label_change"] is False
    assert report["authorize_new_T4_functional_candidate"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
