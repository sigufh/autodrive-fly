import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-loo.json"


def test_T4_source_LOO_is_preregistered_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["preregistration_commit"] == (
        "d885110da55fed14bc4599dae81e777213b99363"
    )
    assert report["protocol"]["preregistration_sha256"] == (
        "ede38392ce650b4c21da6c35a09db10c2f9dac2b0c115ec7a36519c062efe622"
    )
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_T4_source_LOO_preserves_frozen_conditions_and_denominators() -> None:
    report = json.loads(REPORT.read_text())
    assert report["fixed_T4_population_denominator"] == 6861
    assert report["valid_target_count"] == 6749
    assert list(report["condition_results"]) == [
        "intact",
        "drop_Mi1",
        "drop_Tm3",
        "drop_Mi4",
        "drop_C3",
    ]
    assert report["intact_matches_frozen_source_sequence_evidence"] is True
    for condition, result in report["condition_results"].items():
        assert set(result["mode_results"]) == {
            "ordered",
            "temporal_shuffle",
            "static_sham",
        }
        if condition == "intact":
            assert sum(result["removed_matrix_nonzero_count"].values()) == 0
        else:
            assert sum(result["removed_matrix_nonzero_count"].values()) > 0


def test_T4_source_LOO_never_authorizes_source_removal_or_downstream_stage() -> None:
    report = json.loads(REPORT.read_text())
    assert report["condition_summary"] == {
        "intact": {
            "ordered_maximum_direction_pass_count": 2,
            "ordered_maximum_polarity_pass_count": 1,
            "control_maximum_direction_pass_count": 3,
            "ordered_minus_max_control_direction_pass_count": -1,
            "ordered_bilateral_direction_subtypes": [],
            "control_bilateral_direction_subtypes": ["a"],
        },
        "drop_Mi1": {
            "ordered_maximum_direction_pass_count": 3,
            "ordered_maximum_polarity_pass_count": 6,
            "control_maximum_direction_pass_count": 2,
            "ordered_minus_max_control_direction_pass_count": 1,
            "ordered_bilateral_direction_subtypes": ["b"],
            "control_bilateral_direction_subtypes": [],
        },
        "drop_Tm3": {
            "ordered_maximum_direction_pass_count": 3,
            "ordered_maximum_polarity_pass_count": 0,
            "control_maximum_direction_pass_count": 2,
            "ordered_minus_max_control_direction_pass_count": 1,
            "ordered_bilateral_direction_subtypes": [],
            "control_bilateral_direction_subtypes": ["a"],
        },
        "drop_Mi4": {
            "ordered_maximum_direction_pass_count": 5,
            "ordered_maximum_polarity_pass_count": 3,
            "control_maximum_direction_pass_count": 2,
            "ordered_minus_max_control_direction_pass_count": 3,
            "ordered_bilateral_direction_subtypes": ["a", "c"],
            "control_bilateral_direction_subtypes": ["a"],
        },
        "drop_C3": {
            "ordered_maximum_direction_pass_count": 1,
            "ordered_maximum_polarity_pass_count": 0,
            "control_maximum_direction_pass_count": 2,
            "ordered_minus_max_control_direction_pass_count": -1,
            "ordered_bilateral_direction_subtypes": [],
            "control_bilateral_direction_subtypes": [],
        },
    }
    assert report["ablation_evaluated"] is True
    assert report["descriptive_ablation_only"] is True
    assert report["authorize_source_removal"] is False
    assert (
        report[
            "any_source_removal_satisfies_all_eight_direction_and_polarity_populations"
        ]
        is False
    )
    assert (
        report[
            "any_source_removal_has_ordered_bilateral_subtype_absent_from_controls"
        ]
        is True
    )
    assert report["authorize_new_T4_functional_candidate"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
