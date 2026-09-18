import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-crossfit-retinal-symmetry-audit.json"


def test_t4_retinal_symmetry_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["retinal_geometries"] == {
        "reference": "nested_t4_axis_v1",
        "control": "balanced_exact_mirror_v1",
    }
    assert report["retinal_receptor_counts"]["reference"] == {
        "left": 1107,
        "right": 2237,
    }
    assert report["retinal_receptor_counts"]["control"] == {
        "left": 957,
        "right": 957,
    }
    assert report["control_exact_mirror_drive"] is True


def test_t4_retinal_symmetry_audit_does_not_authorize_a_candidate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["fixed_T4_population_denominator"] == 6861
    assert report["valid_target_count"] == 6749
    assert report["maximum_reference_ordered_direction_pass_count"] == 2
    assert report["maximum_balanced_ordered_direction_pass_count"] == 0
    assert all(
        not values
        for values in report[
            "balanced_ordered_bilateral_direction_subtypes_by_reduction"
        ].values()
    )
    assert report["retinal_sampling_imbalance_explains_direction_failure"] is False
    assert report["candidate_selected"] is False
    assert report["authorize_new_functional_candidate"] is False
    assert report["advance_to_three_tuning_conditions"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["boundary"][
        "balanced_retina_is_engineering_ablation_not_biological_reconstruction"
    ]
