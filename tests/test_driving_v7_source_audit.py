import hashlib
from pathlib import Path

from fly_emotion.driving.v7_source_audit import evaluate_v7_t4_source_audit

ROOT = Path(__file__).parents[1]


def test_v7_t4_source_audit_is_anatomy_only_and_non_advancing() -> None:
    report = evaluate_v7_t4_source_audit(ROOT)
    protocol = report["protocol"]
    assert protocol["exploratory"] is True
    assert protocol["advance_allowed"] is False
    assert protocol["driving_data_used"] is False
    assert protocol["visual_stimulus_responses_used"] is False
    assert "retrospective diagnostics" in protocol["selection_bias"]
    assert report["advance_to_columnar_branched_correlator_v2"] is False
    assert report["advance_to_central_complex"] is False


def test_v7_t4_source_audit_binds_config_and_implementation() -> None:
    report = evaluate_v7_t4_source_audit(ROOT)
    assert (
        report["protocol"]["config_sha256"]
        == hashlib.sha256(
            (ROOT / "configs/driving-v7-t4-source-audit.yaml").read_bytes()
        ).hexdigest()
    )
    assert (
        report["protocol"]["implementation_sha256"]
        == hashlib.sha256(
            (ROOT / "src/fly_emotion/driving/v7_source_audit.py").read_bytes()
        ).hexdigest()
    )


def test_v7_t4_source_audit_preserves_branch_and_transmitter_evidence() -> None:
    report = evaluate_v7_t4_source_audit(ROOT)
    metadata = report["source_metadata"]
    assert metadata["Mi1"]["optic_hex_coordinate_fraction"] > 0.99
    assert metadata["Mi4"]["optic_hex_coordinate_fraction"] > 0.99
    assert metadata["Mi9"]["optic_hex_coordinate_fraction"] > 0.99
    assert metadata["C3"]["optic_hex_coordinate_fraction"] > 0.99
    assert metadata["Tm3"]["optic_hex_coordinate_cells"] == 0
    assert metadata["CT1"]["optic_hex_coordinate_cells"] == 0
    assert metadata["Mi1"]["consensus_neurotransmitters"] == {"acetylcholine": 1773}
    assert metadata["Mi4"]["consensus_neurotransmitters"] == {"gaba": 1772}
    assert metadata["C3"]["consensus_neurotransmitters"] == {"gaba": 1779}
    assert metadata["Mi9"]["consensus_neurotransmitters"] == {"glutamate": 1775}


def test_v7_t4_source_audit_quantifies_aggregation_failure_without_claiming_validation() -> None:
    report = evaluate_v7_t4_source_audit(ROOT)
    screen = report["coordinate_branch_screen"]
    assert screen["candidate_count"] == 7
    assert screen["reference_all_coordinate_bearing_delayed_sources"]["offset_sources"] == [
        "Mi4",
        "Mi9",
        "C3",
    ]
    assert screen["accuracy_improvement_over_reference"] > 0.40
    assert screen["best_exploratory_candidate"]["five_fold_accuracy"] > 0.90
    assert report["diagnosis"]["branch_aggregation_degrades_T4_direction_geometry"] is True
    assert report["next_candidate_contract"]["preserve_branches_separately"] is True
    nested = report["nested_branch_validation"]
    assert nested["outer_folds"] == 5
    assert nested["inner_folds"] == 4
    assert nested["outer_cells"] > 6_700
    assert len(nested["selected_source_by_outer_fold"]) == 5
    assert set(nested["preregistered_gates"]) == {
        "outer_accuracy",
        "outer_angle",
        "selection_consensus",
    }
