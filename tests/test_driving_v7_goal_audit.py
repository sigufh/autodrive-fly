import hashlib
import json
from pathlib import Path

from fly_emotion.driving.v7_goal_audit import evaluate_v7_goal_coverage

ROOT = Path(__file__).parents[1]


def test_goal_audit_maps_every_numbered_item_without_unlocking_later_stages() -> None:
    report = evaluate_v7_goal_coverage(ROOT)
    assert [item["item"] for item in report["checks"]] == list(range(9))
    assert report["summary"]["complete_items"] == [0]
    assert report["summary"]["boundary_only_items"] == [8]
    assert report["summary"]["component_pass_items"] == [2, 3]
    assert report["summary"]["failed_items"] == [1]
    assert report["summary"]["incomplete_items"] == [6]
    assert report["summary"]["partially_validated_items"] == [2, 3, 4]
    assert report["summary"]["not_authorized_items"] == [5, 7]
    assert report["summary"]["objective_complete"] is False
    assert report["summary"]["current_stage"] == "controlled_vision"


def test_saved_goal_audit_is_hash_bound_and_matches_recalculation() -> None:
    saved = json.loads((ROOT / "artifacts/v7-goal-audit.json").read_text())
    for path, digest in saved["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    current = evaluate_v7_goal_coverage(ROOT)
    assert current == saved
    visual = saved["checks"][1]
    assert visual["observations"]["visual_inputs_only_at_receptors"] is True
    assert visual["observations"]["controlled_response_gates_pass"] is False
    assert visual["observations"]["T5_biological_PD_code_assigned"] is None
    assert visual["observations"]["T5_external_direction_label_map_verified"] is False
    assert visual["observations"]["T5_model_scoring_allowed"] is False
    assert visual["observations"]["T5_published_fitted_parameter_vectors_available"] is False
    assert visual["observations"]["physical_timebase_identified"] is False
    assert visual["observations"]["strict_scoring_contract_frozen"] is True
    assert visual["observations"]["development_R1_R6_input_gates_pass"] is True
    assert visual["observations"]["development_neural_evaluation_allowed"] is True
    assert visual["observations"]["development_neural_screen_performed"] is True
    assert visual["observations"]["development_response_gates_pass"] is False
    assert visual["observations"]["development_passing_retinal_backends"] == []
    assert visual["observations"]["development_geometry_ab_performed"] is True
    assert visual["observations"]["reversed_geometry_development_gates_pass"] is False
    assert visual["observations"]["reversed_geometry_passing_retinal_backends"] == []
    assert visual["observations"]["T5_targets_with_every_fast_and_any_delayed_source"] == 6717
    assert visual["observations"]["T5_target_count_in_structure_audit"] == 6719
    assert visual["observations"]["looming_targets_with_any_T4_and_any_T5"] == 319
    assert visual["observations"]["looming_target_count_in_structure_audit"] == 445
    assert visual["observations"]["existing_artifacts_rescored_under_new_contract"] is False
    release = saved["checks"][7]
    assert release["observations"]["stage1_stimulus_splits_disjoint"] is True
    assert release["observations"]["reserved_final_evaluated"] is False
    assert release["observations"]["blinded_one_time_final_available"] is False
