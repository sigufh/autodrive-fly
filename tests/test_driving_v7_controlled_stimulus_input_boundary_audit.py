import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-controlled-stimulus-input-boundary-audit.json"


def test_stimulus_input_boundary_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["neural_response_scoring_performed"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_all_requested_stimulus_families_and_splits_are_present() -> None:
    report = json.loads(REPORT.read_text())
    assert report["base_stimuli"]["stimulus_count"] == 20
    assert report["base_stimuli"]["missing_required_conditions"] == []
    assert set(report["independent_splits"]) == {
        "development",
        "validation",
        "ood",
        "final",
    }
    for result in report["independent_splits"].values():
        assert result["stimulus_count"] == 172
        assert result["matches_frozen_manifest"] is True
        assert result["exact_horizontal_mirror_pairs"] is True
    assert report["independent_splits"]["final"]["individual_stimuli_disclosed"] is False
    assert report["typed_mechanism_stimuli"]["condition_count"] == 3
    assert report["typed_mechanism_stimuli"]["stimulus_count"] == 33
    assert report["gates"]["development_sampled_R1_R6_input_gates_pass"] is True
    assert (
        report["typed_mechanism_stimuli"][
            "front_back_labels_are_model_image_trajectory_labels"
        ]
        is True
    )


def test_external_drive_is_restricted_to_r1_r6_without_advancing_functional_gates() -> None:
    report = json.loads(REPORT.read_text())
    boundary = report["input_boundary"]
    assert boundary["retinal_node_types"] == ["R1-R6"]
    assert boundary["target_direct_input_overlap"] == 0
    assert boundary["dynamic_nonretinal_drive_node_count"] == 0
    assert boundary["static_assignment_audit"][
        "all_drive_subscript_assignments_target_retina_indices"
    ] is True
    assert report["stimulus_and_input_boundary_complete"] is True
    assert report["controlled_response_gates_passed"] is False
    assert report["typed_LPLC_LC4_response_gates_passed"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
