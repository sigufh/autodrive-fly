import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / (
    "artifacts/v7-t5-increment-order-control-replication-preregistration.json"
)


def test_increment_order_replication_preregistration_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["frozen_discovery_commit"] == (
        "ef1a3a92dc06a6a23eecef58c1f6d0ab1627f0f7"
    )
    assert report["protocol"]["replication_outputs_observed"] is False


def test_replication_conditions_and_gates_are_frozen() -> None:
    report = json.loads(REPORT.read_text())
    assert report["discovery_condition_id"] == "S1-T01"
    assert report["replication_condition_ids"] == ["S1-T02", "S1-T03"]
    assert report["tested_brain_updates_per_frame"] == [1, 4]
    assert report["settle_frames"] == 10
    assert report["control"]["seed"] == 20260920
    assert report["control"]["preserve_initialization_increment"] is True
    assert report["input_validity"]["terminal_frame_absolute_tolerance"] == 2e-8
    assert report["input_validity"]["R1_R6_energy_ratio_absolute_tolerance"] == 1e-12
    assert report["inherited_output_thresholds"][
        "maximum_control_to_ordered_residual_energy_ratio"
    ] == 0.50
    assert report["inherited_output_thresholds"][
        "maximum_static_to_ordered_energy_ratio"
    ] == 0.50
    assert report["replication_gate"][
        "require_same_candidate_to_pass_both_output_thresholds_every_condition_and_update"
    ] is True


def test_preregistration_does_not_execute_or_advance() -> None:
    report = json.loads(REPORT.read_text())
    assert report["replication_protocol_frozen"] is True
    assert report["replication_evaluated"] is False
    assert report["authorize_direction_scoring"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"]["discovery_condition_excluded_from_replication_gate"] is True
    assert report["boundary"]["not_external_final_validation"] is True
