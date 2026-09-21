import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-measured-kernel-cascade-semantics-audit.json"


def test_cascade_semantics_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["functional_stimulus_evaluated"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_measured_kernel_is_applied_after_typed_recurrent_source_state() -> None:
    report = json.loads(REPORT.read_text())
    assert report["configured_path"] == {
        "retinal_backend": "signed_frame_difference",
        "dynamics_backend": "typed_visual_subgraph_v1",
        "baseline_frames": 2,
        "brain_updates_per_frame": 1,
        "source_types": ["Tm1", "Tm2", "Tm4", "Tm9"],
        "source_leaks": {"Tm1": 0.28, "Tm2": 0.55, "Tm4": 0.34, "Tm9": 0.16},
    }
    assert report["runtime_source_leaks"] == {
        "Tm1": [0.2800000011920929],
        "Tm2": [0.550000011920929],
        "Tm4": [0.3400000035762787],
        "Tm9": [0.1599999964237213],
    }
    gates = report["semantic_gates"]
    assert all(value for name, value in gates.items() if name != "source_state_mapping_available")
    assert gates["source_state_mapping_available"] is False
    assert report["typed_recurrent_source_state_then_measured_kernel_cascade_verified"] is True


def test_cascade_semantics_do_not_claim_source_dynamics_replacement_or_transfer() -> None:
    report = json.loads(REPORT.read_text())
    assert report["measured_kernel_replaces_existing_source_dynamics"] is False
    assert report["single_stage_biological_source_model_interpretation_authorized"] is False
    assert report["external_recording_to_v7_source_state_mapping_available"] is False
    assert report["authorize_physical_source_dynamics_transfer"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
