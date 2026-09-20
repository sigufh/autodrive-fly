import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-source-transfer-synthesis-audit.json"


def test_synthesis_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["functional_stimulus_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_incremental_shape_evidence_is_kept_separate_from_transfer_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["incremental_evidence"] == {
        "voltage_derived_kernel_source_count": 4,
        "exact_type_average_mapping_source_count": 4,
        "Figure5_training_fit_count": 168,
        "moving_bar_condition_count": 24,
    }
    gates = report["gates"]
    assert gates["four_Tm_voltage_derived_kernel_shapes_available"] is True
    assert gates["four_Tm_exact_type_average_mapping_complete"] is True
    assert gates["saline_relative_Tm9_delay_candidate_available"] is True
    assert gates["relative_low_frequency_Tm9_shape_evidence_available"] is True
    assert gates["cross_stimulus_moving_bar_shape_candidate_available"] is True
    assert gates["official_T5_target_model_parameters_available"] is True
    assert gates["official_target_model_to_T5_source_mapping_available"] is False
    assert gates["external_Figure6_axolotl_tmodel_source_available"] is False
    assert gates["related_axolotl_project_identified"] is True
    assert gates["related_axolotl_repository_anonymously_readable"] is False
    assert gates["measured_kernel_temporal_identifiability_passed"] is False
    assert gates["measured_kernel_direction_scoring_performed"] is False
    assert gates["absolute_source_gain_available"] is False
    assert gates["source_to_v7_state_mapping_available"] is False
    assert gates["state_invariant_source_timing_available"] is False
    assert gates["independent_moving_bar_validation_available"] is False
    assert gates["MaleCNS_connectome_weight_transfer_available"] is False
    assert gates["CT1_allowed_dynamics_and_mapping_complete"] is False
    assert gates["original_physical_transfer_gate_passed"] is False


def test_synthesis_preserves_original_transfer_stop() -> None:
    report = json.loads(REPORT.read_text())
    original = report["original_physical_transfer_contract"]
    assert original["ready"] is False
    assert set(original["missing_fields"]) == {
        "v7_camera_angular_calibration",
        "Tm1_Tm2_Tm4_Tm9_membrane_like_kernels",
        "CT1_membrane_like_kernel",
        "stable_source_or_target_cell_to_MaleCNS_mapping",
        "independent_dynamic_validation_cohort",
    }
    assert report["T5_source_transfer_ready"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
