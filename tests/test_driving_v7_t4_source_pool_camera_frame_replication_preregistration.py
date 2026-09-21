import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / (
    "artifacts/v7-t4-source-pool-camera-frame-replication-preregistration.json"
)


def test_camera_frame_replication_preregistration_is_frozen_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["frozen_discovery_commit"] == (
        "4d30a317de081cdab0c4086f33c2a4066a273650"
    )
    assert report["protocol"]["replication_outputs_observed"] is False
    assert report["replication_protocol_frozen"] is True


def test_candidate_conditions_and_stop_gate_are_frozen() -> None:
    report = json.loads(REPORT.read_text())
    assert report["discovery_condition_id"] == "LPLC-T01"
    assert report["replication_condition_ids"] == ["LPLC-T02", "LPLC-T03"]
    assert report["expected_duration_frames"] == {
        "LPLC-T02": 40,
        "LPLC-T03": 48,
    }
    assert report["replication_variation"]["used_by_local_edge_generator"] == [
        "duration_frames"
    ]
    assert report["replication_variation"]["local_edge_noise_standard_deviation"] == 0.0
    assert report["candidate"] == {
        "readout": "pairwise_distal_vector",
        "target_populations": ["T4d_L", "T4d_R"],
        "source_groups": {"center": ["Mi1", "Tm3"], "delayed": ["Mi9"]},
        "source_coordinate_frame": "camera_image_xy",
        "temporal_reduction": "maximum_projected_value",
    }
    assert report["ordered_gate"]["fixed_whole_T4_population_denominator"] == 6861
    assert report["ordered_gate"]["thresholds"] == {
        "minimum_valid_denominator": 1e-6,
        "minimum_valid_cell_fraction": 0.8,
        "minimum_median_signed_contrast": 0.1,
        "minimum_positive_cell_fraction": 0.6,
        "maximum_energy_weighted_mirror_error": 0.2,
        "minimum_active_mirror_pair_fraction": 0.8,
    }
    assert report["controls_after_ordered_gate"]["modes"] == [
        "temporal_shuffle",
        "static_sham",
    ]
    assert report["replication_evaluated"] is False
    assert report["authorize_new_target_formula"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
