import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_neural_channels import (
    CONFIG,
    TUNING_ARTIFACT,
    StructuredNeuralFeatures,
)

ROOT = Path(__file__).parents[1]
CALIBRATION = ROOT / "artifacts/v7-neural-channels-calibration.json"
CONTROLS = ROOT / "artifacts/v7-neural-channel-controls.json"
TOPOLOGY = ROOT / "artifacts/v7-neural-topology-controls.json"


def test_structured_neural_tuning_is_hash_bound_and_closes_nine_obstacles() -> None:
    report = json.loads((ROOT / TUNING_ARTIFACT).read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["visual_input_only_via_R1_R6"] is True
    assert report["protocol"]["teacher_values_injected_into_neural_state"] is False
    assert report["model"]["alpha"] == 1.0
    assert report["model"]["fit_r2"] == [
        0.9108378450037011,
        0.9780301658823125,
        0.9438809366185839,
    ]
    assert len(report["model"]["group_names"]) == 18
    assert report["model"]["feature_dimension_per_parity"] == 450
    assert all(item["success"] and item["obstacles_passed"] == 9 for item in report["episodes"])
    assert report["tuning_passed"] is True
    assert report["final_evaluated"] is False


def test_structured_neural_calibration_passes_once_without_release() -> None:
    report = json.loads(CALIBRATION.read_text())
    tuning = json.loads((ROOT / TUNING_ARTIFACT).read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["seeds"] == [8300, 8301]
    assert report["protocol"]["model_sha256"] == tuning["model_sha256"]
    assert report["protocol"]["model_changed_after_tuning"] is False
    assert all(item["success"] and item["obstacles_passed"] == 9 for item in report["episodes"])
    assert all(report["gates"].values())
    assert report["calibration_passed"] is True
    assert report["final_evaluated"] is False
    assert report["advance_to_navigation_release"] is False


def test_neural_source_controls_identify_required_and_redundant_under_screen() -> None:
    report = json.loads(CONTROLS.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    arms = report["arms"]
    assert arms["full"]["success_count"] == 6
    assert arms["no_T4_T5_spatial"]["success_count"] == 0
    assert arms["no_LPLC1"]["success_count"] == 6
    assert arms["no_LPLC2"]["success_count"] == 4
    assert arms["no_LC4"]["success_count"] == 6
    assert arms["no_all_LPLC_LC"]["success_count"] == 2
    assert arms["no_heading_feedback"]["success_count"] == 0
    assert report["interpretation"]["no_LPLC1"] == "not_necessary_under_this_tuning_screen"
    assert report["interpretation"]["no_LC4"] == "not_necessary_under_this_tuning_screen"
    assert report["final_evaluated"] is False
    assert report["advance_to_navigation_release"] is False


def test_neural_channel_contract_does_not_claim_biological_or_release_gates() -> None:
    import yaml

    config = yaml.safe_load((ROOT / CONFIG).read_text())
    boundary = config["boundary"]
    assert boundary["measured_cell_physiology_gate_passed"] is False
    assert boundary["T5_PD_ND_label_resolved"] is False
    assert boundary["LPLC_typed_mechanism_gate_passed"] is False
    assert boundary["topology_advantage_tested"] is False
    assert boundary["external_final_evaluated"] is False
    assert boundary["navigation_release_allowed"] is False


def test_retina_coordinate_control_changes_the_frozen_input_assignment() -> None:
    import yaml

    config = yaml.safe_load((ROOT / CONFIG).read_text())
    real = StructuredNeuralFeatures(ROOT, config, topology_control="real_malecns")
    shuffled = StructuredNeuralFeatures(
        ROOT, config, topology_control="shuffled_retina_coordinates", control_seed=20260915
    )
    assert np.array_equal(real.probe.retinal_permutation, np.arange(len(real.retina.x)))
    assert not np.array_equal(shuffled.probe.retinal_permutation, np.arange(len(shuffled.retina.x)))
    image = np.arange(24 * 48, dtype=np.float32).reshape(24, 48) / (24 * 48)
    real_values = (real.probe._sample_retina(image) * real.retinal_gain)[
        real.probe.retinal_permutation
    ]
    shuffled_values = (shuffled.probe._sample_retina(image) * shuffled.retinal_gain)[
        shuffled.probe.retinal_permutation
    ]
    assert not np.array_equal(real_values, shuffled_values)


def test_frozen_neural_readout_depends_on_real_topology_and_transmitter_signs() -> None:
    report = json.loads(TOPOLOGY.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    controls = report["controls"]
    assert controls["real_malecns"]["success_count"] == 6
    assert controls["real_malecns"]["total_obstacles_passed"] == 54
    assert controls["shuffled_retina_coordinates"]["success_count"] == 0
    assert controls["source_preserving_target_shuffle"]["success_count"] == 0
    assert controls["shuffled_transmitter_signs"]["success_count"] == 0
    assert all(report["gates"].values())
    assert report["real_topology_advantage_passed"] is True
    assert report["final_evaluated"] is False
    assert report["advance_to_navigation_release"] is False
