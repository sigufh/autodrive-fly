import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-tanaka-mi4-calcium-audit.json"


def test_tanaka_mi4_audit_is_hash_and_revision_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["model_boundary"]["repository_revision"] == (
        "a0f50210404229ae3a1465cf65a8b6bc31a0b0fe"
    )


def test_tanaka_payload_preserves_individual_fly_roi_trial_and_time_axes() -> None:
    report = json.loads(REPORT.read_text())
    inventory = report["payload_inventory"]
    assert inventory["fly_count"] == 10
    assert inventory["fly_ids"] == [
        9176,
        9178,
        9179,
        9180,
        9186,
        9187,
        9188,
        9189,
        9191,
        9192,
    ]
    assert inventory["selected_roi_counts_by_fly"] == [9, 12, 23, 30, 7, 16, 19, 41, 28, 16]
    assert inventory["selected_roi_count"] == 201
    assert inventory["selected_epoch_labels"] == ["2_S", "2_F"]
    assert inventory["time_sample_count"] == 93
    assert inventory["repetition_counts"] == [10]
    assert inventory["all_selected_response_values_finite"] is True
    assert inventory["stable_within_dataset_fly_IDs_available"] is True
    assert inventory["individual_fly_axis_available"] is True
    assert inventory["trial_and_ROI_axes_available"] is True


def test_tanaka_is_independent_mi4_calcium_not_transferable_voltage() -> None:
    report = json.loads(REPORT.read_text())
    assert report["paper"]["doi"] == "10.1016/j.cub.2023.10.011"
    assert report["measurement"] == {
        "source_type": "Mi4",
        "compartment": "medulla_layer_M10_axons",
        "modality": "two_photon_jGCaMP7b_calcium",
        "response_unit": "deltaF_over_F",
        "acquisition_rate_hz": 8.46,
        "stimulus_conditions": [
            "stationary_checkerboard",
            "flickering_checkerboard_15_Hz",
        ],
        "stimulus_duration_seconds": 5.0,
        "baseline_window_seconds": [-0.5, 0.0],
        "right_optic_lobe_only": True,
    }
    assert report["independent_Mi4_numerical_calcium_dynamics_verified"] is True
    assert report["experimental_membrane_voltage"] is False
    assert report["C3_direct_recording_found"] is False
    assert report["recording_to_MaleCNS_body_crosswalk_found"] is False
    assert report["authorize_Mi4_source_dynamics_transfer"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False


def test_tanaka_selection_and_model_provenance_remain_explicit() -> None:
    report = json.loads(REPORT.read_text())
    assert report["ROI_selection"] == {
        "selected_by_flash_probe_response_consistency": True,
        "response_consistency_threshold": 0.4,
        "unselected_payload_retained": True,
    }
    assert report["model_boundary"]["temporal_filter_tau_seconds"] == 0.3
    assert report["model_boundary"]["tau_source"] == "Arenz_2017_not_Figure_7_fit"
    assert report["model_boundary"]["Figure_7_payload_used_to_fit_tau"] is False
    assert report["Dryad"]["full_archive_downloaded"] is False
    assert report["Dryad"]["Figure_7_member_range_extracted"] is True
    assert report["Dryad"]["Figure_7_extracted_payload_crc32_verified"] is True
