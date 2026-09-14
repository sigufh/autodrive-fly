import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_emotion.driving.v7_temporal_audit import (
    audit_edge_window,
    build_step_stimuli,
    independent_pixel_filter,
    summarize_step_response,
)


def test_step_stimuli_share_prehistory_and_have_no_motion_label() -> None:
    first = build_step_stimuli(48, 24)
    second = build_step_stimuli(48, 24)
    assert [item.sha256 for item in first] == [item.sha256 for item in second]
    assert all(item.direction == "none" for item in first)
    assert all(np.array_equal(item.frames[:16], first[0].frames[:16]) for item in first)
    assert np.all(first[2].frames == 0.5)
    assert np.allclose(first[0].frames + first[1].frames, 1.0)


def test_single_pixel_filter_cannot_mix_spatial_channels() -> None:
    frames = np.zeros((4, 2, 2))
    frames[1:, 0, 0] = 1.0
    trace = independent_pixel_filter(frames, leak=0.5, substeps=1)
    np.testing.assert_array_equal(trace[:, 0, 0], [0.0, 0.5, 0.75, 0.875])
    assert np.all(trace[:, 1, :] == 0.0)
    assert np.all(trace[:, :, 1] == 0.0)


def test_non_directional_filter_exposes_global_window_position_confound() -> None:
    audit = audit_edge_window(48, 24, 16)
    for pair in audit["pairs"].values():
        assert pair["global_window_median_absolute_contrast"] > 0.10
        assert pair["lower_coordinate_half_median_contrast"] > 0.10
        assert pair["upper_coordinate_half_median_contrast"] < -0.10
        assert abs(pair["global_window_median_contrast"]) < 0.10
        assert pair["aligned_local_window_maximum_absolute_contrast"] == 0.0
        assert pair["excluded_pixels"] > 0


def test_frame_difference_control_distinguishes_transient_from_sustained_bias() -> None:
    audit = audit_edge_window(48, 24, 16, encoding="signed_frame_difference")
    for pair in audit["pairs"].values():
        assert pair["global_window_median_absolute_contrast"] < 0.01
        assert pair["global_window_maximum_absolute_contrast"] > 0.01
        assert pair["aligned_local_window_maximum_absolute_contrast"] == 0.0


def test_signed_peak_summary_reports_silence_without_correct_sign_credit() -> None:
    delta = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, -1.0, 0.0], [1.0, -3.0, 0.0]])
    summary = summarize_step_response(delta, onset=2, expected_sign=-1)
    assert summary["active_fraction"] == pytest.approx(2 / 3)
    assert summary["fraction_all_cells_expected_peak_sign"] == pytest.approx(1 / 3)
    assert summary["median_peak_latency_frames_active"] == 0.5
    assert summary["fraction_active_peaks_at_window_end"] == 0.5
    assert summary["maximum_pre_step_difference"] == 0.0
    silent = summarize_step_response(delta * 0, onset=2, expected_sign=1)
    assert silent["fraction_all_cells_expected_peak_sign"] == 0.0
    assert silent["median_peak_latency_frames_active"] is None


def test_saved_temporal_audit_binds_dependencies_and_sham_prehistory() -> None:
    root = Path(__file__).parents[1]
    report = json.loads((root / "artifacts/v7-temporal-input-audit.json").read_text())
    protocol = report["protocol"]
    assert protocol["advance_allowed"] is False
    assert protocol["parameter_fitting"] is False
    assert report["advance_to_central_complex"] is False
    for relative, expected_hash in protocol["dependencies_sha256"].items():
        with (root / relative).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected_hash
    for backend in report["step_responses"].values():
        assert len(backend["populations"]) == 24
        for responses in backend["populations"].values():
            for response in responses.values():
                assert response["maximum_pre_step_difference"] == 0.0
                assert len(response["population_mean_trace"]) == 48
