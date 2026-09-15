import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_emotion.driving.v7_phase_motion import phase_average_contrast
from fly_emotion.driving.v7_pixel_sampling import (
    eye_local_coordinates,
    fractional_phase_diagnostic,
    periodic_pixel_filter,
    sample_eye_centres,
    sampling_controls,
)

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("width,height", [(48, 24), (50, 25), (4, 2)])
def test_eye_centres_keep_endpoints_and_never_sample_the_other_eye(width, height) -> None:
    local_x = np.tile([0.0, 0.5, 1.0], 2)
    local_y = np.tile([0.0, 0.5, 1.0], 2)
    side = np.repeat([-1, 1], 3)
    frame = np.zeros((height, width), dtype=np.float32)
    frame[:, width // 2 :] = 1.0
    np.testing.assert_array_equal(
        sample_eye_centres(frame, local_x, local_y, side), [0, 0, 0, 1, 1, 1]
    )
    frame = np.arange(height * width, dtype=np.float32).reshape(height, width)
    values = sample_eye_centres(frame, local_x, local_y, side)
    expected = (
        local_y * (height - 1) * width + local_x * (width // 2 - 1) + (side > 0) * (width // 2)
    )
    np.testing.assert_allclose(values, expected, atol=1e-6)


@pytest.mark.parametrize("width,height", [(48, 24), (50, 25), (4, 2)])
def test_eye_centres_reflect_arbitrary_images_including_half_pixel_ties(width, height) -> None:
    rng = np.random.default_rng(20260915)
    coordinates = np.concatenate(
        ([0, 0.5, 1], (np.arange(width // 2 - 1) + 0.5) / (width // 2 - 1), rng.random(100))
    )
    x, y = np.tile(coordinates, 2), np.tile(coordinates[::-1], 2)
    side = np.repeat([-1, 1], len(coordinates))
    frame = rng.random((height, width), dtype=np.float32)
    half = width // 2
    for horizontal, vertical in ((True, False), (False, True), (True, True)):
        left, right = frame[:, :half], frame[:, half:]
        if horizontal:
            left, right = left[:, ::-1], right[:, ::-1]
        reflected = np.concatenate((left, right), axis=1)
        if vertical:
            reflected = reflected[::-1]
        np.testing.assert_allclose(
            sample_eye_centres(reflected, x, y, side),
            sample_eye_centres(frame, 1 - x if horizontal else x, 1 - y if vertical else y, side),
            atol=1e-7,
        )


@pytest.mark.parametrize("axis,sign", [("x", 1), ("x", -1), ("y", 1), ("y", -1)])
def test_sampled_edge_arrival_follows_distal_centre_proximal_order(axis, sign) -> None:
    positions = np.array([0.15, 0.5, 0.85])
    if sign < 0:
        positions = 1.0 - positions
    x = np.tile(positions if axis == "x" else np.full(3, 0.5), 2)
    y = np.tile(positions if axis == "y" else np.full(3, 0.5), 2)
    side = np.repeat([-1, 1], 3)
    yy, xx = np.mgrid[:24, :48]
    coordinate = xx % 24 if axis == "x" else yy
    if sign < 0:
        coordinate = 23 - coordinate
    frames = [(coordinate <= t).astype(np.float32) for t in range(-1, 24)]
    response = np.stack([sample_eye_centres(frame, x, y, side) for frame in frames])
    arrival = np.argmax(response > 0.5, axis=0).reshape(2, 3)
    assert np.all(np.diff(arrival, axis=1) > 0)
    np.testing.assert_array_equal(arrival[0], arrival[1])


def test_sampler_rejects_unsplittable_images() -> None:
    with pytest.raises(ValueError, match="even width"):
        sample_eye_centres(np.zeros((24, 47)), np.array([0.5]), np.array([0.5]), np.array([-1]))


def test_fractional_pixel_control_catches_bias_missed_by_integer_and_midpoint_grid() -> None:
    side = np.repeat([-1, 1], 3)
    local_x = np.tile([0.25, 0.5, 0.75], 2) / 23
    u = (local_x + (side > 0)) * 0.5
    report = sampling_controls(u, np.zeros(6), side)
    assert report["controls_pass"] is False
    assert report["within_eye_xy_mirror_max_error"] <= 1e-6
    assert (
        report["sampled_independent_pixel_max_absolute_contrast"]["linear_luminance"]["right_left"]
        > 0.01
    )


def test_analytic_periodic_filter_closes_cycle_and_preserves_fractional_scoring_bias() -> None:
    diagnostic = fractional_phase_diagnostic()
    for backend, result in diagnostic["results"].items():
        assert result["maximum_absolute_positive_mean_contrast"] > 0.02
        assert result["maximum_absolute_mean_square_contrast"] < 1e-6
        assert abs(result["periodic_positive_mean_contrast"][8]) > 0.01
        assert abs(result["periodic_positive_mean_contrast"][16]) < 1e-6
        for direction, samples in result["sampled_periods"].items():
            samples = np.asarray(samples)
            for phase, sampled in enumerate(samples):
                drive = (
                    sampled - 0.5
                    if backend == "linear_luminance"
                    else np.clip((sampled - np.roll(sampled, 1, axis=0)) / 0.5, -1, 1)
                )
                trace = periodic_pixel_filter(drive)
                a = (1 - 0.24) ** 4
                np.testing.assert_allclose(trace[0], a * trace[-1] + (1 - a) * drive[0], atol=1e-14)
                np.testing.assert_allclose(
                    trace, result["periodic_response_periods"][direction][phase]
                )


def test_sampling_controls_reject_direction_from_independent_receptors() -> None:
    side = np.repeat([-1, 1], 3)
    u = np.array([0, 0.25, 0.5, 0.5, 0.75, 1], dtype=np.float32)
    v = np.array([0, 0.5, 1, 0, 0.5, 1], dtype=np.float32)
    np.testing.assert_array_equal(eye_local_coordinates(u, side), [0, 0.5, 1, 0, 0.5, 1])
    assert sampling_controls(u, v, side)["controls_pass"] is True


def test_saved_sampling_failure_retains_all_receptors_and_no_neural_results() -> None:
    report = json.loads((ROOT / "artifacts/v7-pixel-sampling.json").read_text())
    reference = json.loads((ROOT / "artifacts/v7-geometry-sign.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / path).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == digest
    assert report["advance_to_central_complex"] is False
    assert report["sampling_controls"]["controls_pass"] is False
    assert report["synthetic_controls"]["controls_pass"] is True
    assert report["neural_evaluation"]["performed"] is False
    assert report["neural_evaluation"]["results"] == {}
    assert report["geometry"]["cross_eye_support_cells"] == 0
    assert report["geometry"]["reference_cross_eye_sample_cells"] == 1
    assert report["geometry"]["receptor_body_ids"] == reference["geometry"]["receptor_body_ids"]
    assert len(set(report["geometry"]["receptor_body_ids"])) == 3344
    assert report["geometry"]["uv"] == reference["geometry"]["reversed_uv"]
    assert len(report["stimulus_sha256"]) == 32
    for key, digest in report["stimulus_sha256"].items():
        assert digest == reference["stimulus_sha256"][key]
    controls = report["sampling_controls"]
    assert len(controls["sampled_luminance_sha256"]) == 32
    for backend, pairs in controls["per_receptor_negative_control"].items():
        for key, values in pairs.items():
            p, o = (
                np.asarray(values[f"{direction}_phase_mean_by_cycle"])
                for direction in ("positive", "negative")
            )
            assert p.shape == o.shape == (2, 3344)
            contrast = (p - o) / (p + o + 1e-12)
            np.testing.assert_allclose(contrast, values["contrast_by_cycle"], atol=1e-14)
            np.testing.assert_allclose(
                np.max(np.abs(contrast)),
                controls["sampled_independent_pixel_max_absolute_contrast"][backend][key],
            )
    for result in report["fractional_phase_diagnostic"]["results"].values():
        p = np.asarray(result["periodic_response_periods"]["right"])
        o = np.asarray(result["periodic_response_periods"]["left"])
        np.testing.assert_allclose(
            phase_average_contrast(np.maximum(p, 0).mean(axis=1), np.maximum(o, 0).mean(axis=1)),
            result["periodic_positive_mean_contrast"],
        )
        np.testing.assert_allclose(
            phase_average_contrast((p**2).mean(axis=1), (o**2).mean(axis=1)),
            result["periodic_mean_square_contrast"],
        )
