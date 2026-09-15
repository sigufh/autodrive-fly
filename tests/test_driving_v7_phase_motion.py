import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_phase_motion import (
    CONDITIONS,
    CYCLES,
    PERIOD,
    PRE_FRAMES,
    motion_controls,
    phase_average_contrast,
    phase_grating,
)

ROOT = Path(__file__).parents[1]


def test_phase_motion_has_common_prehistory_complete_cycles_and_exact_mirrors() -> None:
    for condition in CONDITIONS:
        for phase in range(PERIOD):
            item = phase_grating(condition, phase)
            assert item.frames.shape == (PRE_FRAMES + CYCLES * PERIOD, 24, 48)
            assert np.all(item.frames[:PRE_FRAMES] == 0.5)
            np.testing.assert_array_equal(
                item.frames[PRE_FRAMES : PRE_FRAMES + PERIOD], item.frames[-PERIOD:]
            )
    for phase in range(PERIOD):
        opposite_phase = (-47 - phase) % PERIOD
        np.testing.assert_array_equal(
            phase_grating("right", phase).frames[:, :, ::-1],
            phase_grating("left", opposite_phase).frames,
        )
        vertical_phase = (-23 - phase) % PERIOD
        np.testing.assert_array_equal(
            phase_grating("down", phase).frames[:, ::-1, :],
            phase_grating("up", vertical_phase).frames,
        )
    for forward, backward in (("right", "left"), ("down", "up")):
        first = np.stack([phase_grating(forward, p).frames for p in range(PERIOD)])
        second = np.stack([phase_grating(backward, p).frames for p in range(PERIOD)])
        for y, x in ((0, 0), (7, 12), (23, 47)):
            assert sorted(map(bytes, first[:, :, y, x].copy())) == sorted(
                map(bytes, second[:, :, y, x].copy())
            )


def test_grating_direction_labels_match_pixel_translation_and_static_controls() -> None:
    for condition, axis, sign in (("right", 2, 1), ("left", 2, -1), ("down", 1, 1), ("up", 1, -1)):
        frames = phase_grating(condition, 0).frames[PRE_FRAMES:]
        np.testing.assert_array_equal(frames[1:], np.roll(frames[:-1], sign, axis=axis))
    for condition in ("static_x", "static_y"):
        frames = phase_grating(condition, 0).frames[PRE_FRAMES:]
        assert np.all(frames == frames[0])


def test_phase_controls_reject_independent_pixels_and_detect_spatial_correlator() -> None:
    controls = motion_controls()
    assert controls["controls_pass"] is True
    for values in controls["independent_pixel_maximum_absolute_contrast"].values():
        assert max(values.values()) < 1e-6
    assert min(controls["two_pixel_correlator_median_contrast"].values()) > 0.5


def test_phase_average_precedes_nonlinear_ratio_and_silent_cells_stay_zero() -> None:
    preferred = np.array([[3.0, 0], [0, 0]])
    opposite = np.array([[1.0, 0], [1.0, 0]])
    np.testing.assert_allclose(phase_average_contrast(preferred, opposite), [0.2, 0])


def test_saved_phase_motion_keeps_cells_and_reconstructs_cycle_contrasts() -> None:
    report = json.loads((ROOT / "artifacts/v7-phase-motion.json").read_text())
    assert report["advance_to_central_complex"] is False
    assert report["synthetic_controls"]["controls_pass"] is True
    for relative, expected in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / relative).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    assert len(report["stimulus_sha256"]) == 48
    assert len(report["retinal_drive_sha256"]) == 96
    for condition in CONDITIONS:
        for phase in range(PERIOD):
            assert (
                report["stimulus_sha256"][f"{condition}:{phase}"]
                == phase_grating(condition, phase).sha256
            )
    for result in report["results"].values():
        for model in result["models"].values():
            ids = [body for pop in model["populations"].values() for body in pop["body_ids"]]
            assert len(ids) == len(set(ids)) == model["target_count"] == 13580
            assert len(model["populations"]) == 16
            for pop in model["populations"].values():
                p = np.asarray(pop["per_cell_preferred_cycle_means"])
                o = np.asarray(pop["per_cell_opposite_cycle_means"])
                assert p.shape == o.shape == (2, pop["cells"])
                cycles = (p - o) / (p + o + 1e-12)
                np.testing.assert_allclose(
                    np.median(cycles, axis=1), pop["cycle_median_direction_contrasts"]
                )
                contrast = phase_average_contrast(p, o)
                np.testing.assert_allclose(np.median(contrast), pop["median_direction_contrast"])
                assert pop["direction_positive_fraction"] == float(np.mean(contrast > 0))
                np.testing.assert_allclose(
                    np.median(np.abs(cycles[1] - cycles[0])),
                    pop["median_absolute_cycle_contrast_change"],
                )
                assert pop["fraction_response_sum_above_1e_6"] == float(
                    np.mean(p.mean(axis=0) + o.mean(axis=0) > 1e-6)
                )
            for family in ("T4", "T5"):
                assert model[f"{family}_population_median"] == np.median(
                    [
                        pop["median_direction_contrast"]
                        for name, pop in model["populations"].items()
                        if name.startswith(family)
                    ]
                )
