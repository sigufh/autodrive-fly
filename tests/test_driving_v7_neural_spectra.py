import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_neural_spectra import (
    EyeCentreConductanceProbe,
    EyeCentreTypedProbe,
    _contrast,
    _paired_sampler_summary,
    response_spectrum,
)
from fly_emotion.driving.v7_phase_motion import PERIOD

ROOT = Path(__file__).parents[1]


def test_response_spectrum_retains_cycles_cells_and_parseval() -> None:
    phase, cycle, time, cell = np.mgrid[:8, :2, :8, :3]
    values = 0.2 * (cycle + 1) + (cell + 1) * np.cos(2 * np.pi * (time + phase) / PERIOD)
    result = response_spectrum(values)
    np.testing.assert_allclose(result["dc"], [[0.2] * 3, [0.4] * 3])
    expected_f1 = np.tile(0.5 * np.square(np.arange(1, 4)), (2, 1))
    np.testing.assert_allclose(result["f1_power"], expected_f1)
    np.testing.assert_allclose(
        result["total_power"],
        expected_f1 + np.square(np.asarray([[0.2], [0.4]])),
    )
    assert result["parseval_max_error"] < 1e-14


def test_signed_dc_contrast_keeps_hyperpolarizing_direction_sign() -> None:
    preferred = np.asarray([[-2.0, 2.0], [-4.0, 4.0]])
    opposite = np.asarray([[-1.0, 1.0], [-2.0, 2.0]])
    np.testing.assert_allclose(_contrast(preferred, opposite), [-1 / 3, 1 / 3])


def test_sampler_comparison_is_body_id_aligned_and_metric_separate() -> None:
    def report(offset: float) -> dict:
        metrics = {}
        for name in ("dc", "f1_power", "total_power"):
            preferred = np.asarray([[1.0, 2.0], [3.0, 4.0]]) + offset
            opposite = np.asarray([[0.5, 1.0], [1.5, 2.0]]) + offset
            metrics[name] = {
                "preferred_by_cycle": preferred.tolist(),
                "opposite_by_cycle": opposite.tolist(),
                "phase_and_cycle_averaged_contrast": _contrast(preferred, opposite).tolist(),
            }
        return {"populations": {"T4a_L": {"cells": 2, "body_ids": [1, 2], "metrics": metrics}}}

    paired = _paired_sampler_summary(report(0), report(0.25))
    assert set(paired["T4a_L"]["metrics"]) == {"dc", "f1_power", "total_power"}
    assert paired["T4a_L"]["metrics"]["dc"]["median_preferred_change"] == 0.25


def test_eye_centre_comparator_changes_only_sampling_method() -> None:
    from fly_emotion.driving.v7_geometry_sign import (
        MotionSignConductanceProbe,
        MotionSignTypedProbe,
    )

    assert EyeCentreTypedProbe._advance is MotionSignTypedProbe._advance
    assert EyeCentreConductanceProbe._advance is MotionSignConductanceProbe._advance
    assert EyeCentreTypedProbe._retinal_code is MotionSignTypedProbe._retinal_code
    assert EyeCentreConductanceProbe._retinal_code is MotionSignConductanceProbe._retinal_code


def test_saved_neural_spectra_are_hash_bound_and_reconstructible() -> None:
    report = json.loads((ROOT / "artifacts/v7-neural-spectra.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / path).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == digest
    assert report["advance_to_central_complex"] is False
    assert report["protocol"]["downstream_phase_reference_defined"] is False
    assert report["protocol"]["old_direction_gate_replaced"] is False
    assert len(report["retinal_drive_sha256"]) == 16
    for backend in report["results"].values():
        for name in ("TypedBackgroundProbe", "V7PublishedConductanceProbe"):
            nearest = backend["samplers"]["nearest_reversed_axis"][name]
            interpolated = backend["samplers"]["eye_centre_bilinear"][name]
            assert nearest["target_count"] == interpolated["target_count"] == 13580
            assert nearest["sham_sha256"] == interpolated["sham_sha256"]
            assert nearest["maximum_parseval_error"] < 1e-12
            assert interpolated["maximum_parseval_error"] < 1e-12
            ids = []
            for population, values in nearest["populations"].items():
                other = interpolated["populations"][population]
                assert values["body_ids"] == other["body_ids"]
                assert values["expected_direction"] == other["expected_direction"]
                ids.extend(values["body_ids"])
                for metric in ("dc", "f1_power", "total_power"):
                    item = values["metrics"][metric]
                    p = np.asarray(item["preferred_by_cycle"])
                    o = np.asarray(item["opposite_by_cycle"])
                    assert p.shape == o.shape == (2, values["cells"])
                    contrast = _contrast(p, o)
                    denominator = np.abs(p.mean(axis=0)) + np.abs(o.mean(axis=0))
                    active = denominator > 1e-6
                    np.testing.assert_allclose(contrast, item["phase_and_cycle_averaged_contrast"])
                    np.testing.assert_allclose(denominator, item["response_denominator"])
                    np.testing.assert_allclose(np.median(contrast), item["median_contrast"])
                    np.testing.assert_allclose(
                        np.median(denominator), item["median_response_denominator"]
                    )
                    assert (
                        float(np.mean(active)) == item["fraction_response_denominator_above_1e_6"]
                    )
                    assert item["active_median_contrast_at_1e_6"] == (
                        float(np.median(contrast[active])) if np.any(active) else None
                    )
                    assert float(np.mean(contrast > 0)) == item["positive_contrast_fraction"]
            assert len(ids) == len(set(ids)) == 13580
            for family in ("T4", "T5"):
                populations = [
                    value for key, value in nearest["populations"].items() if key.startswith(family)
                ]
                for metric in ("dc", "f1_power", "total_power"):
                    expected = nearest["family_medians"][family][metric]
                    np.testing.assert_allclose(
                        np.median(
                            [item["metrics"][metric]["median_contrast"] for item in populations]
                        ),
                        expected["eight_population_median_contrast"],
                    )
                    np.testing.assert_allclose(
                        np.median(
                            [
                                item["metrics"][metric]["fraction_response_denominator_above_1e_6"]
                                for item in populations
                            ]
                        ),
                        expected["eight_population_median_active_fraction"],
                    )
            assert backend["paired_sampler_change"][name] == _paired_sampler_summary(
                nearest, interpolated
            )
