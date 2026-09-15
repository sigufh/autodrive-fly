import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE, V7VisualProbe
from fly_emotion.driving.v7_geometry_sign import (
    MotionSignConductanceProbe,
    MotionSignTypedProbe,
    _pixel_indices,
    anatomical_sign_audit,
    reverse_eye_coordinates,
)
from fly_emotion.driving.v7_phase_motion import phase_average_contrast
from fly_emotion.driving.v7_source_audit import CARDINAL_VECTORS, _frozen_consensus_transforms

ROOT = Path(__file__).parents[1]


def _records() -> list[dict]:
    records = []
    for eye, rotation in (("L", np.array([[0, -1], [1, 0]])), ("R", np.diag([-1, 1]))):
        for subtype in "abcd":
            expected = (
                HORIZONTAL_PREFERENCE[(subtype, eye)]
                if subtype in "ab"
                else VERTICAL_PREFERENCE[subtype]
            )
            offset = CARDINAL_VECTORS[expected] @ rotation
            records.append(
                {
                    "body_id": len(records),
                    "side": eye,
                    "subtype": subtype,
                    "expected": expected,
                    "centroids": {"Mi1": np.zeros(2), "Mi4": offset, "C3": offset, "Mi9": -offset},
                    "weights": {source: 1.0 for source in ("Mi1", "Mi4", "C3", "Mi9")},
                }
            )
    return records


def test_distal_centre_proximal_order_requires_opposite_of_legacy_fit() -> None:
    records = _records()
    transforms = _frozen_consensus_transforms(records, ("Mi1",), ("Mi4", "C3"))
    report = anatomical_sign_audit(records, transforms)
    for group in report["groups"].values():
        for axis in group["axes"].values():
            np.testing.assert_allclose(axis["legacy_cosine"], -1)
            np.testing.assert_allclose(axis["reversed_cosine"], 1)
    for record in records:
        projected = np.stack(
            [record["centroids"][name] for name in ("Mi9", "Mi1", "Mi4")]
        ) @ -np.asarray(transforms[record["side"]])
        arrival = projected @ CARDINAL_VECTORS[record["expected"]]
        assert np.all(np.diff(arrival) > 0)


def test_anatomy_retains_missing_and_zero_vectors_in_denominator() -> None:
    records = _records()
    transforms = _frozen_consensus_transforms(records, ("Mi1",), ("Mi4", "C3"))
    records[0]["centroids"]["Mi9"] = np.full(2, np.nan)
    records[1]["centroids"]["Mi9"] = np.zeros(2)
    report = anatomical_sign_audit(records, transforms)
    group = report["groups"]["L"]
    axis = group["axes"]["distal_to_centre"]
    assert group["cells"] == 4
    assert axis["missing_or_zero_cells"] == 2
    assert axis["reversed_cosine"][:2] == [None, None]
    assert axis["reversed_positive_fraction_all_cells"] == 0.5


def test_coordinate_reversal_matches_negated_projection_without_swapping_eyes() -> None:
    projected = np.array([[-2, -3], [-1, 0], [1, 2], [3, 7]], dtype=np.float64)
    x, y = (V7VisualProbe._normalized(projected[:, i]) for i in (0, 1))
    u = np.concatenate((0.5 * x, 0.5 + 0.5 * x))
    v = np.tile(y, 2)
    side = np.repeat([-1, 1], 4)
    ru, rv = reverse_eye_coordinates(u, v, side)
    nx, ny = (V7VisualProbe._normalized(-projected[:, i]) for i in (0, 1))
    np.testing.assert_allclose(ru, np.concatenate((0.5 * nx, 0.5 + 0.5 * nx)), atol=1e-7)
    np.testing.assert_allclose(rv, np.tile(ny, 2), atol=1e-7)
    assert np.all((ru[side < 0] >= 0) & (ru[side < 0] <= 0.5))
    assert np.all((ru[side > 0] >= 0.5) & (ru[side > 0] <= 1))
    uu, vv = reverse_eye_coordinates(ru, rv, side)
    np.testing.assert_allclose(uu, u, atol=1e-7)
    np.testing.assert_allclose(vv, v, atol=1e-7)


def test_pixel_sampling_keeps_historical_rounding_and_exposes_midpoint() -> None:
    u = np.array([0, 0.25, 0.5, 0.5, 0.75, 1], dtype=np.float32)
    v = np.linspace(0, 1, 6, dtype=np.float32)
    ru, rv = reverse_eye_coordinates(u, v, np.array([-1, -1, -1, 1, 1, 1]))
    pixels = _pixel_indices(ru, rv)
    np.testing.assert_array_equal(pixels[:, 0], [24, 12, 0, 47, 35, 24])
    image = np.arange(24 * 48).reshape(24, 48)
    probe = SimpleNamespace(retinal_u=ru, retinal_v=rv)
    np.testing.assert_array_equal(
        V7VisualProbe._sample_retina(probe, image), image[pixels[:, 1], pixels[:, 0]]
    )


def test_geometry_probes_do_not_override_neural_dynamics() -> None:
    from fly_emotion.driving.v7_conductance import V7PublishedConductanceProbe
    from fly_emotion.driving.v7_stability import TypedBackgroundProbe

    assert MotionSignTypedProbe._advance is TypedBackgroundProbe._advance
    assert MotionSignConductanceProbe._advance is V7PublishedConductanceProbe._advance
    assert MotionSignTypedProbe._retinal_code is TypedBackgroundProbe._retinal_code
    assert MotionSignConductanceProbe._retinal_code is V7PublishedConductanceProbe._retinal_code


def test_saved_geometry_comparison_is_hash_bound_and_keeps_all_cells() -> None:
    report = json.loads((ROOT / "artifacts/v7-geometry-sign.json").read_text())
    original = json.loads((ROOT / "artifacts/v7-phase-motion.json").read_text())
    for path, expected in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / path).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    assert report["advance_to_central_complex"] is False
    assert report["protocol"]["normalization_refit"] is False
    assert report["protocol"]["physiological_labels_changed"] is False
    assert report["synthetic_controls"]["controls_pass"] is True
    geometry = report["geometry"]
    assert geometry["receptors"] == len(set(geometry["receptor_body_ids"])) == 3344
    assert sum(group["cells"] for group in report["anatomy"]["groups"].values()) == 6861
    u, v = np.asarray(geometry["legacy_uv"], dtype=np.float32).T
    ru, rv = reverse_eye_coordinates(u, v, np.asarray(geometry["side"]))
    np.testing.assert_array_equal(np.stack((ru, rv), axis=1), geometry["reversed_uv"])
    np.testing.assert_array_equal(_pixel_indices(ru, rv), geometry["reversed_pixels"])
    assert report["stimulus_sha256"] == original["stimulus_sha256"]
    assert len(report["retinal_drive_sha256"]) == 96
    for backend, result in report["results"].items():
        assert (
            result["normalization_sha256"] == original["results"][backend]["normalization_sha256"]
        )
        for name, model in result["models"].items():
            old = original["results"][backend]["models"][name]
            ids = [body for pop in model["populations"].values() for body in pop["body_ids"]]
            assert len(ids) == len(set(ids)) == model["target_count"] == 13580
            assert model["sham_peak_to_peak_max"] == old["sham_peak_to_peak_max"]
            for key, pop in model["populations"].items():
                previous = old["populations"][key]
                assert pop["body_ids"] == previous["body_ids"]
                assert pop["expected_direction"] == previous["expected_direction"]
                p, o = (
                    np.asarray(pop[f"per_cell_{direction}_cycle_means"])
                    for direction in ("preferred", "opposite")
                )
                assert p.shape == o.shape == (2, pop["cells"])
                contrast = phase_average_contrast(p, o)
                old_contrast = phase_average_contrast(
                    previous["per_cell_preferred_cycle_means"],
                    previous["per_cell_opposite_cycle_means"],
                )
                np.testing.assert_allclose(np.median(contrast), pop["median_direction_contrast"])
                np.testing.assert_allclose(
                    np.median(contrast - old_contrast), pop["paired_median_contrast_change"]
                )
                np.testing.assert_allclose(
                    np.median(np.abs(contrast + old_contrast)),
                    pop["median_absolute_residual_to_exact_sign_flip"],
                )
                np.testing.assert_allclose(
                    np.median((p - o) / (p + o + 1e-12), axis=1),
                    pop["cycle_median_direction_contrasts"],
                )
                assert float(np.mean(contrast > 0)) == pop["direction_positive_fraction"]
            for family in ("T4", "T5"):
                assert model[f"{family}_population_median"] == np.median(
                    [
                        pop["median_direction_contrast"]
                        for key, pop in model["populations"].items()
                        if key.startswith(family)
                    ]
                )
