import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_emotion.driving.v7_spectral_controls import (
    compare_spectra,
    directional_controls,
    negative_controls,
    spectral_summary,
)

ROOT = Path(__file__).parents[1]


def test_spectrum_retains_dc_phase_power_and_parseval_without_phase_cancellation() -> None:
    phase, time = np.mgrid[:8, :8]
    drive = np.cos(2 * np.pi * (time + phase) / 8)[:, None, :, None]
    response = 0.3 + 2 * np.cos(2 * np.pi * (time + phase) / 8 - 0.7)[:, None, :, None]
    result = spectral_summary(response, drive)
    np.testing.assert_allclose(result["dc"], [[0.3]])
    np.testing.assert_allclose(result["f1_power"], [[2]])
    np.testing.assert_allclose(result["total_power"], [[2.09]])
    np.testing.assert_allclose(result["transfer_real"], [[2 * np.cos(0.7)]])
    np.testing.assert_allclose(result["transfer_imaginary"], [[-2 * np.sin(0.7)]])
    assert result["transfer_valid"] == [[True]]
    assert result["parseval_max_error"] < 1e-14
    assert abs(np.fft.rfft(response, axis=2)[:, :, 1].mean()) < 1e-14


def test_inverting_potential_preserves_power_but_inverts_dc_and_transfer() -> None:
    rng = np.random.default_rng(42)
    drive, response = rng.normal(size=(2, 8, 2, 8, 3))
    positive, negative = spectral_summary(response, drive), spectral_summary(-response, drive)
    for key in ("f1_power", "total_power"):
        np.testing.assert_allclose(positive[key], negative[key])
    for key in ("dc", "transfer_real", "transfer_imaginary"):
        np.testing.assert_allclose(positive[key], -np.asarray(negative[key]))


def test_no_input_phase_reference_remains_explicitly_invalid() -> None:
    response = np.ones((8, 2, 8, 3))
    result = spectral_summary(response, np.zeros_like(response))
    assert not np.any(result["transfer_valid"])
    assert not np.any(result["transfer_real"])
    np.testing.assert_allclose(result["dc"], 1)
    np.testing.assert_allclose(result["f1_power"], 0)
    with pytest.raises(ValueError, match="aligned"):
        spectral_summary(response[:, :, :-1], response[:, :, :-1])


def test_fractional_grid_spectral_negative_control_and_known_filter_transfer() -> None:
    x = np.tile(np.array([0, 0.25, 0.5, 0.75, 1]) / 23, 2)
    result = negative_controls(x, x[::-1], np.repeat([-1, 1], 5))
    assert result["negative_controls_pass"] is True
    a = (1 - 0.24) ** 4
    expected = (1 - a) / (1 - a * np.exp(-2j * np.pi / 8))
    for modes in result["spectra"].values():
        for spectra in modes["periodic"].values():
            np.testing.assert_allclose(spectra["transfer_real"], expected.real, atol=1e-12)
            np.testing.assert_allclose(spectra["transfer_imaginary"], expected.imag, atol=1e-12)


def test_opponent_direction_is_visible_in_signed_dc_not_power() -> None:
    controls = directional_controls()
    assert controls["directional_controls_pass"] is True
    for item in controls["results"].values():
        contrast = item["comparisons"]["opponent"]
        assert np.min(contrast["signed_dc_contrast"]) > 0.99
        assert contrast["maximum_absolute_total_power_contrast"] < 1e-6
        assert contrast["maximum_absolute_f1_power_contrast"] < 1e-6
        assert np.max(item["comparisons"]["inverted_opponent"]["signed_dc_contrast"]) < -0.99


def test_saved_spectral_evidence_preserves_denominators_and_control_comparisons() -> None:
    report = json.loads((ROOT / "artifacts/v7-spectral-controls.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / path).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == digest
    assert report["measurement_controls_pass"] is True
    assert report["neural_evaluation_performed"] is False
    assert report["advance_to_central_complex"] is False
    assert report["protocol"]["direction_gate_replaced"] is False
    assert len(set(report["geometry"]["receptor_body_ids"])) == 3344
    for name, cells in (
        ("receptor_negative_controls", 3344),
        ("fractional_grid_negative_controls", 66),
    ):
        control = report[name]
        assert control["cells"] == cells
        assert control["negative_controls_pass"] is True
        for backend, modes in control["spectra"].items():
            for mode, conditions in modes.items():
                cycles = 2 if mode == "finite" else 1
                for spectra in conditions.values():
                    for key in ("dc", "f1_power", "total_power", "transfer_valid"):
                        assert np.asarray(spectra[key]).shape == (cycles, cells)
                    assert spectra["parseval_max_error"] < 1e-12
                for p, o in (("right", "left"), ("down", "up")):
                    result = compare_spectra(conditions[p], conditions[o])
                    assert result == control["comparisons"][backend][mode][f"{p}_{o}"]
                    assert result["maximum_absolute_f1_power_contrast"] < 1e-6
                    assert result["maximum_absolute_dc_difference"] < 1e-6
