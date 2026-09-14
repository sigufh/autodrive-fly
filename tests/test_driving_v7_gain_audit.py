import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from fly_emotion.driving.v7_conductance import SourceNormalization, V7PublishedConductanceProbe
from fly_emotion.driving.v7_gain_audit import interface_jacobian, interface_output

ROOT = Path(__file__).parents[1]


def _probe():
    probe = object.__new__(V7PublishedConductanceProbe)
    probe.conductance = {
        "targets": np.array([2]),
        "normalization": SourceNormalization(
            np.zeros(3), np.array([0.1, 0.2, 1.0]), np.array([True, True, False]), "", {}
        ),
        "matrices": {
            "exc": sparse.csr_matrix([[1.0, 0.0, 0.0]]),
            "inh": sparse.csr_matrix([[0.0, 1.0, 0.0]]),
        },
        "model": {
            "leak_conductance": 0.5,
            "reversal_potentials_millivolts": {"leak": -65.0, "exc": -21.0, "inh": -71.0},
            "source_parameters": {
                "exc": {"gain": 0.7, "threshold": 0.1, "reversal": "exc"},
                "inh": {"gain": 0.9, "threshold": 0.2, "reversal": "inh"},
            },
            "output_normalization": {
                "baseline_millivolts": -65.0,
                "excitatory_reversal_millivolts": -21.0,
            },
        },
    }
    return probe


def test_interface_jacobian_matches_finite_difference_with_excitatory_and_inhibitory_signs() -> (
    None
):
    probe = _probe()
    state = np.array([0.05, 0.1, 0.0])
    jacobian, contributions, _ = interface_jacobian(probe, state)
    assert jacobian[0, 0] > 0
    assert jacobian[0, 1] < 0
    assert jacobian[0, 2] == 0
    for index in (0, 1):
        delta = np.zeros(3)
        delta[index] = 1e-4
        measured = (
            interface_output(probe, state + delta) - interface_output(probe, state - delta)
        ) / (2e-4)
        np.testing.assert_allclose(measured, jacobian.toarray()[:, index], rtol=5e-4, atol=1e-4)
    np.testing.assert_allclose(
        sum(contributions.values()), np.asarray(abs(jacobian).sum(axis=1)).ravel()
    )


def test_clipped_or_below_threshold_sources_have_zero_local_derivative() -> None:
    probe = _probe()
    for state in (np.array([-0.1, 0.01, 0.0]), np.array([0.2, 0.3, 0.0])):
        jacobian, _, _ = interface_jacobian(probe, state)
        assert jacobian.nnz == 0


def test_normalization_span_rescales_derivative_without_changing_normalized_output() -> None:
    probe = _probe()
    state = np.array([0.05, 0.1, 0.0])
    output = interface_output(probe, state)
    derivative, _, _ = interface_jacobian(probe, state)
    normalization = probe.conductance["normalization"]
    probe.conductance["normalization"] = SourceNormalization(
        normalization.low * 10,
        normalization.high * 10,
        normalization.source_mask,
        "",
        {},
    )
    scaled_derivative, _, _ = interface_jacobian(probe, state * 10)
    np.testing.assert_allclose(interface_output(probe, state * 10), output)
    np.testing.assert_allclose(scaled_derivative.toarray(), derivative.toarray() / 10)


def test_output_saturation_has_zero_derivative_even_with_active_sources() -> None:
    probe = _probe()
    probe.conductance["model"]["output_normalization"]["excitatory_reversal_millivolts"] = -64.0
    state = np.array([0.05, 0.1, 0.0])
    jacobian, _, detail = interface_jacobian(probe, state)
    assert np.all(detail["raw_output"] > 1)
    assert jacobian.nnz == 0
    np.testing.assert_array_equal(interface_output(probe, state), [1.0])


def test_saved_gain_audit_is_bound_to_original_background_states() -> None:
    report = json.loads((ROOT / "artifacts/v7-normalization-gain.json").read_text())
    reference = json.loads((ROOT / "artifacts/v7-background-stability.json").read_text())
    assert report["advance_to_central_complex"] is False
    for relative, expected in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / relative).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    for backend, result in report["results"].items():
        assert (
            result["normalization_sha256"] == reference["results"][backend]["normalization_sha256"]
        )
        for step, snapshot in result["snapshots"].items():
            assert (
                snapshot["state_sha256"]
                == reference["results"][backend]["models"]["V7PublishedConductanceProbe"][
                    "checkpoints"
                ][step]["state_sha256"]
            )
            assert len(snapshot["target_populations"]) == 8
            assert len(snapshot["source_populations"]) == 10
            for measurement in snapshot["finite_difference"]:
                assert measurement["smooth_targets"] + measurement[
                    "excluded_crossing_or_kink_targets"
                ] == sum(row["cells"] for row in snapshot["target_populations"].values())
                if measurement["epsilon_in_calibration_span_units"] == 1e-3:
                    assert measurement["smooth_targets"] > 6000
                    assert measurement["maximum_absolute_error_smooth"] < 1e-4
            for target in snapshot["top_targets"]:
                np.testing.assert_allclose(
                    target["row_l1_gain"], sum(target["by_source_type"].values())
                )
            for source in snapshot["source_populations"].values():
                assert np.isclose(
                    source["normalization_interior_fraction"]
                    + source["at_or_below_lower_clip_fraction"]
                    + source["at_or_above_upper_clip_fraction"],
                    1.0,
                )
