import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from fly_emotion.driving.v7_conductance import SourceNormalization, V7PublishedConductanceProbe
from fly_emotion.driving.v7_perturbation import (
    advance_with_clamp,
    full_update_jvp,
    perturbation_directions,
    replay_perturbations,
)

ROOT = Path(__file__).parents[1]


def _probe():
    probe = object.__new__(V7PublishedConductanceProbe)
    probe.adjacency = sparse.csr_matrix([[0, 0, 0], [0.5, 0, 0.1], [0, 0.2, 0]], dtype=np.float32)
    probe.leak = np.array([1.0, 0.4, 0.5], dtype=np.float32)
    probe.source_sign = np.ones(3, dtype=np.float32)
    probe.graph = SimpleNamespace(node_count=3)
    probe.retina = SimpleNamespace(node_indices=np.array([0]))
    probe.conductance = {
        "targets": np.array([2]),
        "normalization": SourceNormalization(
            np.zeros(3), np.ones(3) * 0.5, np.array([False, True, False]), "", {}
        ),
        "matrices": {"source": sparse.csr_matrix([[0, 1, 0]], dtype=np.float32)},
        "model": {
            "leak_conductance": 0.5,
            "reversal_potentials_millivolts": {"leak": -65.0, "exc": -21.0},
            "source_parameters": {"source": {"threshold": 0.1, "gain": 0.8, "reversal": "exc"}},
            "output_normalization": {
                "baseline_millivolts": -65.0,
                "excitatory_reversal_millivolts": -21.0,
            },
        },
    }
    return probe


def test_full_update_jvp_matches_runtime_including_recurrence_leak_and_clamp() -> None:
    probe = _probe()
    state = np.array([0.3, 0.2, 0.1], dtype=np.float32)
    drive = np.array([0.3, 0.0, 0.0], dtype=np.float32)
    direction = np.array([[0.0, 0.0], [0.3, -0.2], [-0.1, 0.4]])
    expected = full_update_jvp(probe, state, drive, direction)
    for i in range(2):
        epsilon = 1e-3
        plus = advance_with_clamp(
            probe, (state + epsilon * direction[:, i]).astype(np.float32), drive
        )
        minus = advance_with_clamp(
            probe, (state - epsilon * direction[:, i]).astype(np.float32), drive
        )
        np.testing.assert_allclose(
            (plus - minus) / (2 * epsilon), expected[:, i], rtol=1e-3, atol=3e-5
        )
    assert np.all(expected[0] == 0)
    with pytest.raises(ValueError, match="shape"):
        full_update_jvp(probe, state, drive, np.ones(3))


def test_unit_slope_diagnostic_keeps_t4_leak_and_non_t4_dynamics() -> None:
    probe = _probe()
    state = np.array([0.3, 0.2, 0.1], dtype=np.float32)
    drive = np.array([0.3, 0, 0], dtype=np.float32)
    vector = np.array([[0], [0.3], [-0.1]])
    full = full_update_jvp(probe, state, drive, vector)
    unit = full_update_jvp(probe, state, drive, vector, unit_normalization_slope=True)
    np.testing.assert_array_equal(full[:2], unit[:2])
    leak_term = (1 - probe.leak[2]) * vector[2]
    np.testing.assert_allclose(unit[2] - leak_term, (full[2] - leak_term) * 0.5)


def test_multistep_tangent_matches_paired_replay_and_does_not_mutate_nominal_state() -> None:
    probe = _probe()
    state = np.array([0.3, 0.2, 0.1], dtype=np.float32)
    drive = np.array([0.3, 0, 0], dtype=np.float32)
    original = state.copy()
    direction = np.array([[0.0], [0.2], [-0.1]])
    epsilon = 1e-3
    plus = (state + epsilon * direction[:, 0]).astype(np.float32)
    minus = (state - epsilon * direction[:, 0]).astype(np.float32)
    tangent = ((plus.astype(float) - minus.astype(float)) / (2 * epsilon))[:, None]
    nominal = state.copy()
    for _ in range(8):
        tangent = full_update_jvp(probe, nominal, drive, tangent)
        plus = advance_with_clamp(probe, plus, drive)
        minus = advance_with_clamp(probe, minus, drive)
        nominal = advance_with_clamp(probe, nominal, drive)
    np.testing.assert_allclose((plus - minus) / (2 * epsilon), tangent[:, 0], rtol=0.01, atol=3e-5)
    replay = replay_perturbations(probe, state, drive)
    np.testing.assert_array_equal(state, original)
    for _ in range(24):
        nominal = advance_with_clamp(probe, nominal, drive)
    assert replay["nominal_final_state_sha256"] == hashlib.sha256(nominal.tobytes()).hexdigest()


def test_seeded_directions_are_supported_only_on_unclamped_source_nodes() -> None:
    probe = _probe()
    first = perturbation_directions(probe)
    np.testing.assert_array_equal(first, perturbation_directions(probe))
    assert np.all(first[[0, 2]] == 0)
    np.testing.assert_array_equal(np.abs(first[1]), [0.5, 0.5])


def test_saved_perturbation_report_is_bound_and_keeps_receptors_unperturbed() -> None:
    report = json.loads((ROOT / "artifacts/v7-full-update-perturbation.json").read_text())
    assert report["advance_to_central_complex"] is False
    assert report["protocol"]["parameter_fitting"] is False
    for path, expected in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / path).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    reference = json.loads((ROOT / "artifacts/v7-normalization-gain.json").read_text())
    for backend, result in report["results"].items():
        assert set(result["snapshots"]) == {"128", "2048"}
        for at_step, snapshot in result["snapshots"].items():
            assert (
                snapshot["initial_state_sha256"]
                == reference["results"][backend]["snapshots"][at_step]["state_sha256"]
            )
            assert len(snapshot["linear_propagation"]) == 32
            for epsilon_string, replay in snapshot["nonlinear_replays"].items():
                epsilon = float(epsilon_string)
                assert [row["step"] for row in replay["trace"]] == list(range(1, 33))
                for step in replay["trace"]:
                    assert [row["seed"] for row in step["directions"]] == report["protocol"][
                        "seeds"
                    ]
                    for index, direction in enumerate(step["directions"]):
                        assert direction["retinal_pair_separation_linf"] == 0
                        assert np.isfinite(direction["relative_derivative_error"])
                        initial = replay["initial_pair_separation_norms"][index]["l2"]
                        assert direction["nonlinear_l2_gain"] == pytest.approx(
                            direction["actual_pair_separation"]["l2"] / initial
                        )
                        norm = replay["realized_initial_half_difference_norms"][index]["l2"]
                        assert initial == pytest.approx(2 * epsilon * norm)
                        assert direction["relative_derivative_error"] == pytest.approx(
                            direction["derivative_error_l2"]
                            / max(direction["tangent_l2_gain"] * norm, 1e-30)
                        )
