import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from fly_emotion.driving.v7_conductance import SourceNormalization, V7PublishedConductanceProbe
from fly_emotion.driving.v7_synchronous import SynchronousConductanceProbe

ROOT = Path(__file__).parents[1]


def _tiny_probe(cls):
    probe = object.__new__(cls)
    probe.adjacency = sparse.csr_matrix(([1.0], ([1], [0])), shape=(3, 3))
    probe.source_sign = np.ones(3, dtype=np.float32)
    probe.leak = np.full(3, 0.5, dtype=np.float32)
    model = {
        "leak_conductance": 1.0,
        "reversal_potentials_millivolts": {"leak": -65.0, "exc": -21.0},
        "source_parameters": {"source": {"gain": 1.0, "threshold": 0.0, "reversal": "exc"}},
        "output_normalization": {
            "baseline_millivolts": -65.0,
            "excitatory_reversal_millivolts": -21.0,
        },
    }
    probe.conductance = {
        "targets": np.array([2]),
        "model": model,
        "matrices": {"source": sparse.csr_matrix(([1.0], ([0], [1])), shape=(1, 3))},
        "normalization": SourceNormalization(
            np.zeros(3), np.ones(3), np.ones(3, dtype=bool), "", {}
        ),
    }
    return probe


def test_synchronous_target_waits_until_next_step_for_upstream_change() -> None:
    old = _tiny_probe(V7PublishedConductanceProbe)
    sync = _tiny_probe(SynchronousConductanceProbe)
    state = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    original = state.copy()
    old_history, sync_history = [state.copy()], [state.copy()]
    next_old = old._advance(state, np.zeros(3), old_history)
    next_sync = sync._advance(state, np.zeros(3), sync_history)
    assert next_old[2] > 0
    assert next_sync[2] == 0
    np.testing.assert_array_equal(next_sync[:2], next_old[:2])
    np.testing.assert_array_equal(state, original)
    np.testing.assert_array_equal(sync_history[0], original)
    after = sync._advance(next_sync, np.zeros(3), sync_history)
    assert after[2] > 0
    assert len(sync_history) == 1


def test_update_conventions_agree_for_stationary_source() -> None:
    state = np.array([0.0, 0.4, 0.0], dtype=np.float32)
    drive = np.array([0.0, np.arctanh(0.4), 0.0])
    old = _tiny_probe(V7PublishedConductanceProbe)
    sync = _tiny_probe(SynchronousConductanceProbe)
    np.testing.assert_allclose(
        old._advance(state, drive, [state.copy()]),
        sync._advance(state, drive, [state.copy()]),
        atol=1e-7,
    )


def test_saved_synchronous_comparison_preserves_shared_inputs_and_original_replay() -> None:
    report = json.loads((ROOT / "artifacts/v7-synchronous-update.json").read_text())
    assert report["advance_to_central_complex"] is False
    assert report["protocol"]["parameter_fitting"] is False
    for path, expected in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / path).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    original_background = json.loads((ROOT / "artifacts/v7-background-stability.json").read_text())
    original_visual = json.loads((ROOT / "artifacts/v7-t4-conductance-candidate.json").read_text())
    for backend, result in report["results"].items():
        assert result["original_replay_exact"] is True
        old = result["models"]["V7PublishedConductanceProbe"]
        new = result["models"]["SynchronousConductanceProbe"]
        assert old["scores"] == original_visual["retinal_results"][backend]["scores"]
        assert (
            result["normalization_sha256"]
            == original_visual["retinal_results"][backend]["normalization"]["sha256"]
        )
        for step, checkpoint in old["background"]["checkpoints"].items():
            reference = original_background["results"][backend]["models"][
                "V7PublishedConductanceProbe"
            ]["checkpoints"][step]
            assert checkpoint["state_sha256"] == reference["state_sha256"]
            assert checkpoint["maximum_step_change"] == reference["maximum_step_change"]
        for key in ("stimuli", "target_ids_sha256", "source_matrices_sha256"):
            assert old[key] == new[key]
        for model in result["models"].values():
            assert len(model["stimuli"]) == 20
            assert set(model["background"]["checkpoints"]) == {"128", "512", "2048"}
            direction = [
                v["contrast"]
                for name, v in model["scores"]["direction_selectivity"].items()
                if name.startswith("T4")
            ]
            assert model["T4_median_direction_contrast"] == np.median(direction)
            assert model["gates"]["T4_direction"] == (np.median(direction) >= 0.10)
            assert model["gates"]["mirror"] == (
                model["scores"]["summary"]["maximum_energy_weighted_mirror_response_error"] <= 0.20
            )
            assert model["all_response_gates_pass"] == all(model["gates"].values())
