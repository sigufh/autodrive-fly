import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_emotion.driving.v7_stability import CHECKPOINTS, classify_trace, summarize_tail

ROOT = Path(__file__).parents[1]


def test_trace_classifier_distinguishes_constant_periodic_and_drift() -> None:
    constant = classify_trace(np.ones(128))
    assert constant["status"] == "stationary_within_window"
    assert constant["period_substeps"] is None
    periodic = classify_trace(np.tile([0.0, 0.5, -0.2], 42))
    assert periodic["status"] == "periodic_within_window"
    assert periodic["period_substeps"] == 3
    drifting = classify_trace(np.linspace(0, 1, 128))
    assert drifting["status"] == "unresolved_drift_or_oscillation"
    slow_drift = classify_trace(np.linspace(0, 2e-5, 128))
    assert slow_drift["status"] == "unresolved_drift_or_oscillation"
    drifting_period = classify_trace(np.tile([0.0, 1.0], 64) + np.linspace(0, 1e-3, 128))
    assert drifting_period["status"] == "unresolved_drift_or_oscillation"
    with pytest.raises(ValueError, match="finite"):
        classify_trace(np.array([1, np.nan]))


def test_stability_summary_keeps_rare_oscillator_hidden_by_group_mean() -> None:
    states = np.zeros((128, 4))
    states[:, 0] = np.tile([1.0, -1.0], 64)
    states[:, 1] = -states[:, 0]
    result = summarize_tail(states, np.array(["A", "A", "B", "B"]), np.array([11, 12, 13, 14]))
    assert result["all_cells_stationary_within_window"] is False
    assert result["cells_above_tolerance"] == 2
    assert result["maximum_step_change"] == 2.0
    assert result["top_cells"][0]["body_id"] == 11
    assert result["top_cells"][0]["period_substeps"] == 2
    assert np.all(states.mean(axis=1) == 0)


def test_saved_background_audit_binds_dependencies_and_covers_checkpoints() -> None:
    report = json.loads((ROOT / "artifacts/v7-background-stability.json").read_text())
    assert report["advance_to_central_complex"] is False
    assert report["protocol"]["parameter_fitting"] is False
    for path, expected in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / path).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    original = json.loads((ROOT / "artifacts/v7-conductance-order-comparison.json").read_text())
    for backend, result in report["results"].items():
        assert len(result["models"]) == 3
        for name, model in result["models"].items():
            if name != "TypedBackgroundProbe":
                prior = next(
                    v
                    for v in original["results"][backend]["models"][name]["rest"]
                    if v["background_level"] == 0.5
                )
                assert model["checkpoints"]["128"]["state_sha256"] == prior["state_sha256"]
                assert (
                    model["checkpoints"]["128"]["last_substep_global_change"]
                    == prior["last_step_maximum_state_change"]
                )
            assert set(model["checkpoints"]) == set(map(str, CHECKPOINTS))
            for checkpoint in model["checkpoints"].values():
                groups = checkpoint["populations_ranked_by_step_change"]
                assert sum(group["cells"] for group in groups) == checkpoint["cells"]
                assert (
                    sum(group["cells_above_tolerance"] for group in groups)
                    == checkpoint["cells_above_tolerance"]
                )
                for cell in checkpoint["top_cells"]:
                    assert classify_trace(np.asarray(cell["trace"]))["status"] == cell["status"]
