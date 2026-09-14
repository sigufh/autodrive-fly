import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import sparse

from fly_emotion.driving.v7_branched import V7BranchedT4Probe
from fly_emotion.driving.v7_conductance import V7PublishedConductanceProbe
from fly_emotion.driving.v7_disinhibition import (
    DISTAL_BRANCH,
    MonotoneMi9Probe,
    PresynapticThresholdProbe,
    conductance_truth_table,
    synthetic_mi9_sweep,
)

ROOT = Path(__file__).parents[1]


def test_conductance_truth_table_has_tonic_inhibition_release_and_supralinearity() -> None:
    model = yaml.safe_load((ROOT / "configs/driving-v7-t4-conductance.yaml").read_text())[
        "published_single_compartment_model"
    ]
    table = conductance_truth_table(model)
    for row in table["models"].values():
        assert row["rest_inhibited"] < row["release_only"]
        assert row["excitation_only"] < row["release_and_excitation"]
        assert row["interaction_excess_millivolts"] > 0
    assert (
        table["models"]["V7PublishedConductanceProbe"]
        == table["models"]["PresynapticThresholdProbe"]
    )


def test_presynaptic_threshold_retains_active_source_lost_by_mean_threshold() -> None:
    model = {
        "leak_conductance": 1.0,
        "reversal_potentials_millivolts": {"leak": -65.0, "exc": -21.0},
        "source_parameters": {"example": {"gain": 1.0, "threshold": 0.6, "reversal": "exc"}},
    }
    voltages = []
    for cls in (V7PublishedConductanceProbe, PresynapticThresholdProbe):
        probe = object.__new__(cls)
        probe.conductance = {
            "targets": np.array([0]),
            "model": model,
            "matrices": {"example": sparse.csr_matrix([[0.5, 0.5]])},
        }
        voltage = probe._conductance_voltage(np.array([0.0, 1.0]))[0]
        voltages.append(voltage)
        assert probe._conductance_voltage(np.array([0.2, 0.2]))[0] == -65
    assert voltages[0] == -65
    assert voltages[1] == pytest.approx((-65 + 0.2 * -21) / 1.2)


@pytest.fixture(scope="module")
def probes():
    old = V7BranchedT4Probe(
        ROOT,
        retinal_backend="signed_frame_difference",
        retinal_geometry="nested_t4_axis_v1",
        brain_substeps=4,
        baseline_frames=8,
    )
    revised = MonotoneMi9Probe(ROOT, retinal_backend="signed_frame_difference")
    return old, revised


def test_sign_only_candidate_keeps_graph_targets_and_all_other_parameters(probes) -> None:
    old, revised = probes
    assert (old.adjacency != revised.adjacency).nnz == 0
    assert np.array_equal(old.branched["targets"], revised.branched["targets"])
    for name in ("retinal_u", "retinal_v", "source_sign", "leak"):
        assert np.array_equal(getattr(old, name), getattr(revised, name))
    assert old.branched_config == revised.branched_config
    for name, matrix in old.branched["matrices"].items():
        assert (matrix != revised.branched["matrices"][name]).nnz == 0
        expected = dict(old.branched["branches"][name])
        if name == DISTAL_BRANCH:
            expected["polarity"] = "positive"
        assert expected == revised.branched["branches"][name]
    assert old.branched["gain"] == revised.branched["gain"]
    assert old.branched["history_substeps"] == revised.branched["history_substeps"]


def test_actual_target_update_has_correct_mi9_monotonicity_and_delay(probes) -> None:
    old, revised = probes
    outputs = []
    for probe in (old, revised):
        samples = []
        state = np.zeros(probe.graph.node_count, dtype=np.float32)
        state[np.isin(probe.node_types, ["Mi1", "Tm3"])] = 0.8
        state[np.isin(probe.node_types, ["Mi4", "C3", "CT1"])] = 0.1
        for distal in (-0.4, 0.0, 0.4):
            history = [state.copy() for _ in range(3)]
            history[2][probe.node_types == "Mi9"] = distal
            samples.append(
                probe._advance(state, np.zeros_like(state), history)[probe.branched["targets"]]
            )
        outputs.append(np.asarray(samples))
    assert np.all(outputs[0][0] < outputs[0][1])
    assert np.all(outputs[1][0] == outputs[1][1])
    assert np.all(outputs[1][2] < outputs[1][1])
    assert synthetic_mi9_sweep()["revised_has_no_tonic_inhibition_below_zero"] is True


def test_distal_branch_uses_only_configured_history_slot(probes) -> None:
    _, probe = probes
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy() for _ in range(3)]
    mi9 = probe.node_types == "Mi9"
    state[mi9] = 0.7
    history[0][mi9] = 0.6
    history[1][mi9] = 0.5
    assert np.all(probe._branch_activity(DISTAL_BRANCH, state, history) == 0)
    history[2][mi9] = 0.2
    np.testing.assert_allclose(
        probe._branch_activity(DISTAL_BRANCH, state, history), 0.2, atol=1e-6
    )
    history[2][mi9] = -0.2
    assert np.all(probe._branch_activity(DISTAL_BRANCH, state, history) == 0)


def test_conductance_order_report_preserves_inputs_and_uses_finite_adaptation() -> None:
    report = json.loads((ROOT / "artifacts/v7-conductance-order-comparison.json").read_text())
    assert report["advance_to_central_complex"] is False
    assert report["protocol"]["published_parameters_modified"] is False
    for path, expected in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / path).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    for backend, result in report["results"].items():
        old = result["models"]["V7PublishedConductanceProbe"]
        revised = result["models"]["PresynapticThresholdProbe"]
        assert result["original_scores_reproduced"] is True
        assert old["source_matrices_sha256"] == revised["source_matrices_sha256"]
        assert old["target_ids_sha256"] == revised["target_ids_sha256"]
        assert old["stimuli"] == revised["stimuli"]
        for model in (old, revised):
            assert [row["background_level"] for row in model["rest"]] == [0.08, 0.5, 0.92]
            assert model["all_response_gates_pass"] == all(model["gates"].values())
            if backend == "signed_frame_difference":
                assert len({row["state_sha256"] for row in model["rest"]}) == 1


def test_sign_comparison_artifact_is_hash_bound_and_keeps_input_hashes() -> None:
    report = json.loads((ROOT / "artifacts/v7-mi9-sign-comparison.json").read_text())
    assert report["advance_to_central_complex"] is False
    assert report["protocol"]["parameter_fitting"] is False
    assert report["protocol"]["deployment_enabled"] is False
    for path, expected in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / path).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    assert report["synthetic_sign_check"]["legacy_increases_inhibition_when_mi9_falls_below_zero"]
    original = json.loads((ROOT / "artifacts/v7-branched-t4-candidate.json").read_text())
    for backend, result in report["results"].items():
        assert result["same_inputs_as_frozen_reference"] is True
        assert len(result["stimuli"]) == 20
        reference = original["retinal_results"][f"nested_t4_axis_v1:{backend}"]
        assert result["branch_edges"] == reference["branch_edges"]
        assert result["branched_targets"] == reference["branched_targets"]
        for name, stimulus in result["stimuli"].items():
            assert stimulus["sha256"] == reference["responses"][name]["stimulus_sha256"]
            assert (
                stimulus["retinal_drive_sha256"]
                == reference["responses"][name]["retinal_drive_sha256"]
            )
        direction = [
            v["contrast"]
            for k, v in result["scores"]["direction_selectivity"].items()
            if k.startswith("T4")
        ]
        assert result["T4_summary"]["median_cardinal_direction_contrast"] == np.median(direction)
        assert result["gates"]["T4_cardinal_direction_contrast"] == (np.median(direction) >= 0.10)
        assert result["gates"]["all_T4_populations_positive"] == (
            np.mean(np.asarray(direction) > 0) >= 1.0
        )
        assert result["gates"]["mirror_response_error"] == (
            result["scores"]["summary"]["maximum_energy_weighted_mirror_response_error"] <= 0.20
        )
        assert result["controlled_response_gates_pass"] == all(result["gates"].values())
