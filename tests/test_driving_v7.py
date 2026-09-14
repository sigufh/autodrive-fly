import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from fly_emotion.driving.v7 import (
    V7_TARGET_TYPES,
    V7Contract,
    V7VisualProbe,
    build_controlled_stimuli,
    evaluate_v7_optic_hex_axis_calibration,
    evaluate_v7_typed_visual_candidate,
    load_v7_optic_hex_offsets,
    score_v7_visual_responses,
    write_v7_manifest,
)

ROOT = Path(__file__).parents[1]


def test_v7_contract_freezes_v5_v6_and_disables_deployment() -> None:
    contract = V7Contract.load(ROOT)
    assert contract.payload["version"] == 7
    assert contract.payload["name"] == "v7-experimental"
    assert contract.payload["deployment_enabled"] is False
    assert contract.payload["city_expansion_enabled"] is False
    for baseline in contract.payload["baseline_contracts"].values():
        target = ROOT / baseline["path"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == baseline["sha256"]


def test_v7_contract_rejects_deployment_or_baseline_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load((ROOT / "configs/driving-v7.yaml").read_text())
    target = tmp_path / "configs/driving-v7.yaml"
    target.parent.mkdir(parents=True)
    payload["deployment_enabled"] = True
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="remain isolated"):
        V7Contract.load(tmp_path)


def test_v7_controlled_stimuli_are_deterministic_and_exactly_mirrored() -> None:
    first = build_controlled_stimuli()
    second = build_controlled_stimuli()
    by_name = {item.name: item for item in first}
    assert len(first) == 20
    assert [item.sha256 for item in first] == [item.sha256 for item in second]
    assert {item.family for item in first} == {
        "uniform",
        "moving_edge",
        "looming",
        "receding",
        "static",
        "translation",
        "rotation",
    }
    for item in first:
        assert item.frames.shape == (16, 24, 48)
        assert np.all((item.frames >= 0) & (item.frames <= 1))
        if item.mirror_of:
            assert np.array_equal(item.frames[:, :, ::-1], by_name[item.mirror_of].frames)
    for polarity in ("on", "off"):
        looming = by_name[f"{polarity}_looming"].frames
        receding = by_name[f"{polarity}_receding"].frames
        static = by_name[f"{polarity}_static_disc"].frames
        assert np.array_equal(looming[::-1], receding)
        assert np.array_equal(looming[-1], static[-1])
        assert not np.array_equal(looming[0], static[0])


def test_v7_visual_targets_exist_and_never_receive_direct_input() -> None:
    probe = V7VisualProbe(ROOT, brain_substeps=1)
    assert probe.retina.size == 3344
    for cell_type in V7_TARGET_TYPES:
        assert f"{cell_type}_L" in probe.populations
        assert f"{cell_type}_R" in probe.populations
    targets = np.unique(np.concatenate(list(probe.populations.values())))
    assert not np.intersect1d(probe.retina.node_indices, targets).size


def test_v7_topology_controls_are_deterministic_and_preserve_declared_contracts() -> None:
    real = V7VisualProbe(ROOT, brain_substeps=1)
    retina_a = V7VisualProbe(
        ROOT, brain_substeps=1, control="shuffled_retina_coordinates", control_seed=7
    )
    retina_b = V7VisualProbe(
        ROOT, brain_substeps=1, control="shuffled_retina_coordinates", control_seed=7
    )
    assert np.array_equal(retina_a.retinal_permutation, retina_b.retinal_permutation)
    assert not np.array_equal(retina_a.retinal_permutation, real.retinal_permutation)

    target_shuffle = V7VisualProbe(
        ROOT, brain_substeps=1, control="source_preserving_target_shuffle", control_seed=7
    )
    real_outdegree = np.diff(real.adjacency.tocsc().indptr)
    shuffled_outdegree = np.diff(target_shuffle.adjacency.tocsc().indptr)
    assert np.array_equal(real_outdegree, shuffled_outdegree)
    assert not np.array_equal(real.adjacency.indices, target_shuffle.adjacency.indices)

    sign_shuffle = V7VisualProbe(
        ROOT, brain_substeps=1, control="shuffled_transmitter_signs", control_seed=7
    )
    assert np.array_equal(np.sort(real.source_sign), np.sort(sign_shuffle.source_sign))
    assert not np.array_equal(real.source_sign, sign_shuffle.source_sign)


def test_v7_typed_visual_leak_is_cell_type_specific() -> None:
    legacy = V7VisualProbe(ROOT, brain_substeps=1, dynamics_backend="legacy_uniform_tanh_v1")
    typed = V7VisualProbe(ROOT, brain_substeps=1, dynamics_backend="typed_visual_leak_v1")
    assert np.all(legacy.leak == 0.28)
    assert np.all(typed.leak[typed.node_types == "R1-R6"] == 1.0)
    assert np.all(typed.leak[typed.node_types == "Tm3"] == 0.62)
    assert np.all(typed.leak[typed.node_types == "Mi9"] == 0.12)
    assert np.all(typed.leak[typed.node_types == "T4a"] == 0.50)
    assert not np.array_equal(legacy.leak, typed.leak)


def test_v7_visual_subgraph_uses_only_annotated_visual_nodes_and_real_edges() -> None:
    probe = V7VisualProbe(ROOT, brain_substeps=1, dynamics_backend="typed_visual_subgraph_v1")
    assert 100_000 < np.count_nonzero(probe.visual_subgraph_mask) < probe.graph.node_count
    assert 0 < probe.adjacency.nnz < probe.graph.adjacency.nnz
    rows, columns = probe.adjacency.nonzero()
    assert np.all(probe.visual_subgraph_mask[rows])
    assert np.all(probe.visual_subgraph_mask[columns])
    incoming = np.asarray(probe.adjacency.sum(axis=1)).ravel()
    assert np.allclose(incoming[incoming > 0], 1.0, atol=1e-5)


def test_v7_columnar_delay_uses_only_declared_visual_source_types() -> None:
    probe = V7VisualProbe(ROOT, brain_substeps=1, dynamics_backend="columnar_delay_v1")
    assert int(probe.source_delays.max()) == 3
    assert np.all(probe.source_delays[probe.node_types == "Tm3"] == 0)
    assert np.all(probe.source_delays[probe.node_types == "Mi1"] == 1)
    assert np.all(probe.source_delays[probe.node_types == "Mi4"] == 2)
    assert np.all(probe.source_delays[probe.node_types == "Mi9"] == 3)
    delayed = probe.source_delays > 0
    assert np.all(probe.visual_subgraph_mask[delayed])


def test_v7_columnar_correlator_uses_real_t4_t5_inputs() -> None:
    probe = V7VisualProbe(ROOT, brain_substeps=1, dynamics_backend="columnar_correlator_v1")
    correlator = probe.correlator
    assert correlator is not None
    assert 13_000 < correlator["all_targets"] < 14_000
    assert len(correlator["targets"]) > 13_000
    assert correlator["fast"].nnz > 100_000
    assert correlator["delayed"].nnz > 100_000
    assert correlator["history_substeps"] == 3
    assert np.all(probe.visual_subgraph_mask[correlator["targets"]])


def test_v7_axial_hex_retinal_geometry_is_bounded_and_eye_separated() -> None:
    legacy = V7VisualProbe(ROOT, brain_substeps=1, retinal_geometry="legacy_proxy_v2")
    axial = V7VisualProbe(ROOT, brain_substeps=1, retinal_geometry="axial_hex_cartesian_v1")
    assert np.all((axial.retinal_u >= 0) & (axial.retinal_u <= 1))
    assert np.all((axial.retinal_v >= 0) & (axial.retinal_v <= 1))
    assert np.all(axial.retinal_u[axial.retina.side < 0] <= 0.5)
    assert np.all(axial.retinal_u[axial.retina.side > 0] >= 0.5)
    assert not np.array_equal(axial.retinal_u, legacy.retinal_u)
    assert not np.array_equal(axial.retinal_v, legacy.retinal_v)


def test_v7_optic_hex_offsets_use_real_target_specific_fast_and_delayed_inputs() -> None:
    dataset = load_v7_optic_hex_offsets(ROOT)
    assert dataset.size > 13_000
    assert len(np.unique(dataset.body_ids)) == dataset.size
    assert set(dataset.families) == {"T4", "T5"}
    assert set(dataset.subtypes) == {"a", "b", "c", "d"}
    assert set(dataset.sides) == {"L", "R"}
    assert np.all(np.isfinite(dataset.offsets))
    assert np.all(np.linalg.norm(dataset.offsets, axis=1) > 0)


def test_v7_optic_hex_axis_report_has_disjoint_held_out_and_zero_shot_evidence() -> None:
    report = evaluate_v7_optic_hex_axis_calibration(ROOT)
    implementation_hash = hashlib.sha256(
        (ROOT / "src/fly_emotion/driving/v7.py").read_bytes()
    ).hexdigest()
    assert report["protocol"]["driving_data_used"] is False
    assert report["protocol"]["implementation_sha256"] == implementation_hash
    assert report["protocol"]["T5_used_for_fit_or_model_selection"] is False
    assert report["dataset"]["fit_held_out_body_id_overlap"] == 0
    assert report["dataset"]["T4_fit_cells"] > 3_000
    assert report["dataset"]["T4_held_out_cells"] > 3_000
    assert report["dataset"]["T5_zero_shot_cells"] > 6_000
    assert report["random_orthogonal_baseline"]["count"] == 256
    assert report["advance_to_central_complex"] is False


def test_v7_persisted_visual_candidate_drops_private_per_neuron_scratch() -> None:
    # Keep this assertion structural; the expensive full candidate is covered by the CLI artifact.
    source = evaluate_v7_typed_visual_candidate.__globals__["_public_visual_responses"]
    response = {
        "edge": {
            "name": "edge",
            "population_trace": {"T4a_L": [0.2]},
            "population_response_trace": {"T4a_L": [0.1]},
            "population_mean_abs": {"T4a_L": 0.2},
            "_scratch": [1.0],
        }
    }
    public = source(response)
    assert public["edge"]["name"] == "edge"
    assert public["edge"]["population_mean_abs"] == {"T4a_L": 0.2}
    assert len(public["edge"]["population_response_trace_sha256"]) == 64
    assert "population_trace" not in public["edge"]
    assert "population_response_trace" not in public["edge"]
    assert "_scratch" not in public["edge"]


def _synthetic_visual_responses() -> dict:
    populations = [f"{cell_type}_{side}" for cell_type in V7_TARGET_TYPES for side in ("L", "R")]
    return {
        stimulus.name: {
            "mirror_of": stimulus.mirror_of,
            "population_trace": {name: [0.0, 0.0] for name in populations},
            "population_response_trace": {name: [0.0, 0.0] for name in populations},
            "_population_neuron_positive_mean": {name: [0.0] for name in populations},
        }
        for stimulus in build_controlled_stimuli()
    }


def test_v7_energy_weighted_mirror_metric_does_not_let_silent_population_dominate() -> None:
    responses = _synthetic_visual_responses()
    for name in ("on_edge_left", "on_edge_right"):
        traces = responses[name]["population_response_trace"]
        traces["T4a_L"] = [1.0, -1.0]
        traces["T4a_R"] = [1.0, -1.0]
        traces["T4b_L"] = [1e-6, -1e-6]
    scores = score_v7_visual_responses(responses)
    pair = "on_edge_left<->on_edge_right"
    assert scores["mirror_response_error"][pair] == pytest.approx(1 / len(V7_TARGET_TYPES))
    assert scores["energy_weighted_mirror_response_error"][pair] == pytest.approx(
        2e-6 / (4.0 + 2e-6)
    )
    assert scores["mirror_response_scale"][pair]["populations"] == 2 * len(V7_TARGET_TYPES)


@pytest.mark.parametrize(
    ("left", "right", "expected_error"),
    [
        ([0.0, 0.0], [0.0, 0.0], 0.0),
        ([1.0, -1.0], [1.0, -1.0], 0.0),
        ([1.0, -1.0], [0.0, 0.0], 1.0),
        ([1.0, -1.0], [-1.0, 1.0], 1.0),
    ],
)
def test_v7_mirror_metric_handles_silence_one_sided_activity_and_signed_traces(
    left: list[float], right: list[float], expected_error: float
) -> None:
    responses = _synthetic_visual_responses()
    for name in ("on_edge_left", "on_edge_right"):
        responses[name]["population_response_trace"]["T4a_L"] = left
        responses[name]["population_response_trace"]["T4a_R"] = right
    scores = score_v7_visual_responses(responses)
    assert scores["summary"]["maximum_energy_weighted_mirror_response_error"] == pytest.approx(
        expected_error
    )
    assert scores["summary"]["median_cardinal_direction_contrast"] == 0.0
    assert scores["summary"]["median_on_off_specialization"] == 0.0
    assert scores["summary"]["median_known_looming_contrast"] == 0.0


def test_v7_manifest_is_isolated_and_in_progress() -> None:
    manifest = write_v7_manifest(ROOT)
    assert manifest["version"] == 7
    assert manifest["current_stage"] == "controlled_vision"
    assert manifest["stage_status"] in {"in_progress", "blocked_on_visual_dynamics"}
    assert manifest["advance_to_central_complex"] is False
    assert manifest["default_runtime_changed"] is False
    assert (
        manifest["implementation_sha256"]
        == hashlib.sha256((ROOT / "src/fly_emotion/driving/v7.py").read_bytes()).hexdigest()
    )
    assert manifest["stage_findings"] == {
        "historical_aggregate_optic_hex_axis_passed": False,
        "nested_T4_branch_axis_validation_passed": True,
        "branched_T4_controlled_response_passed": False,
        "published_T4_conductance_response_passed": False,
        "fitted_T4_conductance_validation_passed": False,
        "fitted_T4_one_time_test_evaluated": False,
        "legacy_retinal_input_count_balanced": False,
        "paired_retinal_input_exactly_mirrored": True,
        "active_layer_mirror_error_above_gate": False,
        "full_retina_eye_mass_normalized_mirror_passed": False,
        "balanced_count_mirror_passed": True,
        "balanced_exact_input_mirror_passed": True,
    }
    assert "branched_T4_response_gates_failed" in manifest["blockers"]
    assert "published_T4_conductance_response_gates_failed" in manifest["blockers"]
    assert "fitted_T4_conductance_validation_failed" in manifest["blockers"]
    assert manifest["current_evidence"] == "artifacts/v7-layerwise-mirror-audit.json"
    assert set(manifest["evidence"]) >= {
        "artifacts/v7-layerwise-mirror-audit.json",
        "artifacts/v7-retina-column-audit.json",
        "artifacts/v7-t4-conductance-fit.json",
        "artifacts/v7-t4-conductance-candidate.json",
        "artifacts/v7-branched-t4-candidate.json",
        "artifacts/v7-t4-source-audit.json",
        "artifacts/v7-typed-visual-candidate.json",
    }
