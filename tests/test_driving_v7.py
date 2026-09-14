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


def test_v7_manifest_is_isolated_and_in_progress() -> None:
    manifest = write_v7_manifest(ROOT)
    assert manifest["version"] == 7
    assert manifest["current_stage"] == "controlled_vision"
    assert manifest["stage_status"] in {"in_progress", "blocked_on_visual_dynamics"}
    assert manifest["advance_to_central_complex"] is False
    assert manifest["default_runtime_changed"] is False
