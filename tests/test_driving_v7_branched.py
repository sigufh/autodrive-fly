import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_branched import (
    BRANCHED_CONFIG,
    BRANCHED_IMPLEMENTATION,
    V7BranchedT4Probe,
    evaluate_v7_branched_t4_candidate,
)

ROOT = Path(__file__).parents[1]


def test_v7_branched_t4_uses_real_separate_source_edges() -> None:
    probe = V7BranchedT4Probe(
        ROOT,
        retinal_backend="signed_frame_difference",
        brain_substeps=1,
        baseline_frames=1,
    )
    assert probe.branched["all_targets"] == 6861
    assert len(probe.branched["targets"]) > 6_700
    assert set(probe.branched["matrices"]) == {
        "centre_fast_excitatory",
        "proximal_delayed_on_inhibitory",
        "distal_delayed_off_inhibitory",
    }
    for matrix in probe.branched["matrices"].values():
        assert matrix.nnz > 10_000
        assert np.all(np.diff(matrix.indptr) > 0)
    assert np.all(probe.visual_subgraph_mask[probe.branched["targets"]])


def test_v7_branched_t4_contract_is_frozen_and_non_deployable() -> None:
    probe = V7BranchedT4Probe(
        ROOT,
        retinal_backend="signed_frame_difference",
        brain_substeps=1,
        baseline_frames=1,
    )
    config = probe.branched_config
    assert config["deployment_enabled"] is False
    assert config["parameter_scan_allowed"] is False
    assert config["gain"] == 1.0
    assert config["branches"]["centre_fast_excitatory"]["operation"] == "add"
    assert config["branches"]["proximal_delayed_on_inhibitory"]["operation"] == ("subtract")
    assert config["branches"]["distal_delayed_off_inhibitory"]["operation"] == ("subtract")
    assert hashlib.sha256((ROOT / BRANCHED_CONFIG).read_bytes()).hexdigest()
    assert hashlib.sha256((ROOT / BRANCHED_IMPLEMENTATION).read_bytes()).hexdigest()


def test_v7_nested_t4_geometry_is_bounded_eye_separated_and_audit_bound() -> None:
    legacy = V7BranchedT4Probe(
        ROOT,
        retinal_backend="signed_frame_difference",
        retinal_geometry="legacy_proxy_v2",
        brain_substeps=1,
        baseline_frames=1,
    )
    calibrated = V7BranchedT4Probe(
        ROOT,
        retinal_backend="signed_frame_difference",
        retinal_geometry="nested_t4_axis_v1",
        brain_substeps=1,
        baseline_frames=1,
    )
    assert np.all((calibrated.retinal_u >= 0) & (calibrated.retinal_u <= 1))
    assert np.all((calibrated.retinal_v >= 0) & (calibrated.retinal_v <= 1))
    assert np.all(calibrated.retinal_u[calibrated.retina.side < 0] <= 0.5)
    assert np.all(calibrated.retinal_u[calibrated.retina.side > 0] >= 0.5)
    assert not np.array_equal(calibrated.retinal_u, legacy.retinal_u)
    assert not np.array_equal(calibrated.retinal_v, legacy.retinal_v)
    assert calibrated.retinal_geometry == "nested_t4_axis_v1"


def test_v7_balanced_retina_ablations_separate_count_from_forced_mirror_geometry() -> None:
    count_balanced = V7BranchedT4Probe(
        ROOT,
        retinal_backend="signed_frame_difference",
        retinal_geometry="balanced_count_nested_axis_v1",
        brain_substeps=1,
        baseline_frames=1,
    )
    exact_mirror = V7BranchedT4Probe(
        ROOT,
        retinal_backend="signed_frame_difference",
        retinal_geometry="balanced_exact_mirror_v1",
        brain_substeps=1,
        baseline_frames=1,
    )
    for probe in (count_balanced, exact_mirror):
        assert probe.retina.size == 1914
        assert np.count_nonzero(probe.retina.side < 0) == 957
        assert np.count_nonzero(probe.retina.side > 0) == 957
        assert len(probe.retinal_permutation) == probe.retina.size
    assert np.array_equal(count_balanced.retina.body_ids, exact_mirror.retina.body_ids)
    assert np.array_equal(count_balanced.retina.node_indices, exact_mirror.retina.node_indices)
    assert not np.array_equal(count_balanced.retinal_u, exact_mirror.retinal_u)
    assert not np.array_equal(count_balanced.retinal_v, exact_mirror.retinal_v)
    for branch, matrix in count_balanced.branched["matrices"].items():
        assert (matrix != exact_mirror.branched["matrices"][branch]).nnz == 0
    image = np.arange(24 * 48, dtype=np.float32).reshape(24, 48) / (24 * 48)
    sampled = exact_mirror._sample_retina(image)
    mirrored = exact_mirror._sample_retina(image[:, ::-1])
    baseline = np.full(exact_mirror.retina.size, 0.5, dtype=np.float32)
    drive = exact_mirror._retinal_code(sampled, baseline, baseline)
    mirrored_drive = exact_mirror._retinal_code(mirrored, baseline, baseline)
    assert np.array_equal(drive[0::2], mirrored_drive[1::2])
    assert np.array_equal(drive[1::2], mirrored_drive[0::2])


def test_v7_full_retina_eye_mass_normalization_preserves_every_receptor() -> None:
    calibrated = V7BranchedT4Probe(
        ROOT,
        retinal_backend="signed_frame_difference",
        retinal_geometry="nested_t4_axis_v1",
        brain_substeps=1,
        baseline_frames=1,
    )
    probe = V7BranchedT4Probe(
        ROOT,
        retinal_backend="signed_frame_difference",
        retinal_geometry="full_retina_eye_mass_normalized_v1",
        brain_substeps=1,
        baseline_frames=1,
    )
    assert probe.retina.size == 3344
    assert np.array_equal(probe.retinal_u, calibrated.retinal_u)
    assert np.array_equal(probe.retinal_v, calibrated.retinal_v)
    assert np.isclose(np.sum(probe._retinal_drive_gain[probe.retina.side < 0]), 1672.0)
    assert np.isclose(np.sum(probe._retinal_drive_gain[probe.retina.side > 0]), 1672.0)
    assert np.all(probe._retinal_drive_gain[probe.retina.side < 0] > 1.0)
    assert np.all(probe._retinal_drive_gain[probe.retina.side > 0] < 1.0)
    assert np.array_equal(probe.retina.body_ids, calibrated.retina.body_ids)
    assert (probe.adjacency != calibrated.adjacency).nnz == 0
    baseline = np.full(probe.retina.size, 0.5, dtype=np.float32)
    values = np.linspace(0.1, 0.9, probe.retina.size, dtype=np.float32)
    assert np.array_equal(
        probe._retinal_code(values, baseline, baseline),
        calibrated._retinal_code(values, baseline, baseline) * probe._retinal_drive_gain,
    )


def test_v7_branched_t4_polarity_preserves_on_and_off_channels() -> None:
    values = np.asarray((-0.7, 0.0, 0.4), dtype=np.float32)
    assert np.array_equal(
        V7BranchedT4Probe._polarized(values, "positive"),
        np.asarray((0.0, 0.0, 0.4), dtype=np.float32),
    )
    assert np.array_equal(
        V7BranchedT4Probe._polarized(values, "negative"),
        np.asarray((0.7, 0.0, 0.0), dtype=np.float32),
    )


def test_v7_branched_artifact_is_hash_bound_and_cannot_advance() -> None:
    artifact = json.loads(
        (ROOT / "artifacts/v7-branched-t4-candidate.json").read_text(encoding="utf-8")
    )
    protocol = artifact["protocol"]
    assert (
        protocol["candidate_config_sha256"]
        == hashlib.sha256((ROOT / BRANCHED_CONFIG).read_bytes()).hexdigest()
    )
    assert (
        protocol["candidate_implementation_sha256"]
        == hashlib.sha256((ROOT / BRANCHED_IMPLEMENTATION).read_bytes()).hexdigest()
    )
    assert (
        protocol["source_audit_sha256"]
        == hashlib.sha256((ROOT / "artifacts/v7-t4-source-audit.json").read_bytes()).hexdigest()
    )
    assert artifact["passing_retinal_backends"] == []
    assert artifact["controlled_response_gates_pass"] is False
    assert artifact["advance_to_central_complex"] is False
    for result in artifact["retinal_results"].values():
        scores = result["scores"]["summary"]
        assert result["gates"]["mirror_response_error"] == (
            scores["maximum_energy_weighted_mirror_response_error"] <= 0.20
        )
    for geometry in ("balanced_count_nested_axis_v1", "balanced_exact_mirror_v1"):
        result = artifact["retinal_results"][f"{geometry}:signed_frame_difference"]
        assert result["gates"]["mirror_response_error"] is True
        assert result["gates"]["T4_cardinal_direction_contrast"] is False
        assert result["gates"]["on_off_specialization"] is False
        assert result["controlled_response_gates_pass"] is False
    assert callable(evaluate_v7_branched_t4_candidate)
