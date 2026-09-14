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
    assert callable(evaluate_v7_branched_t4_candidate)
