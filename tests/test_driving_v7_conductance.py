import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    CONDUCTANCE_IMPLEMENTATION,
    V7PublishedConductanceProbe,
    _collect_source_normalization,
    build_conductance_calibration_stimuli,
)

ROOT = Path(__file__).parents[1]


def _config() -> dict:
    return yaml.safe_load((ROOT / CONDUCTANCE_CONFIG).read_text(encoding="utf-8"))


def test_v7_conductance_calibration_has_no_direction_labels_and_is_deterministic() -> None:
    config = _config()
    first = build_conductance_calibration_stimuli(config)
    second = build_conductance_calibration_stimuli(config)
    assert [item.name for item in first] == [
        "conductance_uniform_sweep",
        "conductance_seeded_white_noise",
    ]
    assert [item.sha256 for item in first] == [item.sha256 for item in second]
    assert all(item.direction == "none" for item in first)
    assert all(item.family == "calibration" for item in first)


def test_v7_published_conductance_parameters_match_paper_contract() -> None:
    config = _config()
    assert config["deployment_enabled"] is False
    assert config["parameter_scan_allowed"] is False
    model = config["published_single_compartment_model"]
    assert model["reversal_potentials_millivolts"] == {
        "glutamate": -71.0,
        "acetylcholine": -21.0,
        "gaba": -68.0,
        "leak": -65.0,
    }
    assert model["leak_conductance"] == 0.50
    assert model["source_parameters"] == {
        "Mi9": {"gain": 0.92, "threshold": 0.20, "reversal": "glutamate"},
        "Tm3": {"gain": 0.35, "threshold": 0.35, "reversal": "acetylcholine"},
        "Mi1": {"gain": 0.65, "threshold": 0.88, "reversal": "acetylcholine"},
        "Mi4": {"gain": 1.10, "threshold": 0.44, "reversal": "gaba"},
        "C3": {"gain": 1.49, "threshold": 0.70, "reversal": "gaba"},
    }
    assert len(hashlib.sha256((ROOT / CONDUCTANCE_IMPLEMENTATION).read_bytes()).hexdigest()) == 64


def test_v7_conductance_normalization_is_dynamic_and_real_edge_model_is_complete() -> None:
    config = _config()
    normalization = _collect_source_normalization(
        ROOT,
        retinal_backend="signed_frame_difference",
        retinal_geometry=config["primary_retinal_geometry"],
        config=config,
    )
    assert len(normalization.sha256) == 64
    for source_type, summary in normalization.summary.items():
        assert source_type in {"Mi9", "Tm3", "Mi1", "Mi4", "C3"}
        assert summary["dynamic_cells"] > 0
        assert summary["dynamic_fraction"] > 0
    probe = V7PublishedConductanceProbe(
        ROOT, retinal_backend="signed_frame_difference", normalization=normalization
    )
    assert probe.conductance["all_targets"] > 6_700
    assert len(probe.conductance["targets"]) > 6_500
    for matrix in probe.conductance["matrices"].values():
        assert np.all(np.diff(matrix.indptr) > 0)


def test_v7_single_compartment_voltage_obeys_reversal_limits() -> None:
    config = _config()
    normalization = _collect_source_normalization(
        ROOT,
        retinal_backend="signed_frame_difference",
        retinal_geometry=config["primary_retinal_geometry"],
        config=config,
    )
    probe = V7PublishedConductanceProbe(
        ROOT, retinal_backend="signed_frame_difference", normalization=normalization
    )
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    voltage = probe._conductance_voltage(state)
    assert np.all(voltage >= -71.0)
    assert np.all(voltage <= -21.0)


def test_v7_published_conductance_artifact_is_hash_bound_and_non_advancing() -> None:
    artifact = json.loads(
        (ROOT / "artifacts/v7-t4-conductance-candidate.json").read_text(encoding="utf-8")
    )
    protocol = artifact["protocol"]
    assert (
        protocol["candidate_config_sha256"]
        == hashlib.sha256((ROOT / CONDUCTANCE_CONFIG).read_bytes()).hexdigest()
    )
    assert (
        protocol["candidate_implementation_sha256"]
        == hashlib.sha256((ROOT / CONDUCTANCE_IMPLEMENTATION).read_bytes()).hexdigest()
    )
    assert protocol["published_parameters_modified"] is False
    assert protocol["direction_labels_used_for_normalization"] is False
    assert artifact["passing_retinal_backends"] == []
    assert artifact["advance_to_central_complex"] is False
