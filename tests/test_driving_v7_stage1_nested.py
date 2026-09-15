import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from fly_emotion.driving.v7_stage1_nested import (
    CONFIG,
    build_condition_bundle,
    evaluate_v7_stage1_nested,
)

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-stage1-nested.json"


def test_nested_protocol_has_exactly_three_tuning_one_calibration_one_final() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    roles = [item["role"] for item in config["conditions"]]
    assert len(roles) == 5
    assert roles.count("tuning") == 3
    assert roles.count("calibration") == 1
    assert roles.count("external_final") == 1
    final = next(item for item in config["conditions"] if item["role"] == "external_final")
    assert "generator_parameters" not in final
    assert final["external_manifest"]["retrieval_authorized"] is False
    assert final["external_manifest"]["consumed"] is False


def test_local_bundles_are_complete_disjoint_and_mirrored() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    raw_hashes: set[str] = set()
    for condition in config["conditions"][:-1]:
        stimuli = build_condition_bundle(condition)
        assert len(stimuli) == 23
        assert {item.family for item in stimuli} == {
            "uniform",
            "moving_edge",
            "looming",
            "static",
            "translation",
            "rotation",
        }
        by_identity = {item.identity: item for item in stimuli}
        assert all(
            np.array_equal(item.frames[:, :, ::-1], by_identity[item.mirror_of].frames)
            for item in stimuli
        )
        current = {item.sha256 for item in stimuli}
        assert not current & raw_hashes
        raw_hashes |= current


def test_legacy_is_regression_only_and_cannot_authorize_any_gate() -> None:
    report = json.loads(REPORT.read_text())
    legacy = report["legacy_regression"]
    assert legacy["stimulus_count"] == 172
    assert (
        legacy["aggregate_sha256"]
        == "f0248b99e93efe102aff2b2a4f29663041d808680b80b8afaf61b59bc126172e"
    )
    assert legacy["parameter_selection_eligible"] is False
    assert legacy["calibration_eligible"] is False
    assert legacy["scientific_gate_eligible"] is False
    assert legacy["score_weight"] == 0
    assert report["calibration_authorized"] is False
    assert report["final_authorized"] is False


def test_nested_artifact_is_fresh_and_final_is_not_locally_available() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert len(report["local_condition_manifests"]) == 4
    assert all(item["stimulus_count"] == 23 for item in report["local_condition_manifests"])
    assert report["external_final"]["locally_generatable"] is False
    assert report["external_final"]["committed"] is False
    assert report["external_final"]["evaluated"] is False
    assert report["protocol"]["model_evaluated"] is False
    assert report["protocol"]["parameters_fitted"] is False


def test_external_final_bundle_cannot_be_generated() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    with pytest.raises(ValueError, match="only local"):
        build_condition_bundle(config["conditions"][-1])


def test_artifact_rebuild_is_deterministic() -> None:
    assert (
        evaluate_v7_stage1_nested(ROOT)["local_condition_manifests"]
        == json.loads(REPORT.read_text())["local_condition_manifests"]
    )
