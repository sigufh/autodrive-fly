import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_stage1_split import CONFIG, build_stage1_split

ROOT = Path(__file__).parents[1]


def test_stage1_split_is_deterministic_disjoint_and_exactly_mirrored() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    first = build_stage1_split(config)
    second = build_stage1_split(config)
    assert set(first) == {"development", "validation", "ood", "final"}
    all_identities = []
    split_hashes = {}
    for name, stimuli in first.items():
        assert len(stimuli) == 172
        assert {item.family for item in stimuli} == {
            "uniform",
            "moving_edge",
            "looming",
            "static",
            "translation",
            "rotation",
        }
        assert [item.sha256 for item in stimuli] == [item.sha256 for item in second[name]]
        assert len({item.identity for item in stimuli}) == len(stimuli)
        by_identity = {item.identity: item for item in stimuli}
        assert all(
            np.array_equal(item.frames[:, :, ::-1], by_identity[item.mirror_of].frames)
            for item in stimuli
        )
        for item in stimuli:
            if item.family == "static":
                assert (
                    float(np.mean(item.frames[0])) < 0.2
                    if item.polarity == "on"
                    else float(np.mean(item.frames[0])) > 0.8
                )
                assert not np.array_equal(item.frames[0], item.frames[1])
        all_identities.extend(item.identity for item in stimuli)
        split_hashes[name] = {item.sha256 for item in stimuli}
    assert len(all_identities) == len(set(all_identities))
    names = list(split_hashes)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert not split_hashes[left] & split_hashes[right]


def test_stage1_split_timebase_is_engineering_only() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    timebase = config["engineering_timebase"]
    assert timebase["frame_interval_milliseconds"] == 10.0
    assert timebase["nominal_substep_interval_milliseconds"] == 2.5
    assert timebase["applied_to_current_runtime"] is False
    assert timebase["biologically_calibrated"] is False
    assert config["release_boundary"]["no_target_activity_injection"] is True
    assert config["release_boundary"]["no_parameter_fit_in_this_protocol"] is True


def test_saved_stage1_split_reserves_but_does_not_claim_blinded_final() -> None:
    report = json.loads((ROOT / "artifacts/v7-stage1-split.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["model_evaluated"] is False
    assert report["protocol"]["parameters_fitted"] is False
    assert all(report["exact_mirror_checks"].values())
    assert all(
        values["identity_overlap"] == values["frame_hash_overlap"] == 0
        for values in report["cross_split_overlap"].values()
    )
    final = report["split_manifests"]["final"]
    assert final["evaluable"] is False
    assert final["stimuli"] is None
    assert final["reserved"] is True
    assert final["blinded"] is False
    assert final["one_time_test"] is False
    assert final["evaluated"] is False
    assert len(final["aggregate_sha256"]) == 64
    assert report["advance_to_model_fit"] is False
    assert report["advance_to_final_test"] is False
    assert report["advance_to_central_complex"] is False
