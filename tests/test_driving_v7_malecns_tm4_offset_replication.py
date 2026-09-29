import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

import fly_emotion.driving.v7_malecns_tm4_offset_replication as replication
from fly_emotion.driving.v7_malecns_tm4_offset_replication import (
    _one_sided_exact_sign_pvalue,
    _score_candidate,
    _summarize_records,
)


def test_one_sided_exact_sign_test_uses_only_discordant_pairs() -> None:
    assert _one_sided_exact_sign_pvalue(7, 0) == 0.0078125
    assert _one_sided_exact_sign_pvalue(6, 0) == 0.015625
    assert _one_sided_exact_sign_pvalue(0, 0) == 1.0
    assert _one_sided_exact_sign_pvalue(7, 1) == 9 / 256


def test_fixed_correction_is_coordinate_addition() -> None:
    native = [4, -2]
    assert _score_candidate([5, -1], native, [0, 0]) == {
        "hex": [5, -1],
        "exact_native_match": False,
        "hex_distance_to_native": 2,
    }
    assert _score_candidate([5, -1], native, [-1, -1]) == {
        "hex": native,
        "exact_native_match": True,
        "hex_distance_to_native": 0,
    }


def test_missing_or_tied_candidate_cannot_be_scored_as_exact() -> None:
    assert _score_candidate(None, [1, 2], [-1, -1]) == {
        "hex": None,
        "exact_native_match": False,
        "hex_distance_to_native": None,
    }


def _record(unique: bool, uncorrected: bool, corrected: bool) -> dict:
    return {
        "unique_candidate": unique,
        "uncorrected": {"exact_native_match": uncorrected},
        "corrected": {"exact_native_match": corrected},
    }


def test_replication_gate_requires_every_preregistered_clause() -> None:
    passed = _summarize_records(
        [_record(True, False, True) for _ in range(7)], 7, 0.01
    )
    assert passed["replication_gate_passed"] is True
    assert all(passed["gates"].values())

    tie = _summarize_records(
        [_record(False, False, True), *[_record(True, False, True) for _ in range(6)]],
        7,
        0.01,
    )
    assert tie["gates"]["unique_synapse_count_candidate_for_every_body"] is False
    assert tie["replication_gate_passed"] is False

    worsened = _summarize_records(
        [
            *[_record(True, False, True) for _ in range(7)],
            _record(True, True, False),
        ],
        8,
        0.01,
    )
    assert worsened["gates"]["no_correct_to_incorrect_pairs"] is False
    assert worsened["replication_gate_passed"] is False

    underpowered = _summarize_records(
        [_record(True, False, True) for _ in range(6)], 6, 0.01
    )
    assert underpowered["gates"]["paired_one_sided_exact_sign_test_passed"] is False
    assert underpowered["replication_gate_passed"] is False


def test_passing_replication_never_authorizes_writeback(
    tmp_path: Path, monkeypatch,
) -> None:
    ids = list(range(1, 8))
    rules = {
        "uncorrected": [0, 0],
        "discovery_offset_mode_correction": [-1, -1],
    }
    gate = {
        "require_all_sampled_bodies_retrieved": True,
        "require_unique_synapse_count_candidate_for_every_body": True,
        "require_corrected_exact_count_greater_than_uncorrected_exact_count": True,
        "require_no_correct_to_incorrect_pairs": True,
        "paired_one_sided_exact_sign_test_alpha": 0.01,
        "no_distance_tolerance_acceptance_gate": True,
    }
    prereg_path = Path("prereg.json")
    retrieval_path = Path("retrieval.py")
    snapshot_path = Path("snapshot.json")
    annotation_path = Path("annotations.feather")
    implementation_path = Path("implementation.py")
    config_path = Path("config.yaml")
    prereg = {
        "replication_protocol_frozen": True,
        "replication_sample_body_ids": ids,
        "candidate_rules": rules,
        "replication_gate": gate,
        "protocol": {"replication_outputs_observed": False},
        "boundary": {
            "replication_pass_does_not_authorize_left_Tm4_writeback": True
        },
    }
    (tmp_path / prereg_path).write_text(json.dumps(prereg))
    (tmp_path / retrieval_path).write_text("# frozen retrieval\n")
    (tmp_path / implementation_path).write_text("# evaluator\n")
    (tmp_path / annotation_path).write_bytes(b"frozen annotations")
    snapshot = {
        "protocol": {
            "preregistration_sha256": hashlib.sha256(
                (tmp_path / prereg_path).read_bytes()
            ).hexdigest(),
            "retrieval_implementation_sha256": hashlib.sha256(
                (tmp_path / retrieval_path).read_bytes()
            ).hexdigest(),
            "body_count": len(ids),
        },
        "rows": [{"body_id": body_id} for body_id in ids],
        "column_pin_map": {},
    }
    (tmp_path / snapshot_path).write_text(json.dumps(snapshot))
    config = {
        "name": "synthetic-Tm4-replication",
        "observed_on": "never",
        "preregistration_evidence": str(prereg_path),
        "preregistration_commit": "frozen",
        "preregistration_sha256": hashlib.sha256(
            (tmp_path / prereg_path).read_bytes()
        ).hexdigest(),
        "retrieval_implementation": str(retrieval_path),
        "retrieval_implementation_commit": "frozen",
        "retrieval_implementation_sha256": hashlib.sha256(
            (tmp_path / retrieval_path).read_bytes()
        ).hexdigest(),
        "annotations": {
            "path": str(annotation_path),
            "bytes": (tmp_path / annotation_path).stat().st_size,
            "sha256": hashlib.sha256(
                (tmp_path / annotation_path).read_bytes()
            ).hexdigest(),
        },
        "replication_snapshot": {
            "path": str(snapshot_path),
            "observed": True,
            "bytes": (tmp_path / snapshot_path).stat().st_size,
            "sha256": hashlib.sha256(
                (tmp_path / snapshot_path).read_bytes()
            ).hexdigest(),
        },
        "candidate_rules": rules,
        "replication_gate": gate,
    }
    (tmp_path / config_path).write_text(yaml.safe_dump(config))
    table = pa.table(
        {
            "bodyId": ids,
            "type": ["Tm4"] * len(ids),
            "somaSide": ["R"] * len(ids),
            "assignedOlHex1": [0] * len(ids),
            "assignedOlHex2": [0] * len(ids),
        }
    )
    monkeypatch.setattr(replication, "CONFIG", config_path)
    monkeypatch.setattr(replication, "IMPLEMENTATION", implementation_path)
    frozen_hashes = {
        str(prereg_path): config["preregistration_sha256"],
        str(retrieval_path): config["retrieval_implementation_sha256"],
    }
    monkeypatch.setattr(
        replication, "_git_sha256", lambda *args: frozen_hashes[args[-1]]
    )
    monkeypatch.setattr(replication.feather, "read_table", lambda *args, **kwargs: table)
    monkeypatch.setattr(
        replication,
        "_official_synapse_count_candidate",
        lambda row, pin_map, side: {
            "unique_mode": True,
            "recovered_hex": [1, 1],
        },
    )

    report = replication.evaluate_v7_malecns_tm4_offset_replication(tmp_path)
    assert report["replication_gate_passed"] is True
    assert report["corrected_exact_count"] == 7
    assert report["paired_exact_outcomes"]["one_sided_exact_sign_test_pvalue"] == (
        0.0078125
    )
    assert report["authorize_left_Tm4_coordinate_writeback"] is False
    assert report["authorize_source_mapping_gate_change"] is False
    assert report["advance_to_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False


def test_unobserved_snapshot_fails_closed(tmp_path: Path, monkeypatch) -> None:
    config_path = Path("config.yaml")
    prereg_path = Path("prereg.json")
    retrieval_path = Path("retrieval.py")
    (tmp_path / prereg_path).write_text("{}")
    (tmp_path / retrieval_path).write_text("# retrieval\n")
    prereg_sha = hashlib.sha256((tmp_path / prereg_path).read_bytes()).hexdigest()
    retrieval_sha = hashlib.sha256((tmp_path / retrieval_path).read_bytes()).hexdigest()
    config = {
        "preregistration_evidence": str(prereg_path),
        "preregistration_commit": "frozen",
        "preregistration_sha256": prereg_sha,
        "retrieval_implementation": str(retrieval_path),
        "retrieval_implementation_commit": "frozen",
        "retrieval_implementation_sha256": retrieval_sha,
        "replication_gate": {},
        "replication_snapshot": {
            "path": "absent.json",
            "observed": False,
            "bytes": None,
            "sha256": None,
        },
    }
    (tmp_path / config_path).write_text(yaml.safe_dump(config))
    prereg = {
        "replication_protocol_frozen": True,
        "replication_gate": {},
        "protocol": {"replication_outputs_observed": False},
    }
    (tmp_path / prereg_path).write_text(json.dumps(prereg))
    config["preregistration_sha256"] = hashlib.sha256(
        (tmp_path / prereg_path).read_bytes()
    ).hexdigest()
    (tmp_path / config_path).write_text(yaml.safe_dump(config))
    monkeypatch.setattr(replication, "CONFIG", config_path)
    monkeypatch.setattr(
        replication,
        "_git_sha256",
        lambda *args: (
            config["preregistration_sha256"]
            if args[-1] == str(prereg_path)
            else retrieval_sha
        ),
    )
    with pytest.raises(ValueError, match="snapshot has not been observed"):
        replication.evaluate_v7_malecns_tm4_offset_replication(tmp_path)
