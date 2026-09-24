import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-malecns-tm4-offset-replication-preregistration.json"


def test_Tm4_offset_preregistration_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["frozen_discovery_commit"] == (
        "23a796eefd506816786e4cc8826ab65c859928e0"
    )
    assert report["protocol"]["replication_outputs_observed"] is False


def test_replication_sample_is_independent_and_fixed() -> None:
    report = json.loads(REPORT.read_text())
    assert len(report["discovery_excluded_body_ids"]) == 48
    assert report["replication_universe_count"] == 785
    assert len(report["replication_sample_body_ids"]) == 96
    assert not set(report["discovery_excluded_body_ids"]) & set(
        report["replication_sample_body_ids"]
    )
    assert report["candidate_rules"] == {
        "uncorrected": [0, 0],
        "discovery_offset_mode_correction": [-1, -1],
    }


def test_preregistration_does_not_evaluate_or_authorize() -> None:
    report = json.loads(REPORT.read_text())
    assert report["replication_protocol_frozen"] is True
    assert report["replication_evaluated"] is False
    assert report["authorize_left_Tm4_coordinate_writeback"] is False
    assert report["authorize_source_mapping_gate_change"] is False
    assert report["advance_to_T4_T5_functional_precheck"] is False
    assert report["boundary"]["replication_pass_does_not_authorize_left_Tm4_writeback"] is True
