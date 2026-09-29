import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-pair-preregistration.json"


def test_T4_source_pair_preregistration_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["frozen_LOO_commit"] == (
        "2acd1ed8b9b77295ec017bd54b365a44273c53e1"
    )
    assert report["protocol"]["pair_outputs_observed"] is False


def test_T4_source_pair_grid_is_complete_and_fixed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["pair_conditions"] == [
        {"name": "Mi1_x_Mi4", "fast": "Mi1", "delayed": "Mi4"},
        {"name": "Mi1_x_C3", "fast": "Mi1", "delayed": "C3"},
        {"name": "Tm3_x_Mi4", "fast": "Tm3", "delayed": "Mi4"},
        {"name": "Tm3_x_C3", "fast": "Tm3", "delayed": "C3"},
    ]
    assert report["modes"] == ["ordered", "temporal_shuffle", "static_sham"]
    assert report["temporal_reductions"] == ["positive_peak", "signed_mean"]
    assert report["scoring_contract"]["fixed_T4_population_denominator"] == 6861
    assert report["scoring_contract"]["valid_crossfit_target_count"] == 6749


def test_T4_source_pair_preregistration_does_not_authorize_changes() -> None:
    report = json.loads(REPORT.read_text())
    assert report["pair_protocol_frozen"] is True
    assert report["pair_evaluated"] is False
    assert report["interpretation"]["no_new_acceptance_threshold"] is True
    assert report["authorize_source_pair_candidate"] is False
    assert report["authorize_source_removal"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
