import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-loo-preregistration.json"


def test_T4_source_LOO_preregistration_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["frozen_diagnostic_commit"] == (
        "eadb28847f6e8714d77eaa674c7f9e922c2cb84d"
    )
    assert report["protocol"]["ablation_outputs_observed"] is False


def test_T4_source_LOO_conditions_and_scoring_are_fixed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["source_types"] == ["Mi1", "Tm3", "Mi4", "C3"]
    assert report["source_groups"] == {
        "fast": ["Mi1", "Tm3"],
        "delayed": ["Mi4", "C3"],
    }
    assert report["ablation_conditions"] == [
        "intact",
        "drop_Mi1",
        "drop_Tm3",
        "drop_Mi4",
        "drop_C3",
    ]
    assert report["modes"] == ["ordered", "temporal_shuffle", "static_sham"]
    assert report["temporal_reductions"] == ["positive_peak", "signed_mean"]
    assert report["scoring_contract"]["fixed_T4_population_denominator"] == 6861
    assert report["scoring_contract"]["valid_crossfit_target_count"] == 6749


def test_T4_source_LOO_preregistration_does_not_authorize_changes() -> None:
    report = json.loads(REPORT.read_text())
    assert report["ablation_protocol_frozen"] is True
    assert report["ablation_evaluated"] is False
    assert report["interpretation"]["no_new_acceptance_threshold"] is True
    assert report["authorize_source_removal"] is False
    assert report["authorize_new_T4_functional_candidate"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
