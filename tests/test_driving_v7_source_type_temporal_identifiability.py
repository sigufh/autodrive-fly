import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-source-type-temporal-identifiability.json"


def test_source_type_temporal_audit_is_hash_bound_and_tuning_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False


def test_each_source_type_fails_temporal_identifiability() -> None:
    report = json.loads(REPORT.read_text())
    assert report["families"]["T4"]["source_population_denominator"] == 32
    assert report["families"]["T5"]["source_population_denominator"] == 32
    for family in report["families"].values():
        for population in family["populations"].values():
            for source in population["source_types"].values():
                assert source["comparison_count"] == 4 * population["target_count"]
                assert source["shuffle_to_ordered_residual_energy_ratio"] > 0.5
                assert source["passed"] is False
    assert report["source_type_temporal_identifiability_passed"] is False
    assert report["authorize_source_time_constant_candidate"] is False
    assert report["advance_to_three_tuning_conditions"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
