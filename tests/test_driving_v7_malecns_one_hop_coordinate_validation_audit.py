import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-malecns-one-hop-coordinate-validation-audit.json"


def test_one_hop_validation_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_blind_replay_exposes_source_type_specific_accuracy() -> None:
    report = json.loads(REPORT.read_text())
    sources = report["source_results"]
    assert sources["Tm3"]["native_reference_count"] == 0
    assert sources["Tm3"]["same_type_native_validation_available"] is False
    assert sources["Tm4"]["native_reference_count"] == 833
    assert sources["Tm4"]["overall"]["rounded_exact_fraction"] < 0.32
    assert sources["Tm4"]["overall"]["p95_euclidean_hex_error"] > 1.0
    for source in ("Mi1", "Mi4", "C3", "Tm1", "Tm2", "Tm9"):
        assert sources[source]["overall"]["rounded_exact_fraction"] > 0.98


def test_inferred_coordinates_are_not_promoted_to_native_mapping() -> None:
    report = json.loads(REPORT.read_text())
    assert (
        report["every_inference_dependent_source_has_same_type_native_validation"]
        is False
    )
    assert report["Tm3_same_type_native_validation_available"] is False
    assert report["one_hop_coordinates_are_native_equivalent"] is False
    assert report["authorize_coordinate_rule_as_experimental_mapping"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
