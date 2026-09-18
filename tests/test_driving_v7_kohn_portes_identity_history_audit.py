import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-identity-history-audit.json"


def test_Kohn_Portes_identity_history_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["repository_history"]["commit_count"] == 20
    assert report["repository_history"]["historical_unique_path_count"] == 111
    assert report["repository_history"]["identity_sidecar_paths"] == []


def test_full_history_expands_recording_key_upper_bounds_without_fly_identity() -> None:
    report = json.loads(REPORT.read_text())
    expected = {"Tm1": 8, "Tm2": 7, "Tm4": 7, "Tm9": 13}
    for source, count in expected.items():
        row = report["source_capacity"][source]
        assert row["full_repository_unique_recording_id_upper_bound"] == count
        assert row["biological_individual_count_verified"] is False
        assert row["biological_individual_5_plus_3_split_authorized"] is False
    assert report["source_capacity"]["Tm1"][
        "recording_id_upper_bound_meets_numeric_5_plus_3"
    ] is True
    assert report["source_capacity"]["Tm9"][
        "recording_id_upper_bound_meets_numeric_5_plus_3"
    ] is True
    assert report["source_capacity"]["Tm2"][
        "recording_id_upper_bound_meets_numeric_5_plus_3"
    ] is False
    assert report["source_capacity"]["Tm4"][
        "recording_id_upper_bound_meets_numeric_5_plus_3"
    ] is False


def test_recording_key_is_not_promoted_to_biological_individual() -> None:
    report = json.loads(REPORT.read_text())
    semantics = report["author_identity_semantics"]
    assert semantics["recording_id_is_used_as_cell_aggregation_key"] is True
    assert semantics["multiple_subrecording_numbers_per_recording_id_documented"] is True
    assert semantics["explicit_biological_fly_id_field_in_payloads"] is False
    assert semantics["paper_states_one_recording_cell_per_fly"] is False
    assert semantics["recording_id_promoted_to_biological_individual_id"] is False
    assert report["transfer_gates"]["biological_individual_semantics_verified"] is False
    assert report["T5_biological_individual_split_authorized"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
