import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-pirogova-source-calcium-audit.json"


def test_pirogova_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["repository"]["historical_data_commit"] == (
        "e320792a88e4f5a7d467dccc107e3f1010fe1db2"
    )
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["pickle_security"]["restricted_unpickler_used"] is True
    assert report["pickle_security"]["arbitrary_repository_code_executed"] is False


def test_historical_snapshot_recovers_all_four_numeric_calcium_sources() -> None:
    report = json.loads(REPORT.read_text())
    assert report["history_recovery"]["latest_head_missing_historical_source_files"] == [
        "Tm1_data.pkl",
        "Tm2_data.pkl",
    ]
    assert report["source_coverage"]["covered_required_sources"] == [
        "Mi1",
        "Tm1",
        "Tm2",
        "Tm3",
    ]
    expected_cells = {"Mi1": 98, "Tm3": 65, "Tm1": 24, "Tm2": 22}
    for source, count in expected_cells.items():
        payload = report["source_payloads"][source]
        assert payload["cell_trial_column_group_count"] == count
        assert payload["paper_reported_cell_count"] == count
        assert payload["trial_labels"] == ["0", "1", "2"]
        assert payload["time_sample_count"] == 91
        assert abs(payload["effective_sample_rate_hz"] - 11.9) < 1e-12


def test_cell_labels_are_not_promoted_to_fly_identity_or_voltage() -> None:
    report = json.loads(REPORT.read_text())
    assert report["source_payloads"]["Mi1"][
        "cell_labels_globally_unique_within_payload"
    ] is False
    assert report["source_payloads"]["Tm3"][
        "cell_labels_globally_unique_within_payload"
    ] is False
    for payload in report["source_payloads"].values():
        assert payload["paper_reported_cell_count"] > payload["paper_reported_fly_count"]
        assert payload["cell_to_fly_mapping_available"] is False
        assert payload["biological_individual_id_available"] is False
        assert payload["response_unit_contract_satisfied"] is False
    assert report["transfer_gates"]["allowed_membrane_voltage_response_unit"] is False
    assert report["source_dynamics_transfer_authorized"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
