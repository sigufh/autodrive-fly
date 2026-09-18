import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-edmond-fig3-retrieval-audit.json"


def test_retrieval_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["retrieval_helper"]["workflow_trigger"] == "manual_only"
    assert report["retrieval_helper"]["workflow_artifact_retention_days"] == 1


def test_four_file_manifest_is_exact_and_cross_checked() -> None:
    report = json.loads(REPORT.read_text())
    manifest = report["frozen_file_manifest"]
    assert {name: item["id"] for name, item in manifest.items()} == {
        "fig3_Tm3.npy": 104816,
        "fig3_Mi1.npy": 104811,
        "fig3_Mi4.npy": 104812,
        "fig3_C3.npy": 104809,
    }
    assert all(item["dtype"] == "float64" for item in manifest.values())
    assert all(item["shape"][0] == 2 and item["shape"][2] == 8000 for item in manifest.values())
    assert report["manifest_cross_check"][
        "all_four_files_match_prior_config_report_and_headers"
    ] is True
    assert {name: item["storage_identifier"] for name, item in manifest.items()} == {
        "fig3_Tm3.npy": "54://edmond-objstor-prod:17f0cb650f4-f91d07bbefa2",
        "fig3_Mi1.npy": "54://edmond-objstor-prod:17f0cb645cf-a96b88e2ea5d",
        "fig3_Mi4.npy": "54://edmond-objstor-prod:17f0cb64779-b6b6bcb4d49c",
        "fig3_C3.npy": "54://edmond-objstor-prod:17f0cb642f6-202dfaea2a06",
    }
    history = report["historical_manifest_observation"]
    assert history["dataset_id"] == 104764
    assert history["dataset_version_id"] == 398
    assert history["version_state"] == "RELEASED"
    assert history["response_bytes"] == 61600
    assert history["response_sha256"] == (
        "a61613ae5a62f25b07c5a961a967effab74ff68fbade90aae740536ce192c54a"
    )


def test_network_unavailability_does_not_become_scientific_rejection() -> None:
    report = json.loads(REPORT.read_text())
    assert report["datacite_observation"]["response_status"] == 200
    assert report["datacite_observation"]["content_url_present"] is False
    assert report["mirror_search"]["verified_candidate_count"] == 0
    assert report["gates"]["scientific_source_rejected"] is False
    assert report["boundary"]["network_failure_is_not_scientific_rejection"] is True


def test_missing_payload_cannot_be_replaced_with_workbook_interpolation() -> None:
    report = json.loads(REPORT.read_text())
    assert all(not item["fully_verified"] for item in report["local_candidates"].values())
    boundary = report["workbook_resolution_boundary"]
    assert boundary["sample_interval_milliseconds"] == 10.0
    assert boundary["repository_array_interval_milliseconds"] == 1.0
    assert boundary["workbook_is_repository_array"] is False
    assert boundary["interpolation_authorized_as_repository_array"] is False
    assert report["gates"]["full_resolution_fixed_split_recompute_authorized"] is False
    assert report["prior_fixed_individual_split"]["passing_source_conditions"] == ["on:Tm3"]


def test_all_downstream_gates_remain_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"][
        "T4_T5_LPLC_vehicle_MB_OOD_final_and_deployment_remain_frozen"
    ] is True
