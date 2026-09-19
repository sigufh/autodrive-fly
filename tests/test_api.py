import fly_emotion_api.main as api_module
from fastapi.testclient import TestClient
from fly_emotion_api.main import app


def test_health_reports_driving_task() -> None:
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["connectome"] == "male-cns:v1.0"
    assert response.json()["task"] == "visual-obstacle-and-city-driving"
    assert response.json()["language_model"] == "retired"


def test_health_does_not_mark_obsolete_checkpoint_ready(tmp_path, monkeypatch) -> None:
    import numpy as np

    monkeypatch.setattr(api_module, "ROOT", tmp_path)
    target = tmp_path / "artifacts/checkpoints/driving-policy.npz"
    target.parent.mkdir(parents=True)
    np.savez(target, format_version=np.asarray([2]))
    response = TestClient(app).get("/api/health")
    assert response.json()["policy_checkpoint_ready"] is False
    assert response.json()["policy_checkpoint_version"] == 2
    assert response.json()["required_policy_version"] == 5


def test_v7_status_is_hash_verified_and_explicitly_not_deployed() -> None:
    response = TestClient(app).get("/api/v7/status")
    assert response.status_code == 200
    status = response.json()
    assert status["version"] == "v7-experimental"
    assert status["source"] == "hash-verified-offline-goal-audit"
    assert status["current_stage"] == "controlled_vision"
    assert status["objective_complete"] is False
    assert status["deployment_enabled"] is False
    assert status["default_runtime_changed"] is False
    assert status["gates"]["T4_T5_direction_and_ON_OFF"] is False
    assert status["gates"]["LPLC1_near_collision"] is False
    assert status["gates"]["LPLC2_radial_opponency"] is False
    assert status["gates"]["LC4_angular_speed"] is False
    assert status["gates"]["EPG_PEN_PEG_heading"] is True
    assert status["gates"]["PFL3_DNa_transparent_mapping"] is True
    assert status["contributions"]["upper_planner"]["status"] == "paused"
    assert status["contributions"]["fly_local_core"][
        "active_v7_in_default_runtime"
    ] is False
    boundaries = status["evidence_boundaries"]
    assert boundaries["nine_source_contract_complete"] is False
    assert boundaries["Mi4_C3_direct_numeric_voltage_candidates"] == [
        "Groschner_2022"
    ]
    assert boundaries["Mi4_C3_independent_numeric_voltage_candidate_count"] == 0
    assert boundaries["T5_voltage_field_counts"] == {
        "aggregated_full_field_OFF_flash": 7,
        "raw_white_noise": 8,
        "raw_drifting_grating": 9,
    }
    assert boundaries["Braun_calcium_fly_counts"] == {
        "Tm2": 9,
        "Tm9": 11,
        "CT1": 9,
    }
    assert boundaries["Braun_calcium_condition_grids_complete"] is True
    assert boundaries["Braun_calcium_allowed_voltage_sources"] == []
    assert boundaries["Gou_Dryad_archive_hash_locally_verified"] is True
    assert boundaries["Gou_Dryad_local_processed_calcium_sources"] == [
        "Mi1",
        "Tm3",
        "Tm1",
        "Tm2",
    ]
    assert boundaries["Gou_Dryad_flash_fly_axis_sizes"] == {
        "Mi1": 16,
        "Tm3": 11,
        "Tm1": 9,
        "Tm2": 8,
    }
    assert boundaries["Gou_Dryad_moving_bar_fly_axis_sizes"] == {
        "Mi1": 10,
        "Tm3": 6,
        "Tm1": 8,
        "Tm2": 12,
    }
    assert boundaries["Gou_Dryad_stable_biological_individual_IDs_verified"] is False
    assert boundaries["Gou_Dryad_experimental_membrane_voltage"] is False
    assert boundaries["Gou_DANDI_asset_level_stable_participant_IDs_verified"] is True
    assert boundaries["Gou_DANDI_unique_subject_ID_count"] == 282
    assert boundaries["Gou_Dryad_distinct_fliesUsed_label_count"] == 66
    assert boundaries["Gou_Dryad_DANDI_identity_crosswalk_match_count"] == 0
    assert boundaries["Gou_Dryad_rows_to_DANDI_subject_crosswalk_verified"] is False
    assert boundaries["v7_offline_time_coordinate_contract_complete"] is True
    assert boundaries["v7_offline_horizontal_coordinate_contract_complete"] is True
    assert boundaries["v7_offline_two_dimensional_angular_calibration_complete"] is False
    assert boundaries["v7_offline_frame_interval_milliseconds"] == 10.0
    assert boundaries["v7_offline_substep_interval_milliseconds"] == 2.5
    assert boundaries["T4_source_mapping_mode"] == "exact_type_average"
    assert boundaries["T4_source_recording_level_body_assignment"] is False
    assert boundaries["T4_exact_type_average_mapping_complete"] is True
    assert boundaries["T4_author_minmax_formula_reproduced"] is True
    assert boundaries["T4_state_mapping_held_out_outside_fraction_by_source"] == {
        "Mi1": 0.1295625,
        "Tm3": 0.17478125,
        "Mi4": 0.2870446428571429,
        "C3": 0.18847916666666667,
    }
    assert boundaries["T4_author_minmax_semantics_match_v7_state"] is False
    assert boundaries["T4_millivolts_to_v7_state_mapping_available"] is False
    assert boundaries["T5_record_specific_stimulus_logs_available"] is False
    assert boundaries["Motyxia2_public_history_branch_count"] == 22
    assert boundaries["Motyxia2_public_history_commit_count"] == 447
    assert boundaries["T5_record_log_found_in_Motyxia2_public_history"] is False
    assert boundaries["T5_external_successful_indexes_linked_log_found"] is False
    assert boundaries["T5_PMC_supplement_content_inspected"] is False
    assert boundaries["T5_publisher_supplements_inspected"] is True
    assert boundaries["T5_publisher_supplements_contain_record_log"] is False
    assert boundaries["T5_Figshare_search_accessible"] is False
    assert boundaries["T5_stimulus_log_global_absence_claimed"] is False
    assert boundaries["T5_generator_defaults_used_as_record_fields"] is False
    assert boundaries["T5_stimulus_provenance_complete"] is False
    assert boundaries["CT1_audited_candidate_count"] == 11
    assert boundaries["CT1_incremental_2025_2026_candidate_count"] == 3
    assert boundaries["CT1_direct_experimental_voltage_candidate_found"] is False
    assert boundaries["Tm9_official_synapse_coordinate"] == [15, 2]
    assert boundaries["CT1_per_synapse_Lo1_columnar_retinotopy_available"] is True
    assert boundaries["CT1_complete_official_LO_column_coverage"] is False


def test_autonomy_status_alias_matches_hash_verified_v7_status() -> None:
    client = TestClient(app)
    v7 = client.get("/api/v7/status")
    autonomy = client.get("/api/autonomy/status")
    assert v7.status_code == 200
    assert autonomy.status_code == 200
    assert autonomy.json() == v7.json()


def test_v7_status_rejects_stale_evidence(tmp_path, monkeypatch) -> None:
    import hashlib
    import json

    monkeypatch.setattr(api_module, "ROOT", tmp_path)
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    report = {
        "protocol": {
            "dependencies_sha256": {
                "evidence.json": hashlib.sha256(evidence.read_bytes()).hexdigest()
            }
        },
        "checks": [],
        "summary": {},
    }
    target = tmp_path / "artifacts/v7-goal-audit.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(report))
    evidence.write_text("stale")
    response = TestClient(app).get("/api/v7/status")
    assert response.status_code == 503
    assert response.json()["detail"] == "Stale v7 audit dependency: evidence.json"
    alias = TestClient(app).get("/api/autonomy/status")
    assert alias.status_code == 503
    assert alias.json() == response.json()


def test_status_routes_report_invalid_audit_as_service_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "ROOT", tmp_path)
    target = tmp_path / "artifacts/v7-goal-audit.json"
    target.parent.mkdir(parents=True)
    target.write_text("{not-json", encoding="utf-8")
    client = TestClient(app)
    for route in ("/api/v7/status", "/api/autonomy/status"):
        response = client.get(route)
        assert response.status_code == 503
        assert response.json()["detail"] == "Invalid v7 goal audit"


def test_real_cached_skeleton_endpoint() -> None:
    response = TestClient(app).get("/api/skeleton/10001?max_edges=20000")
    assert response.status_code == 200
    assert response.json()["source_vertices"] == 2975


def test_real_connectome_assets() -> None:
    client = TestClient(app)
    overview = client.get("/api/connectome/overview").json()
    pathways = client.get("/api/connectome/pathways").json()
    assert overview["canonical_nodes"] == 166_700
    assert pathways["all_edges_accounted"] == 25_582_938


def test_driving_endpoints_are_stateful(monkeypatch) -> None:
    calls = []

    class FakeEngine:
        def state(self, **_):
            return {"environment": {"step": len(calls)}}

        def reset(self, seed, *, keep_learning, scenario, control_mode):
            calls.append(("reset", seed, keep_learning, scenario, control_mode))
            return self.state()

        def set_control_mode(self, control_mode):
            calls.append(("control_mode", control_mode))

        def step(self, *, learning, explore, safety_constraints=True, include_activity=True):
            calls.append(("step", learning, explore, safety_constraints))
            return {"environment": {"step": len(calls), "done": False}}

    monkeypatch.setattr(api_module, "engine", lambda: FakeEngine())
    client = TestClient(app)
    reset = client.post(
        "/api/driving/reset", json={"seed": 9, "keep_learning": False}
    )
    assert reset.status_code == 200
    response = client.post("/api/driving/step", json={"steps": 2, "learning": True})
    assert response.json()["environment"]["step"] == 3
    assert calls == [
        ("reset", 9, False, "highway", "assisted"),
        ("step", True, False, True),
        ("step", True, False, True),
    ]


def test_driving_step_defaults_to_frozen_execution(monkeypatch) -> None:
    calls = []

    class FakeEngine:
        def step(self, *, learning, explore, safety_constraints=True, include_activity=True):
            calls.append((learning, explore, safety_constraints))
            return {"environment": {"done": True}}

    monkeypatch.setattr(api_module, "engine", lambda: FakeEngine())
    response = TestClient(app).post("/api/driving/step", json={})
    assert response.status_code == 200
    assert calls == [(False, False, True)]


def test_engine_is_singleton_under_concurrent_first_load(monkeypatch) -> None:
    import threading
    import time
    created = []

    class FakeDrivingEngine:
        def __init__(self, *_args, **_kwargs):
            time.sleep(0.02)
            created.append(self)

    monkeypatch.setattr(api_module, "DrivingEngine", FakeDrivingEngine)
    monkeypatch.setattr(api_module, "_ENGINE", None)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(api_module.engine())) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(created) == 1 and len({id(value) for value in results}) == 1
    monkeypatch.setattr(api_module, "_ENGINE", None)
