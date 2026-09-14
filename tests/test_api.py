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
