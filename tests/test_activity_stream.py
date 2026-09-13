import json
from types import SimpleNamespace

import fly_emotion_api.main as api_module


def test_driving_stream_is_lazy(monkeypatch):
    calls = []

    def step(**_):
        calls.append("step")
        return {"environment": {"done": len(calls) == 2}}

    monkeypatch.setattr(api_module, "engine", lambda: SimpleNamespace(step=step))
    monkeypatch.setattr(api_module.time, "sleep", lambda _: None)
    events = api_module.driving_event_lines(api_module.RunRequest(max_steps=3))
    assert calls == []
    assert json.loads(next(events))["environment"]["done"] is False
    assert calls == ["step"]
    assert json.loads(next(events))["environment"]["done"] is True
    assert calls == ["step", "step"]
    assert json.loads(next(events)) == {"type": "done"}


def test_driving_stream_reports_model_error(monkeypatch):
    def broken(**_):
        raise ValueError("incompatible graph")

    monkeypatch.setattr(api_module, "engine", lambda: SimpleNamespace(step=broken))
    events = api_module.driving_event_lines(api_module.RunRequest(max_steps=1))
    assert json.loads(next(events)) == {
        "type": "error", "message": "incompatible graph"
    }
