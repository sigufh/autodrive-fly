import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/freeze_malecns_tm4_offset_replication.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tm4_replication_freezer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_http_reader_caches_exact_range(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    run = Mock(return_value=subprocess.CompletedProcess([], 0, b"abcd", b""))
    monkeypatch.setattr(module.subprocess, "run", run)
    reader = module._http_reader("https://example.invalid/data")
    assert reader("001.shard", 4, 8) == b"abcd"
    assert reader("001.shard", 4, 8) == b"abcd"
    assert run.call_count == 1
    assert "4-7" in run.call_args.args[0]


def test_http_reader_retries_and_rejects_short_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    run = Mock(return_value=subprocess.CompletedProcess([], 0, b"abc", b""))
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="short HTTP range read"):
        module._http_reader("https://example.invalid/data")("001.shard", 4, 8)
    assert run.call_count == 3


def test_cached_parts_are_bound_to_body_protocol_and_implementation(tmp_path: Path) -> None:
    module = _module()
    part = tmp_path / "12.json"
    payload = {
        "body_id": 12,
        "preregistration_sha256": "pre",
        "retrieval_implementation_sha256": "implementation",
    }
    module._write_json_atomic(part, payload)
    assert not part.with_suffix(".json.part").exists()
    assert json.loads(part.read_text()) == payload
    assert module._load_cached_part(part, 12, "pre", "implementation") == payload
    with pytest.raises(ValueError, match="identity mismatch"):
        module._load_cached_part(part, 13, "pre", "implementation")
    with pytest.raises(ValueError, match="preregistration mismatch"):
        module._load_cached_part(part, 12, "other", "implementation")
    with pytest.raises(ValueError, match="implementation mismatch"):
        module._load_cached_part(part, 12, "pre", "other")
