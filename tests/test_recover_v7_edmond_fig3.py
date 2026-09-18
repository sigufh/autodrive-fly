import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/recover_v7_edmond_fig3.py"
SPEC = importlib.util.spec_from_file_location("recover_v7_edmond_fig3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest(path: Path, payload: bytes) -> None:
    array_path = path.parent / "source.npy"
    array_path.write_bytes(payload)
    values = np.load(array_path, allow_pickle=False)
    report = {
        "frozen_file_manifest": {
            "source.npy": {
                "id": 7,
                "bytes": len(payload),
                "md5": hashlib.md5(payload).hexdigest(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
        }
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    array_path.unlink()


def test_recovery_accepts_only_fully_verified_payload(tmp_path, monkeypatch) -> None:
    source = tmp_path / "valid.npy"
    np.save(source, np.arange(12, dtype=np.float64).reshape(2, 2, 3))
    payload = source.read_bytes()
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, payload)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size=-1):
            if not hasattr(self, "position"):
                self.position = 0
            chunk = payload[self.position : self.position + size]
            self.position += len(chunk)
            return chunk

    monkeypatch.setattr(MODULE.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    output = tmp_path / "output"
    result = MODULE.recover(manifest, output, ["https://example.invalid"], 1.0)
    assert result["all_four_files_verified"] is True
    assert result["files"]["source.npy"]["payload_hash_verified"] is True
    assert (output / "source.npy").read_bytes() == payload


def test_recovery_rejects_wrong_bytes_without_promoting_part_file(tmp_path, monkeypatch) -> None:
    source = tmp_path / "valid.npy"
    np.save(source, np.arange(12, dtype=np.float64).reshape(2, 2, 3))
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, source.read_bytes())

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size=-1):
            if getattr(self, "done", False):
                return b""
            self.done = True
            return b"not-the-frozen-array"

    monkeypatch.setattr(MODULE.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    output = tmp_path / "output"
    result = MODULE.recover(manifest, output, ["https://example.invalid"], 1.0)
    assert result["all_four_files_verified"] is False
    assert result["files"]["source.npy"]["attempts"][0]["payload_hash_verified"] is False
    assert not (output / "source.npy").exists()
    assert not list(output.glob("*.part"))
