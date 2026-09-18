#!/usr/bin/env python3
"""Retrieve the frozen public Edmond Fig. 3 arrays without weakening identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: dict) -> dict:
    actual = {
        "bytes": path.stat().st_size,
        "md5": _digest(path, "md5"),
        "sha256": _digest(path, "sha256"),
    }
    actual["payload_hash_verified"] = all(
        actual[key] == expected[key] for key in ("bytes", "md5", "sha256")
    )
    if not actual["payload_hash_verified"]:
        actual["fully_verified"] = False
        return actual
    values = np.load(path, allow_pickle=False)
    actual.update(
        {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "finite_fraction": float(np.isfinite(values).mean()),
        }
    )
    actual["fully_verified"] = (
        actual["shape"] == expected["shape"] and actual["dtype"] == expected["dtype"]
    )
    return actual


def recover(manifest_path: Path, output: Path, endpoints: list[str], timeout: float) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["frozen_file_manifest"]
    output.mkdir(parents=True, exist_ok=True)
    result = {"endpoints": endpoints, "files": {}}
    for name, expected in manifest.items():
        target = output / name
        if target.is_file():
            existing = _verify(target, expected)
            if existing["fully_verified"]:
                result["files"][name] = {"source": "existing", **existing}
                continue
        attempts = []
        verified = None
        for endpoint in endpoints:
            url = (
                endpoint.format(id=expected["id"])
                if "{id}" in endpoint
                else f"{endpoint.rstrip('/')}/{expected['id']}"
            )
            part = output / f".{name}.{os.getpid()}.part"
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "AutoDrive-Fly-v7-Edmond-recovery/1"}
                )
                with urllib.request.urlopen(request, timeout=timeout) as response, part.open(
                    "wb"
                ) as sink:
                    shutil.copyfileobj(response, sink, length=1024 * 1024)
                inspection = _verify(part, expected)
                attempts.append({"url": url, **inspection})
                if inspection["fully_verified"]:
                    part.replace(target)
                    verified = {"source": url, **inspection}
                    break
            except (OSError, ValueError, urllib.error.URLError) as exc:
                attempts.append({"url": url, "error_type": type(exc).__name__})
            finally:
                part.unlink(missing_ok=True)
        result["files"][name] = verified or {
            "fully_verified": False,
            "attempts": attempts,
        }
    result["all_four_files_verified"] = all(
        item["fully_verified"] for item in result["files"].values()
    )
    (output / "retrieval-attempt.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    report = recover(args.manifest, args.output, args.endpoint, args.timeout)
    print(json.dumps({"all_four_files_verified": report["all_four_files_verified"]}))
    return 0 if report["all_four_files_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
