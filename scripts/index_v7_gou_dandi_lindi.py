#!/usr/bin/env python3
"""Build a bounded LINDI metadata index for the frozen Gou DANDI assets."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

DEFAULT_ASSET_INDEX = Path("data/raw/gou-sparsity/dandi-001205-0.250602.0251/assets-index.json")
DEFAULT_OUTPUT = Path("data/raw/gou-sparsity/dandi-001205-0.250602.0251/lindi-metadata-index.json")
URL_TEMPLATE = "https://lindi.neurosift.org/dandi/dandisets/001205/assets/{asset_id}/nwb.lindi.json"


def _decode_memcpy_scalar(refs: dict[str, Any], path: str) -> Any:
    descriptor = refs.get(f"{path}/.zarray")
    encoded = refs.get(f"{path}/0")
    if not isinstance(descriptor, dict) or not isinstance(encoded, str):
        return None
    raw = (
        base64.b64decode(encoded[7:]) if encoded.startswith("base64:") else encoded.encode("latin1")
    )
    compressor = descriptor.get("compressor")
    if compressor and compressor.get("id") == "blosc":
        if len(raw) < 16:
            raise ValueError(f"short Blosc envelope for {path}")
        uncompressed_bytes = int.from_bytes(raw[4:8], "little")
        compressed_bytes = int.from_bytes(raw[12:16], "little")
        if compressed_bytes != len(raw) or uncompressed_bytes != len(raw) - 16:
            raise ValueError(f"non-memcpy Blosc scalar for {path}")
        raw = raw[16:]
    dtype = np.dtype(descriptor["dtype"])
    return np.frombuffer(raw, dtype=dtype, count=1)[0].item()


def _fetch(asset: dict, timeout_seconds: float, retries: int) -> dict:
    url = URL_TEMPLATE.format(asset_id=asset["asset_id"])
    error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=(10, timeout_seconds))
            response.raise_for_status()
            payload = response.json()
            generation = payload["generationMetadata"]
            if (
                generation["assetId"] != asset["asset_id"]
                or generation["assetPath"] != asset["path"]
                or int(generation["assetSize"]) != int(asset["size"])
            ):
                raise ValueError("LINDI metadata differs from frozen DANDI asset")
            refs = payload["refs"]
            keys = sorted(refs)
            return {
                "asset_id": asset["asset_id"],
                "path": asset["path"],
                "size": int(asset["size"]),
                "status": response.status_code,
                "subject_description": _decode_memcpy_scalar(refs, "general/subject/description"),
                "imaging_location": _decode_memcpy_scalar(
                    refs, "general/optophysiology/imaging_plane/location"
                ),
                "identifier": None,
                "stimulus_keys": [key for key in keys if key.startswith("stimulus/")],
                "interval_keys": [key for key in keys if key.startswith("intervals/")],
                "processing_keys": [key for key in keys if key.startswith("processing/")],
                "scratch_keys": [key for key in keys if key.startswith("scratch/")],
                "error": None,
            }
        except (OSError, ValueError, KeyError, requests.RequestException) as caught:
            error = caught
            if attempt + 1 < retries:
                time.sleep(attempt + 1)
    return {
        "asset_id": asset["asset_id"],
        "path": asset["path"],
        "size": int(asset["size"]),
        "status": None,
        "error": f"{type(error).__name__}: {error}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--asset-index", type=Path, default=DEFAULT_ASSET_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.retries <= 5:
        raise ValueError("workers must be 1..8 and retries must be 1..5")
    root = args.root.resolve()
    source = args.asset_index if args.asset_index.is_absolute() else root / args.asset_index
    output = args.output if args.output.is_absolute() else root / args.output
    assets = json.loads(source.read_text(encoding="utf-8"))["results"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(lambda asset: _fetch(asset, args.timeout_seconds, args.retries), assets)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dandiset": "001205",
                "version": "0.250602.0251",
                "asset_count": len(assets),
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
