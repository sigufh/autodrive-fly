from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from .manifest import DataFile


def download_file(spec: DataFile, root: Path, *, chunk_size: int = 8 * 1024 * 1024) -> Path:
    """Download to a partial file, resume if possible, verify, then atomically publish."""
    target = root / spec.path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not spec.verify(root):
        return target

    partial = target.with_suffix(target.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(spec.url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urllib.request.urlopen(request) as response:  # noqa: S310 - manifest URLs are reviewed
        if offset and response.status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as stream:
            while block := response.read(chunk_size):
                stream.write(block)

    os.replace(partial, target)
    errors = spec.verify(root)
    if errors:
        os.replace(target, partial)
        raise ValueError("; ".join(errors))
    return target
