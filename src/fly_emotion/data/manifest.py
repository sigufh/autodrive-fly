from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataFile:
    dataset: str
    name: str
    path: Path
    url: str
    bytes: int | None = None
    sha256: str | None = None
    md5: str | None = None
    optional: bool = False

    def verify(self, root: Path) -> list[str]:
        target = root / self.path
        errors: list[str] = []
        if not target.is_file():
            return [f"missing: {target}"]
        if self.bytes is not None and target.stat().st_size != self.bytes:
            errors.append(f"size mismatch for {target}: {target.stat().st_size} != {self.bytes}")
        for algorithm, expected in (("sha256", self.sha256), ("md5", self.md5)):
            if (
                expected
                and expected != "TO_BE_FILLED_BY_VERIFICATION"
                and digest(target, algorithm) != expected
            ):
                errors.append(f"{algorithm} mismatch for {target}")
        return errors


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if value.get("schema_version") != 1:
        raise ValueError("unsupported data manifest schema")
    return value


def iter_files(manifest: dict[str, Any], datasets: set[str] | None = None) -> Iterator[DataFile]:
    for dataset, dataset_spec in manifest["datasets"].items():
        if datasets and dataset not in datasets:
            continue
        for name, spec in dataset_spec.get("files", {}).items():
            yield DataFile(
                dataset=dataset,
                name=name,
                path=Path(spec["path"]),
                url=spec["url"],
                bytes=spec.get("bytes"),
                sha256=spec.get("sha256"),
                md5=spec.get("md5"),
                optional=spec.get("optional", False),
            )
