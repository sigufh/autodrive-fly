from pathlib import Path

from fly_emotion.data.manifest import DataFile, digest, iter_files, load_manifest


def test_manifest_has_required_primary_sources() -> None:
    root = Path(__file__).parents[1]
    manifest = load_manifest(root / "configs/data-manifest.yaml")
    names = {(item.dataset, item.name) for item in iter_files(manifest)}
    assert ("malecns", "annotations") in names
    assert ("malecns", "connections") in names
    assert all(dataset == "malecns" for dataset, _ in names)


def test_verify_reports_and_accepts_digest(tmp_path: Path) -> None:
    spec = DataFile(
        dataset="test",
        name="tiny",
        path=Path("tiny.bin"),
        url="https://example.invalid/tiny.bin",
        bytes=3,
        sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )
    assert spec.verify(tmp_path)
    (tmp_path / spec.path).write_bytes(b"abc")
    assert not spec.verify(tmp_path)
    assert digest(tmp_path / spec.path) == spec.sha256
