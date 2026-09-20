"""Distinguish the official MATLAB target model from Figure 6 axolotl code."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-kohn-portes-figure6-model-identity-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_kohn_portes_figure6_model_identity_audit.py"
)


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if "bytes" in spec and path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"file size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"file SHA-256 mismatch: {path}")
    return path


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def evaluate_v7_kohn_portes_figure6_model_identity_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    unified_path = Path(config["unified_model_evidence"])
    unified = json.loads((root / unified_path).read_text(encoding="utf-8"))

    metadata_paths = {
        name: _verify_file(root, spec)
        for name, spec in config["official_metadata"].items()
    }
    metadata = {name: json.loads(path.read_text()) for name, path in metadata_paths.items()}
    model_file = metadata["model_files"][0]
    supporting_file = metadata["supporting_files"][0]
    expected = config["expected"]
    if metadata["model_article"]["id"] != expected["model_article_id"]:
        raise ValueError("official model article ID changed")
    if metadata["supporting_article"]["id"] != expected["supporting_article_id"]:
        raise ValueError("official supporting article ID changed")

    package_paths = {}
    for name, spec in config["packages"].items():
        path = root / spec["path"]
        if path.stat().st_size != int(spec["bytes"]):
            raise ValueError(f"package size mismatch: {path}")
        if _md5(path) != spec["md5"] or _sha256(path) != spec["sha256"]:
            raise ValueError(f"package hash mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"corrupt package: {path}")
        package_paths[name] = path

    manifest_matches = all(
        (observed["id"], observed["size"], observed["md5"])
        == (spec["file_id"], spec["bytes"], spec["md5"])
        for observed, spec in (
            (model_file, config["packages"]["model"]),
            (supporting_file, config["packages"]["supporting"]),
        )
    )
    if not manifest_matches:
        raise ValueError("official widget manifests do not match recovered packages")

    with zipfile.ZipFile(package_paths["model"]) as archive:
        model_members = [
            name for name in archive.namelist() if not name.startswith("__MACOSX/")
        ]
        model_code = archive.read("modelFigure/modelFigureCode.m").decode()
    with zipfile.ZipFile(package_paths["supporting"]) as archive:
        support_members = [
            name for name in archive.namelist() if not name.startswith("__MACOSX/")
        ]
        t5_source = archive.read("supportingFunctions/t5_simple_wrap.m").decode()
        support_text = "\n".join(
            archive.read(name).decode(errors="replace")
            for name in support_members
            if name.endswith((".m", ".py", ".txt", ".md"))
        )
    if not set(expected["model_member_paths"]).issubset(model_members):
        raise ValueError("official model member inventory changed")
    if not set(expected["supporting_member_paths"]).issubset(support_members):
        raise ValueError("official supporting member inventory changed")

    notebook_paths = [_verify_file(root, spec) for spec in config["figure6_notebooks"]]
    notebook_text = "\n".join(_notebook_text(path) for path in notebook_paths)
    axolotl_import_verified = expected["axolotl_import"] in notebook_text
    axolotl_local_path_verified = expected["axolotl_local_path"] in notebook_text
    target_components = ["E", "I", "E2", "I2"]
    target_model_verified = all(
        phrase in t5_source
        for phrase in (
            "[E, I, E2, I2]",
            "params(1:4)",
            "params(5:8)",
            "params(9:12)",
            "params(29)*10",
            "function f = fc_sol",
            "function f = fd_sol",
        )
    )
    source_hits = {
        source: bool(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])",
                model_code + "\n" + support_text,
                flags=re.IGNORECASE,
            )
        )
        for source in expected["T5_source_types"]
    }
    axolotl_source_in_packages = "axolotl.tmodel" in model_code + support_text

    gates = {
        "official_widget_manifests_match_recovered_packages": manifest_matches,
        "official_package_files_verified": unified["files_verified"],
        "official_MATLAB_T5_target_model_verified": target_model_verified,
        "official_MATLAB_model_has_T5_source_type_mapping": any(source_hits.values()),
        "Figure6_axolotl_import_verified": axolotl_import_verified,
        "Figure6_axolotl_local_path_verified": axolotl_local_path_verified,
        "Figure6_axolotl_source_recovered": axolotl_source_in_packages,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(unified_path): _sha256(root / unified_path),
                **{
                    spec["path"]: _sha256(path)
                    for spec, path in zip(
                        config["official_metadata"].values(),
                        metadata_paths.values(),
                        strict=True,
                    )
                },
                **{
                    spec["path"]: _sha256(path)
                    for spec, path in zip(
                        config["packages"].values(), package_paths.values(), strict=True
                    )
                },
                **{
                    spec["path"]: _sha256(path)
                    for spec, path in zip(
                        config["figure6_notebooks"], notebook_paths, strict=True
                    )
                },
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "official_manifests": {
            "model": model_file,
            "supporting": supporting_file,
        },
        "package_inventory": {
            "model_members": model_members,
            "supporting_members": support_members,
        },
        "model_identity": {
            "official_model_language": "MATLAB",
            "official_model_entrypoint": "t5_simple_wrap",
            "official_model_components": target_components,
            "official_model_source_type_mentions": source_hits,
            "Figure6_model_language": "Python",
            "Figure6_import": expected["axolotl_import"],
            "Figure6_notebook_local_package_path": expected["axolotl_local_path"],
            "same_model_implementation": False,
        },
        "gates": gates,
        "official_T5_target_model_parameters_available": target_model_verified,
        "T5_source_mapping_available": False,
        "Figure6_model_reproducible_from_recovered_packages": False,
        "authorize_Tm_to_T5_model_transfer_to_v7": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "verified_MATLAB_target_model_is_not_external_Python_axolotl_source_model",
        "boundary": config["boundary"],
    }
