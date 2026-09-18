"""Verify and inspect the official T4/T5 unified-model package."""

from __future__ import annotations

import hashlib
import importlib.metadata
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-unified-model-package-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_unified_model_package_audit.py")


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_package(path: Path, spec: dict, *, md5_key: str) -> dict:
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"official package size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"official package SHA-256 mismatch: {path}")
    if _md5(path) != spec[md5_key]:
        raise ValueError(f"official package MD5 mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        members = {item.filename: item.file_size for item in archive.infolist()}
    if bad_member is not None:
        raise ValueError(f"corrupt ZIP member: {bad_member}")
    return {
        "path": spec["path"],
        "bytes": path.stat().st_size,
        "md5": _md5(path),
        "sha256": _sha256(path),
        "zip_integrity_passed": True,
        "member_count": len(members),
    }


def _member(archive: zipfile.ZipFile, name: str, expected: dict | str) -> dict:
    raw = archive.read(name)
    expected_sha = expected if isinstance(expected, str) else expected["sha256"]
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"official package member hash mismatch: {name}")
    if isinstance(expected, dict) and len(raw) != int(expected["bytes"]):
        raise ValueError(f"official package member size mismatch: {name}")
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _selected_table(table, objective: str) -> dict:
    if objective != "res_mov_nc_plus_res_mov_pc":
        raise ValueError("unsupported frozen unified-model selection objective")
    score = table["res_mov_nc"] + table["res_mov_pc"]
    order = np.argsort(score.to_numpy(), kind="stable")
    selected = int(order[0])
    minimum = float(score.iloc[selected])
    parameter_columns = [str(name) for name in table.columns if str(name).endswith("_p")]
    quality_columns = [
        "res_aiod",
        "res_spfr_pc",
        "res_mov_pc",
        "res_spfr_nc",
        "res_mov_nc",
        "mse_spfr_pc",
        "mse_mov_pc",
        "mse_spfr_nc",
        "mse_mov_nc",
        "taylor_idx",
        "nan_idx",
        "exitflag",
    ]
    return {
        "shape": list(table.shape),
        "parameter_column_count": len(parameter_columns),
        "parameter_columns": parameter_columns,
        "selected_row_zero_based": selected,
        "selected_row_one_based": selected + 1,
        "selected_score": minimum,
        "second_best_score": float(score.iloc[order[1]]),
        "minimum_is_unique": bool(
            np.count_nonzero(np.isclose(score, minimum, rtol=0.0, atol=1e-12)) == 1
        ),
        "parameters": {name: float(table.iloc[selected][name]) for name in parameter_columns},
        "quality_control": {
            name: (
                bool(table.iloc[selected][name])
                if str(table[name].dtype) == "bool"
                else float(table.iloc[selected][name])
            )
            for name in quality_columns
        },
    }


def evaluate_v7_unified_model_package_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    model_spec = config["model_package"]
    support_spec = config["supporting_package"]
    model_path = root / model_spec["path"]
    support_path = root / support_spec["path"]
    model_result = _verify_package(model_path, model_spec, md5_key="md5")
    support_result = _verify_package(
        support_path, support_spec, md5_key="observed_md5"
    )
    with zipfile.ZipFile(model_path) as model_archive:
        model_members = {
            name: _member(model_archive, name, expected)
            for name, expected in model_spec["members"].items()
        }
        model_code = model_archive.read("modelFigure/modelFigureCode.m").decode()
        with TemporaryDirectory() as directory:
            mat_path = Path(directory) / "optTables.mat"
            mat_path.write_bytes(model_archive.read("modelFigure/optTables.mat"))
            try:
                import matio
            except ImportError as error:
                raise RuntimeError(
                    "the unified-model audit requires the 'mat-io' development dependency"
                ) from error
            tables = matio.load_from_mat(mat_path)
    with zipfile.ZipFile(support_path) as support_archive:
        support_members = {
            name: _member(support_archive, name, expected)
            for name, expected in support_spec["required_members"].items()
        }
        t4_source = support_archive.read(
            "supportingFunctions/t4_simple_wrap.m"
        ).decode()
    expected_tables = config["expected_tables"]
    if sorted(tables) != sorted(expected_tables):
        raise ValueError("unified-model MAT table names differ from frozen manifest")
    shape = list(config["expected_table_shape"])
    if any(list(tables[name].shape) != shape for name in expected_tables):
        raise ValueError("unified-model MAT table shape differs from frozen manifest")
    objective = config["T4_candidate_selection"]["objective"]
    table_results = {name: _selected_table(tables[name], objective) for name in expected_tables}
    author = config["author_selection_reproduction"]
    author_selection_reproduced = (
        table_results[author["table"]]["selected_row_one_based"]
        == int(author["one_based_row"])
        and "relIterT5 = 345" in model_code
        and "res_mov_nc+res_mov_pc" in model_code
    )
    required_source_names = config["source_mapping_requirements"]
    source_mapping_mentions = {name: name in t4_source for name in required_source_names}
    target_component_model = all(
        phrase in t4_source
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
    t4_parameters = {
        name: table_results[name] for name in ("cardT4OptTab", "diagT4OptTab")
    }
    gates = {
        "model_package_verified": True,
        "supporting_package_verified": True,
        "four_tables_decoded": True,
        "author_T5_selection_reproduced": author_selection_reproduced,
        "author_preselected_T4_rows_available": config["T4_candidate_selection"][
            "author_preselected_T4_rows_available"
        ],
        "E_I_E2_I2_to_MaleCNS_source_mapping_available": any(
            source_mapping_mentions.values()
        ),
        "cardinal_diagonal_to_T4_subtype_mapping_available": False,
        "independent_cell_holdout_available": config["boundary"][
            "independent_cell_holdout_available"
        ],
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                "pyproject.toml": _sha256(root / "pyproject.toml"),
            },
            "mat_parser": {
                "package": config["mat_parser"]["package"],
                "version": importlib.metadata.version("mat-io"),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "packages": {
            "model": {**model_result, "doi": model_spec["doi"], "file_id": model_spec["file_id"]},
            "supporting": {
                **support_result,
                "doi": support_spec["doi"],
                "file_id": support_spec["file_id"],
                "md5_provenance": "observed_download_hash_not_repository_checksum",
            },
        },
        "verified_members": {
            "model": model_members,
            "supporting": support_members,
        },
        "tables": {
            "names": expected_tables,
            "all_shapes": {name: list(tables[name].shape) for name in expected_tables},
            "all_parameter_column_counts": {
                name: table_results[name]["parameter_column_count"] for name in expected_tables
            },
            "author_selection_reproduced": author_selection_reproduced,
            "author_selected_cardT5_row_one_based": int(author["one_based_row"]),
            "T4_candidates": t4_parameters,
            "T4_selection_status": config["T4_candidate_selection"]["status"],
        },
        "model_semantics": {
            "target_component_model_verified": target_component_model,
            "components": ["E", "I", "E2", "I2"],
            "parameter_blocks": ["mu", "sigma", "amplitude", "Tr", "Td", "m", "b", "Ti"],
            "parameter_count": int(config["expected_parameter_count"]),
            "source_type_mentions_in_t4_model": source_mapping_mentions,
            "manual_photoreceptor_to_target_delay_milliseconds": 30.0,
            "ON_OFF_parameter_block_reordering_present": (
                "if spfr_data.val == 1" in t4_source
                and "params([3:4,1:2])" in t4_source
            ),
        },
        "transfer_gates": gates,
        "files_verified": True,
        "T4_target_model_parameters_available": True,
        "MaleCNS_source_kernel_transfer_authorized": False,
        "T4_functional_candidate_authorized": False,
        "stop_reason": "verified_target_RF_parameters_lack_MaleCNS_source_mapping",
        "boundary": config["boundary"],
    }
