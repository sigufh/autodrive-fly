"""Audit whether published Fig. 1 tables expose source temporal receptive fields."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-fig1-source-temporal-readiness-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_fig1_source_temporal_readiness_audit.py"
)


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"Fig. 1 source-data size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"Fig. 1 source-data SHA-256 mismatch: {path}")
    return path


def evaluate_v7_fig1_source_temporal_readiness_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    figure_path = _verify_file(root, config["source_data"]["figure_1"])
    extended_path = _verify_file(root, config["source_data"]["extended_figure_1"])
    prior_path = Path(config["prior_evidence"])
    prior = json.loads((root / prior_path).read_text(encoding="utf-8"))
    workbook = pd.ExcelFile(figure_path)
    expected_sheets = config["source_data"]["figure_1"]["expected_sheets"]
    if workbook.sheet_names != expected_sheets:
        raise ValueError("Fig. 1 source-data sheets differ from frozen manifest")
    source_sheets = {}
    for source_type in config["source_types"]:
        sheet = f"Fig. 1d, {source_type}"
        frame = pd.read_excel(figure_path, sheet_name=sheet, header=None)
        source_sheets[source_type] = {
            "sheet": sheet,
            "shape": list(frame.shape),
            "first_axis_label": str(frame.iloc[0, 0]),
            "contains_time_axis": "Time" in str(frame.iloc[0, 0]),
            "population_scope": "source_type_average",
        }
    extended = pd.read_excel(
        extended_path,
        sheet_name=config["source_data"]["extended_figure_1"]["sheet"],
        header=None,
    )
    pattern = re.compile(r"^(Mi9|Tm3|Mi1|Mi4|C3|T4) #(\d+)$")
    labels = [
        match.group(1)
        for value in extended.to_numpy(dtype=object).ravel()
        if isinstance(value, str) and (match := pattern.fullmatch(value))
    ]
    individual_counts = {
        source_type: labels.count(source_type) for source_type in config["source_types"]
    }
    if individual_counts != config["expected_individual_spatial_RF_counts"]:
        raise ValueError("Extended Fig. 1 source-cell counts differ from frozen manifest")
    t4_count = labels.count("T4")
    if t4_count != int(config["expected_T4_spatial_RF_count"]):
        raise ValueError("Extended Fig. 1 T4-cell count differs from frozen manifest")
    payload = config["edmond_payload"]
    prior_file = prior["verified_files"][payload["filename"]]
    prior_header = prior["array_headers"][payload["filename"]]
    for key in ("id", "bytes", "md5", "sha256"):
        if prior_file[key] != payload["file_id" if key == "id" else key]:
            raise ValueError(f"Fig. 1 object payload manifest mismatch: {key}")
    if prior_header != payload["header"]:
        raise ValueError("Fig. 1 object payload header mismatch")
    local_payload = root / payload["local_path"]
    payload_available = local_payload.is_file()
    payload_verified = bool(
        payload_available
        and local_payload.stat().st_size == int(payload["bytes"])
        and _sha256(local_payload) == payload["sha256"]
    )
    source_temporal_fields_in_xlsx = all(
        item["contains_time_axis"] for item in source_sheets.values()
    )
    observations = {
        "official_Fig1_workbooks_verified": True,
        "source_type_average_tables_have_time_axis": source_temporal_fields_in_xlsx,
        "individual_source_tables_have_time_axis": False,
        "official_fig1d_object_payload_locally_available": payload_available,
    }
    fit_gates = {
        "official_fig1d_object_payload_hash_verified": payload_verified,
        "official_fig1d_object_payload_safely_inspected": False,
        "source_temporal_kernel_numerically_available": False,
    }
    fit_authorized = all(fit_gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(prior_path): _sha256(root / prior_path),
                "pyproject.toml": _sha256(root / "pyproject.toml"),
            },
            "paper_doi": config["paper"]["doi"],
            "data_doi": config["paper"]["data_doi"],
            "source_files": {
                "figure_1": {
                    "path": str(figure_path.relative_to(root)),
                    "bytes": figure_path.stat().st_size,
                    "sha256": _sha256(figure_path),
                },
                "extended_figure_1": {
                    "path": str(extended_path.relative_to(root)),
                    "bytes": extended_path.stat().st_size,
                    "sha256": _sha256(extended_path),
                },
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "figure_1_source_tables": source_sheets,
        "extended_figure_1": {
            "sheet": config["source_data"]["extended_figure_1"]["sheet"],
            "shape": list(extended.shape),
            "individual_source_spatial_RF_counts": individual_counts,
            "individual_T4_spatial_RF_count": t4_count,
            "contains_source_time_axis": False,
        },
        "edmond_object_payload": {
            **{
                key: payload[key]
                for key in (
                    "filename",
                    "file_id",
                    "bytes",
                    "md5",
                    "sha256",
                    "header",
                    "local_path",
                )
            },
            "locally_available": payload_available,
            "hash_verified": payload_verified,
            "safely_inspected": False,
        },
        "observations": observations,
        "source_filter_fit_gates": fit_gates,
        "source_temporal_kernel_transfer_authorized": fit_authorized,
        "advance_to_source_filter_fit": fit_authorized,
        "blocking_requirements": [
            name for name, passed in fit_gates.items() if not passed
        ],
        "stop_reason": (
            None
            if fit_authorized
            else "published_Fig1_tables_lack_source_time_axis_and_object_payload_unavailable"
        ),
        "boundary": config["boundary"],
    }
