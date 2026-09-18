"""Audit official Tm4/Tm9/CT1 source data without fitting v7 parameters."""

from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
from statistics import median

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t5-contrast-opponency-source-data-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_contrast_opponency_source_data_audit.py")


def _read_block(sheet, spec: dict, axis: str) -> dict:
    row = int(spec["header_row_zero_based"])
    if sheet.cell_value(row, 0) != spec["expected_header"]:
        raise ValueError(f"unexpected header in {spec['sheet']}: {sheet.cell_value(row, 0)!r}")
    labels = [
        str(sheet.cell_value(row, col)).strip()
        for col in range(1, sheet.ncols)
        if str(sheet.cell_value(row, col)).strip()
    ]
    expected_labels = [f"ROI{i}" for i in range(1, int(spec["expected_roi_count"]) + 1)]
    if labels != expected_labels:
        raise ValueError(f"ROI labels changed in {spec['sheet']}:{row + 1}")
    values = []
    cursor = row + 1
    while cursor < sheet.nrows:
        value = sheet.cell_value(cursor, 0)
        if isinstance(value, str) and value.strip():
            break
        if not isinstance(value, (int, float)):
            break
        values.append(float(value))
        cursor += 1
    expected_samples = int(spec[f"expected_{axis}_samples"])
    if len(values) != expected_samples:
        raise ValueError(f"{axis} sample count changed in {spec['sheet']}:{row + 1}")
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid {axis} values in {spec['sheet']}:{row + 1}")
    unit = "seconds" if axis == "time" else "degrees"
    expected_step = float(spec[f"expected_{axis}_step_{unit}"])
    step = median(right - left for left, right in zip(values, values[1:], strict=False))
    if not math.isclose(step, expected_step, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{axis} step changed in {spec['sheet']}:{row + 1}")
    for key, actual in (
        (f"expected_start_{unit}", values[0]),
        (f"expected_end_{unit}", values[-1]),
    ):
        if key in spec and not math.isclose(actual, float(spec[key]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{axis} endpoint changed in {spec['sheet']}:{row + 1}")
    for data_row in range(row + 1, cursor):
        for col in range(1, len(labels) + 1):
            value = sheet.cell_value(data_row, col)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"non-finite response in {spec['sheet']}:{data_row + 1}")
    return {
        "source_type": spec["source_type"],
        "sheet": spec["sheet"],
        "header_row_one_based": row + 1,
        f"{axis}_samples": len(values),
        "roi_count": len(labels),
        f"{axis}_start_{unit}": values[0],
        f"{axis}_end_{unit}": values[-1],
        f"median_{axis}_step_{unit}": step,
        "response_unit": spec["response_unit"],
        "indicator": spec["indicator"],
    }


def evaluate_v7_t5_contrast_opponency_source_data_audit(root: Path) -> dict:
    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError(
            "the T5 contrast-opponency audit requires the xlrd development dependency"
        ) from exc
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source = config["source_data"]
    source_path = root / source["path"]
    if source_path.stat().st_size != int(source["bytes"]):
        raise ValueError("T5 contrast-opponency workbook size mismatch")
    if _sha256(source_path) != source["sha256"]:
        raise ValueError("T5 contrast-opponency workbook SHA-256 mismatch")
    workbook = xlrd.open_workbook(source_path, on_demand=True)
    if len(workbook.sheet_names()) != int(source["sheet_count"]):
        raise ValueError("T5 contrast-opponency workbook sheet count mismatch")
    dynamic = {
        name: _read_block(workbook.sheet_by_name(spec["sheet"]), spec, "time")
        for name, spec in config["dynamic_blocks"].items()
    }
    spatial = {
        name: _read_block(workbook.sheet_by_name(spec["sheet"]), spec, "position")
        for name, spec in config["spatial_blocks"].items()
    }
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    required = contract["required_families"]["T5"]["source_types"]
    dynamic_sources = sorted({item["source_type"] for item in dynamic.values()})
    spatial_only_sources = sorted(
        {item["source_type"] for item in spatial.values()} - set(dynamic_sources)
    )
    missing_dynamic = [source for source in required if source not in dynamic_sources]
    fields = config["transfer_fields"]
    gates = {
        "official_workbook_verified": True,
        "every_T5_source_has_temporal_dynamics": not missing_dynamic,
        "every_dynamic_source_has_allowed_response_unit": all(
            item["response_unit"] in contract["allowed_response_units"] for item in dynamic.values()
        ),
        "dynamic_ROIs_map_to_stable_individual_ids": bool(
            fields["per_ROI_to_fly_mapping_for_dynamic_blocks_present"]
        ),
        "source_type_to_MaleCNS_body_mapping_available": bool(
            fields["source_type_to_MaleCNS_body_mapping_present"]
        ),
        "required_split_roles_present": bool(
            fields["preregistered_training_validation_external_final_roles_present"]
        ),
    }
    transferable = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
            },
            "paper": config["paper"],
            "source_data": {
                **source,
                "actual_sha256": _sha256(source_path),
                "actual_sheet_count": len(workbook.sheet_names()),
                "xlrd_version": importlib.metadata.version("xlrd"),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "verified_dynamic_blocks": dynamic,
        "verified_spatial_blocks": spatial,
        "published_cohorts": config["published_cohorts"],
        "T5_source_contract": {
            "required_sources": required,
            "sources_with_verified_temporal_blocks": dynamic_sources,
            "sources_with_verified_spatial_but_not_temporal_blocks": spatial_only_sources,
            "missing_temporal_sources": missing_dynamic,
            "temporal_coverage_fraction": len(dynamic_sources) / len(required),
        },
        "transfer_gates": gates,
        "complete_T5_source_dynamics_transfer_authorized": transferable,
        "authorize_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else (
                "Tm4_Tm9_calcium_dynamics_and_CT1_spatial_responses_do_not_supply_"
                "CT1_temporal_or_membrane_voltage_evidence"
            )
        ),
        "boundary": config["boundary"],
    }
