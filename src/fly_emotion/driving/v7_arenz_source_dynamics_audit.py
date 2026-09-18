"""Audit Arenz et al. 2017 source-filter parameters against the v7 T4 contract."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-arenz-source-dynamics-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_arenz_source_dynamics_audit.py")


def evaluate_v7_arenz_source_dynamics_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    supplement_path = root / config["supplement"]["path"]
    if supplement_path.stat().st_size != int(config["supplement"]["bytes"]):
        raise ValueError("Arenz supplement size mismatch")
    if _sha256(supplement_path) != config["supplement"]["sha256"]:
        raise ValueError("Arenz supplement SHA-256 mismatch")
    timebase_path = Path(config["current_timebase_evidence"])
    timebase = json.loads((root / timebase_path).read_text(encoding="utf-8"))
    source_protocol_path = Path(config["current_source_protocol"])
    source_protocol = yaml.safe_load(
        (root / source_protocol_path).read_text(encoding="utf-8")
    )
    current_sources = sorted(
        {
            source
            for group in source_protocol["source_groups"].values()
            for source in group
        }
    )
    parameters = config["control_parameters"]
    covered = sorted(set(current_sources) & set(parameters))
    missing = sorted(set(current_sources) - set(parameters))
    extra = sorted(set(parameters) - set(current_sources))
    raw_fit_gates = {
        name: float(item["raw"]["R2"])
        >= float(config["gates"]["minimum_raw_temporal_fit_R2"])
        for name, item in parameters.items()
    }
    gates = {
        "supplement_verified": True,
        "all_Arenz_raw_temporal_fits_pass": all(raw_fit_gates.values()),
        "every_current_source_type_covered": not missing,
        "physical_v7_timebase_available": timebase["identifiability"][
            "physical_timebase_identified"
        ],
        "stable_recorded_cell_to_MaleCNS_mapping_available": False,
    }
    transferable = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(timebase_path): _sha256(root / timebase_path),
                str(source_protocol_path): _sha256(root / source_protocol_path),
            },
            "paper_doi": config["paper"]["doi"],
            "supplement": {
                "path": config["supplement"]["path"],
                "url": config["supplement"]["url"],
                "bytes": supplement_path.stat().st_size,
                "sha256": _sha256(supplement_path),
                "table": config["supplement"]["table"],
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "paper_model": {
            "GCaMP6f_deconvolution_low_pass_seconds": float(
                config["GCaMP6f_deconvolution_low_pass_seconds"]
            ),
            "simulation_step_seconds": float(config["simulation_step_seconds"]),
            "parameter_source": config["parameter_source"],
            "parameters": parameters,
            "raw_temporal_fit_gates": raw_fit_gates,
        },
        "current_v7_source_contract": {
            "source_types": current_sources,
            "covered_by_Arenz": covered,
            "missing_from_Arenz": missing,
            "Arenz_sources_outside_current_contract": extra,
            "coverage_fraction": len(covered) / len(current_sources),
        },
        "transfer_gates": gates,
        "Arenz_filter_parameters_verified": True,
        "complete_current_source_contract_covered": not missing,
        "physical_time_transfer_authorized": False,
        "source_filter_candidate_authorized": transferable,
        "advance_to_functional_precheck": transferable,
        "blocking_data_requirements": [
            name for name, available in gates.items() if not available
        ],
        "stop_reason": (
            None
            if transferable
            else "Arenz_filters_missing_C3_and_v7_physical_time_mapping"
        ),
        "boundary": config["boundary"],
    }
