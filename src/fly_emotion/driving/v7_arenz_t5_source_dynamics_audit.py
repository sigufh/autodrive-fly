"""Audit Arenz Table S2 temporal filters for the v7 T5 source contract."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-arenz-t5-source-dynamics-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_arenz_t5_source_dynamics_audit.py")


def evaluate_v7_arenz_t5_source_dynamics_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    supplement = root / config["supplement"]["path"]
    if supplement.stat().st_size != int(config["supplement"]["bytes"]):
        raise ValueError("Arenz T5 supplement size mismatch")
    if _sha256(supplement) != config["supplement"]["sha256"]:
        raise ValueError("Arenz T5 supplement SHA-256 mismatch")
    timebase_path = Path(config["timebase_evidence"])
    timebase = json.loads((root / timebase_path).read_text())
    label_path = Path(config["label_evidence"])
    label = json.loads((root / label_path).read_text())
    if label["label_status"]["direction_code_to_PD_ND"] != {"0": "ND", "1": "PD"}:
        raise ValueError("Arenz T5 audit requires verified T5 direction semantics")
    required = list(config["required_sources"])
    parameters = config["control_parameters"]
    if set(parameters) != set(required):
        raise ValueError("Arenz Table S2 source coverage differs from T5 contract")
    raw_gates = {
        name: float(item["raw"]["R2"])
        >= float(config["gates"]["minimum_raw_temporal_fit_R2"])
        for name, item in parameters.items()
    }
    deconvolved_gates = {
        name: float(item["deconvolved"]["R2"])
        >= float(config["gates"]["minimum_deconvolved_temporal_fit_R2"])
        for name, item in parameters.items()
    }
    gates = {
        "supplement_verified": True,
        "every_required_source_covered": set(parameters) == set(required),
        "every_raw_temporal_fit_passed": all(raw_gates.values()),
        "every_deconvolved_temporal_fit_passed": all(deconvolved_gates.values()),
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
                str(label_path): _sha256(root / label_path),
            },
            "paper_doi": config["paper"]["doi"],
            "supplement": {
                **config["supplement"],
                "actual_sha256": _sha256(supplement),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "T5_source_contract": {
            "required_sources": required,
            "covered_sources": sorted(parameters),
            "coverage_fraction": len(parameters) / len(required),
        },
        "paper_model": {
            "GCaMP6f_deconvolution_low_pass_seconds": float(
                config["GCaMP6f_deconvolution_low_pass_seconds"]
            ),
            "parameter_source": config["parameter_source"],
            "parameters": parameters,
            "raw_temporal_fit_gates": raw_gates,
            "deconvolved_temporal_fit_gates": deconvolved_gates,
        },
        "transfer_gates": gates,
        "raw_T5_calcium_filter_contract_complete": bool(
            gates["every_required_source_covered"]
            and gates["every_raw_temporal_fit_passed"]
        ),
        "deconvolved_T5_filter_contract_complete": bool(
            gates["every_required_source_covered"]
            and gates["every_deconvolved_temporal_fit_passed"]
        ),
        "T5_source_filter_transfer_authorized": transferable,
        "advance_to_T5_functional_precheck": transferable,
        "blocking_requirements": [
            name for name, passed in gates.items() if not passed
        ],
        "stop_reason": (
            None
            if transferable
            else "Arenz_T5_filters_fail_deconvolution_or_v7_transfer_identifiability"
        ),
        "boundary": config["boundary"],
    }
