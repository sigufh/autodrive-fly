"""Combined stop/go gate for evidence-backed T4/T5 source dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t4t5-source-dynamics-readiness.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4t5_source_dynamics_readiness.py"
)


def evaluate_v7_t4t5_source_dynamics_readiness(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "t4_transfer_evidence",
            "t4_transfer_protocol",
            "t5_transfer_evidence",
            "t5_transfer_protocol",
            "flyvis_source_time_evidence",
            "flyvis_source_time_protocol",
            "t4_crossfit_axis_evidence",
            "t5_crossfit_axis_evidence",
        )
    }
    reports = {
        name: json.loads((root / path).read_text())
        for name, path in paths.items()
        if name.endswith("_evidence")
    }
    t4 = reports["t4_transfer_evidence"]
    t5 = reports["t5_transfer_evidence"]
    flyvis = reports["flyvis_source_time_evidence"]
    t4_axis = reports["t4_crossfit_axis_evidence"]
    t5_axis = reports["t5_crossfit_axis_evidence"]
    gates = {
        "T4_crossfit_structure_axis": t4_axis["T4_synapse_crossfit_axis_passed"],
        "T5_crossfit_structure_axis": t5_axis["T5_CT1_crossfit_axis_passed"],
        "T4_source_dynamics_transfer": t4["source_dynamics_transfer_authorized"],
        "T5_source_dynamics_transfer": t5["T5_physical_source_transfer_ready"],
        "shared_physical_v7_timebase": bool(
            t4["required_transfer_fields_available"]["physical_v7_sample_interval"]
            and t5["required_fields"]["v7_physical_frame_interval"]
            and t5["required_fields"]["v7_physical_solver_interval"]
        ),
        "independent_dynamic_validation": bool(
            t4["transfer_gates"]["independent_dynamic_validation_available"]
            and t5["required_fields"]["independent_dynamic_validation_cohort"]
        ),
    }
    if list(gates) != config["required_gates"]:
        raise ValueError("combined T4/T5 readiness gates differ from frozen contract")
    passed = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in paths.values()},
            },
            "parameter_fit": False,
            "functional_stimulus_evaluated": False,
            "runtime_modified": False,
        },
        "gates": gates,
        "passing_gates": [name for name, value in gates.items() if value],
        "failing_gates": [name for name, value in gates.items() if not value],
        "FlyVis_source_coverage": flyvis["source_coverage"],
        "FlyVis_time_constants_transferable": flyvis[
            "FlyVis_visual_source_time_constants_transferable"
        ],
        "T4_T5_source_dynamics_ready": passed,
        "authorize_new_T4_or_T5_functional_candidate": passed,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            None if passed else "T4_T5_source_dynamics_evidence_incomplete"
        ),
        "boundary": config["boundary"],
    }
