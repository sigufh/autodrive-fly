"""Audit whether existing physical-time evidence can support a T5 source model."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t5-physical-time-transfer-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_physical_time_transfer_audit.py"
)


def evaluate_v7_t5_physical_time_transfer_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "stage1_split_evidence",
            "stage1_split_protocol",
            "timebase_evidence",
            "timebase_protocol",
            "stimulus_coordinate_evidence",
            "stimulus_coordinate_protocol",
            "t5_target_data_evidence",
            "t5_target_data_protocol",
            "t5_source_filter_evidence",
            "t5_source_filter_protocol",
            "source_identifiability_evidence",
            "source_identifiability_protocol",
            "flyvis_source_time_evidence",
            "flyvis_source_time_protocol",
        )
    }
    reports = {
        name: json.loads((root / path).read_text())
        for name, path in paths.items()
        if name.endswith("_evidence")
    }
    split = reports["stage1_split_evidence"]
    coordinates = reports["stimulus_coordinate_evidence"]
    target = reports["t5_target_data_evidence"]
    source = reports["t5_source_filter_evidence"]
    identifiability = reports["source_identifiability_evidence"]
    flyvis = reports["flyvis_source_time_evidence"]
    scheduler = split["engineering_timebase"]
    target_ready = target["direction_and_identity_readiness"]
    source_gates = source["transfer_gates"]
    fields = {
        "v7_physical_frame_interval": bool(
            coordinates["offline_time_coordinate_contract_complete"]
        ),
        "v7_physical_solver_interval": bool(
            coordinates["offline_time_coordinate_contract_complete"]
        ),
        "v7_camera_angular_calibration": bool(
            coordinates[
                "offline_two_dimensional_stimulus_coordinate_contract_complete"
            ]
        ),
        "Tm1_Tm2_Tm4_Tm9_membrane_like_kernels": bool(
            source_gates["every_deconvolved_temporal_fit_passed"]
        ),
        "CT1_membrane_like_kernel": False,
        "stable_source_or_target_cell_to_MaleCNS_mapping": bool(
            source_gates["stable_recorded_cell_to_MaleCNS_mapping_available"]
            or target_ready["stable_biological_cell_ids_available"]
        ),
        "independent_dynamic_validation_cohort": bool(
            target_ready["independent_cell_holdout_available"]
        ),
    }
    if list(fields) != config["required_fields"]:
        raise ValueError("T5 physical-time readiness fields differ from frozen contract")
    transferable = all(fields.values())
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
        "time_layers": {
            "offline_stimulus_coordinate_contract": {
                "time_coordinates_complete": coordinates[
                    "offline_time_coordinate_contract_complete"
                ],
                "horizontal_coordinates_complete": coordinates[
                    "offline_horizontal_stimulus_coordinate_contract_complete"
                ],
                "two_dimensional_angular_calibration_complete": coordinates[
                    "offline_two_dimensional_stimulus_coordinate_contract_complete"
                ],
                "frame_interval_milliseconds": coordinates["time_coordinates"][
                    "frame_interval_milliseconds"
                ],
                "substep_interval_milliseconds": coordinates["time_coordinates"][
                    "substep_interval_milliseconds"
                ],
                "horizontal_fov_degrees": coordinates["camera_coordinates"][
                    "horizontal_fov_degrees"
                ],
                "angular_sample_spacing_degrees": coordinates[
                    "camera_coordinates"
                ]["angular_sample_spacing_degrees"],
                "biologically_calibrated": coordinates[
                    "biological_timebase_calibrated"
                ],
                "external_recording_alignment_verified": coordinates[
                    "external_recording_alignment_verified"
                ],
            },
            "stage1_scheduler": {
                "frame_interval_milliseconds": scheduler[
                    "frame_interval_milliseconds"
                ],
                "nominal_substep_interval_milliseconds": scheduler[
                    "nominal_substep_interval_milliseconds"
                ],
                "applied_to_current_runtime": scheduler[
                    "applied_to_current_runtime"
                ],
                "biologically_calibrated": scheduler["biologically_calibrated"],
                "permitted_use": scheduler["permitted_use"],
            },
            "T5_target_voltage": {
                "native_sample_intervals_milliseconds": target["data_semantics"][
                    "native_sample_intervals_milliseconds"
                ],
                "native_time_vectors_verified": target_ready[
                    "native_time_vectors_verified"
                ],
                "measures_source_types": False,
            },
            "T5_source_filters": {
                "covered_sources": source["T5_source_contract"]["covered_sources"],
                "raw_calcium_contract_complete": source[
                    "raw_T5_calcium_filter_contract_complete"
                ],
                "deconvolved_contract_complete": source[
                    "deconvolved_T5_filter_contract_complete"
                ],
                "missing_CT1": "CT1"
                not in source["T5_source_contract"]["covered_sources"],
            },
            "FlyVis_source_time_constants": {
                "required_types_present": not flyvis["source_coverage"]["T5"][
                    "missing"
                ],
                "all_above_solver_dt": flyvis["gates"][
                    "every_source_above_solver_dt_in_every_model"
                ],
                "all_cross_model_IQR_stable": flyvis["gates"][
                    "every_source_cross_model_IQR_stable"
                ],
                "transferable": flyvis[
                    "FlyVis_visual_source_time_constants_transferable"
                ],
            },
        },
        "current_source_temporal_identifiability": {
            "passing_T5_source_population_count": identifiability["families"][
                "T5"
            ]["passing_source_population_count"],
            "T5_source_population_denominator": identifiability["families"][
                "T5"
            ]["source_population_denominator"],
            "passed": identifiability[
                "source_type_temporal_identifiability_passed"
            ],
        },
        "required_fields": fields,
        "missing_fields": [name for name, available in fields.items() if not available],
        "T5_physical_source_transfer_ready": transferable,
        "authorize_physical_source_dynamics_candidate": transferable,
        "advance_to_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "stop_reason": (
            None
            if transferable
            else "physical_time_or_source_specific_T5_transfer_evidence_incomplete"
        ),
        "boundary": config["boundary"],
    }
