"""Audit Kohn-Portes T5-input patch-clamp payloads without fitting parameters."""

from __future__ import annotations

import io
import json
import math
import pickle
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-kohn-portes-t5-ephys-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_kohn_portes_t5_ephys_audit.py")


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Permit only the three NumPy constructors used by these plain arrays."""

    _allowed = {
        ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module: str, name: str):
        try:
            return self._allowed[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}") from exc


def _load_restricted(path: Path):
    return _RestrictedNumpyUnpickler(io.BytesIO(path.read_bytes())).load()


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"size mismatch for {spec['path']}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {spec['path']}")
    return path


def evaluate_v7_kohn_portes_t5_ephys_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    support_paths = {
        name: _verify_file(root, spec) for name, spec in config["support_files"].items()
    }
    readme = support_paths["README.md"].read_text(encoding="utf-8")
    analysis_code = support_paths["flash_analysis.py"].read_text(encoding="utf-8")
    plotting_code = support_paths["ephysplot.py"].read_text(encoding="utf-8")
    if "Preprocessed data for Tm1, Tm2, Tm4 and Tm9" not in readme:
        raise ValueError("repository README no longer describes the four Tm payloads")
    per_recording_assignment = (
        "flash_dict[ff['stimulus_name']+' '+ff['bath_solution']]['all'] = mn_all"
    )
    if per_recording_assignment not in analysis_code:
        raise ValueError("author analysis no longer documents per-recording arrays")
    if "# convert to mV" not in plotting_code or "sc = 1e3" not in plotting_code:
        raise ValueError("author plotting code no longer documents the volts-to-mV scale")

    expected_conditions = config["expected_conditions"]
    expected_dt = float(config["expected_sample_interval_seconds"])
    expected_samples = int(config["expected_samples_per_trace"])
    source_payloads = {}
    for source, spec in config["source_files"].items():
        path = _verify_file(root, spec)
        payload = _load_restricted(path)
        if not isinstance(payload, dict) or list(payload) != expected_conditions:
            raise ValueError(f"unexpected condition keys for {source}")
        conditions = {}
        for condition, item in payload.items():
            if not isinstance(item, dict) or set(item) != {
                "mean",
                "std",
                "n",
                "sampling_rate",
                "all",
            }:
                raise ValueError(f"unexpected fields for {source}:{condition}")
            mean = np.asarray(item["mean"], dtype=float)
            std = np.asarray(item["std"], dtype=float)
            traces = np.asarray(item["all"], dtype=float)
            dt = float(item["sampling_rate"])
            if mean.shape != (expected_samples,) or std.shape != (expected_samples,):
                raise ValueError(f"summary shape changed for {source}:{condition}")
            if traces.ndim != 2 or traces.shape[1] != expected_samples:
                raise ValueError(f"trace shape changed for {source}:{condition}")
            if not all(np.isfinite(values).all() for values in (mean, std, traces)):
                raise ValueError(f"non-finite values in {source}:{condition}")
            if not math.isclose(dt, expected_dt, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"sample interval changed for {source}:{condition}")
            conditions[condition] = {
                "reported_cell_count_n": int(item["n"]),
                "anonymous_trace_count": int(traces.shape[0]),
                "reported_n_equals_trace_count": int(item["n"]) == traces.shape[0],
                "sample_count": int(traces.shape[1]),
                "sample_interval_seconds": dt,
                "duration_seconds": traces.shape[1] * dt,
                "stored_response_unit": "volts",
                "contract_response_unit_after_exact_scale": "millivolts",
                "minimum_volts": float(np.min(traces)),
                "maximum_volts": float(np.max(traces)),
            }
        source_payloads[source] = {
            "file": spec,
            "condition_count": len(conditions),
            "conditions": conditions,
            "all_conditions_have_physical_time_axis": True,
            "all_conditions_have_numerical_membrane_voltage": True,
            "stable_recording_ids_retained": False,
            "biological_individual_ids_retained": False,
        }

    covered = list(source_payloads)
    required = contract["required_families"]["T5"]["source_types"]
    missing = [source for source in required if source not in covered]
    fields = config["fields"]
    gates = {
        "Tm1_Tm2_Tm4_Tm9_local_numeric_membrane_voltage_verified": covered
        == ["Tm1", "Tm2", "Tm4", "Tm9"],
        "physical_time_axis_verified": True,
        "allowed_response_unit_documented": "millivolts" in contract["allowed_response_units"],
        "every_T5_source_covered": not missing,
        "stable_recording_ids_retained": bool(fields["stable_recording_ids_retained_in_pickle"]),
        "biological_individual_ids_retained": bool(
            fields["biological_individual_ids_retained_in_pickle"]
        ),
        "required_angular_stimulus_fields_retained": all(
            fields[name]
            for name in (
                "stimulus_direction_retained_in_pickle",
                "stimulus_angular_position_degrees_retained_in_pickle",
                "stimulus_angular_speed_degrees_per_second_retained_in_pickle",
            )
        ),
        "baseline_window_seconds_retained": bool(
            fields["baseline_window_seconds_retained_in_pickle"]
        ),
        "source_to_MaleCNS_body_mapping_present": bool(
            fields["source_to_MaleCNS_body_mapping_present"]
        ),
        "required_split_roles_present": bool(
            fields["preregistered_training_validation_external_final_roles_present"]
        ),
        "external_final_commitment_present": bool(fields["external_final_commitment_present"]),
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
            "repository": config["repository"],
            "support_files": config["support_files"],
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "source_payloads": source_payloads,
        "T5_source_contract": {
            "required_sources": required,
            "sources_with_local_numeric_membrane_voltage": covered,
            "missing_numeric_membrane_voltage_sources": missing,
            "coverage_fraction": len(covered) / len(required),
            "fast_sources": ["Tm1", "Tm2", "Tm4"],
            "delayed_source_kept_separate": "Tm9",
        },
        "transfer_gates": gates,
        "T5_source_dynamics_transfer_authorized": transferable,
        "authorize_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else "four_Tm_voltage_payloads_lack_identity_mapping_splits_and_CT1_coverage"
        ),
        "boundary": config["boundary"],
    }
