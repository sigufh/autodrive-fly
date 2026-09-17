from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_ephys_audit import SOURCE_ORDER, _download_and_verify
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t5_conductance_audit import (
    PREFIXES,
    PROTOCOL_CODES,
    _clone_fixed_repository,
    _load_cell,
    _segments,
    _vector,
    audit_t5_conductance_repository,
)

CONFIG = Path("configs/driving-v7-ephys-interface.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_ephys_interface.py")


@dataclass(frozen=True)
class TimedArray:
    values: np.ndarray
    axes: tuple[str, ...]
    sample_interval_milliseconds: float | None
    role: str

    @property
    def time_milliseconds(self) -> np.ndarray | None:
        if self.sample_interval_milliseconds is None or "time" not in self.axes:
            return None
        length = self.values.shape[self.axes.index("time")]
        return np.arange(length, dtype=float) * self.sample_interval_milliseconds


@dataclass(frozen=True)
class NativeTimedTrace:
    values: np.ndarray
    time_milliseconds: np.ndarray
    role: str
    quantity: str = "baseline_subtracted_membrane_voltage_millivolts"

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.time_milliseconds.ndim != 1:
            raise ValueError("native traces must be one-dimensional")
        if self.values.shape != self.time_milliseconds.shape:
            raise ValueError("native trace values and times must have equal lengths")
        if not np.all(np.diff(self.time_milliseconds) > 0):
            raise ValueError("native trace time must be strictly increasing")
        self.values.setflags(write=False)
        self.time_milliseconds.setflags(write=False)


def load_offline_bundle(source_dir: Path, sample_interval_milliseconds: float) -> dict:
    if sample_interval_milliseconds != 1.0:
        raise ValueError("published processed Fig. 3 interface is frozen at 1 ms per sample")
    bundle = {
        "fig3_inputs": {
            name: TimedArray(
                np.load(source_dir / f"fig3_{name}.npy", allow_pickle=False),
                ("stimulus", "cell", "time"),
                sample_interval_milliseconds,
                "training_reproduction",
            )
            for name in SOURCE_ORDER
        },
        "fig3_T4_voltage": TimedArray(
            np.load(source_dir / "fig3_T4v.npy", allow_pickle=False),
            ("cell", "stimulus", "direction", "time"),
            sample_interval_milliseconds,
            "training_reproduction",
        ),
        "fig3_led": TimedArray(
            np.load(source_dir / "fig3_led.npy", allow_pickle=False),
            ("stimulus", "direction", "time"),
            sample_interval_milliseconds,
            "training_reproduction",
        ),
        "fig5_groups": {},
    }
    for group, role in (
        ("gfp", "external_condition_validation"),
        ("gluclarnai", "mechanism_challenge"),
        ("nmdar1rnai", "measured_negative_control"),
    ):
        bundle["fig5_groups"][group] = {
            "delta_voltage": TimedArray(
                np.load(source_dir / f"fig5_dvm_{group}.npy", allow_pickle=False),
                ("cell", "direction"),
                None,
                role,
            ),
            "absolute_peak_voltage": TimedArray(
                np.load(source_dir / f"fig5c_pvm_{group}.npy", allow_pickle=False),
                ("cell", "direction"),
                None,
                role,
            ),
        }
    return bundle


def _native_trace(values: np.ndarray, time: object, role: str) -> NativeTimedTrace:
    return NativeTimedTrace(
        np.asarray(values, dtype=float).copy(),
        _vector(time).astype(float, copy=True),
        role,
    )


def load_t5_conductance_bundle(source_dir: Path, recorded_cells: int = 17) -> dict:
    cells = {}
    for cell_id in range(1, recorded_cells + 1):
        spfr_data, spfr_protocol = _load_cell(source_dir, cell_id, "spfr")
        all_data, all_protocol = _load_cell(source_dir, cell_id, "all")
        spfr_values, spfr_starts, spfr_stops = _segments(spfr_data, "sb")
        spfr_times = list(_vector(spfr_protocol.t))
        widths = _vector(spfr_protocol.width_sb).astype(int)
        fit_traces = tuple(
            _native_trace(
                spfr_values[start - 1 : stop],
                spfr_times[index],
                "within_cell_fit_source",
            )
            for index, (start, stop, width) in enumerate(
                zip(spfr_starts, spfr_stops, widths, strict=True)
            )
            if width == 2
        )
        protocols = _vector(all_protocol.protocol).astype(int)
        times = list(_vector(all_protocol.t))
        protocol_times = {name: [] for name in PREFIXES}
        for protocol, time in zip(protocols, times, strict=True):
            protocol_times[PROTOCOL_CODES[int(protocol)]].append(time)
        conditions = {}
        for name, prefix in PREFIXES.items():
            if not hasattr(all_data, f"vm_all_{prefix}"):
                conditions[name] = ()
                continue
            values, starts, stops = _segments(all_data, prefix)
            conditions[name] = tuple(
                _native_trace(
                    values[start - 1 : stop],
                    time,
                    "within_cell_condition_generalization",
                )
                for start, stop, time in zip(starts, stops, protocol_times[name], strict=True)
            )
        cells[cell_id] = {
            "fit_width2_single_bar": fit_traces,
            "all_conditions": conditions,
        }
    return {
        "cells": cells,
        "split_kind": "within_cell_stimulus_condition_generalization",
        "fit_allowed": False,
    }


def _array_summary(item: TimedArray) -> dict:
    time = item.time_milliseconds
    return {
        "shape": list(item.values.shape),
        "dtype": str(item.values.dtype),
        "axes": list(item.axes),
        "sample_interval_milliseconds": item.sample_interval_milliseconds,
        "time_start_milliseconds": float(time[0]) if time is not None else None,
        "time_end_milliseconds": float(time[-1]) if time is not None else None,
        "role": item.role,
        "all_finite": bool(np.all(np.isfinite(item.values))),
        "minimum": float(np.min(item.values)),
        "maximum": float(np.max(item.values)),
        "sha256_values": hashlib.sha256(item.values.tobytes()).hexdigest(),
    }


def evaluate_v7_ephys_interface(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    fig3_config = yaml.safe_load(
        (root / config["source_configs"]["fig3"]).read_text(encoding="utf-8")
    )
    fig5_config = yaml.safe_load(
        (root / config["source_configs"]["fig5"]).read_text(encoding="utf-8")
    )
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in config["source_evidence"].items()
    }
    t5_config = yaml.safe_load(
        (root / config["t5_conductance"]["source_config"]).read_text(encoding="utf-8")
    )
    t5_evidence = json.loads(
        (root / config["t5_conductance"]["source_evidence"]).read_text(encoding="utf-8")
    )
    if t5_evidence["advance_to_T5_fit"] or t5_evidence["protocol"]["parameter_fitting"]:
        raise ValueError("T5 interface source evidence must remain non-fitting")
    if t5_evidence["repository"]["commit"] != t5_config["repository"]["commit"]:
        raise ValueError("T5 interface source evidence commit mismatch")
    if not config["exploratory"] or config["advance_allowed"]:
        raise ValueError("offline electrophysiology interface must be non-advancing")
    if evidence["timebase"]["identifiability"]["physical_timebase_identified"]:
        raise ValueError("offline interface must not imply current v7 has a physical timebase")
    required = set(sum(config["dataset_roles"].values(), []))
    manifests = {**fig3_config["files"], **fig5_config["files"]}
    if required != set(manifests) & required:
        raise ValueError("offline data roles contain unknown files")
    download = {
        "dataset": fig3_config["dataset"],
        "files": {name: manifests[name] for name in sorted(required)},
    }
    with tempfile.TemporaryDirectory(prefix="autodrive-v7-ephys-interface-") as temporary:
        source_dir = Path(temporary)
        files = _download_and_verify(download, source_dir)
        bundle = load_offline_bundle(source_dir, float(config["sample_interval_milliseconds"]))
        arrays = {
            "fig3_inputs": {
                name: _array_summary(item) for name, item in bundle["fig3_inputs"].items()
            },
            "fig3_T4_voltage": _array_summary(bundle["fig3_T4_voltage"]),
            "fig3_led": _array_summary(bundle["fig3_led"]),
            "fig5_groups": {
                group: {name: _array_summary(item) for name, item in values.items()}
                for group, values in bundle["fig5_groups"].items()
            },
        }
        t5_source = source_dir / "t5-conductance"
        _clone_fixed_repository(t5_config, t5_source)
        t5_verified = audit_t5_conductance_repository(t5_source, t5_config)
        t5_bundle = load_t5_conductance_bundle(t5_source, int(t5_config["paper"]["recorded_cells"]))
        t5_interface = {
            "role": "within_cell_stimulus_condition_generalization",
            "cell_count": len(t5_bundle["cells"]),
            "fit_allowed": t5_bundle["fit_allowed"],
            "fit_trace_count": sum(
                len(cell["fit_width2_single_bar"]) for cell in t5_bundle["cells"].values()
            ),
            "condition_trace_counts": {
                name: sum(len(cell["all_conditions"][name]) for cell in t5_bundle["cells"].values())
                for name in PREFIXES
            },
            "response_quantity": t5_verified["data_semantics"]["response_quantity"],
            "response_unit": t5_verified["data_semantics"]["response_unit"],
            "native_sample_intervals_milliseconds": t5_verified["data_semantics"][
                "native_sample_intervals_milliseconds"
            ],
            "ragged_trace_lengths": True,
            "repository_commit": t5_verified["repository"]["commit"],
            "manifest_sha256": t5_verified["repository"]["manifest_sha256"],
            "spfr_is_exact_subset_of_all": t5_verified["subset_verification"][
                "all_spfr_traces_byte_exact_with_condition_matched_all_trace"
            ],
            "cells_with_both_moving_bar_direction_codes": t5_verified[
                "direction_and_identity_readiness"
            ]["cells_with_both_moving_bar_direction_codes"],
            "direction_code_to_PD_ND_mapping_verified": t5_verified[
                "direction_and_identity_readiness"
            ]["direction_code_to_PD_ND_mapping_verified"],
            "stable_biological_cell_ids_available": t5_verified[
                "direction_and_identity_readiness"
            ]["stable_biological_cell_ids_available"],
        }
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        **{path: _sha256(root / path) for path in config["source_configs"].values()},
        **{path: _sha256(root / path) for path in config["source_evidence"].values()},
        config["t5_conductance"]["source_config"]: _sha256(
            root / config["t5_conductance"]["source_config"]
        ),
        config["t5_conductance"]["source_evidence"]: _sha256(
            root / config["t5_conductance"]["source_evidence"]
        ),
    }
    return {
        "protocol": {
            "name": config["name"],
            "exploratory": True,
            "advance_allowed": False,
            "dependencies_sha256": dependencies,
            "parameter_fitting": False,
            "runtime_imports_interface": False,
            "raw_files_committed": False,
            "temporary_download_deleted_after_audit": True,
        },
        "dataset_roles": config["dataset_roles"],
        "split_contract": config["split_contract"],
        "interface_boundary": config["interface_boundary"],
        "verified_files": files,
        "arrays": arrays,
        "t5_conductance": t5_interface,
        "available_final_test": False,
        "limitations": [
            "The interface exposes published processed arrays, not raw 10-kHz recordings.",
            "Fig. 3 remains training-data reproduction and cannot score generalization.",
            "Fig. 5 is an external condition but cell identity disjointness is unknown.",
            "Fig. 5 reuses Fig. 3 input traces and is not an independent input dataset.",
            "T5 traces are processed same-cell data, not an independent cell holdout.",
            "T5 native 2.5/5-ms ragged traces remain separate from the T4 1-kHz bundle.",
            "No untouched final test is available in this interface.",
            "No conversion from millivolts to current v7 normalized state is defined.",
        ],
        "advance_to_parameter_fit": False,
        "advance_to_central_complex": False,
    }
