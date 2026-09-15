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
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        **{path: _sha256(root / path) for path in config["source_configs"].values()},
        **{path: _sha256(root / path) for path in config["source_evidence"].values()},
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
        "available_final_test": False,
        "limitations": [
            "The interface exposes published processed arrays, not raw 10-kHz recordings.",
            "Fig. 3 remains training-data reproduction and cannot score generalization.",
            "Fig. 5 is an external condition but cell identity disjointness is unknown.",
            "Fig. 5 reuses Fig. 3 input traces and is not an independent input dataset.",
            "No T5 electrophysiology or untouched final test is available in this interface.",
            "No conversion from millivolts to current v7 normalized state is defined.",
        ],
        "advance_to_parameter_fit": False,
        "advance_to_central_complex": False,
    }
