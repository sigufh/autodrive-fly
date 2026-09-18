"""Audit public Pirogova source-calcium payloads without trusting pickle code."""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import pickletools
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import yaml
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index, _new_Index
from pandas.core.indexes.multi import MultiIndex
from pandas.core.internals.managers import BlockManager

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-pirogova-source-calcium-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_pirogova_source_calcium_audit.py")


class _FrozenNDArray(np.ndarray):
    """Compatibility target for the immutable ndarray used by pandas 0.14."""

    def __new__(cls, data, dtype=None, copy=False):
        return np.array(data, dtype=dtype, copy=copy).view(cls)


class _RestrictedPandasUnpickler(pickle.Unpickler):
    """Allow only the legacy pandas/NumPy containers present in fixed payloads."""

    _allowed = {
        ("__builtin__", "slice"): slice,
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("pandas.core.frame", "DataFrame"): DataFrame,
        ("pandas.core.indexes.base", "Index"): Index,
        ("pandas.core.indexes.base", "_new_Index"): _new_Index,
        ("pandas.core.indexes.frozen", "FrozenNDArray"): _FrozenNDArray,
        ("pandas.core.indexes.multi", "MultiIndex"): MultiIndex,
        ("pandas.core.indexes.numeric", "Float64Index"): Index,
        ("pandas.core.indexes.numeric", "Int64Index"): Index,
        ("pandas.core.internals", "BlockManager"): BlockManager,
    }

    def find_class(self, module: str, name: str):
        try:
            return self._allowed[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}") from exc


def _git_blob(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(prefix + payload, usedforsecurity=False).hexdigest()


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"Pirogova file size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"Pirogova file SHA-256 mismatch: {path}")
    if "git_blob" in spec and _git_blob(path.read_bytes()) != spec["git_blob"]:
        raise ValueError(f"Pirogova file Git blob mismatch: {path}")
    return path


def _load_restricted(path: Path) -> tuple[pd.DataFrame, list[str]]:
    payload = path.read_bytes()
    globals_used = sorted(
        str(value)
        for opcode, value, _ in pickletools.genops(payload)
        if opcode.name == "GLOBAL"
    )
    allowed_globals = {
        f"{module} {name}" for module, name in _RestrictedPandasUnpickler._allowed
    }
    forbidden_globals = set(globals_used) - allowed_globals
    if forbidden_globals:
        raise ValueError(
            f"Pirogova pickle has forbidden globals: {path}: {sorted(forbidden_globals)}"
        )
    frame = _RestrictedPandasUnpickler(io.BytesIO(payload), encoding="latin1").load()
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"Pirogova payload is not a DataFrame: {path}")
    return frame, globals_used


def evaluate_v7_pirogova_source_calcium_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))

    paper_spec = config["paper"]["full_text"]
    paper_path = _verify_file(root, paper_spec)
    paper_text = " ".join("".join(ElementTree.parse(paper_path).getroot().itertext()).split())
    required_phrases = (
        "Imaging was performed at an acquisition rate of 11.8 Hz",
        "Regions of interest (ROIs) were drawn manually",
        "Relative fluorescence change (ΔF/F) was calculated",
        "Mi1 (n = 98 cells/25 flies); Tm1 (n = 24 cells/9 flies);",
        "Tm2 (n = 22 cells/7 flies), and Tm3 (n = 65 cells/16 flies)",
        "Raw data from calcium imaging experiments",
    )
    if any(phrase not in paper_text for phrase in required_phrases):
        raise ValueError("Pirogova paper provenance text changed")

    repository = config["repository"]
    archive_path = _verify_file(root, repository["latest_archive"])
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Pirogova latest archive failed integrity check")
        if archive.comment.decode("ascii") != repository["latest_commit"]:
            raise ValueError("Pirogova latest archive commit changed")
        members = sorted(name for name in archive.namelist() if not name.endswith("/"))
        if len(archive.infolist()) != int(repository["latest_archive"]["archive_member_count"]):
            raise ValueError("Pirogova latest archive member count changed")
        if len(members) != int(repository["latest_archive"]["file_count"]):
            raise ValueError("Pirogova latest archive file count changed")
        latest_basenames = {Path(name).name for name in members}
    latest_missing = [
        name for name in ("Tm1_data.pkl", "Tm2_data.pkl") if name not in latest_basenames
    ]
    if latest_missing != ["Tm1_data.pkl", "Tm2_data.pkl"]:
        raise ValueError("Pirogova latest-head historical-data boundary changed")

    support_paths = {
        name: _verify_file(root, spec) for name, spec in repository["history_support"].items()
    }
    readme = support_paths["README.md"].read_text(encoding="utf-8")
    notebook = json.loads(support_paths["notebook"].read_text(encoding="utf-8"))
    notebook_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    if "Raw data from calcium imaging experiments" not in readme:
        raise ValueError("Pirogova historical README description changed")
    for source in config["source_files"]:
        if f'{source}_data = pd.read_pickle("{source}_data.pkl")' not in notebook_code:
            raise ValueError(f"Pirogova notebook no longer reads {source} historical data")
    if "normalise_before = [-1,-.9]" not in notebook_code:
        raise ValueError("Pirogova notebook baseline window changed")

    expected_dt = float(config["expected_sample_interval_seconds"])
    source_payloads = {}
    globals_by_source = {}
    for source, spec in config["source_files"].items():
        path = _verify_file(root, spec)
        frame, globals_used = _load_restricted(path)
        globals_by_source[source] = globals_used
        if list(frame.shape) != spec["expected_shape"]:
            raise ValueError(f"Pirogova source shape changed: {source}")
        if frame.index.names != config["expected_index_names"]:
            raise ValueError(f"Pirogova source index changed: {source}")
        if frame.columns.names != config["expected_column_names"]:
            raise ValueError(f"Pirogova source columns changed: {source}")
        trial_labels = list(frame.columns.levels[1])
        if trial_labels != config["expected_trial_labels"]:
            raise ValueError(f"Pirogova trial labels changed: {source}")
        columns = list(frame.columns)
        if len(columns) % len(trial_labels):
            raise ValueError(f"Pirogova source columns are not complete trial groups: {source}")
        cell_groups = len(columns) // len(trial_labels)
        ordered_trial_groups = all(
            [columns[index + offset][1] for offset in range(3)] == trial_labels
            and len({columns[index + offset][0] for offset in range(3)}) == 1
            for index in range(0, len(columns), 3)
        )
        if not ordered_trial_groups or cell_groups != int(spec["expected_cell_groups"]):
            raise ValueError(f"Pirogova source cell/trial grouping changed: {source}")
        unique_cell_labels = len(frame.columns.levels[0])
        if unique_cell_labels != int(spec["expected_unique_cell_labels"]):
            raise ValueError(f"Pirogova source cell-label count changed: {source}")
        time = np.sort(np.asarray(frame.index.get_level_values("time").unique(), dtype=float))
        dt = float(np.median(np.diff(time)))
        if time.size != int(config["expected_time_count"]):
            raise ValueError(f"Pirogova source time count changed: {source}")
        if not math.isclose(dt, expected_dt, rel_tol=0.0, abs_tol=1e-14):
            raise ValueError(f"Pirogova source sample interval changed: {source}")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values[~np.isnan(values)]).all():
            raise ValueError(f"Pirogova source contains infinite response values: {source}")
        duplicate_column_positions = len(columns) - len(set(columns))
        source_payloads[source] = {
            "file": spec,
            "shape": list(frame.shape),
            "cell_trial_column_group_count": cell_groups,
            "paper_reported_cell_count": int(spec["expected_cell_groups"]),
            "paper_reported_fly_count": int(spec["paper_fly_count"]),
            "unique_cell_label_count": unique_cell_labels,
            "duplicate_cell_trial_column_positions": duplicate_column_positions,
            "cell_labels_globally_unique_within_payload": duplicate_column_positions == 0,
            "trial_labels": trial_labels,
            "condition_labels": list(frame.index.levels[1]),
            "time_sample_count": int(time.size),
            "time_minimum_seconds": float(time.min()),
            "time_maximum_seconds": float(time.max()),
            "sample_interval_seconds": dt,
            "effective_sample_rate_hz": 1.0 / dt,
            "finite_response_value_count": int(np.isfinite(values).sum()),
            "total_response_value_count": int(values.size),
            "minimum_finite_response": float(np.nanmin(values)),
            "maximum_finite_response": float(np.nanmax(values)),
            "response_modality": config["response_modality"],
            "response_unit": config["response_unit"],
            "response_unit_contract_satisfied": False,
            "stable_recording_unit_label_available": duplicate_column_positions == 0,
            "cell_to_fly_mapping_available": False,
            "biological_individual_id_available": False,
        }

    required_sources = {
        source
        for family in contract["required_families"].values()
        for source in family["source_types"]
    }
    covered_sources = sorted(required_sources & set(source_payloads))
    missing_sources = sorted(required_sources - set(covered_sources))
    gates = {
        "fixed_historical_git_payloads_verified": True,
        "restricted_pickle_constructor_set_verified": True,
        "local_numeric_calcium_time_series_verified": True,
        "physical_time_axis_verified": True,
        "all_nine_required_sources_covered": not missing_sources,
        "allowed_membrane_voltage_response_unit": False,
        "stable_recording_unit_ids_available_for_every_source": all(
            item["stable_recording_unit_label_available"] for item in source_payloads.values()
        ),
        "biological_individual_ids_available": False,
        "source_to_MaleCNS_mapping_available": False,
        "required_split_roles_available": False,
        "external_final_commitment_available": False,
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
            "repository": {
                **repository,
                "actual_latest_archive_sha256": _sha256(archive_path),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "history_recovery": {
            "latest_archive_file_count": len(members),
            "latest_head_missing_historical_source_files": latest_missing,
            "historical_data_commit": repository["historical_data_commit"],
            "historical_notebook_commit": repository["historical_notebook_commit"],
            "historical_notebook_reads_all_four_sources": True,
        },
        "pickle_security": {
            "static_opcode_globals_by_source": globals_by_source,
            "restricted_unpickler_used": True,
            "arbitrary_repository_code_executed": False,
        },
        "source_payloads": source_payloads,
        "source_coverage": {
            "covered_required_sources": covered_sources,
            "missing_required_sources": missing_sources,
            "covered_required_source_count": len(covered_sources),
            "required_source_count": len(required_sources),
        },
        "transfer_gates": gates,
        "source_dynamics_transfer_authorized": transferable,
        "authorize_T4_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else "numeric_calcium_payload_lacks_allowed_unit_fly_identity_and_required_sources"
        ),
        "boundary": config["boundary"],
    }
