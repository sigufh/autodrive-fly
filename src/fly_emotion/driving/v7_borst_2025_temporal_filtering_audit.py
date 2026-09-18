"""Audit provenance of the Borst 2025 temporal-filtering fit target."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-borst-2025-temporal-filtering-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_borst_2025_temporal_filtering_audit.py")


def _array_literal(code: str, name: str) -> list[float]:
    match = re.search(rf"{re.escape(name)}\s*=\s*np\.array\((\[[^\n]+\])\)", code)
    if match is None:
        raise ValueError(f"Borst target parameter missing: {name}")
    return [float(value) for value in ast.literal_eval(match.group(1))]


def evaluate_v7_borst_2025_temporal_filtering_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))

    full_text_spec = config["paper"]["full_text"]
    full_text_path = root / full_text_spec["path"]
    if full_text_path.stat().st_size != int(full_text_spec["bytes"]):
        raise ValueError("Borst 2025 full-text size mismatch")
    if _sha256(full_text_path) != full_text_spec["sha256"]:
        raise ValueError("Borst 2025 full-text SHA-256 mismatch")
    paper_text = " ".join("".join(ElementTree.parse(full_text_path).getroot().itertext()).split())
    required_paper_phrases = (
        "impulse responses of 13 neuronal cell types",
        "determined by calcium imaging, using white noise stimuli and reverse correlation",
        "normalized impulse responses which were integrated over time to result in step responses",
        "contained no information about membrane voltage",
        "arbitrarily set the maximum of all data to 20 mV response magnitude",
        "temporal dynamics over 2 s at a temporal resolution of 10 ms",
        "experimental data were derived from calcium imaging",
        "low-pass filter with 50 ms time-constant",
    )
    if any(phrase not in paper_text for phrase in required_paper_phrases):
        raise ValueError("Borst 2025 paper provenance text changed")

    repository = config["repository"]
    archive_spec = repository["archive"]
    archive_path = root / archive_spec["path"]
    if archive_path.stat().st_size != int(archive_spec["bytes"]):
        raise ValueError("Borst repository archive size mismatch")
    if _sha256(archive_path) != archive_spec["sha256"]:
        raise ValueError("Borst repository archive SHA-256 mismatch")

    with zipfile.ZipFile(archive_path) as zipped:
        if len(zipped.infolist()) != int(repository["archive_member_count"]):
            raise ValueError("Borst repository archive member count changed")
        members = sorted(name for name in zipped.namelist() if not name.endswith("/"))
        if len(members) != int(repository["file_count"]):
            raise ValueError("Borst repository file count changed")
        prefix = repository["commit"]
        if zipped.comment.decode("ascii") != prefix:
            raise ValueError("Borst repository archive commit changed")

        def read_text(relative: str) -> str:
            member = next((name for name in members if name.endswith(relative)), None)
            if member is None:
                raise ValueError(f"Borst repository member missing: {relative}")
            return zipped.read(member).decode("utf-8")

        code = {name: read_text(path) for name, path in repository["code_paths"].items()}
        arrays = []
        array_reports = []
        for relative in repository["data_copies"]:
            member = next((name for name in members if name.endswith(relative)), None)
            if member is None:
                raise ValueError(f"Borst data copy missing: {relative}")
            payload = zipped.read(member)
            if _sha256_bytes(payload) != repository["data_sha256"]:
                raise ValueError(f"Borst data copy SHA-256 mismatch: {relative}")
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            arrays.append(array)
            array_reports.append(
                {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "finite": bool(np.isfinite(array).all()),
                    "minimum": float(array.min()),
                    "maximum": float(array.max()),
                    "pre_stimulus_absolute_maximum": float(np.abs(array[:, :, :50]).max()),
                }
            )

    target = config["target_contract"]
    if any(list(array.shape) != target["shape"] for array in arrays):
        raise ValueError("Borst data target shape changed")
    if any(str(array.dtype) != target["dtype"] for array in arrays):
        raise ValueError("Borst data target dtype changed")
    if not all(np.array_equal(arrays[0], array) for array in arrays[1:]):
        raise ValueError("Borst data target copies differ")

    library = code["library"]
    python_model = code["python_model"]
    pytorch_model = code["pytorch_model"]
    figure_one = code["figure_one"]
    required_library_fragments = (
        "data = np.zeros((13,9,200))",
        "data[i,j] = RecF_data[i,j*5+2]*ImpR_data[i]",
        "signal[50:200] =  1.0",
        "# hp and lp time constants * 10 ms",
    )
    required_model_fragments = (
        "deltat    = 10.0  # in msec",
        "data = ml.read_RecF_data()*20.0 # 20 mV Amplitude",
        "model  = bs.lowpass(model,Ca_tau/deltat)",
    )
    required_pytorch_fragments = (
        "deltat    = 10.0  # in msec",
        "data_amp      = 20.0",
        "mydata = ml.read_RecF_data()*data_amp",
    )
    if any(fragment not in library for fragment in required_library_fragments):
        raise ValueError("Borst target-construction code changed")
    if any(fragment not in python_model for fragment in required_model_fragments):
        raise ValueError("Borst Python model target path changed")
    if any(fragment not in pytorch_model for fragment in required_pytorch_fragments):
        raise ValueError("Borst PyTorch model target path changed")
    required_figure_fragments = (
        "# time constants * 10 msec",
        "myT4title=['Mi1','Tm3','Mi4','Mi9']",
        "myT5title=['Tm1','Tm2','Tm4','Tm9']",
    )
    if any(fragment not in figure_one for fragment in required_figure_fragments):
        raise ValueError("Borst Figure 1 target description changed")

    cells = re.search(r"cell_list=np\.array\((\[[^\n]+\])\)", library)
    if cells is None or list(ast.literal_eval(cells.group(1))) != target["cell_type_order"]:
        raise ValueError("Borst target cell order changed")
    target_parameters = {
        name: _array_literal(library, name)
        for name in (
            "RF_center_width",
            "RF_surrnd_width",
            "RF_surrnd_weight",
            "RF_sign",
            "IR_hp",
            "IR_lp",
        )
    }
    if any(len(values) != len(target["cell_type_order"]) for values in target_parameters.values()):
        raise ValueError("Borst target parameter count changed")

    required_sources = {
        source
        for family in contract["required_families"].values()
        for source in family["source_types"]
    }
    covered_sources = sorted(required_sources & set(target["cell_type_order"]))
    missing_sources = sorted(required_sources - set(covered_sources))
    gates = {
        "fixed_repository_archive_verified": True,
        "paper_full_text_verified": True,
        "local_numeric_parameterized_target_verified": True,
        "all_nine_required_sources_covered": not missing_sources,
        "experimental_membrane_voltage_payload": bool(
            target["source_contains_membrane_voltage_information"]
        ),
        "stable_biological_individual_ids_available": bool(
            target["biological_individual_ids_present"]
        ),
        "source_to_MaleCNS_mapping_available": bool(
            target["source_to_MaleCNS_mapping_present"]
        ),
        "required_split_roles_available": bool(target["preregistered_split_roles_present"]),
        "external_final_commitment_available": bool(
            target["external_final_commitment_present"]
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
            "repository": {**repository, "actual_archive_sha256": _sha256(archive_path)},
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "repository_inventory": {
            "archive_member_count": int(repository["archive_member_count"]),
            "file_count": len(members),
            "data_copies": array_reports,
            "data_copies_identical": True,
        },
        "target_provenance": {
            **target,
            "construction": "hard_coded_spatial_DoG_parameters_times_analytic_temporal_filter",
            "hard_coded_parameters": target_parameters,
            "numeric_target_unit_semantics": "arbitrary_20mV_model_comparison_scale",
            "measured_membrane_voltage": False,
        },
        "v7_source_coverage": {
            "covered_sources": covered_sources,
            "missing_sources": missing_sources,
            "covered_source_count": len(covered_sources),
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
            else "calcium_derived_parameterized_target_is_not_individual_membrane_voltage"
        ),
        "boundary": config["boundary"],
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
