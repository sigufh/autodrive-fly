"""Audit whether TimingModels CT1 filters identify the T5 lobula compartment."""

from __future__ import annotations

import importlib.metadata
import json
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-timing-models-ct1-compartment-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_timing_models_ct1_compartment_audit.py")


def _matlab_string(source: h5py.File, reference: h5py.Reference) -> str:
    return "".join(chr(int(value)) for value in source[reference][()].ravel())


def _load_columns(path: Path, required: list[str]) -> dict:
    with h5py.File(path, "r") as source:
        names = [
            _matlab_string(source, reference) for reference in source["filterList"][()].ravel()
        ]
        labels = [
            _matlab_string(source, reference) for reference in source["filterLabel"][()].ravel()
        ]
        matrix = source["filterMat"][()].T.astype(np.float64)
        sem = source["filterSem"][()].T.astype(np.float64)
        time = source["tSec"][()].ravel().astype(np.float64)
        sample_interval = float(source["dtFilter"][0, 0])
        root_fields = sorted(name for name in source if name != "#refs#")
    if any(name not in names for name in required):
        raise ValueError("TimingModels CT1 comparison source is missing")
    columns = {}
    for name in required:
        index = names.index(name)
        values = matrix[:, index]
        errors = sem[:, index]
        if not np.isfinite(values).all() or not np.isfinite(errors).all():
            raise ValueError(f"TimingModels {name} filter or SEM is non-finite")
        columns[name] = {
            "label": labels[index],
            "values": values,
            "sem": errors,
        }
    return {
        "columns": columns,
        "time": time,
        "sample_interval_seconds": sample_interval,
        "root_fields": root_fields,
    }


def _unit(values: np.ndarray) -> np.ndarray:
    centered = values - values[0]
    return centered / max(float(np.linalg.norm(centered)), np.finfo(float).tiny)


def evaluate_v7_timing_models_ct1_compartment_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    timing_path = Path(config["timing_models_evidence"])
    timing = json.loads((root / timing_path).read_text(encoding="utf-8"))
    timing_protocol_path = Path(config["timing_models_protocol"])
    timing_protocol = yaml.safe_load((root / timing_protocol_path).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    verified_supplements = []
    required_caption_fragments = (
        "Figure S5.",
        "CT1 lobula terminals",
        "wild-type CT1 (CT1 > GC6f, n = 17)",
        "Frames are 1/30 of a second",
    )
    for spec in config["supplements"]:
        path = root / spec["path"]
        if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
            raise ValueError("TimingModels supplement identity mismatch")
        reader = PdfReader(path)
        if len(reader.pages) != int(spec["pages"]):
            raise ValueError("TimingModels supplement page count mismatch")
        caption = " ".join(
            (reader.pages[int(spec["figure_S5_page_one_based"]) - 1].extract_text() or "").split()
        )
        if any(fragment not in caption for fragment in required_caption_fragments):
            raise ValueError("TimingModels supplement Figure S5 caption mismatch")
        attachment_count = len(reader.attachments or {})
        if attachment_count != 0:
            raise ValueError("TimingModels supplement unexpectedly embeds attachments")
        verified_supplements.append(
            {
                **spec,
                "actual_sha256": _sha256(path),
                "actual_pages": len(reader.pages),
                "embedded_attachment_count": attachment_count,
                "Figure_S5_caption_verified": True,
            }
        )
    ct1_name = config["CT1_filter_name"]
    comparisons = list(config["comparison_sources"])
    required_columns = [ct1_name, *comparisons]
    datasets = []
    for spec in timing_protocol["filter_files"]:
        path = root / spec["path"]
        if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
            raise ValueError("TimingModels CT1 source file identity mismatch")
        loaded = _load_columns(path, required_columns)
        if loaded["root_fields"] != [
            "dtFilter",
            "filterLabel",
            "filterList",
            "filterMat",
            "filterSem",
            "tSec",
        ]:
            raise ValueError("TimingModels filter file gained an unreviewed provenance field")
        datasets.append({"spec": spec, **loaded})
    cross_deconvolution = {}
    for first, second in combinations(datasets, 2):
        first_values = _unit(first["columns"][ct1_name]["values"])
        second_values = _unit(second["columns"][ct1_name]["values"])
        key = (
            f"{first['spec']['calcium_deconvolution_milliseconds']}_vs_"
            f"{second['spec']['calcium_deconvolution_milliseconds']}_ms"
        )
        cross_deconvolution[key] = float(np.corrcoef(first_values, second_values)[0, 1])
    distinction = {}
    for dataset in datasets:
        assumption = str(dataset["spec"]["calcium_deconvolution_milliseconds"])
        ct1 = dataset["columns"][ct1_name]["values"]
        distinction[assumption] = {
            source: {
                "exactly_equal": bool(np.array_equal(ct1, dataset["columns"][source]["values"])),
                "correlation": float(np.corrcoef(ct1, dataset["columns"][source]["values"])[0, 1]),
            }
            for source in comparisons
        }
    minimum = min(cross_deconvolution.values())
    provenance = config["provenance"]
    response_unit_allowed = provenance["response_unit"] in contract["allowed_response_units"]
    gates = {
        "repository_and_filter_files_verified": bool(
            timing["transfer_gates"]["repository_and_files_verified"]
        ),
        "CT1_column_present_with_finite_SEM": True,
        "CT1_column_distinct_from_Mi4_and_Tm9": all(
            not item["exactly_equal"]
            for by_assumption in distinction.values()
            for item in by_assumption.values()
        ),
        "CT1_stable_across_deconvolution_assumptions": minimum
        >= float(config["gates"]["minimum_cross_deconvolution_correlation"]),
        "CT1_lobula_L1_Lo1_provenance_available": bool(
            provenance["CT1_column_explicitly_linked_to_lobula_L1_Lo1"]
        ),
        "separate_lobula_L1_Lo1_dynamic_phenotype_published": bool(
            provenance["separate_lobula_L1_Lo1_dynamic_phenotype_published"]
        ),
        "lobula_L1_Lo1_numerical_time_series_payload_available": bool(
            provenance["separate_lobula_L1_Lo1_numerical_payload_verified"]
        ),
        "allowed_response_unit_available": response_unit_allowed,
        "stable_individual_ids_available": bool(provenance["stable_individual_ids_present"]),
        "MaleCNS_mapping_available": bool(provenance["MaleCNS_mapping_present"]),
        "required_split_roles_present": bool(provenance["required_split_roles_present"]),
    }
    transferable = all(gates.values())
    representative = datasets[1]
    values = representative["columns"][ct1_name]["values"]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(timing_path): _sha256(root / timing_path),
                str(timing_protocol_path): _sha256(root / timing_protocol_path),
                str(contract_path): _sha256(root / contract_path),
            },
            "paper": config["paper"],
            "supplements": verified_supplements,
            "pypdf_version": importlib.metadata.version("pypdf"),
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "CT1_filter_evidence": {
            "filter_name": ct1_name,
            "filter_label": representative["columns"][ct1_name]["label"],
            "source_file_root_fields": representative["root_fields"],
            "sample_count": int(values.size),
            "sample_interval_seconds": representative["sample_interval_seconds"],
            "time_start_seconds": float(representative["time"][0]),
            "time_stop_seconds": float(representative["time"][-1]),
            "control_fly_count_reported_in_paper": int(
                config["paper"]["main_CT1_control_fly_count"]
            ),
            "response_unit": provenance["response_unit"],
            "cross_deconvolution_correlations": cross_deconvolution,
            "minimum_cross_deconvolution_correlation": minimum,
            "distinction_from_comparison_sources": distinction,
        },
        "compartment_provenance": provenance,
        "supplement_S5_evidence": config["supplement_S5_evidence"],
        "transfer_gates": gates,
        "CT1_type_average_dynamics_verified": True,
        "T5_lobula_CT1_dynamic_phenotype_published": bool(
            gates["separate_lobula_L1_Lo1_dynamic_phenotype_published"]
        ),
        "T5_lobula_CT1_numerical_payload_verified": bool(
            gates["lobula_L1_Lo1_numerical_time_series_payload_available"]
        ),
        "T5_lobula_CT1_dynamics_verified": bool(
            gates["CT1_lobula_L1_Lo1_provenance_available"]
            and gates["lobula_L1_Lo1_numerical_time_series_payload_available"]
        ),
        "T5_CT1_source_dynamics_transfer_authorized": transferable,
        "authorize_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else "TimingModels_CT1_type_average_has_no_lobula_Lo1_or_individual_identity_provenance"
        ),
        "boundary": config["boundary"],
    }
