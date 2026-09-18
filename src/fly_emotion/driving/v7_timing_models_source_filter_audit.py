"""Audit Clark Lab TimingModels measured source filters against the v7 T4 contract."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-timing-models-source-filter-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_timing_models_source_filter_audit.py"
)


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"TimingModels file size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"TimingModels file SHA-256 mismatch: {path}")
    return path


def _matlab_string(source: h5py.File, reference: h5py.Reference) -> str:
    return "".join(chr(int(value)) for value in source[reference][()].ravel())


def _load_filter_file(path: Path) -> dict:
    with h5py.File(path, "r") as source:
        names = [
            _matlab_string(source, reference)
            for reference in source["filterList"][()].ravel()
        ]
        labels = [
            _matlab_string(source, reference)
            for reference in source["filterLabel"][()].ravel()
        ]
        matrix = source["filterMat"][()].T.astype(np.float64)
        sem = source["filterSem"][()].T.astype(np.float64)
        time = source["tSec"][()].ravel().astype(np.float64)
        sample_interval = float(source["dtFilter"][0, 0])
    return {
        "filter_list": names,
        "filter_labels": labels,
        "matrix": matrix,
        "sem": sem,
        "time": time,
        "sample_interval_seconds": sample_interval,
    }


def _unit(values: np.ndarray) -> np.ndarray:
    centered = values - values[0]
    return centered / max(float(np.linalg.norm(centered)), np.finfo(float).tiny)


def evaluate_v7_timing_models_source_filter_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    head_path = root / config["repository"]["checkout_commit_path"]
    checked_out_commit = head_path.read_text(encoding="utf-8").strip()
    if checked_out_commit != config["repository"]["commit"]:
        raise ValueError("TimingModels checkout commit mismatch")
    for spec in config["code_files"]:
        _verify_file(root, spec)
    datasets = []
    for spec in config["filter_files"]:
        path = _verify_file(root, spec)
        loaded = _load_filter_file(path)
        if loaded["filter_list"] != config["expected_filter_list"]:
            raise ValueError("TimingModels filter list mismatch")
        if list(loaded["matrix"].shape) != config["expected_shape"]:
            raise ValueError("TimingModels filter matrix shape mismatch")
        if not np.isclose(
            loaded["sample_interval_seconds"],
            float(config["expected_sample_interval_seconds"]),
        ):
            raise ValueError("TimingModels sample interval mismatch")
        expected_start, expected_stop = map(float, config["expected_time_seconds"])
        if not np.isclose(loaded["time"][0], expected_start) or not np.isclose(
            loaded["time"][-1], expected_stop
        ):
            raise ValueError("TimingModels time axis mismatch")
        datasets.append({"spec": spec, **loaded})
    available = set(datasets[0]["filter_list"])
    required = set(config["current_source_contract"])
    covered = sorted(required & available)
    missing = sorted(required - available)
    pairwise = {}
    minimum_by_source = {}
    for source_type in covered:
        values = []
        for first, second in combinations(datasets, 2):
            first_kernel = _unit(
                first["matrix"][:, first["filter_list"].index(source_type)]
            )
            second_kernel = _unit(
                second["matrix"][:, second["filter_list"].index(source_type)]
            )
            correlation = float(np.corrcoef(first_kernel, second_kernel)[0, 1])
            key = (
                f"{first['spec']['calcium_deconvolution_milliseconds']}_vs_"
                f"{second['spec']['calcium_deconvolution_milliseconds']}_ms"
            )
            pairwise.setdefault(source_type, {})[key] = correlation
            values.append(correlation)
        minimum_by_source[source_type] = min(values)
    threshold = float(
        config["gates"]["minimum_pairwise_deconvolution_kernel_correlation"]
    )
    stability_gates = {
        source_type: value >= threshold for source_type, value in minimum_by_source.items()
    }
    gates = {
        "repository_and_files_verified": True,
        "covered_source_filters_stable_across_deconvolution_assumptions": all(
            stability_gates.values()
        ),
        "every_current_source_type_covered": not missing,
        "independent_cell_holdout_available": False,
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
            },
            "repository_url": config["repository"]["url"],
            "repository_commit": checked_out_commit,
            "verified_files": [
                {
                    "path": spec["path"],
                    "bytes": (root / spec["path"]).stat().st_size,
                    "sha256": _sha256(root / spec["path"]),
                }
                for spec in [*config["filter_files"], *config["code_files"]]
            ],
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "filter_data": {
            "calcium_deconvolution_milliseconds": [
                item["spec"]["calcium_deconvolution_milliseconds"]
                for item in datasets
            ],
            "filter_list": datasets[0]["filter_list"],
            "filter_shape": list(datasets[0]["matrix"].shape),
            "sample_interval_seconds": datasets[0]["sample_interval_seconds"],
            "time_start_seconds": float(datasets[0]["time"][0]),
            "time_stop_seconds": float(datasets[0]["time"][-1]),
            "pairwise_deconvolution_kernel_correlations": pairwise,
            "minimum_pairwise_correlation_by_current_source": minimum_by_source,
            "stability_gates": stability_gates,
        },
        "current_v7_source_contract": {
            "required_sources": config["current_source_contract"],
            "covered_sources": covered,
            "missing_sources": missing,
            "coverage_fraction": len(covered) / len(required),
            "noncontract_model_sources": sorted(available - required),
        },
        "transfer_gates": gates,
        "covered_source_filter_stability_verified": all(stability_gates.values()),
        "complete_source_filter_candidate_authorized": transferable,
        "advance_to_T4_functional_precheck": transferable,
        "blocking_requirements": [
            name for name, passed in gates.items() if not passed
        ],
        "stop_reason": (
            None
            if transferable
            else "TimingModels_filters_missing_C3_and_independent_identity_validation"
        ),
        "boundary": config["boundary"],
    }
