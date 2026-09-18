"""Audit published C3 STRFs against the v7 T4 source-dynamics contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-c3-strf-source-dynamics-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_c3_strf_source_dynamics_audit.py"
)


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324


def _verify_file(root: Path, spec: dict) -> dict:
    path = root / spec["path"]
    size = path.stat().st_size
    if size != int(spec["bytes"]):
        raise ValueError(f"C3 source file size mismatch: {spec['path']}")
    sha256 = _sha256(path)
    if sha256 != spec["sha256"]:
        raise ValueError(f"C3 source file SHA-256 mismatch: {spec['path']}")
    blob = _git_blob_sha1(path)
    if blob != spec["git_blob_sha1"]:
        raise ValueError(f"C3 source file git blob mismatch: {spec['path']}")
    return {
        "path": spec["path"],
        "bytes": size,
        "git_blob_sha1": blob,
        "sha256": sha256,
    }


def _condition_by_name(records: list[dict], name: str) -> dict:
    matches = [item for item in records if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} STRF condition")
    return matches[0]


def _first_corrcoef_row(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values if values.ndim == 1 else values[0]


def _normalized_hash(values: np.ndarray) -> str:
    values = np.asarray(values, dtype="<f8")
    scale = max(float(np.max(values[:40])), np.finfo(np.float64).tiny)
    return hashlib.sha256((values / scale).tobytes()).hexdigest()


def _selected_traces(
    flies: list[dict], axis: str, time: np.ndarray, contract: dict
) -> tuple[int, list[dict]]:
    threshold = float(contract["author_prediction_correlation_threshold"])
    causal_bins = int(contract["causal_peak_bin_count"])
    selected = []
    total = 0
    for fly_index, fly in enumerate(flies):
        if axis not in fly:
            continue
        axis_data = fly[axis]
        scores = _first_corrcoef_row(axis_data["corrcoefs"])
        strfs = np.asarray(axis_data["STRFs"], dtype=np.float64)
        if strfs.shape[0] != len(scores) or tuple(strfs.shape[1:]) != tuple(
            contract["strf_shape"]
        ):
            raise ValueError(f"unexpected C3 {axis} STRF shape")
        if not np.all(np.isfinite(strfs)) or not np.all(np.isfinite(scores)):
            raise ValueError(f"non-finite C3 {axis} STRF data")
        total += len(scores)
        for roi_index, (score, strf) in enumerate(zip(scores, strfs, strict=True)):
            if score < threshold:
                continue
            space_by_time = strf.T
            if axis == "Az" and contract["azimuth_spatial_flip"]:
                space_by_time = np.flipud(space_by_time)
            spatial_index = int(np.argmax(np.max(space_by_time, axis=1)))
            trace = space_by_time[spatial_index]
            peak_index = int(np.argmax(trace[:causal_bins]))
            selected.append(
                {
                    "fly_index": fly_index,
                    "fly_name": str(fly["Flyname"]),
                    "roi_index": roi_index,
                    "prediction_correlation": float(score),
                    "ON_peak_seconds": float(time[peak_index]),
                    "trace": trace,
                }
            )
    return total, selected


def _audit_axis(
    flies: list[dict], axis: str, time: np.ndarray, contract: dict
) -> tuple[dict, np.ndarray]:
    total, selected = _selected_traces(flies, axis, time, contract)
    retained = [
        item
        for item in selected
        if item["ON_peak_seconds"] > float(contract["retain_peak_after_seconds"])
    ]
    if not retained:
        raise ValueError(f"no retained C3 {axis} STRFs")
    traces = np.stack([item["trace"] for item in retained], axis=1)
    mean_trace = np.mean(traces, axis=1)
    weights = np.asarray(
        [item["prediction_correlation"] ** 10 for item in retained], dtype=np.float64
    )
    weighted_trace = np.average(traces, axis=1, weights=weights)
    peak_times = np.asarray([item["ON_peak_seconds"] for item in retained])
    scores = np.asarray([item["prediction_correlation"] for item in retained])
    causal_bins = int(contract["causal_peak_bin_count"])
    alternating_correlation = float(
        np.corrcoef(
            np.mean(traces[:causal_bins, ::2], axis=1),
            np.mean(traces[:causal_bins, 1::2], axis=1),
        )[0, 1]
    )
    fly_loo = {}
    for fly_name in sorted({item["fly_name"] for item in retained}):
        keep = [
            index
            for index, item in enumerate(retained)
            if item["fly_name"] != fly_name
        ]
        fly_loo[fly_name] = float(
            np.corrcoef(
                mean_trace[:causal_bins],
                np.mean(traces[:causal_bins, keep], axis=1),
            )[0, 1]
        )
    result = {
        "total_ROIs": total,
        "selected_ROIs": len(selected),
        "retained_ROIs": len(retained),
        "retained_fly_count": len({item["fly_name"] for item in retained}),
        "prediction_correlation_minimum": float(np.min(scores)),
        "prediction_correlation_median": float(np.median(scores)),
        "prediction_correlation_maximum": float(np.max(scores)),
        "mean_ON_peak_seconds": float(np.mean(peak_times)),
        "median_ON_peak_seconds": float(np.median(peak_times)),
        "ON_peak_standard_deviation_seconds": float(np.std(peak_times, ddof=1)),
        "alternating_ROI_split_causal_kernel_correlation": alternating_correlation,
        "leave_one_fly_out_causal_kernel_correlations": fly_loo,
        "unweighted_normalized_temporal_kernel_sha256": _normalized_hash(mean_trace),
        "author_weighted_normalized_temporal_kernel_sha256": _normalized_hash(
            weighted_trace
        ),
    }
    return result, mean_trace


def evaluate_v7_c3_strf_source_dynamics_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    repository = config["repository"]
    head_path = root / repository["checkout_head_path"]
    if head_path.stat().st_size != int(repository["checkout_head_bytes"]):
        raise ValueError("C3 source repository HEAD size mismatch")
    if _sha256(head_path) != repository["checkout_head_sha256"]:
        raise ValueError("C3 source repository HEAD SHA-256 mismatch")
    checked_out_revision = head_path.read_text(encoding="utf-8").strip()
    if checked_out_revision != repository["archived_revision"]:
        raise ValueError("C3 source repository revision mismatch")
    source = _verify_file(root, config["source_data"])
    source_code = [_verify_file(root, spec) for spec in config["source_code"]]
    contract = config["analysis_contract"]
    transfer_contract = config["transfer_contract"]
    timebase_path = Path(transfer_contract["current_timebase_evidence"])
    timebase = json.loads((root / timebase_path).read_text(encoding="utf-8"))
    arenz_path = Path(transfer_contract["arenz_source_dynamics_evidence"])
    arenz = json.loads((root / arenz_path).read_text(encoding="utf-8"))
    payload = scipy.io.loadmat(root / source["path"], simplify_cells=True)
    records = payload["RF_DATA"]
    if not isinstance(records, list):
        records = list(records)
    c3 = _condition_by_name(records, contract["condition"])
    sample_interval = float(contract["sample_interval_seconds"])
    time = np.arange(
        float(contract["time_start_seconds"]),
        float(contract["time_stop_seconds"]) + sample_interval / 2.0,
        sample_interval,
    )
    if len(time) != int(contract["strf_shape"][0]):
        raise ValueError("C3 STRF timebase length mismatch")
    axis_outputs = {
        axis: _audit_axis(c3["DATA"], axis, time, contract)
        for axis in contract["axes"]
    }
    axes = {axis: output[0] for axis, output in axis_outputs.items()}
    tolerance = float(contract["reproduction_absolute_tolerance_seconds"])
    reproduction = {
        axis: bool(
            result["selected_ROIs"]
            == int(contract["expected_reproduction"][axis]["selected_ROIs"])
            and result["retained_ROIs"]
            == int(contract["expected_reproduction"][axis]["retained_ROIs"])
            and abs(
                result["mean_ON_peak_seconds"]
                - float(
                    contract["expected_reproduction"][axis]["mean_ON_peak_seconds"]
                )
            )
            <= tolerance
        )
        for axis, result in axes.items()
    }
    causal_bins = int(contract["causal_peak_bin_count"])
    cross_axis_correlation = float(
        np.corrcoef(
            axis_outputs["Az"][1][:causal_bins],
            axis_outputs["El"][1][:causal_bins],
        )[0, 1]
    )
    required_sources = set(transfer_contract["required_source_types"])
    arenz_contract = arenz["current_v7_source_contract"]
    if set(arenz_contract["source_types"]) != required_sources:
        raise ValueError("Arenz audit and C3 audit source contracts differ")
    arenz_sources = set(arenz_contract["covered_by_Arenz"])
    directly_measured_sources = set(arenz_sources)
    directly_measured_sources.add("C3")
    transfer_gates = {
        "published_C3_numerical_STRFs_available": True,
        "author_selection_and_ON_peak_summary_reproduced": all(reproduction.values()),
        "same_measurement_and_parameterization_as_Arenz_filters": False,
        "C3_membrane_voltage_or_validated_deconvolved_kernel_available": False,
        "C3_analytic_filter_parameters_available": False,
        "physical_v7_timebase_available": timebase["identifiability"][
            "physical_timebase_identified"
        ],
        "stable_recorded_cell_to_MaleCNS_mapping_available": False,
        "cross_paper_state_unit_mapping_available": False,
    }
    transferable = all(transfer_gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(timebase_path): _sha256(root / timebase_path),
                str(arenz_path): _sha256(root / arenz_path),
            },
            "paper_doi": config["paper"]["doi"],
            "repository_url": config["repository"]["url"],
            "archived_revision": repository["archived_revision"],
            "checked_out_revision": checked_out_revision,
            "checkout_head": {
                "path": repository["checkout_head_path"],
                "bytes": head_path.stat().st_size,
                "sha256": _sha256(head_path),
            },
            "source_data": source,
            "source_code": source_code,
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "author_analysis_contract": contract,
        "C3_dataset": {
            "fly_count": len(c3["DATA"]),
            "STRF_units": "stimulus_response_correlation",
            "time_bin_seconds": sample_interval,
            "axes": axes,
            "cross_axis_unweighted_causal_kernel_correlation": (
                cross_axis_correlation
            ),
            "author_reproduction_by_axis": reproduction,
        },
        "combined_source_contract": {
            "required_source_types": transfer_contract["required_source_types"],
            "Arenz_analytic_filter_source_types": sorted(arenz_sources),
            "C3_direct_temporal_measurement_available": True,
            "all_source_types_have_some_direct_temporal_evidence": (
                directly_measured_sources == required_sources
            ),
            "all_source_types_share_one_transferable_parameterization": False,
        },
        "transfer_gates": transfer_gates,
        "C3_STRF_numerical_data_verified": True,
        "C3_source_filter_candidate_authorized": transferable,
        "advance_to_functional_precheck": transferable,
        "blocking_data_requirements": [
            name for name, available in transfer_gates.items() if not available
        ],
        "stop_reason": (
            None
            if transferable
            else "C3_calcium_STRF_not_a_cross_paper_or_v7_state_kernel"
        ),
        "boundary": config["boundary"],
    }
