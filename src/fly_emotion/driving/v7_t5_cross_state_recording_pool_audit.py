"""Audit whether saline and octopamine T5 kernels may be pooled."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_kohn_portes_t5_source_kernel_audit import (
    _author_rescaled_temporal,
)

CONFIG = Path("configs/driving-v7-t5-cross-state-recording-pool-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_cross_state_recording_pool_audit.py"
)


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _recording_kernels(path: Path, length: int) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in _load_restricted(path):
        grouped[str(record["recording_id"])].append(
            _author_rescaled_temporal(record, length)
        )
    return {
        recording_id: np.mean(kernels, axis=0)
        for recording_id, kernels in grouped.items()
    }


def evaluate_v7_t5_cross_state_recording_pool_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_config_path = Path(config["source_config"])
    kernel_protocol_path = Path(config["source_kernel_protocol"])
    identity_path = Path(config["identity_history_evidence"])
    source_config = yaml.safe_load(
        (root / source_config_path).read_text(encoding="utf-8")
    )
    identity = json.loads((root / identity_path).read_text(encoding="utf-8"))
    if identity["author_identity_semantics"][
        "recording_id_promoted_to_biological_individual_id"
    ]:
        raise ValueError("recording ID unexpectedly promoted to biological identity")
    kernel_length = int(config["kernel_length"])
    source_paths = {}
    source_results = {}
    for source in config["source_order"]:
        state_kernels = {}
        for state, section in config["states"].items():
            spec = source_config[section][source]
            path = _verify_file(root, spec)
            source_paths[(source, state)] = path
            state_kernels[state] = _recording_kernels(path, kernel_length)
        saline_ids = set(state_kernels["saline"])
        oa_ids = set(state_kernels["OA"])
        shared = sorted(saline_ids & oa_ids)
        if shared != config["expected"]["shared_recording_ids"][source]:
            raise ValueError(f"{source} shared saline/OA recording IDs changed")
        paired = []
        for recording_id in shared:
            saline = state_kernels["saline"][recording_id]
            oa = state_kernels["OA"][recording_id]
            saline_peak = int(np.argmax(np.abs(saline)))
            oa_peak = int(np.argmax(np.abs(oa)))
            paired.append(
                {
                    "recording_id": recording_id,
                    "shape_correlation": float(np.corrcoef(saline, oa)[0, 1]),
                    "OA_minus_saline_absolute_peak_latency_milliseconds": (
                        (oa_peak - saline_peak) * 10.0
                    ),
                    "peak_sign_preserved": bool(
                        np.sign(saline[saline_peak]) == np.sign(oa[oa_peak])
                    ),
                    "OA_to_saline_absolute_peak_ratio": float(
                        np.max(np.abs(oa)) / np.max(np.abs(saline))
                    ),
                }
            )
        source_results[source] = {
            "saline_recording_ids": sorted(saline_ids),
            "OA_recording_ids": sorted(oa_ids),
            "shared_recording_ids": shared,
            "shared_recording_id_count": len(shared),
            "saline_only_recording_ids": sorted(saline_ids - oa_ids),
            "OA_only_recording_ids": sorted(oa_ids - saline_ids),
            "paired_recording_results": paired,
            "paired_shape_correlation_summary": _summary(
                [item["shape_correlation"] for item in paired]
            ),
            "paired_OA_to_saline_absolute_peak_ratio_summary": _summary(
                [item["OA_to_saline_absolute_peak_ratio"] for item in paired]
            ),
            "peak_sign_preserved_for_every_shared_recording_id": all(
                item["peak_sign_preserved"] for item in paired
            ),
        }
    tm2 = source_results["Tm2"]
    if tm2["saline_only_recording_ids"] != config["expected"][
        "Tm2_saline_only_ids"
    ] or tm2["OA_only_recording_ids"] != config["expected"]["Tm2_OA_only_ids"]:
        raise ValueError("Tm2 state-specific recording inventory changed")
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(source_config_path): _sha256(root / source_config_path),
                str(kernel_protocol_path): _sha256(root / kernel_protocol_path),
                str(identity_path): _sha256(root / identity_path),
                **{
                    source_config[config["states"][state]][source]["path"]: (
                        _sha256(path)
                    )
                    for (source, state), path in source_paths.items()
                },
            },
            "kernel_length": kernel_length,
            "kernel_sample_interval_milliseconds": 10.0,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "source_results": source_results,
        "Tm2_OA_only_recording_ids": tm2["OA_only_recording_ids"],
        "Tm2_OA_only_ids_expand_saline_cohort": False,
        "recording_id_is_biological_individual_id": False,
        "saline_and_OA_kernels_exchangeable": False,
        "authorize_cross_state_recording_pool": False,
        "authorize_T5_source_dynamics_fit": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            "OA_only_Tm2_recordings_do_not_expand_saline_cohort_and_recording_"
            "identity_is_not_biological_identity"
        ),
        "boundary": config["boundary"],
    }
