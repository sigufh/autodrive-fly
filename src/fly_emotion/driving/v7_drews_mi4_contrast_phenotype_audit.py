"""Reproduce the published Drews Mi4 contrast phenotype without fitting."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_drews_mi4_sporar_c3_boundary_audit import (
    _load_restricted,
)
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-drews-mi4-contrast-phenotype-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_drews_mi4_contrast_phenotype_audit.py"
)
LOADER_IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_drews_mi4_sporar_c3_boundary_audit.py"
)


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"Drews phenotype input identity changed: {path}")
    return path


def _fly_label(name: str) -> str:
    match = re.match(r"^(fly\d+)_", name)
    if match is None:
        raise ValueError(f"Drews ROI label cannot be reduced to a fly label: {name}")
    return match.group(1)


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Drews phenotype metric changed: {label}")


def evaluate_v7_drews_mi4_contrast_phenotype_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    boundary_path = Path(config["source_boundary_evidence"])
    boundary = json.loads((root / boundary_path).read_text(encoding="utf-8"))
    if (
        not boundary["Drews_2020"]["Mi4_payload"][
            "individual_ROI_trial_numeric_time_series_verified"
        ]
        or boundary["Drews_2020"]["experimental_membrane_voltage_verified"]
        or boundary["Drews_2020"]["C3_direct_neural_recording_verified"]
    ):
        raise ValueError("Drews source boundary changed")

    paper_path = _verify_file(root, config["paper"]["dissertation"])
    paper_text = " ".join(
        " ".join((page.extract_text() or "").split())
        for page in PdfReader(paper_path, strict=False).pages
    )
    required_phrases = (
        "moving for 4 seconds after stimulus onset",
        "the driving foreground contrast frequency was 1 Hz",
        "we evaluated the amplitude of the 1 Hz component of the signal",
        "Amplitudes were averaged over trials and normalized to the maximum",
        "tonic Mi4, Mi9, and Tm9 showed similar tuning as L1–5 and again little "
        "surround-dependency",
    )
    if any(phrase not in paper_text for phrase in required_phrases):
        raise ValueError("Drews phenotype analysis prose changed")

    payload_path = _verify_file(root, config["payload"])
    frame, globals_used = _load_restricted(payload_path)
    analysis = config["analysis"]
    selected = frame.reset_index()
    start, stop = (float(value) for value in analysis["stimulus_motion_window_seconds"])
    selected = selected[
        selected["cell_type"].isin(analysis["cell_types"])
        & (selected["time"] >= start)
        & (selected["time"] < stop)
    ].copy()
    if selected["signal"].isna().any():
        raise ValueError("Drews motion window contains missing values")
    frequency = float(analysis["stimulus_frequency_hz"])
    selected["weighted_signal"] = selected["signal"].to_numpy() * np.exp(
        -2j * np.pi * frequency * selected["time"].to_numpy()
    )
    group_keys = ["cell_type", "name", "fg_contrast", "bg_contrast", "trial"]
    group_sizes = selected.groupby(group_keys, sort=True).size()
    expected_samples = int(analysis["samples_per_trial_in_motion_window"])
    if set(group_sizes.tolist()) != {expected_samples}:
        raise ValueError("Drews motion-window sample count changed")
    amplitudes = (
        selected.groupby(group_keys, sort=True)["weighted_signal"]
        .mean()
        .abs()
        .rename("F1_amplitude")
        .reset_index()
    )
    if sorted(amplitudes["trial"].unique().tolist()) != analysis["trial_labels"]:
        raise ValueError("Drews trial labels changed")
    by_roi = (
        amplitudes.groupby(group_keys[:-1], sort=True)["F1_amplitude"]
        .mean()
        .reset_index()
    )
    by_roi["normalized_F1"] = by_roi["F1_amplitude"] / by_roi.groupby(
        ["cell_type", "name"]
    )["F1_amplitude"].transform("max")
    by_roi["fly"] = by_roi["name"].map(_fly_label)

    results = {}
    for cell_type in analysis["cell_types"]:
        subset = by_roi[by_roi["cell_type"] == cell_type]
        expected = config["expected"][cell_type]
        pairs = subset.pivot(
            index=["fly", "name", "fg_contrast"],
            columns="bg_contrast",
            values="normalized_F1",
        )
        background_zero, background_full = analysis["compared_background_contrasts"]
        fly_delta = (pairs[background_full] - pairs[background_zero]).groupby(
            level="fly"
        ).mean()
        leave_one_fly_out = [
            float(np.median(fly_delta.drop(index=fly))) for fly in fly_delta.index
        ]
        result = {
            "ROI_count": int(subset["name"].nunique()),
            "pseudonymous_fly_count": int(subset["fly"].nunique()),
            "condition_trial_group_count": int(len(group_sizes.xs(cell_type))),
            "median_fly_delta": float(np.median(fly_delta)),
            "positive_fly_count": int((fly_delta > 0).sum()),
            "negative_fly_count": int((fly_delta < 0).sum()),
            "leave_one_fly_out_median_min": min(leave_one_fly_out),
            "leave_one_fly_out_median_max": max(leave_one_fly_out),
        }
        for key in (
            "ROI_count",
            "pseudonymous_fly_count",
            "group_count",
            "positive_fly_count",
            "negative_fly_count",
        ):
            result_key = "condition_trial_group_count" if key == "group_count" else key
            if result[result_key] != int(expected[key]):
                raise ValueError(f"Drews {cell_type} count changed: {key}")
        for key in (
            "median_fly_delta",
            "leave_one_fly_out_median_min",
            "leave_one_fly_out_median_max",
        ):
            _assert_close(result[key], float(expected[key]), f"{cell_type}.{key}")
        results[cell_type] = result

    mi4 = results["Mi4"]
    transient = [results["Mi1"], results["Tm3"]]
    descriptive_reproduction = bool(
        mi4["median_fly_delta"] > 0
        and mi4["leave_one_fly_out_median_min"] > 0
        and all(item["median_fly_delta"] < 0 for item in transient)
        and all(item["leave_one_fly_out_median_max"] < 0 for item in transient)
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(LOADER_IMPLEMENTATION): _sha256(root / LOADER_IMPLEMENTATION),
                str(boundary_path): _sha256(root / boundary_path),
            },
            "raw_snapshot_identity": {
                config["paper"]["dissertation"]["path"]: _sha256(paper_path),
                config["payload"]["path"]: _sha256(payload_path),
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "paper": {key: config["paper"][key] for key in ("title", "doi")},
        "analysis_contract": analysis,
        "pickle_security": {
            "restricted_unpickler_used": True,
            "globals_used": globals_used,
            "arbitrary_repository_code_executed": False,
        },
        "cell_type_results": results,
        "Mi4_tonic_weak_surround_suppression_qualitatively_reproduced": (
            descriptive_reproduction
        ),
        "independent_study_Mi4_calcium_phenotype_available": True,
        "preregistered_independent_dynamic_validation_available": False,
        "full_Mi4_temporal_kernel_identified": False,
        "experimental_Mi4_membrane_voltage_available": False,
        "C3_direct_recording_available": False,
        "recording_to_MaleCNS_body_crosswalk_found": False,
        "authorize_Mi4_source_kernel_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "stop_reason": (
            "retrospective_single_frequency_calcium_phenotype_is_not_a_full_"
            "Mi4_C3_voltage_kernel_validation"
        ),
        "boundary": config["boundary"],
    }
