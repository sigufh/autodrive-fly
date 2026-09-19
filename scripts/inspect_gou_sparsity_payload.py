#!/usr/bin/env python3
"""Freeze a read-only inventory of the Gou Dryad v4 numerical payload.

This script records array structure and finite-value counts.  It deliberately
does not estimate dynamics, fit parameters, or reinterpret fluorescence as
experimental membrane voltage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

FLASH_LEVELS = [1, 2, 4, 8, 15, 30]
MOVING_BAR_LEVELS = [1, 2, 4, 6, 8, 12]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "element_count": int(array.size),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(array).sum()),
        "infinite_count": int(np.isinf(array).sum()),
        "zero_count": int((array == 0).sum()),
        "minimum": float(np.nanmin(array)),
        "maximum": float(np.nanmax(array)),
    }


def _analysis(path: Path) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in loadmat(path, simplify_cells=True).items()
        if not key.startswith("__")
    }
    if len(payload) != 1:
        raise ValueError(f"expected one top-level MAT variable: {path}")
    top_level_name, top_level = next(iter(payload.items()))
    analysis = top_level["analysis"]
    if not isinstance(analysis, dict):
        cells = list(np.asarray(analysis, dtype=object).flat)
        if len(cells) != 1 or not isinstance(cells[0], dict):
            raise ValueError(f"expected one analysis cell: {path}")
        analysis = cells[0]
    return {"top_level_name": top_level_name, "analysis": analysis}


def _identity_summary(analysis: dict[str, Any], fly_axis_size: int) -> dict[str, Any]:
    labels = np.asarray(analysis["fliesUsed"]).reshape(-1)
    values = [int(value) for value in labels]
    unique_values = list(dict.fromkeys(values))
    ind_fly = analysis.get("indFly")
    if ind_fly is None:
        ind_fly_summary = {"present": False}
    else:
        items = list(np.asarray(ind_fly, dtype=object).reshape(-1))
        ind_fly_summary = {
            "present": True,
            "entry_count": len(items),
            "all_entries_empty": all(np.asarray(item).size == 0 for item in items),
        }
    return {
        "numFlies": int(analysis["numFlies"]),
        "fly_axis_size": fly_axis_size,
        "numFlies_matches_fly_axis": int(analysis["numFlies"]) == fly_axis_size,
        "fliesUsed_dtype": str(labels.dtype),
        "fliesUsed_values": values,
        "fliesUsed_entry_count": len(values),
        "fliesUsed_unique_count": len(unique_values),
        "fliesUsed_duplicate_count": len(values) - len(unique_values),
        "fliesUsed_entry_count_matches_fly_axis": len(values) == fly_axis_size,
        "indFly": ind_fly_summary,
        "stable_biological_individual_identity_verified": False,
        "identity_boundary": (
            "fliesUsed contains opaque integer labels whose entry count can differ from "
            "numFlies and the processed fly axis; no archive field defines their biological "
            "identity semantics"
        ),
    }


def _flash(root: Path, relative: str, source: str) -> dict[str, Any]:
    path = root / relative
    loaded = _analysis(path)
    analysis = loaded["analysis"]
    array_names = ["indFilters"] + [
        f"Hz{level}{suffix}"
        for level in FLASH_LEVELS
        for suffix in ("All", "LightImpulse", "DarkImpulse")
    ]
    arrays = {name: _array_summary(analysis[name]) for name in array_names}
    fly_axis_size = arrays["indFilters"]["shape"][0]
    return {
        "source_type": source,
        "protocol": "flash_sparsity_sweep",
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "top_level_name": loaded["top_level_name"],
        "identity": _identity_summary(analysis, fly_axis_size),
        "totROI": int(analysis["totROI"]),
        "condition_axis": {
            "field_prefixes": [f"Hz{level}" for level in FLASH_LEVELS],
            "fill_fractions": [f"{level}/30" for level in FLASH_LEVELS],
        },
        "arrays": arrays,
    }


def _moving_bar(root: Path, relative: str, source: str) -> dict[str, Any]:
    path = root / relative
    loaded = _analysis(path)
    analysis = loaded["analysis"]
    consumed_array_names = [
        "indFilters",
        "kernels",
        "LightImpulseResps",
        "DarkImpulseResps",
    ]
    unconsumed_array_names = [
        "whiteKernels",
        "blackKernels",
        "probeResps",
        "temporal_shift_L_allFlies_out",
    ]
    array_names = consumed_array_names + unconsumed_array_names
    arrays = {name: _array_summary(analysis[name]) for name in array_names}
    fly_axis_size = arrays["kernels"]["shape"][2]
    return {
        "source_type": source,
        "protocol": "moving_bar_sparsity_sweep",
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "top_level_name": loaded["top_level_name"],
        "identity": _identity_summary(analysis, fly_axis_size),
        "totROI": int(analysis["totROI"]),
        "condition_axis": {
            "axis_index_zero_based": 1,
            "fill_fractions": [f"{level}/12" for level in MOVING_BAR_LEVELS],
            "direction_axis_present": False,
            "direction_order_recoverable_from_file_or_Fig6_script": False,
        },
        "field_semantics": {
            "published_Fig6_script_consumed_array_names": consumed_array_names,
            "additional_unconsumed_array_names": unconsumed_array_names,
            "direction_named_field_names": [
                name
                for name in analysis
                if re.search(
                    r"(?:direction|left|right|(?:^|_)pd(?:_|$)|(?:^|_)nd(?:_|$))",
                    name,
                    re.I,
                )
            ],
            "unconsumed_12_condition_arrays_have_published_column_labels": False,
        },
        "arrays": arrays,
    }


def build_inventory(root: Path) -> dict[str, Any]:
    raw = root / "data/raw/gou-sparsity"
    extracted = raw / "extracted/sparsity_Dryad_upload"
    outer = raw / "dryad-version-402168.zip"
    readme = raw / "version-402168/README.md"
    inner = raw / "version-402168/sparsity_Dryad_upload.zip"
    with zipfile.ZipFile(inner) as archive:
        bad_member = archive.testzip()
        members = archive.infolist()
        real_members = [
            item
            for item in members
            if not item.is_dir()
            and "/__MACOSX/" not in f"/{item.filename}"
            and not Path(item.filename).name.startswith("._")
            and Path(item.filename).name != ".DS_Store"
        ]
        real_mat_members = [item for item in real_members if item.filename.endswith(".mat")]

    data_prefix = "data/raw/gou-sparsity/extracted/sparsity_Dryad_upload/data"
    flash_paths = {
        "Mi1": f"{data_prefix}/fig1_Mi1GC6f_flash.mat",
        "Tm1": f"{data_prefix}/fig4_otherNeurons_flash/Tm1GC6f_flash.mat",
        "Tm2": f"{data_prefix}/fig4_otherNeurons_flash/Tm2GC6f_flash.mat",
        "Tm3": f"{data_prefix}/fig4_otherNeurons_flash/Tm3GC6f_flash.mat",
    }
    moving_paths = {
        source: f"{data_prefix}/fig6_otherNeurons_movingBar/{source}GC6f_movingbar.mat"
        for source in ("Mi1", "Tm1", "Tm2", "Tm3")
    }
    flash = {source: _flash(root, path, source) for source, path in flash_paths.items()}
    moving = {source: _moving_bar(root, path, source) for source, path in moving_paths.items()}
    absent_names = ["Mi4", "C3", "Tm4", "Tm9", "CT1"]
    real_names = [item.filename for item in real_members]
    return {
        "protocol": {
            "name": "gou-dryad-v4-numerical-payload-inventory",
            "dataset_doi": "10.5061/dryad.t1g1jwtbs",
            "dataset_id": 142517,
            "version_id": 402168,
            "version_number": 4,
            "read_only": True,
            "parameter_fit": False,
        },
        "archive": {
            "outer_version_download": {
                "path": str(outer.relative_to(root)),
                "bytes": outer.stat().st_size,
                "sha256": _sha256(outer),
            },
            "publisher_archive": {
                "path": str(inner.relative_to(root)),
                "bytes": inner.stat().st_size,
                "sha256": _sha256(inner),
                "zip_integrity_passed": bad_member is None,
                "entry_count_including_directories_and_metadata": len(members),
                "real_file_count": len(real_members),
                "real_mat_file_count": len(real_mat_members),
            },
            "readme": {
                "path": str(readme.relative_to(root)),
                "bytes": readme.stat().st_size,
                "sha256": _sha256(readme),
            },
            "extracted_root": str(extracted.relative_to(root)),
        },
        "source_coverage": {
            "covered_required_sources": ["Mi1", "Tm3", "Tm1", "Tm2"],
            "missing_required_sources": absent_names,
            "missing_source_name_occurrences_in_real_member_paths": {
                name: sum(name.lower() in member.lower() for member in real_names)
                for name in absent_names
            },
        },
        "flash": flash,
        "moving_bar": moving,
        "script_semantics": {
            "flash": {
                "analysis_rate_hz": 30,
                "filter_num_frames_forward": 10,
                "filter_length": 60,
                "filter_point_count": 70,
                "filter_time_start_seconds": -10 / 30,
                "filter_time_end_seconds": 59 / 30,
                "impulse_point_count": 71,
                "impulse_time_start_seconds": -10 / 30,
                "impulse_time_end_seconds": 60 / 30,
                "other_neuron_filter_baseline_matlab_indices_inclusive": [1, 9],
                "impulse_sparse_baseline_matlab_indices_inclusive": [1, 10],
            },
            "moving_bar": {
                "analysis_rate_hz": 60,
                "num_frames_forward": 20,
                "kernel_length": 120,
                "point_count": 141,
                "time_start_seconds": -20 / 60,
                "time_end_seconds": 120 / 60,
                "impulse_baseline_matlab_indices_inclusive": [1, 20],
                "condition_order": [f"{level}/12" for level in MOVING_BAR_LEVELS],
                "direction_order_not_encoded_in_arrays_or_Fig6_script": True,
                "Fig6_consumes_six_sparsity_not_twelve_unlabelled_conditions": True,
            },
            "analysis_axes_are_not_exact_acquisition_timestamps": True,
        },
        "identity_conclusion": {
            "processed_fly_axis_verified": True,
            "opaque_fliesUsed_labels_present": True,
            "stable_biological_individual_ids_verified": False,
            "fit_held_out_identity_disjointness_authorized": False,
        },
        "measurement_boundary": {
            "validated_arrays_are_processed_GCaMP6f_fluorescence": True,
            "experimental_membrane_voltage": False,
            "cross_modality_transfer_authorized": False,
            "target_activity_injection_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/gou-sparsity/gou-v4-numerical-inventory.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_inventory(root), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
