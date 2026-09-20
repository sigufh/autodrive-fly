"""Audit whether Gou Fig. 6 source arrays expose usable motion directions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import yaml
from scipy.io import loadmat

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-gou-moving-bar-direction-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_gou_moving_bar_direction_audit.py")


def _verified_path(root: Path, spec: dict, label: str) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError(f"Gou {label} payload changed")
    return path


def _analysis(path: Path) -> dict:
    payload = {
        key: value
        for key, value in loadmat(path, simplify_cells=True).items()
        if not key.startswith("__")
    }
    if set(payload) != {"sparseSweep_Analysis"}:
        raise ValueError("unexpected Gou moving-bar top-level MAT variables")
    analysis = payload["sparseSweep_Analysis"]["analysis"]
    if not isinstance(analysis, dict):
        items = list(np.asarray(analysis, dtype=object).flat)
        if len(items) != 1 or not isinstance(items[0], dict):
            raise ValueError("unexpected Gou moving-bar analysis structure")
        analysis = items[0]
    return analysis


def evaluate_v7_gou_moving_bar_direction_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_audit_path = Path(config["source_audit"])
    source_audit = json.loads((root / source_audit_path).read_text(encoding="utf-8"))
    dandi_path = Path(config["dandi_stimulus_metadata_audit"])
    dandi = json.loads((root / dandi_path).read_text(encoding="utf-8"))
    inventory_path = _verified_path(root, config["inventory"], "inventory")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    readme_path = _verified_path(root, config["readme"], "README")
    fig6_path = _verified_path(root, config["fig6_script"], "Fig. 6 script")
    fig7_path = _verified_path(
        root, config["fig7_direction_reference_script"], "Fig. 7 direction script"
    )
    readme = readme_path.read_text(encoding="utf-8")
    fig6 = fig6_path.read_text(encoding="utf-8")
    fig7 = fig7_path.read_text(encoding="utf-8")
    required_fig6_snippets = (
        "sparsity=[1/12 2/12 4/12 6/12 8/12 12/12]",
        "kernels = sparseSweep_Analysis.analysis{1, 1}.kernels*60",
        "LightImpulseResps = sparseSweep_Analysis.analysis{1, 1}.LightImpulseResps",
        "DarkImpulseResps = sparseSweep_Analysis.analysis{1, 1}.DarkImpulseResps",
    )
    if not all(snippet in fig6 for snippet in required_fig6_snippets):
        raise ValueError("Gou Fig. 6 script semantics changed")
    if "Conditions are different sparsity levels" not in readme:
        raise ValueError("Gou README moving-bar condition semantics changed")
    fig7_direction_rule_present = all(
        snippet in fig7
        for snippet in (
            "T4Left_kernels(:,2:2:end",
            "T4Right_kernels(:,1:2:end",
            "T4_PD_kernels",
            "T4_ND_kernels",
        )
    )
    if not fig7_direction_rule_present:
        raise ValueError("Gou Fig. 7 target direction rule changed")

    consumed = list(config["expected_consumed_arrays"])
    unconsumed = list(config["expected_unconsumed_arrays"])
    source_results = {}
    payload_paths = {}
    direction_pattern = re.compile(
        r"(?:direction|left|right|(?:^|_)pd(?:_|$)|(?:^|_)nd(?:_|$))", re.I
    )
    for source in config["required_sources"]:
        spec = config["moving_bar_files"][source]
        path = _verified_path(root, spec, f"{source} moving-bar")
        payload_paths[source] = path
        analysis = _analysis(path)
        shapes = {name: list(np.asarray(analysis[name]).shape) for name in consumed + unconsumed}
        expected_shapes = {
            name: [*shape, int(spec["fly_axis_size"])]
            for name, shape in {
                **config["expected_consumed_arrays"],
                **config["expected_unconsumed_arrays"],
            }.items()
        }
        if shapes != expected_shapes:
            raise ValueError(f"Gou {source} moving-bar array shapes changed")
        source_results[source] = {
            "fly_axis_size": int(spec["fly_axis_size"]),
            "array_shapes": shapes,
            "Fig6_consumed_arrays": consumed,
            "Fig6_unconsumed_arrays": unconsumed,
            "direction_named_fields": sorted(
                name for name in analysis if direction_pattern.search(name)
            ),
            "six_condition_axis_identified_as_sparsity": True,
            "twelve_condition_array_column_labels_available": False,
            "record_level_direction_axis_identifiable": False,
        }

    source_statements = source_audit["source_evidence"]
    if any(
        source_statements[source]["processed_moving_bar_direction_labels_available"]
        for source in config["required_sources"]
    ):
        raise ValueError("Gou source audit overstates moving-bar direction labels")
    direction_identifiable = all(
        item["record_level_direction_axis_identifiable"] for item in source_results.values()
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(source_audit_path): _sha256(root / source_audit_path),
                str(dandi_path): _sha256(root / dandi_path),
                str(inventory_path.relative_to(root)): _sha256(inventory_path),
                str(readme_path.relative_to(root)): _sha256(readme_path),
                str(fig6_path.relative_to(root)): _sha256(fig6_path),
                str(fig7_path.relative_to(root)): _sha256(fig7_path),
                **{str(path.relative_to(root)): _sha256(path) for path in payload_paths.values()},
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "source_results": source_results,
        "script_semantics": {
            "Fig6_condition_order": config["expected_sparsity_order"],
            "Fig6_condition_axis_is_sparsity": True,
            "Fig6_reads_unconsumed_twelve_condition_arrays": False,
            "Fig7_target_direction_rule_explicit": fig7_direction_rule_present,
            "Fig7_target_direction_rule_applied_to_Fig6_source_arrays": False,
        },
        "inventory_consistency": {
            "direction_order_not_encoded_in_arrays_or_Fig6_script": inventory["script_semantics"][
                "moving_bar"
            ]["direction_order_not_encoded_in_arrays_or_Fig6_script"],
            "processed_source_direction_labels_available": False,
        },
        "DANDI_stimulus_metadata": {
            "Mi1_asset_count": dandi["source_results"]["Mi1"]["asset_count"],
            "Tm3_asset_count": dandi["source_results"]["Tm3"]["asset_count"],
            "all_282_assets_indexed": dandi["complete_index"]["successful_index_count"] == 282,
            "Mi1_Tm3_stimulus_metadata_available": dandi[
                "Mi1_Tm3_NWB_stimulus_metadata_available"
            ],
            "Mi1_Tm3_direction_or_position_fields_available": dandi[
                "Mi1_Tm3_NWB_direction_or_position_fields_available"
            ],
            "Dryad_row_crosswalk_available": dandi[
                "Dryad_processed_rows_to_DANDI_subject_crosswalk_available"
            ],
        },
        "Gou_Mi1_Tm3_record_level_direction_axis_identifiable": direction_identifiable,
        "authorize_direction_invariance_audit": direction_identifiable,
        "direction_invariance_evaluated": False,
        "authorize_T4_source_kernel_transfer": False,
        "stop_reason": "Fig6_source_array_direction_labels_not_identifiable",
        "boundary": config["boundary"],
    }
