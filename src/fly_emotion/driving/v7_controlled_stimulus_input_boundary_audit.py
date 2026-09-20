"""Audit controlled-stimulus coverage and the external R1-R6 input boundary."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import (
    V7_TARGET_TYPES,
    V7VisualProbe,
    build_controlled_stimuli,
)
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc_typed_screen import _condition_stimuli
from fly_emotion.driving.v7_stage1_split import build_stage1_split

CONFIG = Path(
    "configs/driving-v7-controlled-stimulus-input-boundary-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_controlled_stimulus_input_boundary_audit.py"
)


class _DriveRecordingProbe(V7VisualProbe):
    """Record drive support while avoiding an unnecessary full graph propagation."""

    def __init__(self, root: Path) -> None:
        super().__init__(
            root,
            brain_substeps=1,
            baseline_frames=1,
            retinal_backend="linear_luminance",
        )
        self.drive_call_count = 0
        self.outside_retina_nodes: set[int] = set()

    def _advance(
        self, state: np.ndarray, drive: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        self.drive_call_count += 1
        support = np.flatnonzero(drive)
        outside = np.setdiff1d(support, self.retina.node_indices, assume_unique=False)
        self.outside_retina_nodes.update(int(node) for node in outside)
        return state.copy()


def _assignment_slice(target: ast.expr) -> str | None:
    if not isinstance(target, ast.Subscript) or not isinstance(target.ctx, ast.Store):
        return None
    if not isinstance(target.value, ast.Name) or target.value.id != "drive":
        return None
    return ast.unparse(target.slice)


def _audit_drive_assignments(root: Path) -> dict:
    records = []
    paths = sorted((root / "src/fly_emotion/driving").glob("v7*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                index = _assignment_slice(target)
                if index is not None:
                    records.append(
                        {
                            "path": str(path.relative_to(root)),
                            "line": int(node.lineno),
                            "index_expression": index,
                        }
                    )
    allowed = {
        "self.retina.node_indices",
        "probe.retina.node_indices",
        "self.probe.retina.node_indices",
    }
    violations = [record for record in records if record["index_expression"] not in allowed]
    return {
        "scanned_module_count": len(paths),
        "drive_assignment_count": len(records),
        "files_with_drive_assignments": len({record["path"] for record in records}),
        "index_expression_counts": dict(
            sorted(Counter(record["index_expression"] for record in records).items())
        ),
        "violations": violations,
        "all_drive_subscript_assignments_target_retina_indices": not violations,
        "module_inventory_sha256": hashlib.sha256(
            "\n".join(str(path.relative_to(root)) for path in paths).encode()
        ).hexdigest(),
    }


def _condition_counts(stimuli) -> dict[str, int]:
    return dict(sorted(Counter(item.family for item in stimuli).items()))


def evaluate_v7_controlled_stimulus_input_boundary_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    v7_config_path = Path(config["v7_config"])
    controlled_evidence_path = Path(config["controlled_vision_evidence"])
    split_config_path = Path(config["stage1_split_config"])
    split_evidence_path = Path(config["stage1_split_evidence"])
    split_implementation_path = Path(config["stage1_split_implementation"])
    input_evidence_path = Path(config["stage1_input_evidence"])
    typed_config_path = Path(config["typed_lplc_config"])
    typed_evidence_path = Path(config["typed_lplc_evidence"])
    typed_implementation_path = Path(config["typed_lplc_implementation"])
    v7 = yaml.safe_load((root / v7_config_path).read_text(encoding="utf-8"))
    controlled_evidence = json.loads(
        (root / controlled_evidence_path).read_text(encoding="utf-8")
    )
    split_config = yaml.safe_load((root / split_config_path).read_text(encoding="utf-8"))
    split_evidence = json.loads((root / split_evidence_path).read_text(encoding="utf-8"))
    input_evidence = json.loads((root / input_evidence_path).read_text(encoding="utf-8"))
    typed_config = yaml.safe_load((root / typed_config_path).read_text(encoding="utf-8"))
    typed_evidence = json.loads((root / typed_evidence_path).read_text(encoding="utf-8"))
    expected = config["expected"]

    visual = v7["controlled_vision"]
    base = build_controlled_stimuli(
        width=int(visual["width"]),
        height=int(visual["height"]),
        frames=int(visual["frames_per_stimulus"]),
    )
    base_conditions = [[item.family, item.polarity, item.direction] for item in base]
    required_base = expected["required_base_conditions"]
    missing_base = [condition for condition in required_base if condition not in base_conditions]
    base_by_name = {item.name: item for item in base}
    base_mirror_exact = all(
        item.mirror_of is None
        or (
            item.mirror_of in base_by_name
            and np.array_equal(item.frames[:, :, ::-1], base_by_name[item.mirror_of].frames)
        )
        for item in base
    )
    base_finite_bounded = all(
        np.isfinite(item.frames).all()
        and np.all(item.frames >= 0.0)
        and np.all(item.frames <= 1.0)
        for item in base
    )
    if len(base) != int(expected["base_stimulus_count"]):
        raise ValueError("base controlled-stimulus count changed")
    if missing_base:
        raise ValueError(f"required base controlled stimuli missing: {missing_base}")

    splits = build_stage1_split(split_config)
    if list(splits) != expected["split_names"]:
        raise ValueError("stage-1 split order changed")
    split_results = {}
    for name, stimuli in splits.items():
        manifest = split_evidence["split_manifests"][name]
        digest = hashlib.sha256()
        for item in stimuli:
            digest.update(item.identity.encode())
            digest.update(bytes.fromhex(item.sha256))
        by_identity = {item.identity: item for item in stimuli}
        mirror_exact = all(
            item.mirror_of in by_identity
            and np.array_equal(item.frames[:, :, ::-1], by_identity[item.mirror_of].frames)
            for item in stimuli
        )
        counts = _condition_counts(stimuli)
        split_results[name] = {
            "stimulus_count": len(stimuli),
            "family_counts": counts,
            "aggregate_sha256": digest.hexdigest(),
            "matches_frozen_manifest": (
                len(stimuli) == manifest["stimulus_count"]
                and counts == manifest["families"]
                and digest.hexdigest() == manifest["aggregate_sha256"]
            ),
            "exact_horizontal_mirror_pairs": mirror_exact,
            "individual_stimuli_disclosed": manifest["stimuli"] is not None,
        }
        if len(stimuli) != int(expected["stimuli_per_split"]):
            raise ValueError(f"stage-1 stimulus count changed: {name}")
        if counts != expected["split_family_counts"]:
            raise ValueError(f"stage-1 family counts changed: {name}")
        if not split_results[name]["matches_frozen_manifest"] or not mirror_exact:
            raise ValueError(f"stage-1 split manifest or mirrors changed: {name}")

    typed_names = []
    typed_hashes = {}
    for condition in typed_config["conditions"]:
        stimuli = _condition_stimuli(condition, typed_config)
        typed_names.extend(item.name for item in stimuli.values())
        typed_hashes.update(
            {
                item.name: hashlib.sha256(item.frames.tobytes()).hexdigest()
                for item in stimuli.values()
            }
        )
    frozen_typed = {
        item["identity"]: item["frame_sha256"]
        for item in typed_evidence["stimulus_manifest"]
    }
    required_typed = set(expected["required_typed_lplc_stimuli"] )
    typed_suffixes = {name.split(":", 1)[1] for name in typed_names}
    typed_complete = required_typed == typed_suffixes
    typed_matches_manifest = typed_hashes == frozen_typed
    if len(typed_config["conditions"]) != int(expected["typed_lplc_condition_count"]):
        raise ValueError("typed LPLC condition count changed")
    if len(typed_names) != int(expected["typed_lplc_condition_count"]) * int(
        expected["typed_lplc_stimuli_per_condition"]
    ):
        raise ValueError("typed LPLC stimulus count changed")
    if not typed_complete or not typed_matches_manifest:
        raise ValueError("typed LPLC stimulus inventory changed")

    probe = _DriveRecordingProbe(root)
    for stimulus in base:
        probe.run(stimulus)
    retinal_types = set(probe.node_types[probe.retina.node_indices].tolist())
    target_nodes = np.unique(np.concatenate(list(probe.populations.values())))
    target_overlap = int(np.intersect1d(probe.retina.node_indices, target_nodes).size)
    static_assignments = _audit_drive_assignments(root)
    v7_module_paths = sorted((root / "src/fly_emotion/driving").glob("v7*.py"))
    if retinal_types != {"R1-R6"}:
        raise ValueError("retinal input nodes are not exclusively R1-R6")
    if probe.outside_retina_nodes or target_overlap:
        raise ValueError("external visual drive reached a non-retinal or target node")
    if not static_assignments["all_drive_subscript_assignments_target_retina_indices"]:
        raise ValueError("v7 drive assignment outside retina indices detected")

    gates = {
        "required_base_stimulus_conditions_complete": not missing_base,
        "base_stimuli_finite_and_bounded": base_finite_bounded,
        "base_stimulus_mirrors_exact": base_mirror_exact,
        "all_four_split_manifests_reproduced": all(
            result["matches_frozen_manifest"] for result in split_results.values()
        ),
        "all_four_split_mirrors_exact": all(
            result["exact_horizontal_mirror_pairs"] for result in split_results.values()
        ),
        "development_sampled_R1_R6_input_gates_pass": input_evidence[
            "input_gates_pass"
        ],
        "typed_LPLC1_LPLC2_LC4_stimulus_inventory_complete": typed_complete,
        "typed_stimulus_frames_match_frozen_evidence": typed_matches_manifest,
        "dynamic_external_drive_support_is_R1_R6_only": not probe.outside_retina_nodes,
        "all_static_drive_assignments_target_retina_indices": static_assignments[
            "all_drive_subscript_assignments_target_retina_indices"
        ],
        "target_populations_have_zero_direct_external_drive_overlap": target_overlap == 0,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(v7_config_path): _sha256(root / v7_config_path),
                config["v7_implementation"]: _sha256(root / config["v7_implementation"]),
                str(controlled_evidence_path): _sha256(root / controlled_evidence_path),
                str(split_config_path): _sha256(root / split_config_path),
                str(split_evidence_path): _sha256(root / split_evidence_path),
                str(split_implementation_path): _sha256(root / split_implementation_path),
                str(input_evidence_path): _sha256(root / input_evidence_path),
                str(typed_config_path): _sha256(root / typed_config_path),
                str(typed_evidence_path): _sha256(root / typed_evidence_path),
                str(typed_implementation_path): _sha256(root / typed_implementation_path),
                **{
                    str(path.relative_to(root)): _sha256(path)
                    for path in v7_module_paths
                    if path != root / IMPLEMENTATION
                },
            },
            "parameter_fit": False,
            "neural_response_scoring_performed": False,
            "runtime_modified": False,
        },
        "base_stimuli": {
            "stimulus_count": len(base),
            "conditions": base_conditions,
            "missing_required_conditions": missing_base,
        },
        "independent_splits": split_results,
        "typed_mechanism_stimuli": {
            "condition_count": len(typed_config["conditions"]),
            "stimulus_count": len(typed_names),
            "stimulus_names_per_condition": sorted(typed_suffixes),
            "front_back_labels_are_model_image_trajectory_labels": True,
        },
        "input_boundary": {
            "retinal_node_count": int(probe.retina.size),
            "retinal_node_types": sorted(retinal_types),
            "target_population_types": list(V7_TARGET_TYPES),
            "target_direct_input_overlap": target_overlap,
            "dynamic_probe_stimulus_count": len(base),
            "dynamic_probe_advance_call_count": probe.drive_call_count,
            "dynamic_nonretinal_drive_node_count": len(probe.outside_retina_nodes),
            "static_assignment_audit": static_assignments,
        },
        "gates": gates,
        "stimulus_and_input_boundary_complete": all(gates.values()),
        "controlled_response_gates_passed": controlled_evidence[
            "controlled_response_gates_pass"
        ],
        "typed_LPLC_LC4_response_gates_passed": typed_evidence[
            "typed_lplc_lc4_gates_passed"
        ],
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "stimulus_and_input_boundary_complete_but_neural_response_gates_failed",
        "boundary": config["boundary"],
    }
