"""Evaluate the preregistered T4 camera-frame source-pool replication."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import (
    _infer_t4_t5_positions,
    _node_annotations,
)
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4_source_pool_local import (
    _pairwise_vectors,
    _projected_traces,
    _projection_rows,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE, _local_edge

CONFIG = Path("configs/driving-v7-t4-source-pool-camera-frame-replication.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_source_pool_camera_frame_replication.py"
)


def _git_sha256(root: Path, revision: str, path: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def _controlled_stimulus(
    stimulus: VisualStimulus, mode: str, seed: int, baseline_frames: int
) -> VisualStimulus:
    prefix = stimulus.frames[:baseline_frames]
    moving = stimulus.frames[baseline_frames:]
    if mode == "temporal_shuffle":
        moving = moving[np.random.default_rng(seed).permutation(len(moving))]
    elif mode == "static_sham":
        moving = np.repeat(moving[:1], len(moving), axis=0)
    elif mode != "ordered":
        raise ValueError(f"unknown mode: {mode}")
    return VisualStimulus(
        name=stimulus.name,
        family=stimulus.family,
        polarity=stimulus.polarity,
        direction=stimulus.direction,
        frames=np.concatenate((prefix, moving)),
        mirror_of=stimulus.mirror_of,
    )


def _candidate_scores(
    root: Path,
    probe: MassBalancedVisualProbe,
    config: dict,
    condition: dict,
    scoring: dict,
    mode: str,
    seed: int,
) -> dict:
    populations = {name: probe.populations[name] for name in ("T4d_L", "T4d_R")}
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    target_ids = probe.graph.body_ids[targets]
    target_population = np.concatenate(
        [np.full(len(nodes), name) for name, nodes in populations.items()]
    )
    matrices = {}
    source_sides, source_coordinates = _node_annotations(root, probe)
    source_coordinates[source_sides == "L", 0] *= -1.0
    source_coordinates[:, 1] *= -1.0
    for name, types in config["source_groups"].items():
        matrices[f"pair_{name}"] = _projection_rows(
            probe, targets, tuple(types), coordinates=source_coordinates
        )
        for moment, suffix in ((0, "x"), (1, "y")):
            matrices[f"pair_{name}_{suffix}"] = _projection_rows(
                probe,
                targets,
                tuple(types),
                coordinates=source_coordinates,
                moment=moment,
            )
    source = yaml.safe_load((root / config["source_protocol"]).read_text())
    positions, _ = _infer_t4_t5_positions(root, probe, source)
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = finite.min(axis=0), finite.max(axis=0)
    local = yaml.safe_load((root / config["local_edge_config"]).read_text())
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    camera = (positions[targets] - low) / np.maximum(high - low, 1e-12) * (47, 23)
    assigned = np.stack(
        (
            np.argmin(np.abs(x_centers[:, None] - camera[:, 0]), axis=0),
            np.argmin(np.abs(y_centers[:, None] - camera[:, 1]), axis=0),
        ),
        axis=1,
    )
    responses = {}
    for x_index, center_x in enumerate(x_centers):
        for y_index, center_y in enumerate(y_centers):
            for direction in ("up", "down"):
                stimulus = _local_edge(
                    int(condition["duration_frames"]),
                    float(center_x),
                    float(center_y),
                    float(local["stimulus"]["aperture_radius_pixels"]),
                    float(local["stimulus"]["bar_half_width_pixels"]),
                    "on",
                    direction,
                    int(local["common_background_frames"]),
                )
                controlled = _controlled_stimulus(
                    stimulus, mode, seed, int(local["common_background_frames"])
                )
                traces = _projected_traces(probe, controlled, matrices)
                responses[(x_index, y_index, direction)] = _pairwise_vectors(
                    traces, ""
                )["distal"]
    scores = {}
    for population in populations:
        group = np.flatnonzero(target_population == population)
        rows = assigned[group, 1] * len(x_centers) + assigned[group, 0]
        columns = group

        def gather(
            direction: str,
            *,
            selected_rows: np.ndarray = rows,
            selected_columns: np.ndarray = columns,
        ) -> np.ndarray:
            vector = (0.0, -1.0) if direction == "up" else (0.0, 1.0)
            peaks = []
            for y_index in range(len(y_centers)):
                for x_index in range(len(x_centers)):
                    values = responses[(x_index, y_index, direction)]
                    projected = vector[0] * values[0] + vector[1] * values[1]
                    finite_values = np.any(np.isfinite(projected), axis=0)
                    peak = np.max(
                        np.where(np.isfinite(projected), projected, -np.inf), axis=0
                    )
                    peak[~finite_values] = np.nan
                    peaks.append(peak)
            return np.stack(peaks)[selected_rows, selected_columns]

        preferred = "down"
        score = strict_contrast_summary(
            target_ids[group],
            gather(preferred),
            gather(OPPOSITE[preferred]),
            scoring["thresholds"],
        )
        scores[population] = {
            key: score[key]
            for key in (
                "cell_count",
                "valid_cell_count",
                "valid_cell_fraction",
                "median_signed_contrast",
                "positive_cell_fraction",
                "invalid_cell_ids",
                "gates",
                "passed",
            )
        }
    return {
        "mode": mode,
        "population_scores": scores,
        "bilateral_T4d_passed": all(item["passed"] for item in scores.values()),
    }


def evaluate_v7_t4_source_pool_camera_frame_replication(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    prereg_path = Path(audit["preregistration_evidence"])
    if _sha256(root / prereg_path) != audit["preregistration_sha256"]:
        raise ValueError("preregistration artifact hash changed")
    revision = audit["preregistration_commit"]
    if _git_sha256(root, revision, str(prereg_path)) != audit["preregistration_sha256"]:
        raise ValueError("preregistration was not frozen at the declared commit")
    prereg = json.loads((root / prereg_path).read_text())
    if prereg["protocol"]["replication_outputs_observed"]:
        raise ValueError("preregistration already contains replication output")
    candidate = prereg["candidate"]
    if candidate["readout"] != "pairwise_distal_vector":
        raise ValueError("replication candidate changed")
    source_config_path = Path(audit["source_pool_config"])
    config = yaml.safe_load((root / source_config_path).read_text())
    if config["source_groups"]["center"] != candidate["source_groups"]["center"]:
        raise ValueError("center source group changed")
    if config["source_groups"]["distal"] != candidate["source_groups"]["delayed"]:
        raise ValueError("distal source group changed")
    local_path = Path(audit["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text())
    typed_path = Path(local["typed_stimulus_protocol"])
    typed = yaml.safe_load((root / typed_path).read_text())
    conditions = {item["condition_id"]: item for item in typed["conditions"]}
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    if scoring["thresholds"] != prereg["ordered_gate"]["thresholds"]:
        raise ValueError("scoring thresholds changed after preregistration")
    probe = MassBalancedVisualProbe(root, config)
    fixed_whole_T4_population_denominator = sum(
        len(probe.populations[f"T4{subtype}_{side}"])
        for subtype in "abcd"
        for side in "LR"
    )
    if fixed_whole_T4_population_denominator != int(
        prereg["ordered_gate"]["fixed_whole_T4_population_denominator"]
    ):
        raise ValueError("fixed whole-T4 denominator changed")
    ordered = {
        condition_id: _candidate_scores(
            root, probe, config, conditions[condition_id], scoring, "ordered", 0
        )
        for condition_id in prereg["replication_condition_ids"]
    }
    ordered_passed = all(
        result["bilateral_T4d_passed"] for result in ordered.values()
    )
    controls = {}
    if ordered_passed:
        seed = int(prereg["controls_after_ordered_gate"]["temporal_shuffle_seed"])
        for condition_id in prereg["replication_condition_ids"]:
            controls[condition_id] = {
                mode: _candidate_scores(
                    root, probe, config, conditions[condition_id], scoring, mode, seed
                )
                for mode in prereg["controls_after_ordered_gate"]["modes"]
            }
    controls_passed = bool(
        ordered_passed
        and all(
            not result["bilateral_T4d_passed"]
            for by_mode in controls.values()
            for result in by_mode.values()
        )
    )
    replication_passed = ordered_passed and controls_passed
    local_implementation_path = Path(audit["local_edge_implementation"])
    source_implementation_path = Path(audit["source_pool_implementation"])
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(prereg_path): _sha256(root / prereg_path),
        str(source_config_path): _sha256(root / source_config_path),
        str(source_implementation_path): _sha256(root / source_implementation_path),
        str(local_path): _sha256(root / local_path),
        str(local_implementation_path): _sha256(root / local_implementation_path),
        str(typed_path): _sha256(root / typed_path),
        str(scoring_path): _sha256(root / scoring_path),
    }
    return {
        "protocol": {
            "name": audit["name"],
            "observed_on": audit["observed_on"],
            "dependencies_sha256": dependencies,
            "preregistration_commit": revision,
            "preregistration_sha256": audit["preregistration_sha256"],
            "discovery_condition_excluded": True,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "candidate": candidate,
        "replication_evaluated": True,
        "fixed_whole_T4_population_denominator": (
            fixed_whole_T4_population_denominator
        ),
        "candidate_target_population_denominator": sum(
            len(probe.populations[name]) for name in candidate["target_populations"]
        ),
        "ordered_results": ordered,
        "ordered_replication_passed": ordered_passed,
        "controls_evaluated": bool(controls),
        "control_results": controls,
        "controls_passed": controls_passed,
        "replication_gate_passed": replication_passed,
        "authorize_new_target_formula": replication_passed,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            "replicated_candidate_requires_full_T4_gate_before_calibration"
            if replication_passed
            else (
                "preregistered_ordered_replication_failed_controls_not_run"
                if not ordered_passed
                else "preregistered_control_specificity_failed"
            )
        ),
        "boundary": audit["boundary"],
    }
