"""T4 antisymmetric precheck using target-wise out-of-fold synapse axes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import yaml
from scipy import sparse

from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_synapse_axis_calibration import _collect_axes, _split
from fly_emotion.driving.v7_t4_synapse_antisymmetric_precheck import (
    _moving_edges,
    _run_responses,
    _score_candidates,
)

CONFIG = Path("configs/driving-v7-t4-synapse-crossfit-precheck.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t4_synapse_crossfit_precheck.py"
)
SOURCE_IMPLEMENTATIONS = (
    Path("src/fly_emotion/driving/v7_conductance.py"),
    Path("src/fly_emotion/driving/v7_geometry_sign.py"),
    Path("src/fly_emotion/driving/v7_synapse_axis_calibration.py"),
    Path("src/fly_emotion/driving/v7_t4_synapse_correlator_precheck.py"),
    Path("src/fly_emotion/driving/v7_t4_synapse_antisymmetric_precheck.py"),
)


def _row_matrix(rows: list[dict[int, float]], body_ids: np.ndarray) -> sparse.csr_matrix:
    row_indices, columns, values = [], [], []
    for row_index, entries in enumerate(rows):
        for body_id, value in entries.items():
            node = int(np.searchsorted(body_ids, body_id))
            if node < len(body_ids) and body_ids[node] == body_id:
                row_indices.append(row_index)
                columns.append(node)
                values.append(value)
    return sparse.csr_matrix(
        (values, (row_indices, columns)),
        shape=(len(rows), len(body_ids)),
        dtype=np.float64,
    )


def _target_transforms(
    root: Path, spatial: dict, crossfit: dict, seed: int, target_bodies: np.ndarray
) -> tuple[dict[int, np.ndarray], str]:
    dataset = _collect_axes(root, spatial)
    t4 = dataset["family"] == "T4"
    dataset = {name: values[t4] for name, values in dataset.items()}
    first, second = _split(dataset, seed, 0.5)
    fold_by_body = {
        int(body): 0 if first[index] else 1
        for index, body in enumerate(dataset["body_id"])
    }
    row_by_body = {int(body): index for index, body in enumerate(dataset["body_id"])}
    transforms = {}
    axes = []
    axis_bodies = []
    for body in target_bodies:
        body = int(body)
        if body not in fold_by_body:
            continue
        row = row_by_body[body]
        side = str(dataset["side"][row])
        fold = fold_by_body[body]
        key = "fit_fold_1_apply_fold_0" if fold == 0 else "fit_fold_0_apply_fold_1"
        transform = np.asarray(crossfit["transforms_by_eye_and_training_fold"][side][key])
        transforms[body] = transform
        axis = dataset["offset"][row] @ transform
        axes.append(axis / np.linalg.norm(axis))
        axis_bodies.append(body)
    order = np.argsort(axis_bodies)
    digest = hashlib.sha256()
    digest.update(np.asarray(axis_bodies, dtype=np.int64)[order].tobytes())
    digest.update(np.asarray(axes, dtype=np.float64)[order].tobytes())
    return transforms, digest.hexdigest()


def _crossfit_moments(root: Path, probe, config: dict, crossfit: dict) -> dict:
    spatial = yaml.safe_load((root / config["synapse_spatial_protocol"]).read_text())
    annotations = feather.read_table(
        root / spatial["annotations"], columns=["bodyId", "type", "somaSide"]
    ).to_pandas()
    types = dict(zip(annotations.bodyId.astype(int), annotations.type, strict=True))
    target_bodies = probe.graph.body_ids[probe.conductance["targets"]]
    target_index = {int(body): index for index, body in enumerate(target_bodies)}
    transforms, axis_digest = _target_transforms(
        root, spatial, crossfit, int(config["crossfit_split_seed"]), target_bodies
    )
    source_sets = {
        channel: {
            int(body)
            for body, cell_type in types.items()
            if cell_type in config["source_groups"][channel]
        }
        for channel in ("fast", "delayed")
    }
    rows = {
        f"{channel}_{moment}": [dict() for _ in target_bodies]
        for channel in ("fast", "delayed")
        for moment in ("mass", "x", "y")
    }
    source_union = pa.array(sorted(source_sets["fast"] | source_sets["delayed"]), type=pa.int64())
    target_union = pa.array(target_bodies, type=pa.int64())
    sums: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], int] = {}
    with pa.memory_map(str(root / spatial["synapse_partners"]), "r") as stream:
        reader = ipc.open_file(stream)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            selected = batch.filter(
                pc.and_(
                    pc.is_in(batch["body_pre"], value_set=source_union),
                    pc.is_in(batch["body_post"], value_set=target_union),
                )
            )
            if not selected.num_rows:
                continue
            pre = selected["body_pre"].to_numpy()
            post = selected["body_post"].to_numpy()
            xyz = np.column_stack(
                [selected[name].to_numpy() for name in ("x_post", "y_post", "z_post")]
            ).astype(np.float64)
            for source, target in np.unique(np.column_stack((pre, post)), axis=0):
                keep = (pre == source) & (post == target)
                key = (int(source), int(target))
                sums[key] = sums.get(key, np.zeros(3)) + np.sum(xyz[keep], axis=0)
                counts[key] = counts.get(key, 0) + int(np.count_nonzero(keep))
    grouped: dict[int, list[tuple[int, np.ndarray, int]]] = {}
    for (source, target), total in sums.items():
        grouped.setdefault(target, []).append(
            (source, total / counts[(source, target)], counts[(source, target)])
        )
    axes = np.full((len(target_bodies), 2), np.nan)
    for target, entries in grouped.items():
        if target not in transforms:
            continue
        row = target_index[target]
        transform = transforms[target]
        centers = {}
        for channel in ("fast", "delayed"):
            selected = [item for item in entries if item[0] in source_sets[channel]]
            total = float(sum(item[2] for item in selected))
            if total <= 0.0:
                continue
            center = np.zeros(2)
            for source, xyz, count in selected:
                position = xyz @ transform
                weight = count / total
                rows[f"{channel}_mass"][row][source] = weight
                rows[f"{channel}_x"][row][source] = weight * position[0]
                rows[f"{channel}_y"][row][source] = weight * position[1]
                center += weight * position
            centers[channel] = center
        if set(centers) == {"fast", "delayed"}:
            axis = centers["fast"] - centers["delayed"]
            axes[row] = axis / np.linalg.norm(axis)
    matrices = {name: _row_matrix(entries, probe.graph.body_ids) for name, entries in rows.items()}
    valid = (np.diff(matrices["fast_mass"].indptr) > 0) & (
        np.diff(matrices["delayed_mass"].indptr) > 0
    ) & np.all(np.isfinite(axes), axis=1)
    return {
        "matrices": matrices,
        "axes": axes,
        "valid": valid,
        "target_bodies": target_bodies,
        "target_index": target_index,
        "crossfit_subset_axes_sha256": axis_digest,
    }


def evaluate_v7_t4_synapse_crossfit_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "synapse_spatial_protocol",
            "crossfit_axis_evidence",
            "crossfit_axis_protocol",
            "prior_antisymmetric_evidence",
            "prior_antisymmetric_protocol",
            "conductance_protocol",
            "stage1_protocol",
            "scoring_config",
        )
    }
    crossfit = json.loads((root / paths["crossfit_axis_evidence"]).read_text())
    prior = json.loads((root / paths["prior_antisymmetric_evidence"]).read_text())
    if not crossfit["authorize_T4_crossfit_single_condition_precheck"]:
        raise ValueError("T4 functional cross-fit requires passed structure gate")
    if prior["advance_to_T4_calibration"]:
        raise ValueError("T4 cross-fit requires preserved prior candidate failure")
    crossfit_protocol = yaml.safe_load((root / paths["crossfit_axis_protocol"]).read_text())
    config["crossfit_split_seed"] = crossfit_protocol["split_seed"]
    conductance = yaml.safe_load((root / paths["conductance_protocol"]).read_text())
    stage1 = yaml.safe_load((root / paths["stage1_protocol"]).read_text())
    scoring = yaml.safe_load((root / paths["scoring_config"]).read_text())
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("T4 cross-fit precheck may consume tuning only")
    normalization = _collect_source_normalization(
        root,
        retinal_backend="linear_luminance",
        retinal_geometry=conductance["primary_retinal_geometry"],
        config=conductance,
    )
    probe = MotionSignConductanceProbe(
        root, retinal_backend="linear_luminance", normalization=normalization
    )
    moments = _crossfit_moments(root, probe, config, crossfit)
    stimuli = _moving_edges(condition)
    specs = [
        (reduction, float(gain))
        for reduction in config["correlator"]["temporal_reductions"]
        for gain in config["correlator"]["additive_gains"]
    ]
    ordered_responses = _run_responses(probe, stimuli, moments, "ordered", 0)
    ordered = _score_candidates(probe, stimuli, ordered_responses, moments, scoring, specs)
    minimum = int(config["stop_gate"]["minimum_bilateral_direction_subtypes_to_run_controls"])
    eligible = [
        item for item in ordered if len(item["bilateral_direction_subtypes"]) >= minimum
    ]
    strict = [item for item in ordered if item["strict_ordered_gate_passed"]]
    controls = {}
    if eligible:
        eligible_specs = [(item["temporal_reduction"], item["gain"]) for item in eligible]
        for mode in config["controls"]["modes"]:
            responses = _run_responses(
                probe, stimuli, moments, mode, int(config["controls"]["temporal_shuffle_seed"])
            )
            controls[mode] = {
                item["name"]: item
                for item in _score_candidates(
                    probe, stimuli, responses, moments, scoring, eligible_specs
                )
            }
    control_passing = [
        item["name"]
        for item in strict
        if all(
            not controls[mode][item["name"]]["bilateral_direction_subtypes"]
            for mode in controls
        )
    ]
    all_t4 = {
        name: probe.populations[name]
        for name in scoring["direction_populations"]
        if name.startswith("T4")
    }
    spatial = yaml.safe_load((root / paths["synapse_spatial_protocol"]).read_text())
    dependency_paths = [
        *paths.values(),
        Path(spatial["annotations"]),
        Path(spatial["synapse_partners"]),
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in SOURCE_IMPLEMENTATIONS},
                **{str(path): _sha256(root / path) for path in dependency_paths},
            },
            "condition_id": config["condition_id"],
            "candidate_count": len(specs),
            "parameter_fit_to_neural_response": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "diagnostic_target_count": len(moments["target_bodies"]),
        "diagnostic_valid_target_count": int(np.count_nonzero(moments["valid"])),
        "fixed_T4_population_denominator": int(sum(len(nodes) for nodes in all_t4.values())),
        "crossfit_subset_axes_sha256": moments["crossfit_subset_axes_sha256"],
        "ordered_candidates": ordered,
        "control_eligible_candidates": [item["name"] for item in eligible],
        "strict_ordered_candidates": [item["name"] for item in strict],
        "controls_evaluated": bool(eligible),
        "control_results": controls,
        "control_passing_candidates": control_passing,
        "candidate_passed": bool(control_passing),
        "three_condition_evaluation_performed": False,
        "advance_to_T4_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            None
            if control_passing
            else "T4_crossfit_antisymmetric_candidate_failed_strict_or_control_gate"
        ),
        "boundary": config["boundary"],
    }
