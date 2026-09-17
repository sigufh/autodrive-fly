"""Nested T4 calibration of 3D synapse-cluster axes with zero-shot T5 audit."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import yaml

from fly_emotion.driving.v7 import V7Contract
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_synapse_spatial_audit import (
    IMPLEMENTATION as SPATIAL_IMPLEMENTATION,
)

CONFIG = Path("configs/driving-v7-synapse-axis-calibration.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_synapse_axis_calibration.py")
DIRECTION_ORDER = ("left", "right", "up", "down")
DIRECTION_VECTORS = {
    "left": np.asarray((-1.0, 0.0)),
    "right": np.asarray((1.0, 0.0)),
    "up": np.asarray((0.0, -1.0)),
    "down": np.asarray((0.0, 1.0)),
}


def _expected(population: str) -> str:
    mapping = {
        ("a", "L"): "left",
        ("a", "R"): "right",
        ("b", "L"): "right",
        ("b", "R"): "left",
        ("c", "L"): "up",
        ("c", "R"): "up",
        ("d", "L"): "down",
        ("d", "R"): "down",
    }
    return mapping[(population[2], population[-1])]


def _collect_axes(root: Path, spatial_config: dict) -> dict[str, np.ndarray]:
    annotations = feather.read_table(
        root / spatial_config["annotations"], columns=["bodyId", "type", "somaSide"]
    ).to_pandas()
    body_ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    types = annotations["type"].fillna("").to_numpy(dtype=object)
    sides = annotations["somaSide"].fillna("").to_numpy(dtype=object)
    type_by_body = dict(zip(body_ids.tolist(), types.tolist(), strict=True))
    side_by_body = dict(zip(body_ids.tolist(), sides.tolist(), strict=True))
    source_sets = {}
    target_sets = {}
    for family, specification in spatial_config["families"].items():
        source_sets[family] = {
            channel: set(body_ids[np.isin(types, specification[f"{channel}_sources"])] )
            for channel in ("fast", "delayed")
        }
        target_sets[family] = set(body_ids[np.isin(types, specification["target_types"])] )
    source_union = pa.array(
        sorted(
            set().union(
                *(values for family in source_sets.values() for values in family.values())
            )
        ),
        type=pa.int64(),
    )
    target_union = pa.array(sorted(set().union(*target_sets.values())), type=pa.int64())
    sums = {}
    for family in target_sets:
        targets = np.asarray(sorted(target_sets[family]), dtype=np.int64)
        sums[family] = {
            "targets": targets,
            "index": {int(body): index for index, body in enumerate(targets)},
            "fast_count": np.zeros(len(targets), dtype=np.int64),
            "delayed_count": np.zeros(len(targets), dtype=np.int64),
            "fast_sum": np.zeros((len(targets), 3)),
            "delayed_sum": np.zeros((len(targets), 3)),
        }
    with pa.memory_map(str(root / spatial_config["synapse_partners"]), "r") as source:
        reader = ipc.open_file(source)
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
            for family, data in sums.items():
                family_targets = np.isin(post, list(target_sets[family]))
                for channel in ("fast", "delayed"):
                    keep = family_targets & np.isin(pre, list(source_sets[family][channel]))
                    if not np.any(keep):
                        continue
                    rows = np.searchsorted(data["targets"], post[keep])
                    np.add.at(data[f"{channel}_count"], rows, 1)
                    for axis in range(3):
                        np.add.at(data[f"{channel}_sum"][:, axis], rows, xyz[keep, axis])
    records = []
    for family, data in sums.items():
        complete = (data["fast_count"] > 0) & (data["delayed_count"] > 0)
        fast = data["fast_sum"][complete] / data["fast_count"][complete, None]
        delayed = data["delayed_sum"][complete] / data["delayed_count"][complete, None]
        offsets = fast - delayed
        offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)
        for body, offset in zip(data["targets"][complete], offsets, strict=True):
            population = f"{type_by_body[int(body)]}_{side_by_body[int(body)]}"
            records.append(
                (
                    family,
                    int(body),
                    population[2],
                    population[-1],
                    offset,
                    DIRECTION_VECTORS[_expected(population)],
                )
            )
    return {
        "family": np.asarray([row[0] for row in records]),
        "body_id": np.asarray([row[1] for row in records], dtype=np.int64),
        "subtype": np.asarray([row[2] for row in records]),
        "side": np.asarray([row[3] for row in records]),
        "offset": np.stack([row[4] for row in records]),
        "expected": np.stack([row[5] for row in records]),
    }


def _split(dataset: dict, seed: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    fit = np.zeros(len(dataset["body_id"]), dtype=bool)
    for subtype in "abcd":
        for side in "LR":
            group = np.flatnonzero(
                (dataset["family"] == "T4")
                & (dataset["subtype"] == subtype)
                & (dataset["side"] == side)
            )
            keys = np.asarray(
                [
                    hashlib.sha256(f"{seed}:{int(dataset['body_id'][index])}".encode()).digest()
                    for index in group
                ],
                dtype="S32",
            )
            order = np.argsort(keys, kind="stable")
            count = int(np.floor(len(group) * fraction))
            fit[group[order[:count]]] = True
    return fit, (dataset["family"] == "T4") & ~fit


def _fit(dataset: dict, mask: np.ndarray) -> dict[str, np.ndarray]:
    transforms = {}
    for side in "LR":
        covariance = np.zeros((3, 2))
        for subtype in "abcd":
            group = mask & (dataset["side"] == side) & (dataset["subtype"] == subtype)
            covariance += (
                dataset["offset"][group].T @ dataset["expected"][group]
            ) / np.count_nonzero(group)
        left, _, right = np.linalg.svd(covariance, full_matrices=False)
        transforms[side] = left @ right
    return transforms


def _metrics(dataset: dict, mask: np.ndarray, transforms: dict[str, np.ndarray]) -> dict:
    indices = np.flatnonzero(mask)
    predicted = np.empty((len(indices), 2))
    for side in "LR":
        local = dataset["side"][indices] == side
        predicted[local] = dataset["offset"][indices[local]] @ transforms[side]
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    expected = dataset["expected"][indices]
    cardinal = np.stack([DIRECTION_VECTORS[name] for name in DIRECTION_ORDER])
    predicted_label = np.argmax(predicted @ cardinal.T, axis=1)
    expected_label = np.argmax(expected @ cardinal.T, axis=1)
    angles = np.degrees(
        np.arccos(np.clip(np.sum(predicted * expected, axis=1), -1.0, 1.0))
    )
    groups = {}
    for family in sorted(set(dataset["family"][indices])):
        for subtype in "abcd":
            for side in "LR":
                group = (
                    (dataset["family"][indices] == family)
                    & (dataset["subtype"][indices] == subtype)
                    & (dataset["side"][indices] == side)
                )
                if np.any(group):
                    groups[f"{family}{subtype}_{side}"] = {
                        "count": int(np.count_nonzero(group)),
                        "accuracy": float(np.mean(predicted_label[group] == expected_label[group])),
                        "median_angle_error_degrees": float(np.median(angles[group])),
                    }
    return {
        "count": len(indices),
        "accuracy": float(np.mean(predicted_label == expected_label)),
        "median_angle_error_degrees": float(np.median(angles)),
        "mean_angle_error_degrees": float(np.mean(angles)),
        "by_population": groups,
    }


def _mirror(dataset: dict, transforms: dict[str, np.ndarray]) -> dict:
    values = {}
    for family in ("T4", "T5"):
        for subtype in "abcd":
            means = {}
            for side in "LR":
                group = (
                    (dataset["family"] == family)
                    & (dataset["subtype"] == subtype)
                    & (dataset["side"] == side)
                )
                vectors = dataset["offset"][group] @ transforms[side]
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                mean = np.mean(vectors, axis=0)
                means[side] = mean / np.linalg.norm(mean)
            reflected = means["L"] * np.asarray((-1.0, 1.0))
            values[f"{family}{subtype}"] = float(
                np.degrees(np.arccos(np.clip(reflected @ means["R"], -1.0, 1.0)))
            )
    return {
        "by_population_degrees": values,
        "maximum_T4_degrees": max(value for name, value in values.items() if name.startswith("T4")),
        "maximum_T5_degrees": max(value for name, value in values.items() if name.startswith("T5")),
    }


def evaluate_v7_synapse_axis_calibration(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    spatial_path = Path(config["synapse_spatial_protocol"])
    spatial_config = yaml.safe_load((root / spatial_path).read_text(encoding="utf-8"))
    evidence_path = Path(config["synapse_spatial_evidence"])
    evidence = __import__("json").loads((root / evidence_path).read_text(encoding="utf-8"))
    if not evidence["strict_synapse_spatial_structure_gate_passed"]:
        raise ValueError("synapse-axis calibration requires passed spatial structure")
    contract = V7Contract.load(root)
    gates = contract.payload["controlled_vision"]["optic_hex_axis_calibration_v1"]["gates"]
    dataset = _collect_axes(root, spatial_config)
    fit, held_out = _split(dataset, int(config["split_seed"]), float(config["fit_fraction"]))
    transforms = _fit(dataset, fit)
    fit_metrics = _metrics(dataset, fit, transforms)
    held_metrics = _metrics(dataset, held_out, transforms)
    t5_mask = dataset["family"] == "T5"
    t5_metrics = _metrics(dataset, t5_mask, transforms)
    mirror = _mirror(dataset, transforms)

    rng = np.random.default_rng(int(config["split_seed"]))
    random_scores = []
    for _ in range(int(config["random_orthogonal_baselines"])):
        random_transforms = {}
        for side in "LR":
            matrix = rng.normal(size=(3, 2))
            left, _, right = np.linalg.svd(matrix, full_matrices=False)
            random_transforms[side] = left @ right
        random_scores.append(_metrics(dataset, held_out, random_transforms)["accuracy"])
    random_p95 = float(np.quantile(random_scores, 0.95))
    preregistered = {
        "held_out_T4_accuracy": held_metrics["accuracy"]
        >= float(gates["minimum_held_out_T4_accuracy"]),
        "held_out_T4_angle": held_metrics["median_angle_error_degrees"]
        <= float(gates["maximum_held_out_T4_median_angle_error_degrees"]),
        "T4_cross_eye_mirror": mirror["maximum_T4_degrees"]
        <= float(gates["maximum_cross_eye_mirror_angle_error_degrees"]),
        "held_out_T4_above_random_p95": held_metrics["accuracy"] > random_p95,
    }
    t4_passed = all(preregistered.values())
    zero_shot_T5_passed = bool(
        t5_metrics["accuracy"] >= float(gates["minimum_zero_shot_T5_accuracy"])
        and t5_metrics["median_angle_error_degrees"]
        <= float(gates["maximum_zero_shot_T5_median_angle_error_degrees"])
        and mirror["maximum_T5_degrees"]
        <= float(gates["maximum_cross_eye_mirror_angle_error_degrees"])
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(spatial_path): _sha256(root / spatial_path),
                str(SPATIAL_IMPLEMENTATION): _sha256(root / SPATIAL_IMPLEMENTATION),
                str(evidence_path): _sha256(root / evidence_path),
                "configs/driving-v7.yaml": _sha256(root / "configs/driving-v7.yaml"),
            },
            "T5_used_for_fit_or_model_selection": False,
            "random_orthogonal_baseline_count": len(random_scores),
            "parameter_fit": True,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "dataset": {
            "T4_fit_count": int(np.count_nonzero(fit)),
            "T4_held_out_count": int(np.count_nonzero(held_out)),
            "T5_zero_shot_count": int(np.count_nonzero(t5_mask)),
            "fit_held_out_body_id_overlap": int(
                np.intersect1d(dataset["body_id"][fit], dataset["body_id"][held_out]).size
            ),
        },
        "transforms_by_eye": {side: transform.tolist() for side, transform in transforms.items()},
        "T4_fit": fit_metrics,
        "held_out_T4": held_metrics,
        "zero_shot_T5": t5_metrics,
        "cross_eye_mirror": mirror,
        "random_orthogonal_baseline": {
            "count": len(random_scores),
            "held_out_T4_accuracy_p95": random_p95,
        },
        "preregistered_T4_gates": preregistered,
        "T4_synapse_axis_calibration_passed": t4_passed,
        "zero_shot_T5_mapping_passed": zero_shot_T5_passed,
        "authorize_T4_single_condition_precheck": t4_passed,
        "authorize_T5_mapping_application": False,
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
