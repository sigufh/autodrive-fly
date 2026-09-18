"""Map local CT1 terminals and audit a label-blind T5 suppressor axis."""

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

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_source_audit import CARDINAL_VECTORS

CONFIG = Path("configs/driving-v7-t5-ct1-terminal-axis-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_ct1_terminal_axis_audit.py")


def _fold(body_id: int, seed: int, folds: int = 5) -> int:
    digest = hashlib.sha256(f"{seed}:{body_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _fit_affine(xyz: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((xyz, np.ones(len(xyz))))
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _predict(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.column_stack((xyz, np.ones(len(xyz)))) @ transform


def evaluate_v7_t5_ct1_terminal_axis_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    label_path = Path(config["label_evidence"])
    label = json.loads((root / label_path).read_text())
    if label["label_status"]["direction_code_to_PD_ND"] != {"0": "ND", "1": "PD"}:
        raise ValueError("T5 CT1 axis audit requires verified direction semantics")
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    if float(config["gates"]["minimum_target_valid_fraction"]) != float(
        scoring["thresholds"]["minimum_valid_cell_fraction"]
    ):
        raise ValueError("CT1 terminal axis valid-fraction gate differs from scoring")
    if float(config["gates"]["minimum_median_expected_cosine"]) != float(
        scoring["thresholds"]["minimum_median_signed_contrast"]
    ):
        raise ValueError("CT1 terminal axis cosine gate differs from scoring")
    if float(config["gates"]["minimum_positive_alignment_fraction"]) != float(
        scoring["thresholds"]["minimum_positive_cell_fraction"]
    ):
        raise ValueError("CT1 terminal axis alignment gate differs from scoring")
    annotations = feather.read_table(
        root / config["annotations"],
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    body_ids = np.load(root / config["body_ids"])
    node_type = np.full(len(body_ids), "", dtype=object)
    node_side = np.full(len(body_ids), "", dtype=object)
    optic = np.full((len(body_ids), 2), np.nan, dtype=np.float64)
    ann_ids = annotations.bodyId.to_numpy(dtype=np.int64)
    nodes = np.searchsorted(body_ids, ann_ids)
    valid = nodes < len(body_ids)
    valid[valid] &= body_ids[nodes[valid]] == ann_ids[valid]
    rows = annotations.iloc[np.flatnonzero(valid)]
    nodes = nodes[valid]
    node_type[nodes] = rows.type.fillna("").to_numpy(dtype=object)
    node_side[nodes] = rows.somaSide.fillna("").to_numpy(dtype=object)
    hex1 = rows.assignedOlHex1.to_numpy(dtype=np.float64)
    hex2 = rows.assignedOlHex2.to_numpy(dtype=np.float64)
    optic[nodes, 0] = 1.5 * hex2
    optic[nodes, 1] = np.sqrt(3.0) * (hex1 + hex2 / 2.0)
    optic[node_side == "L", 0] *= -1.0
    target_nodes = np.flatnonzero(np.isin(node_type, [f"T5{x}" for x in "abcd"]))
    target_set = pa.array(body_ids[target_nodes], type=pa.int64())
    source_types = set(config["mapping"]["anchors"] + config["mapping"]["predicted_sources"])
    source_nodes = np.flatnonzero(np.isin(node_type, list(source_types)))
    source_set = pa.array(body_ids[source_nodes], type=pa.int64())
    sums: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], int] = {}
    by_target: dict[int, list[tuple[int, np.ndarray, int]]] = {}
    source_totals: dict[int, np.ndarray] = {}
    source_counts: dict[int, int] = {}
    with pa.memory_map(str(root / config["synapse_partners"]), "r") as stream:
        reader = ipc.open_file(stream)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            selected = batch.filter(
                pc.and_(
                    pc.is_in(batch["body_pre"], value_set=source_set),
                    pc.is_in(batch["body_post"], value_set=target_set),
                )
            )
            if not selected.num_rows:
                continue
            pre = selected["body_pre"].to_numpy()
            post = selected["body_post"].to_numpy()
            xyz = np.column_stack(
                [selected[name].to_numpy() for name in ("x_post", "y_post", "z_post")]
            ).astype(np.float64)
            for source in np.unique(pre):
                keep_source = pre == source
                source_totals[int(source)] = source_totals.get(int(source), np.zeros(3)) + np.sum(
                    xyz[keep_source], axis=0
                )
                source_counts[int(source)] = source_counts.get(int(source), 0) + int(
                    np.count_nonzero(keep_source)
                )
            for source, target in np.unique(np.column_stack((pre, post)), axis=0):
                keep = (pre == source) & (post == target)
                key = (int(source), int(target))
                sums[key] = sums.get(key, np.zeros(3)) + np.sum(xyz[keep], axis=0)
                counts[key] = counts.get(key, 0) + int(np.count_nonzero(keep))
    for (source_body, target_body), xyz_sum in sums.items():
        by_target.setdefault(target_body, []).append(
            (source_body, xyz_sum, counts[(source_body, target_body)])
        )
    id_to_node = {int(value): index for index, value in enumerate(body_ids)}
    anchors = []
    for source, total in source_totals.items():
        node = id_to_node[source]
        if node_type[node] in config["mapping"]["anchors"] and np.all(np.isfinite(optic[node])):
            anchors.append(
                {
                    "body_id": source,
                    "side": str(node_side[node]),
                    "xyz": total / source_counts[source],
                    "optic": optic[node],
                }
            )
    seed = int(config["mapping"]["split_seed"])
    holdout_predictions = []
    transforms = {}
    for side in ("L", "R"):
        side_anchors = [item for item in anchors if item["side"] == side]
        xyz = np.stack([item["xyz"] for item in side_anchors])
        target = np.stack([item["optic"] for item in side_anchors])
        ids = np.asarray([item["body_id"] for item in side_anchors])
        folds = np.asarray([_fold(int(value), seed) for value in ids])
        for fold in range(5):
            train = folds != fold
            test = folds == fold
            transform = _fit_affine(xyz[train], target[train])
            predicted = _predict(xyz[test], transform)
            for body, expected, value in zip(ids[test], target[test], predicted, strict=True):
                holdout_predictions.append(
                    {
                        "body_id": int(body),
                        "side": side,
                        "expected": expected,
                        "predicted": value,
                    }
                )
        transforms[side] = _fit_affine(xyz, target)
    mapping_results = {}
    mapping_passed = True
    max_error = float(config["gates"]["maximum_holdout_median_error_hex_units"])
    min_r2 = float(config["gates"]["minimum_holdout_R2_each_coordinate"])
    for side in ("L", "R"):
        records = [item for item in holdout_predictions if item["side"] == side]
        expected = np.stack([item["expected"] for item in records])
        predicted = np.stack([item["predicted"] for item in records])
        residual = expected - predicted
        r2 = 1.0 - np.sum(residual**2, axis=0) / np.sum(
            (expected - np.mean(expected, axis=0)) ** 2, axis=0
        )
        errors = np.linalg.norm(residual, axis=1)
        gates = {
            "each_coordinate_R2": bool(np.all(r2 >= min_r2)),
            "median_error": float(np.median(errors)) <= max_error,
        }
        mapping_results[side] = {
            "source_body_count": len(records),
            "R2_by_coordinate": r2.tolist(),
            "median_error_hex_units": float(np.median(errors)),
            "q95_error_hex_units": float(np.quantile(errors, 0.95)),
            "gates": gates,
            "passed": all(gates.values()),
        }
        mapping_passed &= all(gates.values())
    target_records = []
    population_axes: dict[str, list[np.ndarray]] = {}
    fast_types = set(config["axis"]["fast_sources"])
    for target in target_nodes:
        side = str(node_side[target])
        if side not in transforms:
            continue
        fast_sum = np.zeros(2)
        fast_count = 0
        ct1_sum = np.zeros(2)
        ct1_count = 0
        for source_body, xyz_sum, count in by_target.get(int(body_ids[target]), []):
            source_node = id_to_node[source_body]
            predicted = _predict((xyz_sum / count)[None, :], transforms[side])[0]
            if node_type[source_node] in fast_types:
                fast_sum += predicted * count
                fast_count += count
            elif node_type[source_node] == "CT1":
                ct1_sum += predicted * count
                ct1_count += count
        axis = (
            fast_sum / fast_count - ct1_sum / ct1_count
            if fast_count and ct1_count
            else np.full(2, np.nan)
        )
        norm = float(np.linalg.norm(axis))
        valid_axis = bool(np.isfinite(norm) and norm > 1e-12)
        unit = axis / norm if valid_axis else np.full(2, np.nan)
        population = f"{node_type[target]}_{side}"
        if valid_axis:
            population_axes.setdefault(population, []).append(unit)
        target_records.append(
            {
                "body_id": int(body_ids[target]),
                "population": population,
                "fast_synapse_count": fast_count,
                "CT1_synapse_count": ct1_count,
                "unit_axis": unit.tolist() if valid_axis else None,
            }
        )
    population_results = {}
    medians = {}
    thresholds = config["gates"]
    for subtype in "abcd":
        for side in "LR":
            population = f"T5{subtype}_{side}"
            vectors = np.asarray(population_axes.get(population, []))
            total = int(np.count_nonzero((node_type == f"T5{subtype}") & (node_side == side)))
            expected_name = (
                HORIZONTAL_PREFERENCE[(subtype, side)]
                if subtype in {"a", "b"}
                else VERTICAL_PREFERENCE[subtype]
            )
            expected = CARDINAL_VECTORS[expected_name]
            cosines = vectors @ expected if len(vectors) else np.empty(0)
            center = np.median(vectors, axis=0) if len(vectors) else np.full(2, np.nan)
            center /= max(float(np.linalg.norm(center)), np.finfo(float).tiny)
            gates = {
                "valid_fraction": len(vectors) / total
                >= float(thresholds["minimum_target_valid_fraction"]),
                "median_expected_cosine": bool(
                    len(cosines)
                    and np.median(cosines)
                    >= float(thresholds["minimum_median_expected_cosine"])
                ),
                "positive_alignment_fraction": float(np.count_nonzero(cosines > 0) / total)
                >= float(thresholds["minimum_positive_alignment_fraction"]),
            }
            medians[population] = center
            population_results[population] = {
                "target_count": total,
                "valid_axis_count": len(vectors),
                "valid_axis_fraction": len(vectors) / total,
                "median_expected_cosine": float(np.median(cosines)) if len(cosines) else None,
                "positive_alignment_fraction_all_targets": float(
                    np.count_nonzero(cosines > 0) / total
                ),
                "median_unit_axis": center.tolist() if np.all(np.isfinite(center)) else None,
                "gates": gates,
                "passed": all(gates.values()),
            }
    mirrors = {}
    for subtype in "abcd":
        left = medians[f"T5{subtype}_L"]
        right = medians[f"T5{subtype}_R"]
        reflected = np.asarray((-right[0], right[1]))
        error = float(np.linalg.norm(left - reflected))
        mirrors[subtype] = {
            "error": error,
            "passed": error <= float(thresholds["maximum_population_mirror_axis_error"]),
        }
    strict = bool(
        mapping_passed
        and all(item["passed"] for item in population_results.values())
        and all(item["passed"] for item in mirrors.values())
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(label_path): _sha256(root / label_path),
                str(scoring_path): _sha256(root / scoring_path),
                config["annotations"]: _sha256(root / config["annotations"]),
                config["synapse_partners"]: _sha256(root / config["synapse_partners"]),
                config["body_ids"]: _sha256(root / config["body_ids"]),
            },
            "parameter_fit_to_neural_response": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "mapping_contract": config["mapping"],
        "mapping_holdout": mapping_results,
        "mapping_passed": mapping_passed,
        "axis_contract": config["axis"],
        "population_results": population_results,
        "population_mirror": mirrors,
        "target_record_count": len(target_records),
        "target_records": target_records,
        "strict_CT1_terminal_axis_gate_passed": strict,
        "authorize_CT1_spatial_dynamics_candidate": strict,
        "advance_to_T5_functional_precheck": strict,
        "stop_reason": None if strict else "CT1_terminal_axis_failed_mapping_or_label_alignment",
        "boundary": config["boundary"],
    }
