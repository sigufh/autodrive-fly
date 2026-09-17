"""Structure-only audit of T4/T5 fast and delayed synapse-site clusters."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import yaml

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-synapse-spatial-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_synapse_spatial_audit.py")


def _file_md5(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "md5").hexdigest()


def _accumulator(size: int) -> dict[str, np.ndarray]:
    return {
        "count": np.zeros(size, dtype=np.int64),
        "sum": np.zeros((size, 3), dtype=np.float64),
        "sum_square": np.zeros((size, 3), dtype=np.float64),
    }


def _add(accumulator: dict[str, np.ndarray], target_rows: np.ndarray, xyz: np.ndarray) -> None:
    np.add.at(accumulator["count"], target_rows, 1)
    for axis in range(3):
        np.add.at(accumulator["sum"][:, axis], target_rows, xyz[:, axis])
        np.add.at(accumulator["sum_square"][:, axis], target_rows, xyz[:, axis] ** 2)


def _cluster(accumulator: dict[str, np.ndarray], index: int) -> tuple[np.ndarray, float]:
    count = int(accumulator["count"][index])
    if not count:
        return np.full(3, np.nan), np.nan
    center = accumulator["sum"][index] / count
    variance = np.maximum(accumulator["sum_square"][index] / count - center**2, 0.0)
    return center, float(np.sqrt(np.sum(variance)))


def evaluate_v7_synapse_spatial_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    manifest_path = Path(config["data_manifest"])
    manifest = yaml.safe_load((root / manifest_path).read_text(encoding="utf-8"))
    partner_spec = manifest["datasets"]["malecns"]["files"]["synapse_partners"]
    partner_path = Path(config["synapse_partners"])
    local_partner = root / partner_path
    if local_partner.stat().st_size != int(partner_spec["bytes"]):
        raise ValueError("synapse-partner file size differs from manifest")
    if _file_md5(local_partner) != partner_spec["md5"]:
        raise ValueError("synapse-partner file MD5 differs from manifest")

    annotation_path = Path(config["annotations"])
    annotations = feather.read_table(
        root / annotation_path, columns=["bodyId", "type", "somaSide"], memory_map=True
    ).to_pandas()
    body_ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    cell_types = annotations["type"].fillna("").to_numpy(dtype=object)
    sides = annotations["somaSide"].fillna("").to_numpy(dtype=object)
    type_by_body = dict(zip(body_ids.tolist(), cell_types.tolist(), strict=True))
    side_by_body = dict(zip(body_ids.tolist(), sides.tolist(), strict=True))

    families = {}
    source_sets = {}
    target_sets = {}
    for family, specification in config["families"].items():
        source_sets[family] = {
            channel: set(body_ids[np.isin(cell_types, specification[f"{channel}_sources"])] )
            for channel in ("fast", "delayed")
        }
        target_sets[family] = set(
            body_ids[np.isin(cell_types, specification["target_types"])]
        )
        ordered_targets = np.asarray(sorted(target_sets[family]), dtype=np.int64)
        families[family] = {
            "target_body_ids": ordered_targets,
            "target_index": {int(body): index for index, body in enumerate(ordered_targets)},
            "channels": {
                channel: _accumulator(len(ordered_targets)) for channel in ("fast", "delayed")
            },
        }

    source_union = pa.array(
        sorted(
            set().union(
                *(values for family in source_sets.values() for values in family.values())
            )
        ),
        type=pa.int64(),
    )
    target_union = pa.array(sorted(set().union(*target_sets.values())), type=pa.int64())
    selected_rows = 0
    neuropils: Counter[str] = Counter()
    with pa.memory_map(str(local_partner), "r") as source:
        reader = ipc.open_file(source)
        total_rows = sum(
            reader.get_batch(index).num_rows for index in range(reader.num_record_batches)
        )
        schema = [str(field) for field in reader.schema]
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            broad = pc.and_(
                pc.is_in(batch["body_pre"], value_set=source_union),
                pc.is_in(batch["body_post"], value_set=target_union),
            )
            selected = batch.filter(broad)
            if not selected.num_rows:
                continue
            pre = selected["body_pre"].to_numpy()
            post = selected["body_post"].to_numpy()
            xyz = np.column_stack(
                [selected[name].to_numpy() for name in ("x_post", "y_post", "z_post")]
            ).astype(np.float64)
            for family in config["families"]:
                target_mask = np.isin(post, list(target_sets[family]))
                ordered_targets = families[family]["target_body_ids"]
                for channel in ("fast", "delayed"):
                    keep = target_mask & np.isin(pre, list(source_sets[family][channel]))
                    if not np.any(keep):
                        continue
                    rows = np.searchsorted(ordered_targets, post[keep])
                    _add(families[family]["channels"][channel], rows, xyz[keep])
                    selected_rows += int(np.count_nonzero(keep))
            for name in selected["primary_post"].to_pylist():
                neuropils[str(name)] += 1

    graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
    thresholds = yaml.safe_load(
        (root / config["thresholds_from"]).read_text(encoding="utf-8")
    )["thresholds"]
    reports = {}
    for family, data in families.items():
        targets = data["target_body_ids"]
        fast = data["channels"]["fast"]
        delayed = data["channels"]["delayed"]
        target_nodes = np.searchsorted(graph.body_ids, targets)
        fast_nodes = np.searchsorted(graph.body_ids, sorted(source_sets[family]["fast"]))
        delayed_nodes = np.searchsorted(graph.body_ids, sorted(source_sets[family]["delayed"]))
        adjacency = graph.adjacency[target_nodes, :]
        aggregate_edges = adjacency[:, np.concatenate((fast_nodes, delayed_nodes))]
        aggregate_weight = int(aggregate_edges.sum(dtype=np.int64))
        aggregate_pairs = int(aggregate_edges.nnz)
        records = []
        unit_vectors = {}
        normalized_separation = []
        complete = []
        for index, target in enumerate(targets):
            fast_center, fast_radius = _cluster(fast, index)
            delayed_center, delayed_radius = _cluster(delayed, index)
            is_complete = bool(fast["count"][index] and delayed["count"][index])
            displacement = fast_center - delayed_center
            separation = float(np.linalg.norm(displacement)) if is_complete else None
            within_scale = (
                float(np.sqrt((fast_radius**2 + delayed_radius**2) / 2.0))
                if is_complete
                else None
            )
            normalized = (
                separation / within_scale
                if is_complete and within_scale and within_scale > 0
                else None
            )
            complete.append(is_complete)
            normalized_separation.append(normalized if normalized is not None else np.nan)
            if is_complete and separation and separation > 0:
                unit_vectors[int(target)] = displacement / separation
            records.append(
                {
                    "body_id": int(target),
                    "type": type_by_body[int(target)],
                    "soma_side": side_by_body[int(target)],
                    "fast_synapse_count": int(fast["count"][index]),
                    "delayed_synapse_count": int(delayed["count"][index]),
                    "fast_postsynaptic_centroid_voxels": (
                        fast_center.tolist() if is_complete else None
                    ),
                    "delayed_postsynaptic_centroid_voxels": (
                        delayed_center.tolist() if is_complete else None
                    ),
                    "fast_cluster_rms_radius_voxels": fast_radius if is_complete else None,
                    "delayed_cluster_rms_radius_voxels": delayed_radius if is_complete else None,
                    "centroid_separation_voxels": separation,
                    "normalized_cluster_separation": normalized,
                }
            )
        complete_array = np.asarray(complete)
        normalized_array = np.asarray(normalized_separation)
        target_ids_sha256 = hashlib.sha256(targets.astype("<i8").tobytes()).hexdigest()
        missing_target_body_ids = targets[~complete_array].tolist()
        population_results = {}
        mirrors = {}
        for subtype in "abcd":
            medians = {}
            for side in ("L", "R"):
                selected_records = [
                    item
                    for item in records
                    if item["type"] == f"{family}{subtype}" and item["soma_side"] == side
                ]
                values = np.asarray(
                    [item["normalized_cluster_separation"] for item in selected_records],
                    dtype=np.float64,
                )
                valid = np.isfinite(values)
                separated = valid & (
                    values >= float(thresholds["minimum_median_signed_contrast"])
                )
                vectors = np.asarray(
                    [
                        unit_vectors.get(item["body_id"], np.full(3, np.nan))
                        for item in selected_records
                    ]
                )
                vector_valid = np.all(np.isfinite(vectors), axis=1)
                median_vector = np.median(vectors[vector_valid], axis=0)
                median_vector /= np.linalg.norm(median_vector)
                medians[side] = median_vector
                gates = {
                    "complete_target_fraction": float(np.mean(valid))
                    >= float(thresholds["minimum_valid_cell_fraction"]),
                    "median_normalized_cluster_separation": float(np.median(values[valid]))
                    >= float(thresholds["minimum_median_signed_contrast"]),
                    "separated_target_fraction": float(np.mean(separated))
                    >= float(thresholds["minimum_positive_cell_fraction"]),
                }
                population_results[f"{family}{subtype}_{side}"] = {
                    "target_count": len(selected_records),
                    "complete_target_count": int(np.count_nonzero(valid)),
                    "complete_target_fraction": float(np.mean(valid)),
                    "median_centroid_separation_voxels": float(
                        np.median(
                            [
                                item["centroid_separation_voxels"]
                                for item in selected_records
                                if item["centroid_separation_voxels"] is not None
                            ]
                        )
                    ),
                    "median_normalized_cluster_separation": float(np.median(values[valid])),
                    "separated_target_fraction": float(np.mean(separated)),
                    "median_unit_displacement": median_vector.tolist(),
                    "gates": gates,
                    "passed": bool(all(gates.values())),
                }
            reflected_right = medians["R"].copy()
            reflected_right[0] *= -1.0
            error = float(np.linalg.norm(medians["L"] - reflected_right))
            mirrors[subtype] = {
                "median_unit_displacement_mirror_error": error,
                "passed": error
                <= float(thresholds["maximum_energy_weighted_mirror_error"]),
            }
        report = {
            "target_count": len(targets),
            "complete_target_count": int(np.count_nonzero(complete_array)),
            "complete_target_fraction": float(np.mean(complete_array)),
            "median_normalized_cluster_separation": float(
                np.nanmedian(normalized_array)
            ),
            "separated_target_fraction": float(
                np.mean(
                    np.isfinite(normalized_array)
                    & (normalized_array >= float(thresholds["minimum_median_signed_contrast"]))
                )
            ),
            "selected_synapse_row_count": int(
                np.sum(fast["count"]) + np.sum(delayed["count"])
            ),
            "aggregate_graph_synapse_weight": aggregate_weight,
            "aggregate_graph_unique_pair_count": aggregate_pairs,
            "synapse_rows_match_aggregate_graph_weight": int(
                np.sum(fast["count"]) + np.sum(delayed["count"])
            )
            == aggregate_weight,
            "population_results": population_results,
            "population_mirror": mirrors,
            "all_population_structure_gates_passed": all(
                item["passed"] for item in population_results.values()
            ),
            "all_population_mirror_gates_passed": all(item["passed"] for item in mirrors.values()),
            "target_body_ids_sha256": target_ids_sha256,
            "missing_target_body_ids": missing_target_body_ids,
            "target_record_examples": records[:8],
            "stored_example_count": min(8, len(records)),
            "all_targets_in_aggregate_statistics": True,
        }
        report["strict_structure_gate_passed"] = bool(
            report["synapse_rows_match_aggregate_graph_weight"]
            and report["all_population_structure_gates_passed"]
            and report["all_population_mirror_gates_passed"]
        )
        reports[family] = report
    all_passed = all(report["strict_structure_gate_passed"] for report in reports.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(manifest_path): _sha256(root / manifest_path),
                str(annotation_path): _sha256(root / annotation_path),
                str(config["raw_adjacency"]): _sha256(root / config["raw_adjacency"]),
                str(config["body_ids"]): _sha256(root / config["body_ids"]),
            },
            "source_file": {
                "path": str(partner_path),
                "bytes": local_partner.stat().st_size,
                "md5": partner_spec["md5"],
                "source_url": partner_spec["url"],
                "full_row_count": total_rows,
                "selected_row_count": selected_rows,
                "schema": schema,
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "primary_post_neuropil_counts": dict(sorted(neuropils.items())),
        "families": reports,
        "strict_synapse_spatial_structure_gate_passed": bool(all_passed),
        "authorize_single_condition_synapse_spatial_precheck": bool(all_passed),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
