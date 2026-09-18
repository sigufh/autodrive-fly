"""T5 precheck using cross-fit CT1 terminal axes and source activity."""

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

from fly_emotion.driving.v7 import HORIZONTAL_PREFERENCE, VERTICAL_PREFERENCE
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_radial_opponency import _infer_t4_t5_positions
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE
from fly_emotion.driving.v7_t5_ct1_axis_calibration import _fit_split
from fly_emotion.driving.v7_t5_ct1_multiplicative_precheck import _compact
from fly_emotion.driving.v7_t5_ct1_source_dynamics_precheck import _variant
from fly_emotion.driving.v7_t5_ct1_terminal_axis_audit import _fit_affine, _predict
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe
from fly_emotion.driving.v7_three_hop_moment import _local_step_edge

CONFIG = Path("configs/driving-v7-t5-ct1-axis-aware-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_ct1_axis_aware_precheck.py")
SOURCE_IMPLEMENTATIONS = (
    Path("src/fly_emotion/driving/v7_t5_ct1_axis_calibration.py"),
    Path("src/fly_emotion/driving/v7_t5_ct1_crossfit_axis_audit.py"),
    Path("src/fly_emotion/driving/v7_t5_ct1_multiplicative_precheck.py"),
    Path("src/fly_emotion/driving/v7_t5_ct1_source_dynamics_precheck.py"),
    Path("src/fly_emotion/driving/v7_t5_ct1_terminal_axis_audit.py"),
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


def _annotations(root: Path, config: dict) -> dict:
    table = feather.read_table(
        root / config["annotations"],
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    records = {}
    for row in table.itertuples(index=False):
        optic = None
        if np.isfinite(row.assignedOlHex1) and np.isfinite(row.assignedOlHex2):
            optic = np.asarray(
                (
                    1.5 * float(row.assignedOlHex2),
                    np.sqrt(3.0)
                    * (float(row.assignedOlHex1) + float(row.assignedOlHex2) / 2.0),
                )
            )
            if row.somaSide == "L":
                optic[0] *= -1.0
        records[int(row.bodyId)] = {
            "type": str(row.type),
            "side": str(row.somaSide),
            "optic": optic,
        }
    return records


def _crossfit_axes(target_bodies: np.ndarray, terminal: dict, crossfit: dict, seed: int):
    by_body = {item["body_id"]: item for item in terminal["target_records"]}
    axes = np.full((len(target_bodies), 2), np.nan, dtype=np.float64)
    fold_counts = {"0": 0, "1": 0}
    for index, body in enumerate(target_bodies):
        record = by_body.get(int(body))
        if record is None or record["unit_axis"] is None:
            continue
        fold = int(_fit_split(int(body), seed))
        side = record["population"][-1]
        key = "fit_fold_0_apply_fold_1" if fold else "fit_fold_1_apply_fold_0"
        transform = np.asarray(
            crossfit["transforms_by_eye_and_training_fold"][side][key]
        )
        axis = np.asarray(record["unit_axis"]) @ transform
        axes[index] = axis / np.linalg.norm(axis)
        fold_counts[str(fold)] += 1
    return axes, fold_counts


def _terminal_moments(
    root: Path, probe, config: dict, targets: np.ndarray, terminal: dict, crossfit: dict
) -> dict:
    annotations = _annotations(root, config)
    target_bodies = probe.graph.body_ids[targets]
    target_index = {int(body): index for index, body in enumerate(target_bodies)}
    source_types = set(
        config["source_roles"]["mapping_anchors"]
        + config["source_roles"]["fast_excitation"]
        + config["source_roles"]["slow_inhibition"]
    )
    source_bodies = [body for body, item in annotations.items() if item["type"] in source_types]
    sums: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], int] = {}
    source_totals: dict[int, np.ndarray] = {}
    source_counts: dict[int, int] = {}
    with pa.memory_map(str(root / config["synapse_partners"]), "r") as stream:
        reader = ipc.open_file(stream)
        source_set = pa.array(source_bodies, type=pa.int64())
        target_set = pa.array(target_bodies, type=pa.int64())
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
                keep = pre == source
                source_totals[int(source)] = source_totals.get(int(source), np.zeros(3)) + np.sum(
                    xyz[keep], axis=0
                )
                source_counts[int(source)] = source_counts.get(int(source), 0) + int(
                    np.count_nonzero(keep)
                )
            for source, target in np.unique(np.column_stack((pre, post)), axis=0):
                keep = (pre == source) & (post == target)
                key = (int(source), int(target))
                sums[key] = sums.get(key, np.zeros(3)) + np.sum(xyz[keep], axis=0)
                counts[key] = counts.get(key, 0) + int(np.count_nonzero(keep))
    base_transforms = {}
    anchors = set(config["source_roles"]["mapping_anchors"])
    for side in ("L", "R"):
        anchor_bodies = [
            body
            for body in source_totals
            if annotations[body]["type"] in anchors
            and annotations[body]["side"] == side
            and annotations[body]["optic"] is not None
        ]
        xyz = np.stack([source_totals[body] / source_counts[body] for body in anchor_bodies])
        optic = np.stack([annotations[body]["optic"] for body in anchor_bodies])
        base_transforms[side] = _fit_affine(xyz, optic)
    grouped: dict[int, list[tuple[int, np.ndarray, int]]] = {}
    for (source, target), total in sums.items():
        grouped.setdefault(target, []).append(
            (source, total / counts[(source, target)], counts[(source, target)])
        )
    rows = {
        f"{channel}_{moment}": [dict() for _ in targets]
        for channel in ("fast", "CT1")
        for moment in ("mass", "x", "y")
    }
    fast_types = set(config["source_roles"]["fast_excitation"])
    for target_body, entries in grouped.items():
        row = target_index[target_body]
        side = annotations[target_body]["side"]
        for channel, allowed in (("fast", fast_types), ("CT1", {"CT1"})):
            selected = [entry for entry in entries if annotations[entry[0]]["type"] in allowed]
            total = float(sum(entry[2] for entry in selected))
            if total <= 0.0:
                continue
            for source, xyz, count in selected:
                position = _predict(xyz[None, :], base_transforms[side])[0]
                weight = count / total
                rows[f"{channel}_mass"][row][source] = weight
                rows[f"{channel}_x"][row][source] = weight * position[0]
                rows[f"{channel}_y"][row][source] = weight * position[1]
    matrices = {name: _row_matrix(entries, probe.graph.body_ids) for name, entries in rows.items()}
    axes, fold_counts = _crossfit_axes(
        target_bodies, terminal, crossfit, int(config["crossfit_split_seed"])
    )
    valid = (np.diff(matrices["fast_mass"].indptr) > 0) & (
        np.diff(matrices["CT1_mass"].indptr) > 0
    ) & np.all(np.isfinite(axes), axis=1)
    canonical = np.argsort(target_bodies[valid])
    digest = hashlib.sha256()
    digest.update(target_bodies[valid][canonical].astype(np.int64).tobytes())
    digest.update(axes[valid][canonical].astype(np.float64).tobytes())
    return {
        "matrices": matrices,
        "axes": axes,
        "valid": valid,
        "fold_counts": fold_counts,
        "crossfit_axes_sha256": digest.hexdigest(),
    }


def _axis_pair_component(
    current: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    component: str,
    reverse_order_coefficient: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    terms = (
        current[f"fast_{component}"] * previous["CT1_mass"],
        current["fast_mass"] * previous[f"CT1_{component}"],
        current[f"CT1_{component}"] * previous["fast_mass"],
        current["CT1_mass"] * previous[f"fast_{component}"],
    )
    return (
        terms[0]
        - terms[1]
        + reverse_order_coefficient * (terms[2] - terms[3]),
        terms,
    )


def _trace(
    probe, stimulus, moments: dict, tm9, reverse_order_coefficient: float
) -> dict[str, np.ndarray]:
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    baseline_image = stimulus.frames[0]
    baseline_values = probe._sample_retina(baseline_image)[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(baseline_values, baseline_values, baseline_values)
    for _ in range(probe.baseline_frames):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = baseline_drive
    baseline_state = state.astype(np.float64)
    previous_retina = baseline_values.copy()
    previous = None
    tm9_values, sequence_values = [], []
    matrices = moments["matrices"]
    axes = moments["axes"]
    for image in stimulus.frames[probe.baseline_frames :]:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous_retina)
        previous_retina = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        positive = np.maximum(state.astype(np.float64) - baseline_state, 0.0)
        current = {name: matrix @ positive for name, matrix in matrices.items()}
        tm9_values.append(tm9 @ positive)
        if previous is not None:
            components = []
            denominator_terms = []
            for component in ("x", "y"):
                value, terms = _axis_pair_component(
                    current, previous, component, reverse_order_coefficient
                )
                components.append(value)
                denominator_terms.extend(terms)
            denominator = sum(np.abs(term) for term in denominator_terms) + 1e-9
            sequence_values.append(
                (axes[:, 0] * components[0] + axes[:, 1] * components[1]) / denominator
            )
        previous = current
    return {
        "Tm9_peak": np.max(np.stack(tm9_values), axis=0),
        "positive_axis_sequence_peak": np.max(
            np.maximum(np.stack(sequence_values), 0.0), axis=0
        ),
    }


def _score(
    probe, responses, populations, names, valid, assignments, gain, x_count, y_count, scoring
) -> dict:
    scores = {}
    for population, nodes in populations.items():
        group = np.flatnonzero(names == population)
        subtype, side = population[2], population[-1]
        preferred = (
            HORIZONTAL_PREFERENCE[(subtype, side)]
            if subtype in {"a", "b"}
            else VERTICAL_PREFERENCE[subtype]
        )

        def value(
            polarity: str,
            direction: str,
            *,
            selected_group: np.ndarray = group,
            selected_population: str = population,
        ) -> np.ndarray:
            grid = np.stack(
                [
                    responses[(xi, yi, polarity, direction)]["Tm9_peak"]
                    * (
                        1.0
                        + gain
                        * responses[(xi, yi, polarity, direction)][
                            "positive_axis_sequence_peak"
                        ]
                    )
                    for yi in range(y_count)
                    for xi in range(x_count)
                ]
            )[:, selected_group]
            assignment = assignments[selected_population]
            rows = assignment[:, 1] * x_count + assignment[:, 0]
            output = grid[rows, np.arange(len(selected_group))]
            output[~valid[selected_group]] = np.nan
            return output

        preferred_response = value("off", preferred)
        scores[population] = {
            "direction": _compact(
                strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("off", OPPOSITE[preferred]),
                    scoring["thresholds"],
                )
            ),
            "polarity": _compact(
                strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("on", preferred),
                    scoring["thresholds"],
                )
            ),
        }
    mirror = {
        subtype: {
            head: abs(
                scores[f"T5{subtype}_L"][head]["median_signed_contrast"]
                - scores[f"T5{subtype}_R"][head]["median_signed_contrast"]
            )
            for head in ("direction", "polarity")
        }
        for subtype in "abcd"
    }
    return {
        "gain": gain,
        "direction_pass_count": int(
            sum(item["direction"]["passed"] for item in scores.values())
        ),
        "polarity_pass_count": int(
            sum(item["polarity"]["passed"] for item in scores.values())
        ),
        "bilateral_direction_subtypes": [
            subtype
            for subtype in "abcd"
            if scores[f"T5{subtype}_L"]["direction"]["passed"]
            and scores[f"T5{subtype}_R"]["direction"]["passed"]
        ],
        "mirror_absolute_median_contrast_errors": mirror,
        "mirror_gate_passed": all(
            error
            <= float(scoring["thresholds"]["maximum_energy_weighted_mirror_error"])
            for values in mirror.values()
            for error in values.values()
        ),
        "population_scores": scores,
    }


def _evaluate(
    root: Path,
    config_path: Path = CONFIG,
    implementation_path: Path = IMPLEMENTATION,
) -> dict:
    config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    config["crossfit_split_seed"] = yaml.safe_load(
        (root / config["crossfit_axis_protocol"]).read_text()
    )["split_seed"]
    crossfit = json.loads((root / config["crossfit_axis_evidence"]).read_text())
    terminal = json.loads((root / config["terminal_axis_evidence"]).read_text())
    prior = json.loads((root / config["prior_multiplicative_evidence"]).read_text())
    if not crossfit["authorize_axis_aware_single_condition_precheck"]:
        raise ValueError("axis-aware precheck requires passed cross-fit structure gate")
    if prior["candidate_passed"]:
        raise ValueError("axis-aware precheck requires preserved scalar candidate failure")
    lamina = yaml.safe_load((root / config["lamina_split_protocol"]).read_text())
    lamina_evidence = json.loads((root / config["lamina_split_evidence"]).read_text())
    if not all(
        item["polarity"]["passing_condition_count"] == 3
        for item in lamina_evidence["population_consistency"].values()
    ):
        raise ValueError("axis-aware precheck requires passed OFF polarity component")
    stage1 = yaml.safe_load((root / config["stage1_protocol"]).read_text())
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("axis-aware precheck may consume tuning only")
    scoring = yaml.safe_load((root / config["scoring_config"]).read_text())
    probe = LaminaSplitProbe(root, lamina)
    populations = {
        f"T5{subtype}_{side}": probe.populations[f"T5{subtype}_{side}"]
        for subtype in "abcd"
        for side in "LR"
    }
    targets = np.concatenate(list(populations.values())).astype(np.int32)
    names = np.concatenate([np.full(len(nodes), name) for name, nodes in populations.items()])
    moments = _terminal_moments(root, probe, config, targets, terminal, crossfit)
    if moments["crossfit_axes_sha256"] != crossfit["out_of_fold_predictions_sha256"]:
        raise ValueError("axis-aware precheck did not reproduce cross-fit target axes")
    tm9 = probe._normalized_target_inputs(
        targets, tuple(config["source_roles"]["enhancer_base"])
    )
    valid = moments["valid"] & (np.diff(tm9.indptr) > 0)
    source_position = yaml.safe_load(
        (root / "configs/driving-v7-lplc2-radial-opponency.yaml").read_text()
    )
    positions, _ = _infer_t4_t5_positions(root, probe, source_position)
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    low, high = finite.min(axis=0), finite.max(axis=0)
    local_path = Path(lamina["local_edge_config"])
    local = yaml.safe_load((root / local_path).read_text())
    x_centers = np.asarray(local["centers"]["x"], dtype=np.float64)
    y_centers = np.asarray(local["centers"]["y"], dtype=np.float64)
    assignments = {}
    for population, nodes in populations.items():
        camera = (positions[nodes] - low) / np.maximum(high - low, 1e-12) * (47, 23)
        assignments[population] = np.stack(
            (
                np.argmin(np.abs(x_centers[:, None] - camera[:, 0]), axis=0),
                np.argmin(np.abs(y_centers[:, None] - camera[:, 1]), axis=0),
            ),
            axis=1,
        )
    speed = float(condition["generator_parameters"]["edge_speed_pixels_per_frame"])

    def run_mode(mode: str) -> dict:
        responses = {}
        for xi, center_x in enumerate(x_centers):
            for yi, center_y in enumerate(y_centers):
                for polarity in ("on", "off"):
                    for direction in ("left", "right", "up", "down"):
                        stimulus = _local_step_edge(
                            float(center_x),
                            float(center_y),
                            float(local["stimulus"]["aperture_radius_pixels"]),
                            speed,
                            polarity,
                            direction,
                            int(local["common_background_frames"]),
                        )
                        responses[(xi, yi, polarity, direction)] = _trace(
                            probe,
                            _variant(
                                stimulus,
                                mode,
                                int(config["controls"]["temporal_shuffle_seed"]),
                            ),
                            moments,
                            tm9,
                            float(config["mechanism"]["reverse_order_coefficient"]),
                        )
        return responses

    ordered = run_mode("ordered")
    candidates = [
        _score(
            probe,
            ordered,
            populations,
            names,
            valid,
            assignments,
            float(gain),
            len(x_centers),
            len(y_centers),
            scoring,
        )
        for gain in config["mechanism"]["gains"]
    ]
    minimum = int(config["stop_gate"]["minimum_bilateral_direction_subtypes_to_run_controls"])
    eligible = [
        item for item in candidates if len(item["bilateral_direction_subtypes"]) >= minimum
    ]
    strict = [
        item
        for item in candidates
        if item["direction_pass_count"] == len(populations)
        and item["polarity_pass_count"] == len(populations)
        and item["mirror_gate_passed"]
    ]
    controls = {}
    if eligible:
        for mode in config["controls"]["modes"]:
            responses = run_mode(mode)
            controls[mode] = {
                str(item["gain"]): _score(
                    probe,
                    responses,
                    populations,
                    names,
                    valid,
                    assignments,
                    item["gain"],
                    len(x_centers),
                    len(y_centers),
                    scoring,
                )
                for item in eligible
            }
    control_passing = [
        item["gain"]
        for item in strict
        if all(
            not controls[mode][str(item["gain"])]["bilateral_direction_subtypes"]
            for mode in controls
        )
    ]
    dependency_paths = [
        Path(config[name])
        for name in (
            "crossfit_axis_evidence",
            "crossfit_axis_protocol",
            "terminal_axis_evidence",
            "terminal_axis_protocol",
            "prior_multiplicative_evidence",
            "prior_multiplicative_protocol",
            "lamina_split_protocol",
            "lamina_split_evidence",
            "stage1_protocol",
            "scoring_config",
            "annotations",
            "synapse_partners",
        )
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(config_path): _sha256(root / config_path),
                str(implementation_path): _sha256(root / implementation_path),
                **{str(path): _sha256(root / path) for path in SOURCE_IMPLEMENTATIONS},
                **{str(path): _sha256(root / path) for path in dependency_paths},
                str(local_path): _sha256(root / local_path),
                "configs/driving-v7-lplc2-radial-opponency.yaml": _sha256(
                    root / "configs/driving-v7-lplc2-radial-opponency.yaml"
                ),
            },
            "condition_id": config["condition_id"],
            "position_count": int(len(x_centers) * len(y_centers)),
            "stimulus_count": len(ordered),
            "candidate_count": len(candidates),
            "parameter_fit_to_neural_response": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "source_roles": config["source_roles"],
        "mechanism": config["mechanism"],
        "crossfit_axis_fold_counts": moments["fold_counts"],
        "crossfit_axes_sha256": moments["crossfit_axes_sha256"],
        "joint_source_and_axis_present_count": int(np.count_nonzero(valid)),
        "fixed_target_denominator": len(targets),
        "ordered_candidates": candidates,
        "control_eligible_gains": [item["gain"] for item in eligible],
        "strict_ordered_gains": [item["gain"] for item in strict],
        "controls_requested": config["controls"]["modes"],
        "controls_evaluated": bool(eligible),
        "control_results": controls,
        "control_passing_strict_gains": control_passing,
        "candidate_passed": bool(control_passing),
        "three_condition_evaluation_performed": False,
        "advance_to_three_tuning_conditions": False,
        "advance_to_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "stop_reason": (
            None
            if control_passing
            else "axis_aware_CT1_candidate_failed_strict_or_control_gate"
        ),
        "boundary": config["boundary"],
    }


def evaluate_v7_t5_ct1_axis_aware_precheck(root: Path) -> dict:
    return _evaluate(root)
