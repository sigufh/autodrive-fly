"""T4-only functional precheck using held-out-calibrated synapse axes."""

from __future__ import annotations

import json
from functools import partial
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
from fly_emotion.driving.v7_stage1_development import _as_visual_stimulus
from fly_emotion.driving.v7_stage1_nested import build_condition_bundle
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary
from fly_emotion.driving.v7_t4t5_local_edge_precheck import OPPOSITE

CONFIG = Path("configs/driving-v7-t4-synapse-correlator-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_synapse_correlator_precheck.py")


def _row_matrix(rows: list[dict[int, float]], body_ids: np.ndarray) -> sparse.csr_matrix:
    row_indices = []
    columns = []
    values = []
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


def _synapse_moment_matrices(root: Path, probe, config: dict, transforms: dict) -> dict:
    spatial_path = Path(config["synapse_spatial_protocol"])
    spatial = yaml.safe_load((root / spatial_path).read_text(encoding="utf-8"))
    annotations = feather.read_table(
        root / spatial["annotations"], columns=["bodyId", "type", "somaSide"]
    ).to_pandas()
    types = dict(zip(annotations.bodyId.astype(int), annotations.type, strict=True))
    sides = dict(zip(annotations.bodyId.astype(int), annotations.somaSide, strict=True))
    target_bodies = probe.graph.body_ids[probe.conductance["targets"]]
    target_index = {int(body): index for index, body in enumerate(target_bodies)}
    source_sets = {
        channel: {
            int(body)
            for body, cell_type in types.items()
            if cell_type in config["source_groups"][channel]
        }
        for channel in ("fast", "delayed")
    }
    source_union = pa.array(sorted(source_sets["fast"] | source_sets["delayed"]), type=pa.int64())
    target_union = pa.array(target_bodies, type=pa.int64())
    rows = {
        f"{channel}_{moment}": [dict() for _ in target_bodies]
        for channel in ("fast", "delayed")
        for moment in ("mass", "x", "y")
    }
    with pa.memory_map(str(root / spatial["synapse_partners"]), "r") as source:
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
            for target in np.unique(post):
                target_rows = post == target
                eye = str(sides[int(target)])
                projected = xyz[target_rows] @ np.asarray(transforms[eye])
                sources = pre[target_rows]
                output_row = target_index[int(target)]
                for channel in ("fast", "delayed"):
                    keep = np.isin(sources, list(source_sets[channel]))
                    if not np.any(keep):
                        continue
                    selected_sources = sources[keep]
                    selected_positions = projected[keep]
                    source_ids, counts = np.unique(selected_sources, return_counts=True)
                    centroids = np.asarray(
                        [
                            np.mean(selected_positions[selected_sources == source_id], axis=0)
                            for source_id in source_ids
                        ]
                    )
                    denominator = float(np.sum(counts))
                    mass = counts / denominator
                    rows[f"{channel}_mass"][output_row] = dict(
                        zip(source_ids.tolist(), mass.tolist(), strict=True)
                    )
                    rows[f"{channel}_x"][output_row] = dict(
                        zip(source_ids.tolist(), (mass * centroids[:, 0]).tolist(), strict=True)
                    )
                    rows[f"{channel}_y"][output_row] = dict(
                        zip(source_ids.tolist(), (mass * centroids[:, 1]).tolist(), strict=True)
                    )
    matrices = {
        name: _row_matrix(entries, probe.graph.body_ids) for name, entries in rows.items()
    }
    valid = (np.diff(matrices["fast_mass"].indptr) > 0) & (
        np.diff(matrices["delayed_mass"].indptr) > 0
    )
    ones = np.ones(probe.graph.node_count)
    axes = np.column_stack(
        (
            matrices["fast_x"] @ ones - matrices["delayed_x"] @ ones,
            matrices["fast_y"] @ ones - matrices["delayed_y"] @ ones,
        )
    )
    axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
    return {
        "matrices": matrices,
        "axes": axes,
        "valid": valid,
        "target_bodies": target_bodies,
        "target_index": target_index,
    }


def _response(probe, stimulus, moments: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    baseline_target = state[probe.conductance["targets"]].astype(np.float64)
    previous_retina = baseline_values.copy()
    previous = None
    base_values = []
    projected_values = []
    matrices = moments["matrices"]
    axes = moments["axes"]
    for image in stimulus.frames:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous_retina)
        previous_retina = sampled
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        source_state = probe._normalized_source_state(state)
        current = {name: matrix @ source_state for name, matrix in matrices.items()}
        base_values.append(
            state[probe.conductance["targets"]].astype(np.float64) - baseline_target
        )
        if previous is not None:
            cross_x = -(
                current["delayed_x"] * previous["fast_mass"]
                - current["delayed_mass"] * previous["fast_x"]
                - current["fast_x"] * previous["delayed_mass"]
                + current["fast_mass"] * previous["delayed_x"]
            )
            cross_y = -(
                current["delayed_y"] * previous["fast_mass"]
                - current["delayed_mass"] * previous["fast_y"]
                - current["fast_y"] * previous["delayed_mass"]
                + current["fast_mass"] * previous["delayed_y"]
            )
            terms = (
                current["delayed_x"] * previous["fast_mass"],
                current["delayed_mass"] * previous["fast_x"],
                current["fast_x"] * previous["delayed_mass"],
                current["fast_mass"] * previous["delayed_x"],
                current["delayed_y"] * previous["fast_mass"],
                current["delayed_mass"] * previous["fast_y"],
                current["fast_y"] * previous["delayed_mass"],
                current["fast_mass"] * previous["delayed_y"],
            )
            denominator = sum(np.abs(term) for term in terms) + 1e-6
            projected_values.append((axes[:, 0] * cross_x + axes[:, 1] * cross_y) / denominator)
        previous = current
    projected = np.stack(projected_values)
    return (
        np.max(np.stack(base_values), axis=0),
        np.max(np.maximum(projected, 0.0), axis=0),
        np.mean(projected, axis=0),
    )


def _compact(score: dict) -> dict:
    return {
        key: score[key]
        for key in (
            "cell_count",
            "valid_cell_count",
            "valid_cell_fraction",
            "median_signed_contrast",
            "positive_cell_fraction",
            "passed",
        )
    }


def _candidate_response(
    responses: dict,
    lookup: dict,
    reduction: str,
    gain: float,
    target_rows: np.ndarray,
    valid: np.ndarray,
    polarity: str,
    direction: str,
) -> np.ndarray:
    base, positive_peak, signed_mean = responses[lookup[(polarity, direction)]]
    head = positive_peak if reduction == "positive_peak" else signed_mean
    candidate = base + gain * head
    output = np.full(len(target_rows), np.nan)
    present = target_rows >= 0
    output[present] = candidate[target_rows[present]]
    valid_rows = np.maximum(target_rows, 0)
    output[present & ~valid[valid_rows]] = np.nan
    return output


def evaluate_v7_t4_synapse_correlator_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    axis_path = Path(config["synapse_axis_evidence"])
    axis = json.loads((root / axis_path).read_text(encoding="utf-8"))
    if not axis["authorize_T4_single_condition_precheck"]:
        raise ValueError("T4 synapse-axis precheck requires passed held-out calibration")
    if axis["authorize_T5_mapping_application"]:
        raise ValueError("T5 mapping must remain forbidden in the T4-only precheck")
    conductance_path = Path(config["conductance_protocol"])
    conductance = yaml.safe_load((root / conductance_path).read_text(encoding="utf-8"))
    stage1_path = Path(config["stage1_protocol"])
    stage1 = yaml.safe_load((root / stage1_path).read_text(encoding="utf-8"))
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    condition = next(
        item for item in stage1["conditions"] if item["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("T4 synapse-axis precheck may consume tuning only")
    stimuli = [
        item
        for item in build_condition_bundle(condition)
        if item.family == "moving_edge"
    ]
    normalization = _collect_source_normalization(
        root,
        retinal_backend="linear_luminance",
        retinal_geometry=conductance["primary_retinal_geometry"],
        config=conductance,
    )
    probe = MotionSignConductanceProbe(
        root, retinal_backend="linear_luminance", normalization=normalization
    )
    moments = _synapse_moment_matrices(root, probe, config, axis["transforms_by_eye"])
    responses = {
        item.identity: _response(probe, _as_visual_stimulus(item), moments) for item in stimuli
    }
    lookup = {(item.polarity, item.direction): item.identity for item in stimuli}
    all_t4 = {
        name: probe.populations[name]
        for name in scoring["direction_populations"]
        if name.startswith("T4")
    }
    candidate_results = []
    reductions = config["correlator"]["temporal_reductions"]
    for reduction_index, reduction in enumerate(reductions, start=1):
        for gain in config["correlator"]["additive_gains"]:
            population_scores = {}
            for population, nodes in all_t4.items():
                preferred = scoring["direction_populations"][population]
                target_rows = np.asarray(
                    [
                        moments["target_index"].get(int(body_id), -1)
                        for body_id in probe.graph.body_ids[nodes]
                    ]
                )

                value = partial(
                    _candidate_response,
                    responses,
                    lookup,
                    reduction,
                    float(gain),
                    target_rows,
                    moments["valid"],
                )

                preferred_response = value("on", preferred)
                direction = strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("on", OPPOSITE[preferred]),
                    scoring["thresholds"],
                )
                polarity = strict_contrast_summary(
                    probe.graph.body_ids[nodes],
                    preferred_response,
                    value("off", preferred),
                    scoring["thresholds"],
                )
                population_scores[population] = {
                    "direction": _compact(direction),
                    "polarity": _compact(polarity),
                }
            bilateral = [
                subtype
                for subtype in "abcd"
                if population_scores[f"T4{subtype}_L"]["direction"]["passed"]
                and population_scores[f"T4{subtype}_R"]["direction"]["passed"]
            ]
            mirror_errors = {}
            for subtype in "abcd":
                mirror_errors[subtype] = {
                    head: abs(
                        population_scores[f"T4{subtype}_L"][head]["median_signed_contrast"]
                        - population_scores[f"T4{subtype}_R"][head]["median_signed_contrast"]
                    )
                    for head in ("direction", "polarity")
                }
            mirror_gate = all(
                error <= float(scoring["thresholds"]["maximum_energy_weighted_mirror_error"])
                for values in mirror_errors.values()
                for error in values.values()
            )
            candidate_results.append(
                {
                    "name": f"{reduction}:gain={float(gain):g}",
                    "temporal_reduction": reduction,
                    "reduction_index": reduction_index,
                    "gain": float(gain),
                    "direction_pass_count": int(
                        sum(item["direction"]["passed"] for item in population_scores.values())
                    ),
                    "polarity_pass_count": int(
                        sum(item["polarity"]["passed"] for item in population_scores.values())
                    ),
                    "bilateral_direction_subtypes": bilateral,
                    "mirror_absolute_median_contrast_errors": mirror_errors,
                    "mirror_gate_passed": mirror_gate,
                    "population_scores": population_scores,
                }
            )
    minimum_bilateral = int(
        config["stop_gate"]["minimum_bilateral_direction_subtypes_to_run_controls"]
    )
    control_eligible = [
        item["name"]
        for item in candidate_results
        if len(item["bilateral_direction_subtypes"]) >= minimum_bilateral
    ]
    advancing = [
        item["name"]
        for item in candidate_results
        if item["direction_pass_count"] == len(all_t4)
        and item["polarity_pass_count"] == len(all_t4)
        and item["mirror_gate_passed"]
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(axis_path): _sha256(root / axis_path),
                str(conductance_path): _sha256(root / conductance_path),
                str(stage1_path): _sha256(root / stage1_path),
                str(scoring_path): _sha256(root / scoring_path),
                str(config["synapse_spatial_protocol"]): _sha256(
                    root / config["synapse_spatial_protocol"]
                ),
            },
            "condition_id": config["condition_id"],
            "stimulus_count": len(stimuli),
            "candidate_count": len(candidate_results),
            "T5_evaluated": False,
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "diagnostic_target_count": len(moments["target_bodies"]),
        "diagnostic_valid_target_count": int(np.count_nonzero(moments["valid"])),
        "fixed_T4_population_denominator": int(sum(len(nodes) for nodes in all_t4.values())),
        "candidates": candidate_results,
        "control_eligible_candidates": control_eligible,
        "advancing_candidates": advancing,
        "main_gate_passed": bool(advancing),
        "controls_requested": config["controls_after_main_gate"],
        "controls_evaluated": False,
        "three_condition_evaluation_performed": False,
        "advance_to_calibration": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            None
            if advancing
            else "no_bilateral_T4_direction_subtype_passed_synapse_axis_main_gate"
        ),
        "boundary": config["boundary"],
    }
