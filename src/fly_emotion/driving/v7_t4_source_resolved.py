from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    _collect_source_normalization,
)
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_retina_audit import build_balanced_retina_control
from fly_emotion.driving.v7_stage1_development import (
    OPPOSITE_DIRECTION,
    _as_visual_stimulus,
    run_signed_cell_responses,
)
from fly_emotion.driving.v7_stage1_nested import CONFIG as NESTED_CONFIG
from fly_emotion.driving.v7_stage1_nested import build_condition_bundle
from fly_emotion.driving.v7_stage1_scoring import (
    mirror_error_summary,
    strict_contrast_summary,
)
from fly_emotion.driving.v7_t5_spatial_order import (
    _contrast,
    _fixed_sample,
    _optic_coordinates,
    _row_matrix,
    _source_traces,
    _weighted_median,
)

CONFIG = Path("configs/driving-v7-t4-source-resolved.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_source_resolved.py")


class SourceResolvedT4Probe(MassBalancedVisualProbe):
    def __init__(self, root: Path, config: dict, variant: dict):
        super().__init__(root, config)
        if variant["implementation"] != "source_resolved":
            raise ValueError("source-resolved probe requires a source-resolved variant")
        geometry = variant.get("retinal_geometry", "mass_balanced_hungarian")
        if geometry in {"frozen_branch_axis", "frozen_branch_axis_reversed"}:
            self._apply_frozen_branch_axis_geometry(root, config)
            if geometry == "frozen_branch_axis_reversed":
                self.retinal_u = np.where(self.retina.side < 0, 0.5, 1.5) - self.retinal_u
                self.retinal_v = 1.0 - self.retinal_v
                self.retina = type(self.retina)(
                    self.retina.node_indices,
                    self.retina.body_ids,
                    self.retinal_u.astype(np.float32),
                    self.retinal_v.astype(np.float32),
                    self.retina.side,
                    self.retina.mapping_version,
                )
        elif geometry != "mass_balanced_hungarian":
            raise ValueError(f"unknown source-resolved retinal geometry: {geometry}")
        groups = config["source_groups"]
        active_groups = set(variant["active_groups"])
        if not active_groups <= set(groups):
            raise ValueError("source-resolved variant names an unknown source group")
        source_types = tuple(source for values in groups.values() for source in values)
        if len(source_types) != len(set(source_types)):
            raise ValueError("T4 source groups must be disjoint")
        target_nodes = np.unique(
            np.concatenate(
                [nodes for name, nodes in self.populations.items() if name.startswith("T4")]
            )
        ).astype(np.int32)
        source_mask = np.isin(self.node_types, source_types).astype(np.float32)
        union = self.adjacency[target_nodes, :].multiply(source_mask).tocsr()
        totals = np.asarray(np.abs(union).sum(axis=1)).ravel().astype(np.float32)
        scale = np.zeros_like(totals)
        np.divide(1.0, totals, out=scale, where=totals > 0)
        normalized = (sparse.diags(scale, format="csr") @ union).tocsr()
        matrices = {}
        for source_type in source_types:
            mask = (self.node_types == source_type).astype(np.float32)
            matrices[source_type] = normalized.multiply(mask).tocsr()
        configured_delays = {
            name: int(delay) for name, delay in config["source_delays_substeps"].items()
        }
        if set(configured_delays) != set(source_types):
            raise ValueError("every source type must have exactly one declared delay")
        delays = (
            configured_delays
            if variant["delays"] == "declared"
            else {name: 0 for name in source_types}
        )
        if variant["delays"] not in {"declared", "zero"}:
            raise ValueError("unknown source delay mode")
        active_types = {source for group in active_groups for source in groups[group]}
        self.source_resolved = {
            "targets": target_nodes,
            "matrices": matrices,
            "delays": delays,
            "active_types": active_types,
            "gain": float(config["target_gain"]),
            "valid": totals > 0,
        }
        self.source_delays = np.zeros(self.graph.node_count, dtype=np.int8)
        for source_type, delay in delays.items():
            self.source_delays[self.node_types == source_type] = delay
        self.correlator = None
        self.dynamics_backend = config["dynamics_backend"]

    def _apply_frozen_branch_axis_geometry(self, root: Path, config: dict) -> None:
        audit_path = Path(config["source_audit_evidence"])
        audit = json.loads((root / audit_path).read_text())
        if not audit["nested_branch_validation"]["nested_validation_pass"]:
            raise ValueError("branch-axis retinal geometry requires passed anatomy audit")
        transforms = audit["frozen_axis_calibration"]["transforms_by_eye"]
        balanced = build_balanced_retina_control(root)
        if not np.array_equal(self.retina.body_ids, balanced.retina.body_ids):
            raise ValueError("branch-axis comparison changed balanced receptor body IDs")
        coordinates = balanced.pair_coordinates.astype(np.float64)
        u = np.empty(self.retina.size, dtype=np.float32)
        v = np.empty(self.retina.size, dtype=np.float32)
        for eye, positions, side in (
            ("L", balanced.left_positions, -1),
            ("R", balanced.right_positions, 1),
        ):
            projected = coordinates @ np.asarray(transforms[eye], dtype=np.float64)
            low = np.min(projected, axis=0)
            high = np.max(projected, axis=0)
            normalized = (projected - low) / np.maximum(high - low, 1e-12)
            u[positions] = (
                0.5 * normalized[:, 0]
                if side < 0
                else 0.5 + 0.5 * normalized[:, 0]
            )
            v[positions] = normalized[:, 1]
        self.probe_retinal_geometry = "frozen_branch_axis"
        self.retina = type(self.retina)(
            self.retina.node_indices,
            self.retina.body_ids,
            u,
            v,
            self.retina.side,
            self.retina.mapping_version,
        )
        self.retinal_u, self.retinal_v = u, v

    def _advance(
        self, state: np.ndarray, drive: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        recurrent = self.adjacency @ (state * self.source_sign)
        updated = (
            (1.0 - self.leak) * state + self.leak * np.tanh(1.8 * recurrent + drive)
        ).astype(np.float32)
        target_drive = np.zeros(len(self.source_resolved["targets"]), dtype=np.float32)
        for source_type in sorted(self.source_resolved["active_types"]):
            delay = self.source_resolved["delays"][source_type]
            source_state = state if delay == 0 else history[min(delay - 1, len(history) - 1)]
            signed_state = source_state * self.source_sign
            target_drive += self.source_resolved["matrices"][source_type] @ signed_state
        target_drive = np.tanh(self.source_resolved["gain"] * target_drive).astype(np.float32)
        targets = self.source_resolved["targets"]
        updated[targets] = (
            (1.0 - self.leak[targets]) * state[targets]
            + self.leak[targets] * target_drive
        )
        history.insert(0, state.copy())
        del history[max(1, max(self.source_resolved["delays"].values())) :]
        return updated


def _compact_score(score: dict) -> dict:
    contrast = np.asarray(
        [value if value is not None else np.nan for value in score["contrast"]],
        dtype=np.float64,
    )
    success = np.isfinite(contrast) & (contrast >= 0.10)
    return {
        "cell_count": score["cell_count"],
        "valid_cell_count": score["valid_cell_count"],
        "valid_cell_fraction": score["valid_cell_fraction"],
        "median_signed_contrast": score["median_signed_contrast"],
        "positive_cell_fraction": score["positive_cell_fraction"],
        "all_target_success_count": int(np.count_nonzero(success)),
        "all_target_success_fraction": float(np.mean(success)),
        "passed": score["passed"],
    }


def _joint_coverage(scores: list[dict]) -> dict:
    ids = np.asarray(scores[0]["cell_ids"], dtype=np.int64)
    if any(not np.array_equal(np.asarray(score["cell_ids"]), ids) for score in scores):
        raise ValueError("T4 target body IDs are misaligned across conditions")
    contrasts = np.asarray(
        [
            [value if value is not None else np.nan for value in score["contrast"]]
            for score in scores
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(contrasts)
    success = valid & (contrasts >= 0.10)
    return {
        "target_count": int(len(ids)),
        "body_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "joint_valid_count": int(np.count_nonzero(np.all(valid, axis=0))),
        "joint_valid_fraction": float(np.mean(np.all(valid, axis=0))),
        "all_condition_success_count": int(np.count_nonzero(np.all(success, axis=0))),
        "all_condition_success_fraction": float(np.mean(np.all(success, axis=0))),
    }


def _score_variant(probe, condition_stimuli: dict, scoring: dict) -> dict:
    responses = {}
    for items in condition_stimuli.values():
        for item in items:
            responses[item.identity] = run_signed_cell_responses(
                probe, _as_visual_stimulus(item)
            )
    per_condition = {}
    raw_scores = {}
    mirror_scores = {}
    for condition_id, stimuli in condition_stimuli.items():
        lookup = {(item.polarity, item.direction): item for item in stimuli}
        raw_scores[condition_id] = {"direction": {}, "polarity": {}}
        for population in scoring["target_populations"]:
            preferred_direction = scoring["direction_populations"][population]
            preferred = lookup[("on", preferred_direction)]
            opposite = lookup[("on", OPPOSITE_DIRECTION[preferred_direction])]
            off = lookup[("off", preferred_direction)]
            body_ids = probe.graph.body_ids[probe.populations[population]]
            raw_scores[condition_id]["direction"][population] = strict_contrast_summary(
                body_ids,
                responses[preferred.identity]["peaks"][population],
                responses[opposite.identity]["peaks"][population],
                scoring["thresholds"],
            )
            raw_scores[condition_id]["polarity"][population] = strict_contrast_summary(
                body_ids,
                responses[preferred.identity]["peaks"][population],
                responses[off.identity]["peaks"][population],
                scoring["thresholds"],
            )
        mirror_scores[condition_id] = {}
        for polarity in ("on", "off"):
            right = lookup[(polarity, "right")]
            left = lookup[(polarity, "left")]
            original = np.stack(
                [
                    responses[right.identity]["population_traces"][f"T4{subtype}_L"]
                    for subtype in ("a", "b")
                ]
            )
            reflected = np.stack(
                [
                    responses[left.identity]["population_traces"][f"T4{subtype}_R"]
                    for subtype in ("a", "b")
                ]
            )
            mirror_scores[condition_id][polarity] = mirror_error_summary(
                original, reflected, scoring["thresholds"]
            )
        per_condition[condition_id] = {
            head: {population: _compact_score(score) for population, score in values.items()}
            for head, values in raw_scores[condition_id].items()
        }
    joint = {
        head: {
            population: _joint_coverage(
                [raw_scores[name][head][population] for name in condition_stimuli]
            )
            for population in scoring["target_populations"]
        }
        for head in ("direction", "polarity")
    }
    gates = scoring["gates"]
    strict_passed = (
        all(
            score["passed"]
            for condition in per_condition.values()
            for head in condition.values()
            for score in head.values()
        )
        and all(
            item["joint_valid_fraction"] >= float(gates["minimum_joint_valid_fraction"])
            and item["all_condition_success_fraction"]
            >= float(gates["minimum_all_condition_success_fraction"])
            for head in joint.values()
            for item in head.values()
        )
        and all(
            score["passed"]
            for condition in mirror_scores.values()
            for score in condition.values()
        )
    )
    return {
        "per_condition_scores": per_condition,
        "joint_target_coverage": joint,
        "mirror_scores": mirror_scores,
        "strict_t4ab_tuning_passed": bool(strict_passed),
    }


def _lag(values: np.ndarray, frames: int) -> np.ndarray:
    if frames == 0:
        return values
    return np.concatenate((np.zeros_like(values[:frames]), values[:-frames]), axis=0)


def _spatial_response(
    traces: dict[str, np.ndarray], lag: int, form: str, output_sign: int
) -> np.ndarray:
    center_low = traces["center_low"]
    center_high = traces["center_high"]
    proximal_low = _lag(traces["proximal_low"], lag)
    proximal_high = _lag(traces["proximal_high"], lag)
    distal_low = _lag(traces["distal_low"], lag)
    distal_high = _lag(traces["distal_high"], lag)
    center_proximal = center_high * proximal_low - center_low * proximal_high
    if form == "center_proximal":
        drive = center_proximal
    elif form == "proximal_center":
        drive = proximal_high * _lag(center_low, lag) - proximal_low * _lag(center_high, lag)
    elif form == "center_distal":
        drive = center_high * distal_low - center_low * distal_high
    elif form == "combined":
        drive = center_proximal + center_high * distal_low - center_low * distal_high
    else:
        raise ValueError(f"unknown spatial correlation form: {form}")
    return np.max(float(output_sign) * drive, axis=0)


def _spatial_reachability_pilot(
    root: Path,
    config: dict,
    probe: MassBalancedVisualProbe,
    condition_stimuli: dict,
    scoring: dict,
) -> dict:
    pilot = config["spatial_pilot"]
    coordinates = _optic_coordinates(root, probe)
    target_nodes: list[int] = []
    target_body_ids: list[int] = []
    target_populations: list[str] = []
    for population in scoring["target_populations"]:
        nodes = probe.populations[population]
        body_ids = probe.graph.body_ids[nodes]
        selected = _fixed_sample(
            body_ids, int(pilot["targets_per_population"]), int(pilot["target_sample_seed"])
        )
        target_nodes.extend(nodes[selected].tolist())
        target_body_ids.extend(body_ids[selected].tolist())
        target_populations.extend([population] * len(selected))
    pool_names = (
        "center_low",
        "center_high",
        "proximal_low",
        "proximal_high",
        "distal_low",
        "distal_high",
        "ct1",
    )
    pools: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in pool_names}
    structurally_valid = []
    for target, population in zip(target_nodes, target_populations, strict=True):
        row = probe.adjacency.getrow(target)
        sources = row.indices
        weights = np.abs(row.data).astype(np.float64)
        horizontal = coordinates[sources, 0].copy()
        if population.endswith("_L"):
            horizontal *= -1.0
        located = np.isfinite(horizontal)
        center = np.isin(probe.node_types[sources], ("Mi1", "Tm3")) & located
        proximal = np.isin(probe.node_types[sources], ("Mi4", "C3")) & located
        distal = (probe.node_types[sources] == "Mi9") & located
        ct1 = probe.node_types[sources] == "CT1"

        def centroid(
            mask: np.ndarray,
            positions: np.ndarray = horizontal,
            source_weights: np.ndarray = weights,
        ) -> float:
            return (
                float(np.average(positions[mask], weights=source_weights[mask]))
                if np.any(mask)
                else np.nan
            )

        orientation = np.sign(centroid(center) - centroid(proximal))
        orientation = 1.0 if not np.isfinite(orientation) or orientation == 0 else orientation
        axis = horizontal * orientation
        spatial = center | proximal | distal
        split = (
            _weighted_median(axis[spatial], weights[spatial]) if np.any(spatial) else np.nan
        )
        low, high = axis <= split, axis > split

        def append_pool(
            name: str,
            mask: np.ndarray,
            denominator_mask: np.ndarray,
            source_nodes: np.ndarray = sources,
            source_weights: np.ndarray = weights,
        ) -> None:
            denominator = float(np.sum(source_weights[denominator_mask]))
            pools[name].append(
                (source_nodes[mask], source_weights[mask] / denominator)
                if denominator > 0
                else (np.empty(0, dtype=np.int32), np.empty(0))
            )

        append_pool("center_low", center & low, center)
        append_pool("center_high", center & high, center)
        append_pool("proximal_low", proximal & low, proximal)
        append_pool("proximal_high", proximal & high, proximal)
        append_pool("distal_low", distal & low, distal)
        append_pool("distal_high", distal & high, distal)
        append_pool("ct1", ct1, ct1)
        structurally_valid.append(
            all(
                np.any(mask)
                for mask in (center & low, center & high, proximal & low, proximal & high)
            )
        )
    matrices = {
        name: _row_matrix(rows, probe.graph.node_count) for name, rows in pools.items()
    }
    traces = {
        item.identity: _source_traces(probe, _as_visual_stimulus(item), matrices)
        for items in condition_stimuli.values()
        for item in items
    }
    lookup = {
        (condition_id, item.polarity, item.direction): item.identity
        for condition_id, items in condition_stimuli.items()
        for item in items
    }
    candidates = [
        (int(lag), str(form), int(sign))
        for lag in pilot["lags_frames"]
        for form in pilot["correlation_forms"]
        for sign in pilot["output_signs"]
    ]
    population_results = {}
    structural = np.asarray(structurally_valid, dtype=bool)
    names = np.asarray(target_populations)
    for population in scoring["target_populations"]:
        group = np.flatnonzero(names == population)
        preferred_direction = scoring["direction_populations"][population]
        opposite_direction = OPPOSITE_DIRECTION[preferred_direction]
        candidate_success = []
        for lag, form, sign in candidates:
            comparisons = []
            for condition_id in condition_stimuli:
                preferred_on = _spatial_response(
                    traces[lookup[(condition_id, "on", preferred_direction)]],
                    lag,
                    form,
                    sign,
                )[group]
                opposite_on = _spatial_response(
                    traces[lookup[(condition_id, "on", opposite_direction)]],
                    lag,
                    form,
                    sign,
                )[group]
                preferred_off = _spatial_response(
                    traces[lookup[(condition_id, "off", preferred_direction)]],
                    lag,
                    form,
                    sign,
                )[group]
                comparisons.extend(
                    (
                        _contrast(preferred_on, opposite_on, 1e-6),
                        _contrast(preferred_on, preferred_off, 1e-6),
                    )
                )
            contrast = np.stack(comparisons, axis=1)
            candidate_success.append(
                np.isfinite(contrast) & (contrast >= float(pilot["minimum_contrast"]))
            )
        success = np.stack(candidate_success)
        reachable = np.any(np.all(success, axis=2), axis=0) & structural[group]
        best_count = np.max(np.sum(success, axis=2), axis=0)
        population_results[population] = {
            "target_count": int(len(group)),
            "body_ids_sha256": hashlib.sha256(
                np.asarray(target_body_ids, dtype=np.int64)[group].tobytes()
            ).hexdigest(),
            "structurally_valid_count": int(np.count_nonzero(structural[group])),
            "all_six_comparisons_reachable_count": int(np.count_nonzero(reachable)),
            "best_comparison_count_histogram_0_to_6": np.bincount(
                best_count, minlength=7
            ).tolist(),
        }
    return {
        "status": "pilot_informed_reachability_envelope_only",
        "candidate_count": len(candidates),
        "targetwise_label_based_candidate_selection": True,
        "may_authorize_candidate": False,
        "anatomical_orientation": (
            "per-target sign of center-minus-proximal optic-hex horizontal centroid"
        ),
        "ct1_used_in_spatial_pool": False,
        "population_results": population_results,
        "advance_to_full_population": False,
        "advance_to_calibration": False,
    }


def evaluate_v7_t4_source_resolved(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested = yaml.safe_load((root / NESTED_CONFIG).read_text())
    scoring_path = Path(config["scoring_config"])
    source_audit_path = Path(config["source_audit_evidence"])
    strict_scoring = yaml.safe_load((root / scoring_path).read_text())
    condition_ids = list(config["condition_ids"])
    conditions = {item["condition_id"]: item for item in nested["conditions"]}
    if any(conditions[name]["role"] != "tuning" for name in condition_ids):
        raise ValueError("source-resolved T4 screen may consume tuning conditions only")
    condition_stimuli = {
        name: [
            item
            for item in build_condition_bundle(conditions[name])
            if item.family == "moving_edge" and item.direction in {"left", "right"}
        ]
        for name in condition_ids
    }
    scoring = {
        "target_populations": list(config["target_populations"]),
        "direction_populations": strict_scoring["direction_populations"],
        "thresholds": strict_scoring["thresholds"],
        "gates": config["gates"],
    }
    results = {}
    for variant in config["variants"]:
        if variant["implementation"] == "pooled_typed":
            probe = MassBalancedVisualProbe(root, config)
        elif variant["implementation"] == "published_conductance_reversed_branch_axis":
            conductance_config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text())
            normalization = _collect_source_normalization(
                root,
                retinal_backend=config["retinal_backend"],
                retinal_geometry=conductance_config["primary_retinal_geometry"],
                config=conductance_config,
            )
            probe = MotionSignConductanceProbe(
                root, retinal_backend=config["retinal_backend"], normalization=normalization
            )
            if (
                probe.brain_substeps != int(variant["brain_substeps_per_frame"])
                or probe.baseline_frames != int(variant["baseline_frames"])
            ):
                raise ValueError("published conductance native timebase drifted")
        else:
            probe = SourceResolvedT4Probe(root, config, variant)
        results[variant["name"]] = {
            "contract": variant,
            **_score_variant(probe, condition_stimuli, scoring),
        }
    passing = [
        name for name, result in results.items() if result["strict_t4ab_tuning_passed"]
    ]
    spatial_pilot = _spatial_reachability_pilot(
        root, config, MassBalancedVisualProbe(root, config), condition_stimuli, scoring
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(NESTED_CONFIG): _sha256(root / NESTED_CONFIG),
                str(scoring_path): _sha256(root / scoring_path),
                str(source_audit_path): _sha256(root / source_audit_path),
                str(CONDUCTANCE_CONFIG): _sha256(root / CONDUCTANCE_CONFIG),
            },
            "condition_ids": condition_ids,
            "moving_edge_stimulus_count": sum(map(len, condition_stimuli.values())),
            "variant_count": len(results),
            "parameter_fit": False,
            "labels_used_by_dynamics": False,
            "subtype_conditioned_dynamics": False,
            "runtime_modified": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
        },
        "source_contract": {
            "groups": config["source_groups"],
            "delays_substeps": config["source_delays_substeps"],
            "transmitter_signs_from_malecns": True,
            "single_union_normalization_preserves_relative_source_mass": True,
            "ct1_kept_separate": True,
            "geometry_ab_holds_receptor_ids_mass_and_dynamics_fixed": True,
            "branch_axis_used_for_diagnosis_only": True,
        },
        "variants": results,
        "spatial_reachability_pilot": spatial_pilot,
        "passing_variants": passing,
        "advance_to_calibration": bool(passing),
        "advance_to_runtime_integration": False,
        "advance_to_navigation": False,
        "boundary": config["boundary"],
    }
