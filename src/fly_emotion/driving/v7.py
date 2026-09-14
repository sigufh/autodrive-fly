"""Isolated v7 causal-vision experiments; never used by the deployed v6 engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml
from scipy import sparse

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.engine import INHIBITORY_TRANSMITTERS, MODULATORY_TRANSMITTERS
from fly_emotion.driving.retina import load_or_build_retina_map

V7_CONFIG = Path("configs/driving-v7.yaml")
V7_IMPLEMENTATION = Path("src/fly_emotion/driving/v7.py")
V7_TARGET_TYPES = (
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
    "LPLC1",
    "LPLC2",
    "LC4",
    "LC6",
    "LC16",
)
HORIZONTAL_PREFERENCE = {
    ("a", "L"): "left",
    ("a", "R"): "right",
    ("b", "L"): "right",
    ("b", "R"): "left",
}
VERTICAL_PREFERENCE = {"c": "up", "d": "down"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class V7Contract:
    path: Path
    payload: dict

    @classmethod
    def load(cls, root: Path) -> V7Contract:
        path = root / V7_CONFIG
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload.get("version") != 7 or payload.get("name") != "v7-experimental":
            raise ValueError("invalid v7 experiment contract")
        if payload.get("deployment_enabled") or payload.get("city_expansion_enabled"):
            raise ValueError("v7 must remain isolated until every release gate passes")
        for baseline in payload["baseline_contracts"].values():
            target = root / baseline["path"]
            if not target.exists() or _sha256(target) != baseline["sha256"]:
                raise ValueError(f"v7 frozen baseline mismatch: {baseline['path']}")
        return cls(path, payload)

    @property
    def sha256(self) -> str:
        return _sha256(self.path)


@dataclass(frozen=True)
class VisualStimulus:
    name: str
    family: str
    polarity: str
    direction: str
    frames: np.ndarray
    mirror_of: str | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.frames.astype(np.float32, copy=False).tobytes()).hexdigest()


@dataclass(frozen=True)
class OpticHexOffsetDataset:
    """Per-cell fast-minus-delayed input offsets from real MaleCNS edges."""

    body_ids: np.ndarray
    node_indices: np.ndarray
    families: np.ndarray
    subtypes: np.ndarray
    sides: np.ndarray
    offsets: np.ndarray
    expected_directions: np.ndarray

    @property
    def size(self) -> int:
        return int(self.body_ids.size)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        for values in (
            self.body_ids.astype(np.int64, copy=False),
            self.offsets.astype(np.float64, copy=False),
        ):
            digest.update(values.tobytes())
        for values in (self.families, self.subtypes, self.sides, self.expected_directions):
            digest.update("\n".join(values.tolist()).encode("utf-8"))
        return digest.hexdigest()


def _moving_edge(
    *, width: int, height: int, frames: int, axis: str, direction: int, bright: bool
) -> np.ndarray:
    background, foreground = (0.08, 0.92) if bright else (0.92, 0.08)
    output = np.full((frames, height, width), background, dtype=np.float32)
    extent = width if axis == "x" else height
    positions = np.linspace(2, extent - 3, frames)
    if direction < 0:
        positions = positions[::-1]
    for index, position in enumerate(positions):
        centre = int(round(float(position)))
        if axis == "x":
            if direction > 0:
                output[index, :, : centre + 1] = foreground
            else:
                output[index, :, centre:] = foreground
        else:
            if direction > 0:
                output[index, : centre + 1, :] = foreground
            else:
                output[index, centre:, :] = foreground
    return output


def _looming(*, width: int, height: int, frames: int, bright: bool) -> np.ndarray:
    background, foreground = (0.08, 0.92) if bright else (0.92, 0.08)
    yy, xx = np.mgrid[:height, :width]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    output = np.full((frames, height, width), background, dtype=np.float32)
    for index, radius in enumerate(np.linspace(1.0, min(width, height) * 0.46, frames)):
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        output[index, mask] = foreground
    return output


def _static_disc(*, width: int, height: int, frames: int, bright: bool) -> np.ndarray:
    final = _looming(width=width, height=height, frames=frames, bright=bright)[-1]
    return np.repeat(final[None, :, :], frames, axis=0)


def _grating_translation(*, width: int, height: int, frames: int, direction: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    output = []
    for phase in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        image = 0.5 + 0.4 * np.sin(2 * np.pi * xx / 8 - direction * phase)
        output.append(np.broadcast_to(image, (height, width)).astype(np.float32))
    return np.stack(output)


def _rotation(*, width: int, height: int, frames: int, direction: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    x = (xx - (width - 1) / 2) / width
    y = (yy - (height - 1) / 2) / height
    radius = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)
    output = []
    for phase in np.linspace(0, np.pi / 2, frames):
        pattern = np.sin(18 * radius + 5 * (angle - direction * phase))
        output.append((0.5 + 0.4 * pattern).astype(np.float32))
    return np.stack(output)


def build_controlled_stimuli(
    *, width: int = 48, height: int = 24, frames: int = 16
) -> list[VisualStimulus]:
    """Deterministic visual battery with no road or privileged state input."""
    stimuli = [
        VisualStimulus(
            "uniform_dark",
            "uniform",
            "dark",
            "none",
            np.full((frames, height, width), 0.08, np.float32),
        ),
        VisualStimulus(
            "uniform_bright",
            "uniform",
            "bright",
            "none",
            np.full((frames, height, width), 0.92, np.float32),
        ),
    ]
    for polarity, bright in (("on", True), ("off", False)):
        horizontal = _moving_edge(
            width=width, height=height, frames=frames, axis="x", direction=1, bright=bright
        )
        for direction, image, mirror in (
            ("right", horizontal, f"{polarity}_edge_left"),
            ("left", horizontal[:, :, ::-1].copy(), f"{polarity}_edge_right"),
        ):
            name = f"{polarity}_edge_{direction}"
            stimuli.append(
                VisualStimulus(
                    name,
                    "moving_edge",
                    polarity,
                    direction,
                    image,
                    mirror_of=mirror,
                )
            )
        for direction, sign in (("down", 1), ("up", -1)):
            name = f"{polarity}_edge_{direction}"
            stimuli.append(
                VisualStimulus(
                    name,
                    "moving_edge",
                    polarity,
                    direction,
                    _moving_edge(
                        width=width,
                        height=height,
                        frames=frames,
                        axis="y",
                        direction=sign,
                        bright=bright,
                    ),
                    mirror_of=name,
                )
            )
        stimuli.extend(
            [
                VisualStimulus(
                    f"{polarity}_looming",
                    "looming",
                    polarity,
                    "expansion",
                    _looming(width=width, height=height, frames=frames, bright=bright),
                    mirror_of=f"{polarity}_looming",
                ),
                VisualStimulus(
                    f"{polarity}_receding",
                    "receding",
                    polarity,
                    "contraction",
                    _looming(width=width, height=height, frames=frames, bright=bright)[::-1].copy(),
                    mirror_of=f"{polarity}_receding",
                ),
                VisualStimulus(
                    f"{polarity}_static_disc",
                    "static",
                    polarity,
                    "none",
                    _static_disc(width=width, height=height, frames=frames, bright=bright),
                    mirror_of=f"{polarity}_static_disc",
                ),
            ]
        )
    grating_right = _grating_translation(width=width, height=height, frames=frames, direction=1)
    rotation_cw = _rotation(width=width, height=height, frames=frames, direction=1)
    stimuli.extend(
        [
            VisualStimulus(
                "grating_right",
                "translation",
                "mixed",
                "right",
                grating_right,
                mirror_of="grating_left",
            ),
            VisualStimulus(
                "grating_left",
                "translation",
                "mixed",
                "left",
                grating_right[:, :, ::-1].copy(),
                mirror_of="grating_right",
            ),
            VisualStimulus(
                "rotation_cw",
                "rotation",
                "mixed",
                "clockwise",
                rotation_cw,
                mirror_of="rotation_ccw",
            ),
            VisualStimulus(
                "rotation_ccw",
                "rotation",
                "mixed",
                "counterclockwise",
                rotation_cw[:, :, ::-1].copy(),
                mirror_of="rotation_cw",
            ),
        ]
    )
    return stimuli


_DIRECTION_ORDER = ("left", "right", "up", "down")
_DIRECTION_VECTORS = {
    "left": np.asarray((-1.0, 0.0), dtype=np.float64),
    "right": np.asarray((1.0, 0.0), dtype=np.float64),
    "up": np.asarray((0.0, -1.0), dtype=np.float64),
    "down": np.asarray((0.0, 1.0), dtype=np.float64),
}


def load_v7_optic_hex_offsets(root: Path) -> OpticHexOffsetDataset:
    """Extract target-specific spatial offsets without using stimulus or driving labels."""
    contract = V7Contract.load(root)
    config = contract.payload["controlled_vision"]["optic_hex_axis_calibration_v1"]
    processed = root / "data/processed/malecns-v1.0"
    annotations_path = root / "data/raw/malecns-v1.0/body-annotations.feather"
    graph = load_graph(processed, normalized=False)
    annotations = feather.read_table(
        annotations_path,
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    body_ids = annotations["bodyId"].to_numpy(dtype=np.int64)
    nodes = np.searchsorted(graph.body_ids, body_ids)
    valid = nodes < graph.node_count
    valid[valid] &= graph.body_ids[nodes[valid]] == body_ids[valid]
    annotations = annotations.iloc[np.flatnonzero(valid)].copy()
    annotations["node"] = nodes[valid]

    node_types = np.full(graph.node_count, "", dtype=object)
    node_types[annotations["node"].to_numpy(dtype=np.int32)] = (
        annotations["type"].fillna("").to_numpy(dtype=object)
    )
    coordinates = np.full((graph.node_count, 2), np.nan, dtype=np.float64)
    annotated_nodes = annotations["node"].to_numpy(dtype=np.int32)
    coordinates[annotated_nodes, 0] = annotations["assignedOlHex1"].to_numpy(dtype=float)
    coordinates[annotated_nodes, 1] = annotations["assignedOlHex2"].to_numpy(dtype=float)

    records: list[tuple[int, int, str, str, str, np.ndarray, str]] = []
    for family in ("T4", "T5"):
        fast_types = tuple(config[f"{family}_fast_sources"])
        delayed_types = tuple(config[f"{family}_delayed_sources"])
        for subtype in ("a", "b", "c", "d"):
            for side in ("L", "R"):
                rows = annotations.loc[
                    annotations["type"].eq(f"{family}{subtype}") & annotations["somaSide"].eq(side),
                    ["bodyId", "node"],
                ]
                expected = (
                    HORIZONTAL_PREFERENCE[(subtype, side)]
                    if subtype in {"a", "b"}
                    else VERTICAL_PREFERENCE[subtype]
                )
                for body_id, target in rows.itertuples(index=False, name=None):
                    row = graph.adjacency.getrow(int(target))
                    sources = row.indices
                    weights = np.abs(row.data).astype(np.float64)

                    def centroid(
                        source_types: tuple[str, ...],
                        source_nodes: np.ndarray = sources,
                        source_weights: np.ndarray = weights,
                    ) -> np.ndarray:
                        keep = np.isin(node_types[source_nodes], source_types)
                        keep &= np.all(np.isfinite(coordinates[source_nodes]), axis=1)
                        if not np.any(keep):
                            return np.full(2, np.nan, dtype=np.float64)
                        return np.average(
                            coordinates[source_nodes[keep]],
                            axis=0,
                            weights=source_weights[keep],
                        )

                    offset = centroid(fast_types) - centroid(delayed_types)
                    if np.all(np.isfinite(offset)) and np.linalg.norm(offset) > 1e-12:
                        records.append(
                            (
                                int(body_id),
                                int(target),
                                family,
                                subtype,
                                side,
                                offset,
                                expected,
                            )
                        )
    return OpticHexOffsetDataset(
        body_ids=np.asarray([item[0] for item in records], dtype=np.int64),
        node_indices=np.asarray([item[1] for item in records], dtype=np.int32),
        families=np.asarray([item[2] for item in records], dtype=str),
        subtypes=np.asarray([item[3] for item in records], dtype=str),
        sides=np.asarray([item[4] for item in records], dtype=str),
        offsets=np.asarray([item[5] for item in records], dtype=np.float64),
        expected_directions=np.asarray([item[6] for item in records], dtype=str),
    )


def _stratified_axis_split(
    dataset: OpticHexOffsetDataset, *, seed: int, fit_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    fit = np.zeros(dataset.size, dtype=bool)
    for subtype in ("a", "b", "c", "d"):
        for side in ("L", "R"):
            group = np.flatnonzero(
                (dataset.families == "T4") & (dataset.subtypes == subtype) & (dataset.sides == side)
            )
            hashes = np.asarray(
                [
                    hashlib.sha256(f"{seed}:{int(dataset.body_ids[index])}".encode()).digest()
                    for index in group
                ],
                dtype="S32",
            )
            order = np.argsort(hashes, kind="stable")
            count = max(1, min(len(group) - 1, int(np.floor(len(group) * fit_fraction))))
            fit[group[order[:count]]] = True
    held_out = (dataset.families == "T4") & ~fit
    return fit, held_out


def _orthogonal_axis_transform(
    offsets: np.ndarray, targets: np.ndarray, strata: np.ndarray
) -> np.ndarray:
    normalized = offsets / np.maximum(np.linalg.norm(offsets, axis=1, keepdims=True), 1e-12)
    covariance = np.zeros((2, 2), dtype=np.float64)
    for stratum in sorted(set(strata)):
        group = strata == stratum
        covariance += normalized[group].T @ targets[group] / np.count_nonzero(group)
    left, _, right = np.linalg.svd(covariance, full_matrices=False)
    return left @ right


def _axis_metrics(
    dataset: OpticHexOffsetDataset, mask: np.ndarray, transforms: dict[str, np.ndarray]
) -> dict:
    indices = np.flatnonzero(mask)
    predicted = np.empty((len(indices), 2), dtype=np.float64)
    for side in ("L", "R"):
        local = dataset.sides[indices] == side
        predicted[local] = dataset.offsets[indices[local]] @ transforms[side]
    norms = np.linalg.norm(predicted, axis=1, keepdims=True)
    predicted = predicted / np.maximum(norms, 1e-12)
    expected = np.stack([_DIRECTION_VECTORS[name] for name in dataset.expected_directions[indices]])
    cardinal = np.stack([_DIRECTION_VECTORS[name] for name in _DIRECTION_ORDER])
    predicted_labels = np.argmax(predicted @ cardinal.T, axis=1)
    expected_labels = np.argmax(expected @ cardinal.T, axis=1)
    angles = np.degrees(np.arccos(np.clip(np.sum(predicted * expected, axis=1), -1.0, 1.0)))
    by_group = {}
    for family in sorted(set(dataset.families[indices])):
        for subtype in ("a", "b", "c", "d"):
            for side in ("L", "R"):
                group = (
                    (dataset.families[indices] == family)
                    & (dataset.subtypes[indices] == subtype)
                    & (dataset.sides[indices] == side)
                )
                if np.any(group):
                    by_group[f"{family}{subtype}_{side}"] = {
                        "count": int(np.count_nonzero(group)),
                        "accuracy": float(
                            np.mean(predicted_labels[group] == expected_labels[group])
                        ),
                        "median_angle_error_degrees": float(np.median(angles[group])),
                    }
    return {
        "count": int(len(indices)),
        "accuracy": float(np.mean(predicted_labels == expected_labels)),
        "median_angle_error_degrees": float(np.median(angles)),
        "mean_angle_error_degrees": float(np.mean(angles)),
        "by_group": by_group,
        "_indices": indices,
        "_predicted": predicted,
    }


def _cross_eye_axis_error(
    dataset: OpticHexOffsetDataset, transforms: dict[str, np.ndarray]
) -> dict:
    values = {}
    errors = []
    for family in ("T4", "T5"):
        for subtype in ("a", "b", "c", "d"):
            means = {}
            for side in ("L", "R"):
                mask = (
                    (dataset.families == family)
                    & (dataset.subtypes == subtype)
                    & (dataset.sides == side)
                )
                vectors = dataset.offsets[mask] @ transforms[side]
                vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                means[side] = np.mean(vectors, axis=0)
                means[side] /= max(np.linalg.norm(means[side]), 1e-12)
            mirrored_left = means["L"] * np.asarray((-1.0, 1.0))
            error = float(
                np.degrees(np.arccos(np.clip(np.dot(mirrored_left, means["R"]), -1.0, 1.0)))
            )
            values[f"{family}{subtype}"] = error
            errors.append(error)
    return {
        "by_population_degrees": values,
        "median_degrees": float(np.median(errors)),
        "maximum_degrees": float(np.max(errors)),
    }


def _serializable_axis_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def evaluate_v7_optic_hex_axis_calibration(root: Path) -> dict:
    """Fit on half of T4 and test held-out T4 plus zero-shot T5."""
    contract = V7Contract.load(root)
    config = contract.payload["controlled_vision"]["optic_hex_axis_calibration_v1"]
    dataset = load_v7_optic_hex_offsets(root)
    fit, held_out = _stratified_axis_split(
        dataset, seed=int(config["split_seed"]), fit_fraction=float(config["fit_fraction"])
    )
    expected = np.stack([_DIRECTION_VECTORS[name] for name in dataset.expected_directions])
    fitted = {}
    for side in ("L", "R"):
        mask = fit & (dataset.sides == side)
        fitted[side] = _orthogonal_axis_transform(
            dataset.offsets[mask], expected[mask], dataset.subtypes[mask]
        )

    t5 = dataset.families == "T5"
    fit_metrics = _axis_metrics(dataset, fit, fitted)
    held_out_metrics = _axis_metrics(dataset, held_out, fitted)
    t5_metrics = _axis_metrics(dataset, t5, fitted)
    mirror = _cross_eye_axis_error(dataset, fitted)

    standard_axial = np.asarray(
        ((0.0, -np.sqrt(3.0)), (1.5, -np.sqrt(3.0) / 2.0)), dtype=np.float64
    )
    deterministic_baselines = {
        "standard_axial_same_both_eyes": {"L": standard_axial, "R": standard_axial},
        "legacy_proxy_eye_mirrored": {
            "L": np.asarray(((-1.0, 0.0), (0.0, -1.0))),
            "R": np.asarray(((1.0, 0.0), (0.0, -1.0))),
        },
    }
    baseline_results = {}
    for name, transforms in deterministic_baselines.items():
        baseline_results[name] = {
            "held_out_T4": _serializable_axis_metrics(_axis_metrics(dataset, held_out, transforms)),
            "zero_shot_T5": _serializable_axis_metrics(_axis_metrics(dataset, t5, transforms)),
            "cross_eye_mirror": _cross_eye_axis_error(dataset, transforms),
        }

    signed_permutations = []
    for swap in (False, True):
        for first_sign in (-1.0, 1.0):
            for second_sign in (-1.0, 1.0):
                base = (
                    np.asarray(((0.0, first_sign), (second_sign, 0.0)))
                    if swap
                    else np.asarray(((first_sign, 0.0), (0.0, second_sign)))
                )
                signed_permutations.append(base)
    permutation_scores = []
    for left in signed_permutations:
        for right in signed_permutations:
            transforms = {"L": left, "R": right}
            permutation_scores.append(
                (
                    _axis_metrics(dataset, held_out, transforms)["accuracy"],
                    _axis_metrics(dataset, t5, transforms)["accuracy"],
                )
            )
    permutation_scores_array = np.asarray(permutation_scores)

    rng = np.random.default_rng(int(config["split_seed"]))
    random_scores = []
    for _ in range(int(config["random_orthogonal_baselines"])):
        transforms = {}
        for side in ("L", "R"):
            angle = rng.uniform(-np.pi, np.pi)
            reflection = rng.choice((-1.0, 1.0))
            transforms[side] = np.asarray(
                (
                    (np.cos(angle), -reflection * np.sin(angle)),
                    (np.sin(angle), reflection * np.cos(angle)),
                )
            )
        random_scores.append(
            (
                _axis_metrics(dataset, held_out, transforms)["accuracy"],
                _axis_metrics(dataset, t5, transforms)["accuracy"],
            )
        )
    random_scores_array = np.asarray(random_scores)
    random_p95 = {
        "held_out_T4_accuracy": float(np.quantile(random_scores_array[:, 0], 0.95)),
        "zero_shot_T5_accuracy": float(np.quantile(random_scores_array[:, 1], 0.95)),
    }

    thresholds = config["gates"]
    gates = {
        "held_out_T4_accuracy": held_out_metrics["accuracy"]
        >= thresholds["minimum_held_out_T4_accuracy"],
        "zero_shot_T5_accuracy": t5_metrics["accuracy"]
        >= thresholds["minimum_zero_shot_T5_accuracy"],
        "held_out_T4_angle": held_out_metrics["median_angle_error_degrees"]
        <= thresholds["maximum_held_out_T4_median_angle_error_degrees"],
        "zero_shot_T5_angle": t5_metrics["median_angle_error_degrees"]
        <= thresholds["maximum_zero_shot_T5_median_angle_error_degrees"],
        "cross_eye_mirror": mirror["maximum_degrees"]
        <= thresholds["maximum_cross_eye_mirror_angle_error_degrees"],
        "held_out_T4_above_random_p95": held_out_metrics["accuracy"]
        > random_p95["held_out_T4_accuracy"],
        "zero_shot_T5_above_random_p95": t5_metrics["accuracy"]
        > random_p95["zero_shot_T5_accuracy"],
    }
    passed = all(gates.values())
    return {
        "protocol": {
            "version": 7,
            "mode": "v7-experimental",
            "deployment_enabled": False,
            "config_sha256": contract.sha256,
            "implementation_sha256": _sha256(root / V7_IMPLEMENTATION),
            "fit_source": "T4 only, stratified by subtype and eye",
            "fit_objective": "unit offsets; equal weight per T4 subtype within each eye",
            "validation_source": "disjoint T4 bodyIds plus all T5 bodyIds",
            "driving_data_used": False,
            "stimulus_direction_labels_used": False,
            "T5_used_for_fit_or_model_selection": False,
        },
        "dataset": {
            "sha256": dataset.sha256,
            "total_cells": dataset.size,
            "T4_fit_cells": int(np.count_nonzero(fit)),
            "T4_held_out_cells": int(np.count_nonzero(held_out)),
            "T5_zero_shot_cells": int(np.count_nonzero(t5)),
            "fit_held_out_body_id_overlap": int(
                np.intersect1d(dataset.body_ids[fit], dataset.body_ids[held_out]).size
            ),
        },
        "fitted_transforms": {side: fitted[side].tolist() for side in ("L", "R")},
        "fit_T4": _serializable_axis_metrics(fit_metrics),
        "held_out_T4": _serializable_axis_metrics(held_out_metrics),
        "zero_shot_T5": _serializable_axis_metrics(t5_metrics),
        "cross_eye_mirror": mirror,
        "deterministic_baselines": baseline_results,
        "signed_axis_permutation_baseline": {
            "count": int(len(permutation_scores_array)),
            "best_held_out_T4_accuracy": float(np.max(permutation_scores_array[:, 0])),
            "best_zero_shot_T5_accuracy": float(np.max(permutation_scores_array[:, 1])),
            "selection_note": "diagnostic exhaustive baseline; not used to choose fitted transform",
        },
        "random_orthogonal_baseline": {
            "count": int(len(random_scores_array)),
            "p95": random_p95,
            "mean_held_out_T4_accuracy": float(np.mean(random_scores_array[:, 0])),
            "mean_zero_shot_T5_accuracy": float(np.mean(random_scores_array[:, 1])),
        },
        "preregistered_gates": gates,
        "axis_calibration_pass": passed,
        "advance_to_columnar_correlator_v2": passed,
        "advance_to_central_complex": False,
        "stop_reason": (
            "axis cross-validation passed; controlled visual response gates still required"
            if passed
            else "optic-hex axis calibration failed cross-validation"
        ),
    }


class V7VisualProbe:
    """Read-only population probe driven exclusively through mapped R1-R6 input."""

    def __init__(
        self,
        root: Path,
        *,
        brain_substeps: int = 4,
        baseline_frames: int = 8,
        retinal_backend: str = "legacy_absolute_contrast",
        retinal_geometry: str = "legacy_proxy_v2",
        dynamics_backend: str = "legacy_uniform_tanh_v1",
        control: str = "real_malecns",
        control_seed: int = 20260914,
    ):
        self.root = root
        self.contract = V7Contract.load(root)
        processed = root / "data/processed/malecns-v1.0"
        raw = root / "data/raw/malecns-v1.0"
        self.graph = load_graph(processed)
        self.adjacency = self.graph.adjacency
        self.retina = load_or_build_retina_map(
            self.graph, raw / "body-annotations.feather", processed / "retina_map.npz"
        )
        if retinal_geometry not in self.contract.payload["controlled_vision"]["retinal_geometries"]:
            raise ValueError(f"unknown v7 retinal geometry: {retinal_geometry}")
        self.retinal_geometry = retinal_geometry
        self.retinal_u, self.retinal_v = self._retinal_coordinates(
            raw / "body-annotations.feather", processed
        )
        self.source_sign = self._source_sign(raw / "body-neurotransmitters.feather")
        self.populations = self._populations(raw / "body-annotations.feather")
        self.node_types = self._node_types(raw / "body-annotations.feather")
        self.node_superclasses = self._node_labels(raw / "body-annotations.feather", "superclass")
        self.brain_substeps = brain_substeps
        self.baseline_frames = baseline_frames
        if retinal_backend not in {
            "legacy_absolute_contrast",
            "linear_luminance",
            "signed_frame_difference",
        }:
            raise ValueError(f"unknown v7 retinal backend: {retinal_backend}")
        self.retinal_backend = retinal_backend
        if dynamics_backend not in self.contract.payload["controlled_vision"]["dynamics_backends"]:
            raise ValueError(f"unknown v7 dynamics backend: {dynamics_backend}")
        self.dynamics_backend = dynamics_backend
        self.leak = self._leak_vector()
        self.source_delays = self._source_delay_vector()
        self.visual_subgraph_mask = np.ones(self.graph.node_count, dtype=bool)
        if dynamics_backend in {
            "typed_visual_subgraph_v1",
            "columnar_delay_v1",
            "columnar_correlator_v1",
        }:
            self.adjacency, self.visual_subgraph_mask = self._visual_subgraph_adjacency(processed)
        self.correlator = self._build_correlator()
        self.control = control
        self.control_seed = control_seed
        self.retinal_permutation = np.arange(self.retina.size, dtype=np.int32)
        rng = np.random.default_rng(control_seed)
        if control == "shuffled_retina_coordinates":
            self.retinal_permutation = rng.permutation(self.retina.size).astype(np.int32)
        elif control == "source_preserving_target_shuffle":
            target_permutation = rng.permutation(self.graph.node_count)
            self.adjacency = sparse.csr_matrix(self.graph.adjacency[target_permutation, :])
        elif control == "shuffled_transmitter_signs":
            self.source_sign = self.source_sign[rng.permutation(self.graph.node_count)]
        elif control != "real_malecns":
            raise ValueError(f"unknown v7 topology control: {control}")
        target_nodes = np.unique(np.concatenate(list(self.populations.values())))
        if np.intersect1d(self.retina.node_indices, target_nodes).size:
            raise ValueError("v7 target population overlaps direct retinal inputs")

    def _source_sign(self, path: Path) -> np.ndarray:
        table = feather.read_table(path, columns=["body", "consensus_nt"], memory_map=True)
        ids = table["body"].to_numpy(zero_copy_only=False).astype(np.int64)
        nodes = np.searchsorted(self.graph.body_ids, ids)
        valid = nodes < self.graph.node_count
        valid[valid] &= self.graph.body_ids[nodes[valid]] == ids[valid]
        signs = np.ones(self.graph.node_count, dtype=np.float32)
        names = np.asarray(table["consensus_nt"].to_pylist(), dtype=object)
        for node, name in zip(nodes[valid], names[valid], strict=True):
            if name in INHIBITORY_TRANSMITTERS:
                signs[node] = -1.0
            elif name in MODULATORY_TRANSMITTERS:
                signs[node] = 0.0
        return signs

    def _populations(self, path: Path) -> dict[str, np.ndarray]:
        table = feather.read_table(
            path, columns=["bodyId", "type", "somaSide"], memory_map=True
        ).to_pandas()
        populations = {}
        for cell_type in V7_TARGET_TYPES:
            for side in ("L", "R"):
                ids = table.loc[
                    table["type"].eq(cell_type) & table["somaSide"].eq(side), "bodyId"
                ].to_numpy(np.int64)
                nodes = np.searchsorted(self.graph.body_ids, ids).astype(np.int32)
                valid = nodes < self.graph.node_count
                valid[valid] &= self.graph.body_ids[nodes[valid]] == ids[valid]
                if np.any(valid):
                    populations[f"{cell_type}_{side}"] = nodes[valid]
        return populations

    def _node_labels(self, path: Path, column: str) -> np.ndarray:
        table = feather.read_table(path, columns=["bodyId", column], memory_map=True)
        ids = table["bodyId"].to_numpy(zero_copy_only=False).astype(np.int64)
        names = np.asarray(table[column].to_pylist(), dtype=object)
        nodes = np.searchsorted(self.graph.body_ids, ids)
        valid = nodes < self.graph.node_count
        valid[valid] &= self.graph.body_ids[nodes[valid]] == ids[valid]
        result = np.full(self.graph.node_count, "", dtype=object)
        result[nodes[valid]] = np.asarray(
            [name if isinstance(name, str) else "" for name in names[valid]], dtype=object
        )
        return result

    def _node_types(self, path: Path) -> np.ndarray:
        return self._node_labels(path, "type")

    def _visual_subgraph_adjacency(self, processed: Path) -> tuple[sparse.csr_matrix, np.ndarray]:
        allowed = np.isin(
            self.node_superclasses,
            ("ol_sensory", "ol_intrinsic", "visual_projection", "visual_projection_tbc"),
        )
        raw = load_graph(processed, normalized=False).adjacency.tocoo()
        keep = allowed[raw.row] & allowed[raw.col]
        induced = sparse.csr_matrix(
            (raw.data[keep].astype(np.float32), (raw.row[keep], raw.col[keep])),
            shape=raw.shape,
        )
        incoming = np.asarray(induced.sum(axis=1)).ravel().astype(np.float32)
        scale = np.zeros_like(incoming)
        np.divide(1.0, incoming, out=scale, where=incoming > 0)
        normalized = (sparse.diags(scale, format="csr") @ induced).astype(np.float32)
        if not np.all(allowed[self.retina.node_indices]):
            raise ValueError("visual subgraph excludes mapped R1-R6 inputs")
        target_nodes = np.unique(np.concatenate(list(self.populations.values())))
        if not np.all(allowed[target_nodes]):
            raise ValueError("visual subgraph excludes requested target populations")
        return normalized.tocsr(), allowed

    def _retinal_coordinates(
        self, annotations_path: Path, processed: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.retinal_geometry == "legacy_proxy_v2":
            return self.retina.u.copy(), self.retina.v.copy()
        table = feather.read_table(
            annotations_path,
            columns=["bodyId", "assignedOlHex1", "assignedOlHex2"],
            memory_map=True,
        ).to_pandas()
        coordinates = table.dropna(subset=["assignedOlHex1", "assignedOlHex2"])
        ids = coordinates["bodyId"].to_numpy(dtype=np.int64)
        nodes = np.searchsorted(self.graph.body_ids, ids)
        valid = nodes < self.graph.node_count
        valid[valid] &= self.graph.body_ids[nodes[valid]] == ids[valid]
        hex1 = np.full(self.graph.node_count, np.nan, dtype=np.float64)
        hex2 = np.full(self.graph.node_count, np.nan, dtype=np.float64)
        rows = coordinates.iloc[np.flatnonzero(valid)]
        hex1[nodes[valid]] = rows["assignedOlHex1"].to_numpy(dtype=np.float64)
        hex2[nodes[valid]] = rows["assignedOlHex2"].to_numpy(dtype=np.float64)
        raw = load_graph(processed, normalized=False).adjacency[:, self.retina.node_indices].tocsc()
        inferred1 = np.full(self.retina.size, np.nan, dtype=np.float64)
        inferred2 = np.full(self.retina.size, np.nan, dtype=np.float64)
        for column in range(self.retina.size):
            start, end = raw.indptr[column : column + 2]
            targets = raw.indices[start:end]
            weights = np.abs(raw.data[start:end]).astype(np.float64)
            known = np.isfinite(hex1[targets]) & np.isfinite(hex2[targets])
            if np.any(known):
                inferred1[column] = np.average(hex1[targets[known]], weights=weights[known])
                inferred2[column] = np.average(hex2[targets[known]], weights=weights[known])
        if not np.all(np.isfinite(inferred1) & np.isfinite(inferred2)):
            raise ValueError("axial retinal geometry cannot map every retained receptor")
        cartesian_x = 1.5 * inferred2
        cartesian_y = -np.sqrt(3.0) * (inferred1 + inferred2 / 2.0)
        u = np.zeros(self.retina.size, dtype=np.float32)
        v = np.zeros(self.retina.size, dtype=np.float32)
        for side in (-1, 1):
            mask = self.retina.side == side
            local_x = self._normalized(cartesian_x[mask])
            local_y = self._normalized(cartesian_y[mask])
            u[mask] = (1.0 - local_x) * 0.5 if side < 0 else 0.5 + local_x * 0.5
            v[mask] = 1.0 - local_y
        return u, v

    @staticmethod
    def _normalized(values: np.ndarray) -> np.ndarray:
        low, high = float(values.min()), float(values.max())
        if high <= low:
            return np.full(values.shape, 0.5, dtype=np.float32)
        return ((values - low) / (high - low)).astype(np.float32)

    def _sample_retina(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        x = np.clip(np.rint(self.retinal_u * (width - 1)).astype(np.int32), 0, width - 1)
        y = np.clip(np.rint(self.retinal_v * (height - 1)).astype(np.int32), 0, height - 1)
        return np.asarray(image[y, x], dtype=np.float32)

    def _leak_vector(self) -> np.ndarray:
        if self.dynamics_backend == "legacy_uniform_tanh_v1":
            return np.full(self.graph.node_count, 0.28, dtype=np.float32)
        config = self.contract.payload["controlled_vision"]["typed_visual_leak_v1"]
        leak = np.full(self.graph.node_count, config["default"], dtype=np.float32)
        exact_types = (
            "R1-R6",
            "L1",
            "L2",
            "L3",
            "L5",
            "Mi1",
            "Tm3",
            "Mi4",
            "Mi9",
            "CT1",
            "C3",
            "Tm1",
            "Tm2",
            "Tm4",
            "Tm9",
        )
        for cell_type in exact_types:
            leak[self.node_types == cell_type] = config[cell_type]
        for prefix in ("T4", "T5", "LPLC"):
            mask = np.fromiter(
                (name.startswith(prefix) for name in self.node_types),
                dtype=bool,
                count=self.graph.node_count,
            )
            leak[mask] = config[prefix]
        lc_mask = np.fromiter(
            (
                name.startswith("LC") and not name.startswith(("LPLC", "LLPC"))
                for name in self.node_types
            ),
            dtype=bool,
            count=self.graph.node_count,
        )
        leak[lc_mask] = config["LC"]
        return leak

    def _source_delay_vector(self) -> np.ndarray:
        delays = np.zeros(self.graph.node_count, dtype=np.int8)
        if self.dynamics_backend != "columnar_delay_v1":
            return delays
        config = self.contract.payload["controlled_vision"]["columnar_delay_v1"]
        for cell_type, delay in config["source_delays"].items():
            delays[self.node_types == cell_type] = int(delay)
        if int(delays.max()) > int(config["max_delay_substeps"]):
            raise ValueError("v7 source delay exceeds configured history")
        return delays

    def _normalized_target_inputs(
        self, target_nodes: np.ndarray, source_types: tuple[str, ...]
    ) -> sparse.csr_matrix:
        source_mask = np.isin(self.node_types, source_types).astype(np.float32)
        matrix = self.adjacency[target_nodes, :].multiply(source_mask).tocsr()
        totals = np.asarray(np.abs(matrix).sum(axis=1)).ravel().astype(np.float32)
        scale = np.zeros_like(totals)
        np.divide(1.0, totals, out=scale, where=totals > 0)
        return (sparse.diags(scale, format="csr") @ matrix).tocsr()

    def _build_correlator(self) -> dict | None:
        if self.dynamics_backend != "columnar_correlator_v1":
            return None
        config = self.contract.payload["controlled_vision"]["columnar_correlator_v1"]
        all_targets = np.flatnonzero(
            np.fromiter(
                (name.startswith(("T4", "T5")) for name in self.node_types),
                dtype=bool,
                count=self.graph.node_count,
            )
        ).astype(np.int32)
        t4_mask = np.fromiter(
            (self.node_types[node].startswith("T4") for node in all_targets),
            dtype=bool,
            count=len(all_targets),
        )
        targets = np.concatenate([all_targets[t4_mask], all_targets[~t4_mask]])
        fast = sparse.vstack(
            (
                self._normalized_target_inputs(
                    all_targets[t4_mask], tuple(config["T4_fast_sources"])
                ),
                self._normalized_target_inputs(
                    all_targets[~t4_mask], tuple(config["T5_fast_sources"])
                ),
            ),
            format="csr",
        )
        delayed = sparse.vstack(
            (
                self._normalized_target_inputs(
                    all_targets[t4_mask], tuple(config["T4_delayed_sources"])
                ),
                self._normalized_target_inputs(
                    all_targets[~t4_mask], tuple(config["T5_delayed_sources"])
                ),
            ),
            format="csr",
        )
        valid = (np.diff(fast.indptr) > 0) & (np.diff(delayed.indptr) > 0)
        return {
            "targets": targets[valid].astype(np.int32),
            "fast": fast[valid],
            "delayed": delayed[valid],
            "gain": float(config["gain"]),
            "history_substeps": int(config["history_substeps"]),
            "all_targets": int(len(targets)),
        }

    def _advance(
        self, state: np.ndarray, drive: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        transmitted = state.copy()
        if self.dynamics_backend == "columnar_delay_v1":
            for delay in range(1, int(self.source_delays.max()) + 1):
                mask = self.source_delays == delay
                transmitted[mask] = history[min(delay - 1, len(history) - 1)][mask]
        recurrent = self.adjacency @ (transmitted * self.source_sign)
        updated = ((1.0 - self.leak) * state + self.leak * np.tanh(1.8 * recurrent + drive)).astype(
            np.float32
        )
        if self.correlator is not None:
            previous = history[min(self.correlator["history_substeps"] - 1, len(history) - 1)]
            current_positive = np.maximum(state, 0.0)
            previous_positive = np.maximum(previous, 0.0)
            fast_now = self.correlator["fast"] @ current_positive
            delayed_now = self.correlator["delayed"] @ current_positive
            fast_then = self.correlator["fast"] @ previous_positive
            delayed_then = self.correlator["delayed"] @ previous_positive
            correlation = np.maximum(fast_now * delayed_then - fast_then * delayed_now, 0.0)
            targets = self.correlator["targets"]
            target_drive = np.tanh(self.correlator["gain"] * correlation).astype(np.float32)
            updated[targets] = (1.0 - self.leak[targets]) * state[targets] + self.leak[
                targets
            ] * target_drive
        history.insert(0, state.copy())
        history_length = max(
            1,
            int(self.source_delays.max()),
            int(self.correlator["history_substeps"]) if self.correlator is not None else 0,
        )
        del history[history_length:]
        return updated

    def _retinal_code(
        self, values: np.ndarray, baseline: np.ndarray, previous: np.ndarray
    ) -> np.ndarray:
        if self.retinal_backend == "legacy_absolute_contrast":
            return 0.25 * values + 0.75 * np.abs(values - float(values.mean()))
        if self.retinal_backend == "linear_luminance":
            return values.astype(np.float32, copy=False)
        return np.clip((values - previous) / np.maximum(baseline, 0.05), -1.0, 1.0).astype(
            np.float32
        )

    def run(self, stimulus: VisualStimulus) -> dict:
        state = np.zeros(self.graph.node_count, dtype=np.float32)
        history_length = max(
            1,
            int(self.source_delays.max()),
            int(self.correlator["history_substeps"]) if self.correlator is not None else 0,
        )
        history = [state.copy() for _ in range(history_length)]
        traces = {name: [] for name in self.populations}
        neuron_positive_sum = {
            name: np.zeros(len(nodes), dtype=np.float64) for name, nodes in self.populations.items()
        }
        retinal_hash = hashlib.sha256()
        baseline_image = stimulus.frames[0].copy()
        baseline_values = self._sample_retina(baseline_image)[self.retinal_permutation]
        baseline_drive = self._retinal_code(baseline_values, baseline_values, baseline_values)
        for _ in range(self.baseline_frames):
            drive = np.zeros_like(state)
            drive[self.retina.node_indices] = baseline_drive
            for _ in range(self.brain_substeps):
                state = self._advance(state, drive, history)
                state[self.retina.node_indices] = baseline_drive
        neuron_baseline = {name: state[nodes].copy() for name, nodes in self.populations.items()}
        population_baseline = {
            name: float(np.mean(values)) for name, values in neuron_baseline.items()
        }
        previous_values = baseline_values.copy()
        for image in stimulus.frames:
            sampled = self._sample_retina(image)[self.retinal_permutation]
            receptor_values = self._retinal_code(sampled, baseline_values, previous_values)
            previous_values = sampled
            retinal_hash.update(receptor_values.tobytes())
            drive = np.zeros_like(state)
            drive[self.retina.node_indices] = receptor_values
            for _ in range(self.brain_substeps):
                state = self._advance(state, drive, history)
                state[self.retina.node_indices] = receptor_values
            for name, nodes in self.populations.items():
                traces[name].append(float(np.mean(state[nodes])))
                neuron_positive_sum[name] += np.maximum(state[nodes] - neuron_baseline[name], 0.0)
        return {
            "name": stimulus.name,
            "family": stimulus.family,
            "polarity": stimulus.polarity,
            "direction": stimulus.direction,
            "mirror_of": stimulus.mirror_of,
            "stimulus_sha256": stimulus.sha256,
            "retinal_drive_sha256": retinal_hash.hexdigest(),
            "retinal_backend": self.retinal_backend,
            "retinal_geometry": self.retinal_geometry,
            "dynamics_backend": self.dynamics_backend,
            "topology_control": self.control,
            "visual_subgraph_nodes": int(np.count_nonzero(self.visual_subgraph_mask)),
            "visual_subgraph_edges": int(self.adjacency.nnz),
            "correlator_targets": (
                int(len(self.correlator["targets"])) if self.correlator is not None else 0
            ),
            "maximum_source_delay_substeps": int(self.source_delays.max()),
            "baseline_frames": self.baseline_frames,
            "population_trace": traces,
            "population_response_trace": {
                name: [float(value - population_baseline[name]) for value in values]
                for name, values in traces.items()
            },
            "_population_neuron_positive_mean": {
                name: (values / len(stimulus.frames)).tolist()
                for name, values in neuron_positive_sum.items()
            },
            "population_mean_abs": {
                name: float(np.mean(np.abs(values))) for name, values in traces.items()
            },
            "population_peak_abs": {
                name: float(np.max(np.abs(values))) for name, values in traces.items()
            },
        }


def _response_energy(response: dict, population: str) -> float:
    values = np.asarray(response["_population_neuron_positive_mean"][population])
    return float(np.mean(values))


def _contrast(preferred: float, opposite: float) -> float:
    return float((preferred - opposite) / (abs(preferred) + abs(opposite) + 1e-12))


def _neuron_contrasts(
    preferred_response: dict, opposite_response: dict, population: str
) -> np.ndarray:
    preferred = np.asarray(
        preferred_response["_population_neuron_positive_mean"][population], dtype=np.float64
    )
    opposite = np.asarray(
        opposite_response["_population_neuron_positive_mean"][population], dtype=np.float64
    )
    return (preferred - opposite) / (np.abs(preferred) + np.abs(opposite) + 1e-12)


def score_v7_visual_responses(responses: dict[str, dict]) -> dict:
    """Compute preregistered response contrasts without fitting to the result."""
    direction_scores = {}
    polarity_scores = {}
    for family in ("T4", "T5"):
        polarity = "on" if family == "T4" else "off"
        opposite_polarity = "off" if polarity == "on" else "on"
        for subtype in ("a", "b", "c", "d"):
            for side in ("L", "R"):
                population = f"{family}{subtype}_{side}"
                preferred = (
                    HORIZONTAL_PREFERENCE[(subtype, side)]
                    if subtype in {"a", "b"}
                    else VERTICAL_PREFERENCE[subtype]
                )
                opposite = {
                    "left": "right",
                    "right": "left",
                    "up": "down",
                    "down": "up",
                }[preferred]
                preferred_response = responses[f"{polarity}_edge_{preferred}"]
                opposite_response = responses[f"{polarity}_edge_{opposite}"]
                preferred_energy = _response_energy(preferred_response, population)
                opposite_energy = _response_energy(opposite_response, population)
                neuron_dsi = _neuron_contrasts(preferred_response, opposite_response, population)
                direction_scores[population] = {
                    "expected_direction": preferred,
                    "preferred_energy": preferred_energy,
                    "opposite_energy": opposite_energy,
                    "contrast": float(np.median(neuron_dsi)),
                    "fraction_cells_expected_direction": float(np.mean(neuron_dsi > 0)),
                }
                matched_other_response = responses[f"{opposite_polarity}_edge_{preferred}"]
                matched_other = _response_energy(matched_other_response, population)
                neuron_osi = _neuron_contrasts(
                    preferred_response, matched_other_response, population
                )
                polarity_scores[population] = {
                    "expected_polarity": polarity,
                    "preferred_polarity_energy": preferred_energy,
                    "opposite_polarity_energy": matched_other,
                    "contrast": float(np.median(neuron_osi)),
                    "fraction_cells_expected_polarity": float(np.mean(neuron_osi > 0)),
                }

    looming_scores = {}
    for cell_type in ("LPLC1", "LPLC2", "LC4", "LC6", "LC16"):
        for side in ("L", "R"):
            population = f"{cell_type}_{side}"
            by_polarity = {}
            for polarity in ("on", "off"):
                looming_response = responses[f"{polarity}_looming"]
                receding_response = responses[f"{polarity}_receding"]
                looming = _response_energy(looming_response, population)
                receding = _response_energy(receding_response, population)
                neuron_looming = _neuron_contrasts(looming_response, receding_response, population)
                by_polarity[polarity] = {
                    "looming_energy": looming,
                    "receding_energy": receding,
                    "contrast": float(np.median(neuron_looming)),
                    "fraction_cells_looming_preferred": float(np.mean(neuron_looming > 0)),
                }
            looming_scores[population] = by_polarity

    mirror_errors = {}
    stimulus_by_name = {name: response for name, response in responses.items()}
    for name, response in responses.items():
        mirror_name = response["mirror_of"]
        if not mirror_name or mirror_name not in stimulus_by_name or name > mirror_name:
            continue
        mirrored = stimulus_by_name[mirror_name]
        errors = []
        for population in response["population_trace"]:
            if not population.endswith(("_L", "_R")):
                continue
            counterpart = (
                population[:-2] + "_R" if population.endswith("_L") else population[:-2] + "_L"
            )
            a = np.asarray(response["population_response_trace"][population], dtype=np.float64)
            b = np.asarray(mirrored["population_response_trace"][counterpart], dtype=np.float64)
            scale = float(np.mean(np.abs(a)) + np.mean(np.abs(b)) + 1e-12)
            errors.append(float(np.mean(np.abs(a - b)) / scale))
        mirror_errors[f"{name}<->{mirror_name}"] = float(np.mean(errors))

    known_looming = [
        max(values["on"]["contrast"], values["off"]["contrast"])
        for name, values in looming_scores.items()
        if name.startswith(("LPLC1_", "LPLC2_", "LC4_"))
    ]
    return {
        "direction_selectivity": direction_scores,
        "on_off_specialization": polarity_scores,
        "looming_vs_static": looming_scores,
        "mirror_response_error": mirror_errors,
        "summary": {
            "median_cardinal_direction_contrast": float(
                np.median([value["contrast"] for value in direction_scores.values()])
            ),
            "fraction_expected_direction_positive": float(
                np.mean([value["contrast"] > 0 for value in direction_scores.values()])
            ),
            "median_on_off_specialization": float(
                np.median([value["contrast"] for value in polarity_scores.values()])
            ),
            "fraction_expected_polarity_positive": float(
                np.mean([value["contrast"] > 0 for value in polarity_scores.values()])
            ),
            "median_known_looming_contrast": float(np.median(known_looming)),
            "maximum_mirror_response_error": max(mirror_errors.values(), default=0.0),
        },
    }


def _public_visual_responses(responses: dict[str, dict]) -> dict[str, dict]:
    """Persist hashes and summaries, not repeated per-frame/per-neuron scratch arrays."""
    compact = {}
    for stimulus, response in responses.items():
        traces = response["population_response_trace"]
        trace_hash = hashlib.sha256(
            b"".join(
                np.asarray(traces[population], dtype=np.float32).tobytes()
                for population in sorted(traces)
            )
        ).hexdigest()
        compact[stimulus] = {
            key: value
            for key, value in response.items()
            if not key.startswith("_")
            and key not in {"population_trace", "population_response_trace"}
        }
        compact[stimulus]["population_response_trace_sha256"] = trace_hash
    return compact


def evaluate_v7_controlled_vision(root: Path) -> dict:
    contract = V7Contract.load(root)
    visual = contract.payload["controlled_vision"]
    stimuli = build_controlled_stimuli(
        width=visual["width"], height=visual["height"], frames=visual["frames_per_stimulus"]
    )
    stimulus_by_name = {item.name: item for item in stimuli}
    mirror_checks = {}
    for item in stimuli:
        if item.mirror_of and item.mirror_of in stimulus_by_name:
            mirror_checks[item.name] = bool(
                np.array_equal(item.frames[:, :, ::-1], stimulus_by_name[item.mirror_of].frames)
            )
    thresholds = visual["gates"]
    backend_results = {}
    for backend in visual["retinal_backends"]:
        probe = V7VisualProbe(
            root,
            brain_substeps=visual["brain_substeps_per_frame"],
            baseline_frames=visual["baseline_frames"],
            retinal_backend=backend,
            control="real_malecns",
            control_seed=visual["topology_control_seed"],
        )
        responses = {item.name: probe.run(item) for item in stimuli}
        scores = score_v7_visual_responses(responses)
        gates = {
            "retina_only_direct_input": True,
            "all_requested_populations_present": all(
                any(name.startswith(f"{cell_type}_") for name in probe.populations)
                for cell_type in V7_TARGET_TYPES
            ),
            "stimulus_mirrors_exact": all(mirror_checks.values()),
            "cardinal_direction_contrast": (
                scores["summary"]["median_cardinal_direction_contrast"]
                >= thresholds["minimum_cardinal_direction_contrast"]
            ),
            "on_off_specialization": (
                scores["summary"]["median_on_off_specialization"]
                >= thresholds["minimum_on_off_specialization"]
            ),
            "looming_contrast": (
                scores["summary"]["median_known_looming_contrast"]
                >= thresholds["minimum_looming_contrast"]
            ),
            "mirror_response_error": (
                scores["summary"]["maximum_mirror_response_error"]
                <= thresholds["maximum_mirror_response_error"]
            ),
        }
        backend_results[backend] = {
            "scores": scores,
            "gates": gates,
            "controlled_response_gates_pass": all(gates.values()),
            "responses": _public_visual_responses(responses),
        }
    control_results = {}
    for control in visual["topology_controls"]:
        control_results[control] = {}
        for backend in visual["retinal_backends"]:
            probe = V7VisualProbe(
                root,
                brain_substeps=visual["brain_substeps_per_frame"],
                baseline_frames=visual["baseline_frames"],
                retinal_backend=backend,
                control=control,
                control_seed=visual["topology_control_seed"],
            )
            responses = {item.name: probe.run(item) for item in stimuli}
            scores = score_v7_visual_responses(responses)
            control_results[control][backend] = {
                "summary": scores["summary"],
                "response_sha256": hashlib.sha256(
                    b"".join(
                        np.asarray(
                            response["population_response_trace"][population],
                            dtype=np.float32,
                        ).tobytes()
                        for response in responses.values()
                        for population in sorted(response["population_response_trace"])
                    )
                ).hexdigest(),
            }
    passing_backends = [
        name for name, result in backend_results.items() if result["controlled_response_gates_pass"]
    ]
    topology_advantage = {}
    for backend in visual["retinal_backends"]:
        real = backend_results[backend]["scores"]["summary"]
        comparisons = {}
        for control in visual["topology_controls"]:
            shuffled = control_results[control][backend]["summary"]
            comparisons[control] = {
                "direction_contrast_higher": (
                    real["median_cardinal_direction_contrast"]
                    > shuffled["median_cardinal_direction_contrast"]
                ),
                "on_off_specialization_higher": (
                    real["median_on_off_specialization"] > shuffled["median_on_off_specialization"]
                ),
                "looming_contrast_higher": (
                    real["median_known_looming_contrast"]
                    > shuffled["median_known_looming_contrast"]
                ),
                "mirror_error_lower": (
                    real["maximum_mirror_response_error"]
                    < shuffled["maximum_mirror_response_error"]
                ),
            }
            comparisons[control]["all_metrics_better"] = all(comparisons[control].values())
        topology_advantage[backend] = {
            "by_control": comparisons,
            "beats_every_control": all(
                result["all_metrics_better"] for result in comparisons.values()
            ),
        }
    return {
        "protocol": {
            "version": 7,
            "mode": "v7-experimental",
            "deployment_enabled": False,
            "config_sha256": contract.sha256,
            "implementation_sha256": _sha256(root / V7_IMPLEMENTATION),
            "dynamics_backend": "legacy_uniform_tanh_v1",
            "retinal_backends": visual["retinal_backends"],
            "direct_input_type": "R1-R6",
            "direct_input_nodes": int(V7VisualProbe(root, brain_substeps=1).retina.size),
            "target_direct_input_overlap": 0,
            "brain_substeps_per_frame": visual["brain_substeps_per_frame"],
            "baseline_frames": visual["baseline_frames"],
            "stimulus_count": len(stimuli),
        },
        "population_counts": {
            name: int(len(nodes))
            for name, nodes in V7VisualProbe(root, brain_substeps=1).populations.items()
        },
        "stimuli": [
            {
                "name": item.name,
                "family": item.family,
                "polarity": item.polarity,
                "direction": item.direction,
                "mirror_of": item.mirror_of,
                "sha256": item.sha256,
            }
            for item in stimuli
        ],
        "stimulus_mirror_checks": mirror_checks,
        "backends": backend_results,
        "topology_controls": control_results,
        "topology_advantage": topology_advantage,
        "passing_retinal_backends": passing_backends,
        "controlled_response_gates_pass": bool(passing_backends),
        "stage1_topology_controls_complete": True,
        "strict_degree_preserving_control_complete": False,
        "real_topology_advantage": any(
            result["beats_every_control"] for result in topology_advantage.values()
        ),
        "advance_to_central_complex": False,
        "stop_reason": (
            "controlled visual response gates failed"
            if not passing_backends
            else "topology controls have not yet been evaluated"
        ),
    }


def evaluate_v7_typed_visual_candidate(root: Path) -> dict:
    """Screen type-specific visual time constants before topology controls."""
    contract = V7Contract.load(root)
    visual = contract.payload["controlled_vision"]
    stimuli = build_controlled_stimuli(
        width=visual["width"], height=visual["height"], frames=visual["frames_per_stimulus"]
    )
    thresholds = visual["gates"]
    backends = {}
    for dynamics_backend in (
        "typed_visual_leak_v1",
        "typed_visual_subgraph_v1",
        "columnar_delay_v1",
        "columnar_correlator_v1",
    ):
        backends[dynamics_backend] = {}
        for retinal_backend in visual["retinal_backends"]:
            probe = V7VisualProbe(
                root,
                brain_substeps=visual["brain_substeps_per_frame"],
                baseline_frames=visual["baseline_frames"],
                retinal_backend=retinal_backend,
                dynamics_backend=dynamics_backend,
                control="real_malecns",
                control_seed=visual["topology_control_seed"],
            )
            responses = {item.name: probe.run(item) for item in stimuli}
            scores = score_v7_visual_responses(responses)
            gates = {
                "cardinal_direction_contrast": (
                    scores["summary"]["median_cardinal_direction_contrast"]
                    >= thresholds["minimum_cardinal_direction_contrast"]
                ),
                "on_off_specialization": (
                    scores["summary"]["median_on_off_specialization"]
                    >= thresholds["minimum_on_off_specialization"]
                ),
                "looming_contrast": (
                    scores["summary"]["median_known_looming_contrast"]
                    >= thresholds["minimum_looming_contrast"]
                ),
                "mirror_response_error": (
                    scores["summary"]["maximum_mirror_response_error"]
                    <= thresholds["maximum_mirror_response_error"]
                ),
            }
            backends[dynamics_backend][retinal_backend] = {
                "scores": scores,
                "gates": gates,
                "controlled_response_gates_pass": all(gates.values()),
                "responses": _public_visual_responses(responses),
                "visual_subgraph_nodes": int(np.count_nonzero(probe.visual_subgraph_mask)),
                "visual_subgraph_edges": int(probe.adjacency.nnz),
            }
    axial_name = "columnar_delay_axial_hex_v1"
    backends[axial_name] = {}
    for retinal_backend in visual["retinal_backends"]:
        probe = V7VisualProbe(
            root,
            brain_substeps=visual["brain_substeps_per_frame"],
            baseline_frames=visual["baseline_frames"],
            retinal_backend=retinal_backend,
            retinal_geometry="axial_hex_cartesian_v1",
            dynamics_backend="columnar_delay_v1",
            control="real_malecns",
            control_seed=visual["topology_control_seed"],
        )
        responses = {item.name: probe.run(item) for item in stimuli}
        scores = score_v7_visual_responses(responses)
        gates = {
            "cardinal_direction_contrast": (
                scores["summary"]["median_cardinal_direction_contrast"]
                >= thresholds["minimum_cardinal_direction_contrast"]
            ),
            "on_off_specialization": (
                scores["summary"]["median_on_off_specialization"]
                >= thresholds["minimum_on_off_specialization"]
            ),
            "looming_contrast": (
                scores["summary"]["median_known_looming_contrast"]
                >= thresholds["minimum_looming_contrast"]
            ),
            "mirror_response_error": (
                scores["summary"]["maximum_mirror_response_error"]
                <= thresholds["maximum_mirror_response_error"]
            ),
        }
        backends[axial_name][retinal_backend] = {
            "scores": scores,
            "gates": gates,
            "controlled_response_gates_pass": all(gates.values()),
            "responses": _public_visual_responses(responses),
            "retinal_geometry": probe.retinal_geometry,
            "visual_subgraph_nodes": int(np.count_nonzero(probe.visual_subgraph_mask)),
            "visual_subgraph_edges": int(probe.adjacency.nnz),
        }
    passing = [
        {"dynamics": dynamics, "retina": retina}
        for dynamics, retinal_results in backends.items()
        for retina, result in retinal_results.items()
        if result["controlled_response_gates_pass"]
    ]
    return {
        "protocol": {
            "version": 7,
            "mode": "v7-experimental",
            "deployment_enabled": False,
            "config_sha256": contract.sha256,
            "implementation_sha256": _sha256(root / V7_IMPLEMENTATION),
            "dynamics_backends": [
                "typed_visual_leak_v1",
                "typed_visual_subgraph_v1",
                "columnar_delay_v1",
                "columnar_correlator_v1",
                axial_name,
            ],
            "retinal_geometries": visual["retinal_geometries"],
            "retinal_backends": visual["retinal_backends"],
            "direct_input_type": "R1-R6",
            "target_direct_input_overlap": 0,
            "topology_control": "real_malecns",
            "topology_controls_deferred_until_response_gate": True,
            "parameter_source": "literature-constrained ordering; not fitted to driving",
        },
        "typed_leak": visual["typed_visual_leak_v1"],
        "backends": backends,
        "passing_retinal_backends": passing,
        "controlled_response_gates_pass": bool(passing),
        "advance_to_topology_controls": bool(passing),
        "advance_to_central_complex": False,
        "stop_reason": (
            "typed visual response gates failed"
            if not passing
            else "topology controls required before central-complex stage"
        ),
    }


def write_v7_manifest(root: Path) -> dict:
    contract = V7Contract.load(root)
    implementation_sha256 = _sha256(root / V7_IMPLEMENTATION)
    evidence_paths = (
        root / "artifacts/v7-t4-conductance-fit.json",
        root / "artifacts/v7-t4-conductance-candidate.json",
        root / "artifacts/v7-branched-t4-candidate.json",
        root / "artifacts/v7-t4-source-audit.json",
        root / "artifacts/v7-optic-hex-axis-calibration.json",
        root / "artifacts/v7-typed-visual-candidate.json",
        root / "artifacts/v7-controlled-vision.json",
    )
    stage_status = "in_progress"
    advance = False
    blockers: list[str] = []
    evidence_used: list[str] = []
    current_evidence: dict[str, dict] = {}
    for evidence_path in evidence_paths:
        if not evidence_path.exists():
            continue
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        protocol = evidence.get("protocol", {})
        if evidence_path.name == "v7-t4-conductance-fit.json":
            source_audit_path = root / "artifacts/v7-t4-source-audit.json"
            valid_evidence = (
                protocol.get("v7_config_sha256") == contract.sha256
                and protocol.get("v7_implementation_sha256") == implementation_sha256
                and protocol.get("branched_implementation_sha256")
                == _sha256(root / "src/fly_emotion/driving/v7_branched.py")
                and protocol.get("conductance_implementation_sha256")
                == _sha256(root / "src/fly_emotion/driving/v7_conductance.py")
                and protocol.get("config_sha256")
                == _sha256(root / "configs/driving-v7-t4-fit.yaml")
                and protocol.get("implementation_sha256")
                == _sha256(root / "src/fly_emotion/driving/v7_fit.py")
                and source_audit_path.exists()
                and protocol.get("source_audit_sha256") == _sha256(source_audit_path)
            )
        elif evidence_path.name == "v7-t4-conductance-candidate.json":
            source_audit_path = root / "artifacts/v7-t4-source-audit.json"
            valid_evidence = (
                protocol.get("v7_config_sha256") == contract.sha256
                and protocol.get("v7_implementation_sha256") == implementation_sha256
                and protocol.get("branched_implementation_sha256")
                == _sha256(root / "src/fly_emotion/driving/v7_branched.py")
                and protocol.get("candidate_config_sha256")
                == _sha256(root / "configs/driving-v7-t4-conductance.yaml")
                and protocol.get("candidate_implementation_sha256")
                == _sha256(root / "src/fly_emotion/driving/v7_conductance.py")
                and source_audit_path.exists()
                and protocol.get("source_audit_sha256") == _sha256(source_audit_path)
            )
        elif evidence_path.name == "v7-t4-source-audit.json":
            valid_evidence = protocol.get("config_sha256") == _sha256(
                root / "configs/driving-v7-t4-source-audit.yaml"
            ) and protocol.get("implementation_sha256") == _sha256(
                root / "src/fly_emotion/driving/v7_source_audit.py"
            )
        elif evidence_path.name == "v7-branched-t4-candidate.json":
            source_audit_path = root / "artifacts/v7-t4-source-audit.json"
            valid_evidence = (
                protocol.get("v7_config_sha256") == contract.sha256
                and protocol.get("v7_implementation_sha256") == implementation_sha256
                and protocol.get("candidate_config_sha256")
                == _sha256(root / "configs/driving-v7-branched-t4.yaml")
                and protocol.get("candidate_implementation_sha256")
                == _sha256(root / "src/fly_emotion/driving/v7_branched.py")
                and source_audit_path.exists()
                and protocol.get("source_audit_sha256") == _sha256(source_audit_path)
            )
        else:
            valid_evidence = (
                protocol.get("config_sha256") == contract.sha256
                and protocol.get("implementation_sha256") == implementation_sha256
            )
        if valid_evidence:
            relative = str(evidence_path.relative_to(root))
            evidence_used.append(relative)
            current_evidence[evidence_path.name] = evidence

    axis = current_evidence.get("v7-optic-hex-axis-calibration.json")
    source_audit = current_evidence.get("v7-t4-source-audit.json")
    branched = current_evidence.get("v7-branched-t4-candidate.json")
    conductance = current_evidence.get("v7-t4-conductance-candidate.json")
    fitted = current_evidence.get("v7-t4-conductance-fit.json")
    controlled = current_evidence.get("v7-typed-visual-candidate.json") or (
        current_evidence.get("v7-controlled-vision.json")
    )
    if source_audit is None:
        blockers.append("T4_source_audit_evidence_stale_or_missing")
    elif not source_audit.get("nested_branch_validation", {}).get("nested_validation_pass"):
        blockers.append("nested_T4_branch_axis_validation_failed")
    if branched is None:
        blockers.append("branched_T4_response_evidence_stale_or_missing")
    elif not branched.get("controlled_response_gates_pass"):
        blockers.append("branched_T4_response_gates_failed")
    if conductance is None:
        blockers.append("published_T4_conductance_evidence_stale_or_missing")
    elif not conductance.get("controlled_response_gates_pass"):
        blockers.append("published_T4_conductance_response_gates_failed")
    if fitted is None:
        blockers.append("fitted_T4_conductance_evidence_stale_or_missing")
    elif not fitted.get("validation_passed"):
        blockers.append("fitted_T4_conductance_validation_failed")
    if controlled is None:
        blockers.append("controlled_visual_response_evidence_stale_or_missing")
    else:
        if not controlled.get("controlled_response_gates_pass"):
            blockers.append("controlled_visual_response_gates_failed")
        if not controlled.get("real_topology_advantage"):
            blockers.append("real_topology_advantage_not_demonstrated")
    advance = bool(
        source_audit is not None
        and source_audit.get("nested_branch_validation", {}).get("nested_validation_pass")
        and branched is not None
        and branched.get("controlled_response_gates_pass")
        and branched.get("topology_controls_complete")
        and conductance is not None
        and conductance.get("controlled_response_gates_pass")
        and conductance.get("topology_controls_complete")
        and fitted is not None
        and fitted.get("validation_passed")
        and fitted.get("test", {}).get("evaluated")
        and controlled is not None
        and controlled.get("advance_to_central_complex")
    )
    if evidence_used:
        stage_status = "passed" if advance else "blocked_on_visual_dynamics"
    return {
        "version": contract.payload["version"],
        "name": contract.payload["name"],
        "deployment_enabled": contract.payload["deployment_enabled"],
        "city_expansion_enabled": contract.payload["city_expansion_enabled"],
        "config_sha256": contract.sha256,
        "implementation_sha256": implementation_sha256,
        "baseline_contracts": contract.payload["baseline_contracts"],
        "stage_order": contract.payload["stage_order"],
        "current_stage": "controlled_vision",
        "stage_status": stage_status,
        "advance_to_central_complex": advance,
        "blockers": blockers,
        "stage_findings": {
            "historical_aggregate_optic_hex_axis_passed": bool(
                axis and axis.get("axis_calibration_pass")
            ),
            "nested_T4_branch_axis_validation_passed": bool(
                source_audit
                and source_audit.get("nested_branch_validation", {}).get("nested_validation_pass")
            ),
            "branched_T4_controlled_response_passed": bool(
                branched and branched.get("controlled_response_gates_pass")
            ),
            "published_T4_conductance_response_passed": bool(
                conductance and conductance.get("controlled_response_gates_pass")
            ),
            "fitted_T4_conductance_validation_passed": bool(
                fitted and fitted.get("validation_passed")
            ),
            "fitted_T4_one_time_test_evaluated": bool(
                fitted and fitted.get("test", {}).get("evaluated")
            ),
        },
        "current_evidence": evidence_used[0] if evidence_used else None,
        "evidence": evidence_used,
        "default_runtime_changed": False,
    }
