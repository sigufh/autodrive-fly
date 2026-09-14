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


class V7VisualProbe:
    """Read-only population probe driven exclusively through mapped R1-R6 input."""

    def __init__(
        self,
        root: Path,
        *,
        brain_substeps: int = 4,
        baseline_frames: int = 8,
        retinal_backend: str = "legacy_absolute_contrast",
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
        self.visual_subgraph_mask = np.ones(self.graph.node_count, dtype=bool)
        if dynamics_backend == "typed_visual_subgraph_v1":
            self.adjacency, self.visual_subgraph_mask = self._visual_subgraph_adjacency(processed)
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

    def _advance(self, state: np.ndarray, drive: np.ndarray) -> np.ndarray:
        recurrent = self.adjacency @ (state * self.source_sign)
        return ((1.0 - self.leak) * state + self.leak * np.tanh(1.8 * recurrent + drive)).astype(
            np.float32
        )

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
        traces = {name: [] for name in self.populations}
        retinal_hash = hashlib.sha256()
        baseline_image = stimulus.frames[0].copy()
        baseline_values = self.retina.encode(baseline_image)[self.retinal_permutation]
        baseline_drive = self._retinal_code(baseline_values, baseline_values, baseline_values)
        for _ in range(self.baseline_frames):
            drive = np.zeros_like(state)
            drive[self.retina.node_indices] = baseline_drive
            for _ in range(self.brain_substeps):
                state = self._advance(state, drive)
                state[self.retina.node_indices] = baseline_drive
        population_baseline = {
            name: float(np.mean(state[nodes])) for name, nodes in self.populations.items()
        }
        previous_values = baseline_values.copy()
        for image in stimulus.frames:
            sampled = self.retina.encode(image)[self.retinal_permutation]
            receptor_values = self._retinal_code(sampled, baseline_values, previous_values)
            previous_values = sampled
            retinal_hash.update(receptor_values.tobytes())
            drive = np.zeros_like(state)
            drive[self.retina.node_indices] = receptor_values
            for _ in range(self.brain_substeps):
                state = self._advance(state, drive)
                state[self.retina.node_indices] = receptor_values
            for name, nodes in self.populations.items():
                traces[name].append(float(np.mean(state[nodes])))
        return {
            "name": stimulus.name,
            "family": stimulus.family,
            "polarity": stimulus.polarity,
            "direction": stimulus.direction,
            "mirror_of": stimulus.mirror_of,
            "stimulus_sha256": stimulus.sha256,
            "retinal_drive_sha256": retinal_hash.hexdigest(),
            "retinal_backend": self.retinal_backend,
            "dynamics_backend": self.dynamics_backend,
            "topology_control": self.control,
            "visual_subgraph_nodes": int(np.count_nonzero(self.visual_subgraph_mask)),
            "visual_subgraph_edges": int(self.adjacency.nnz),
            "baseline_frames": self.baseline_frames,
            "population_trace": traces,
            "population_response_trace": {
                name: [float(value - population_baseline[name]) for value in values]
                for name, values in traces.items()
            },
            "population_mean_abs": {
                name: float(np.mean(np.abs(values))) for name, values in traces.items()
            },
            "population_peak_abs": {
                name: float(np.max(np.abs(values))) for name, values in traces.items()
            },
        }


def _response_energy(response: dict, population: str) -> float:
    values = np.asarray(response["population_response_trace"][population], dtype=np.float64)
    return float(np.mean(np.maximum(values, 0.0)))


def _contrast(preferred: float, opposite: float) -> float:
    return float((preferred - opposite) / (abs(preferred) + abs(opposite) + 1e-12))


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
                preferred_energy = _response_energy(
                    responses[f"{polarity}_edge_{preferred}"], population
                )
                opposite_energy = _response_energy(
                    responses[f"{polarity}_edge_{opposite}"], population
                )
                direction_scores[population] = {
                    "expected_direction": preferred,
                    "preferred_energy": preferred_energy,
                    "opposite_energy": opposite_energy,
                    "contrast": _contrast(preferred_energy, opposite_energy),
                }
                matched_other = _response_energy(
                    responses[f"{opposite_polarity}_edge_{preferred}"], population
                )
                polarity_scores[population] = {
                    "expected_polarity": polarity,
                    "preferred_polarity_energy": preferred_energy,
                    "opposite_polarity_energy": matched_other,
                    "contrast": _contrast(preferred_energy, matched_other),
                }

    looming_scores = {}
    for cell_type in ("LPLC1", "LPLC2", "LC4", "LC6", "LC16"):
        for side in ("L", "R"):
            population = f"{cell_type}_{side}"
            by_polarity = {}
            for polarity in ("on", "off"):
                looming = _response_energy(responses[f"{polarity}_looming"], population)
                receding = _response_energy(responses[f"{polarity}_receding"], population)
                by_polarity[polarity] = {
                    "looming_energy": looming,
                    "receding_energy": receding,
                    "contrast": _contrast(looming, receding),
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
            "responses": responses,
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
            "dynamics_backend": V7VisualProbe.backend,
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
    for dynamics_backend in ("typed_visual_leak_v1", "typed_visual_subgraph_v1"):
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
                "responses": responses,
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
            "dynamics_backends": ["typed_visual_leak_v1", "typed_visual_subgraph_v1"],
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
    evidence_paths = (
        root / "artifacts/v7-typed-visual-candidate.json",
        root / "artifacts/v7-controlled-vision.json",
    )
    stage_status = "in_progress"
    advance = False
    blockers: list[str] = []
    evidence_used = None
    for evidence_path in evidence_paths:
        if not evidence_path.exists():
            continue
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("protocol", {}).get("config_sha256") == contract.sha256:
            advance = bool(evidence.get("advance_to_central_complex"))
            stage_status = "passed" if advance else "blocked_on_visual_dynamics"
            evidence_used = str(evidence_path.relative_to(root))
            if not evidence.get("controlled_response_gates_pass"):
                blockers.append("controlled_visual_response_gates_failed")
            if not evidence.get("real_topology_advantage"):
                blockers.append("real_topology_advantage_not_demonstrated")
            break
    return {
        "version": contract.payload["version"],
        "name": contract.payload["name"],
        "deployment_enabled": contract.payload["deployment_enabled"],
        "city_expansion_enabled": contract.payload["city_expansion_enabled"],
        "config_sha256": contract.sha256,
        "baseline_contracts": contract.payload["baseline_contracts"],
        "stage_order": contract.payload["stage_order"],
        "current_stage": "controlled_vision",
        "stage_status": stage_status,
        "advance_to_central_complex": advance,
        "blockers": blockers,
        "current_evidence": evidence_used,
        "default_runtime_changed": False,
    }
