"""Published-parameter single-compartment T4 conductance candidate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import (
    V7_IMPLEMENTATION,
    V7Contract,
    V7VisualProbe,
    VisualStimulus,
    _public_visual_responses,
    build_controlled_stimuli,
    score_v7_visual_responses,
)
from fly_emotion.driving.v7_branched import (
    BRANCHED_IMPLEMENTATION,
    SOURCE_AUDIT,
    V7BranchedT4Probe,
)

CONDUCTANCE_CONFIG = Path("configs/driving-v7-t4-conductance.yaml")
CONDUCTANCE_IMPLEMENTATION = Path("src/fly_emotion/driving/v7_conductance.py")
SOURCE_TYPES = ("Mi9", "Tm3", "Mi1", "Mi4", "C3")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceNormalization:
    low: np.ndarray
    high: np.ndarray
    source_mask: np.ndarray
    sha256: str
    summary: dict


def build_conductance_calibration_stimuli(config: dict) -> list[VisualStimulus]:
    calibration = config["calibration"]
    width = int(calibration["width"])
    height = int(calibration["height"])
    uniform_frames = int(calibration["uniform_frames"])
    levels = np.concatenate(
        (
            np.linspace(0.02, 0.98, uniform_frames // 2),
            np.linspace(0.98, 0.02, uniform_frames - uniform_frames // 2),
        )
    )
    uniform = np.stack([np.full((height, width), level, dtype=np.float32) for level in levels])
    rng = np.random.default_rng(int(calibration["seed"]))
    coarse = rng.integers(
        0,
        2,
        size=(
            int(calibration["white_noise_frames"]),
            int(calibration["checker_rows"]),
            int(calibration["checker_columns"]),
        ),
    ).astype(np.float32)
    row_repeat = height // int(calibration["checker_rows"])
    column_repeat = width // int(calibration["checker_columns"])
    white_noise = np.repeat(
        np.repeat(0.08 + 0.84 * coarse, row_repeat, axis=1),
        column_repeat,
        axis=2,
    )
    return [
        VisualStimulus(
            "conductance_uniform_sweep",
            "calibration",
            "mixed",
            "none",
            uniform,
        ),
        VisualStimulus(
            "conductance_seeded_white_noise",
            "calibration",
            "mixed",
            "none",
            white_noise.astype(np.float32),
        ),
    ]


def _collect_source_normalization(
    root: Path, *, retinal_backend: str, retinal_geometry: str, config: dict
) -> SourceNormalization:
    probe = V7BranchedT4Probe(
        root,
        retinal_backend=retinal_backend,
        retinal_geometry=retinal_geometry,
        brain_substeps=int(config["brain_substeps_per_frame"]),
        baseline_frames=int(config["baseline_frames"]),
    )
    # Calibration estimates upstream ranges with the ordinary typed visual subgraph.
    # The candidate T4 equation is not used to define its own input normalization.
    probe.correlator = None
    probe.dynamics_backend = "typed_visual_subgraph_v1"
    source_mask = np.isin(probe.node_types, SOURCE_TYPES)
    low = np.full(probe.graph.node_count, np.inf, dtype=np.float32)
    high = np.full(probe.graph.node_count, -np.inf, dtype=np.float32)
    for stimulus in build_conductance_calibration_stimuli(config):
        state = np.zeros(probe.graph.node_count, dtype=np.float32)
        history = [state.copy()]
        baseline = probe._sample_retina(stimulus.frames[0])
        baseline = baseline[probe.retinal_permutation]
        baseline_drive = probe._retinal_code(baseline, baseline, baseline)
        for _ in range(probe.baseline_frames):
            drive = np.zeros_like(state)
            drive[probe.retina.node_indices] = baseline_drive
            for _ in range(probe.brain_substeps):
                state = V7VisualProbe._advance(probe, state, drive, history)
                state[probe.retina.node_indices] = baseline_drive
        previous = baseline.copy()
        for image in stimulus.frames:
            sampled = probe._sample_retina(image)[probe.retinal_permutation]
            receptor_values = probe._retinal_code(sampled, baseline, previous)
            previous = sampled
            drive = np.zeros_like(state)
            drive[probe.retina.node_indices] = receptor_values
            for _ in range(probe.brain_substeps):
                state = V7VisualProbe._advance(probe, state, drive, history)
                state[probe.retina.node_indices] = receptor_values
                low[source_mask] = np.minimum(low[source_mask], state[source_mask])
                high[source_mask] = np.maximum(high[source_mask], state[source_mask])
    span = high - low
    active = source_mask & np.isfinite(span) & (span > 1e-6)
    low[~active] = 0.0
    high[~active] = 1.0
    digest = hashlib.sha256()
    digest.update(low[source_mask].tobytes())
    digest.update(high[source_mask].tobytes())
    summary = {}
    for source_type in SOURCE_TYPES:
        mask = probe.node_types == source_type
        summary[source_type] = {
            "cells": int(np.count_nonzero(mask)),
            "dynamic_cells": int(np.count_nonzero(active & mask)),
            "dynamic_fraction": float(np.mean(active[mask])),
            "median_range": float(np.median(span[active & mask])) if np.any(active & mask) else 0.0,
        }
    return SourceNormalization(low, high, source_mask, digest.hexdigest(), summary)


class V7PublishedConductanceProbe(V7BranchedT4Probe):
    """Apply published gains, thresholds and reversal potentials to real source edges."""

    def __init__(
        self,
        root: Path,
        *,
        retinal_backend: str,
        normalization: SourceNormalization,
    ):
        self.conductance_config = yaml.safe_load(
            (root / CONDUCTANCE_CONFIG).read_text(encoding="utf-8")
        )
        if self.conductance_config.get("deployment_enabled"):
            raise ValueError("published T4 conductance candidate must not be deployed")
        if self.conductance_config.get("parameter_scan_allowed"):
            raise ValueError("published T4 conductance parameters must remain frozen")
        super().__init__(
            root,
            retinal_backend=retinal_backend,
            retinal_geometry=self.conductance_config["primary_retinal_geometry"],
            brain_substeps=int(self.conductance_config["brain_substeps_per_frame"]),
            baseline_frames=int(self.conductance_config["baseline_frames"]),
        )
        model = self.conductance_config["published_single_compartment_model"]
        matrices = {
            source_type: self._normalized_target_inputs(self.branched["targets"], (source_type,))
            for source_type in SOURCE_TYPES
        }
        valid = np.ones(len(self.branched["targets"]), dtype=bool)
        for matrix in matrices.values():
            valid &= np.diff(matrix.indptr) > 0
        self.conductance = {
            "targets": self.branched["targets"][valid],
            "matrices": {name: matrix[valid] for name, matrix in matrices.items()},
            "model": model,
            "normalization": normalization,
            "all_targets": int(len(self.branched["targets"])),
        }
        self.correlator = None
        self.dynamics_backend = self.conductance_config["name"]

    def _normalized_source_state(self, state: np.ndarray) -> np.ndarray:
        normalization = self.conductance["normalization"]
        return np.clip(
            (state - normalization.low) / np.maximum(normalization.high - normalization.low, 1e-6),
            0.0,
            1.0,
        ).astype(np.float32)

    def _conductance_voltage(self, source_state: np.ndarray) -> np.ndarray:
        model = self.conductance["model"]
        reversal = model["reversal_potentials_millivolts"]
        numerator = np.full(
            len(self.conductance["targets"]),
            float(reversal["leak"]) * float(model["leak_conductance"]),
            dtype=np.float64,
        )
        denominator = np.full(
            len(self.conductance["targets"]),
            float(model["leak_conductance"]),
            dtype=np.float64,
        )
        for source_type, parameters in model["source_parameters"].items():
            signal = self.conductance["matrices"][source_type] @ source_state
            conductance = float(parameters["gain"]) * np.maximum(
                signal - float(parameters["threshold"]), 0.0
            )
            denominator += conductance
            numerator += float(reversal[parameters["reversal"]]) * conductance
        return numerator / np.maximum(denominator, 1e-12)

    def _advance(
        self, state: np.ndarray, drive: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        recurrent = self.adjacency @ (state * self.source_sign)
        updated = ((1.0 - self.leak) * state + self.leak * np.tanh(1.8 * recurrent + drive)).astype(
            np.float32
        )
        source_state = self._normalized_source_state(updated)
        voltage = self._conductance_voltage(source_state)
        output = self.conductance["model"]["output_normalization"]
        target_drive = np.clip(
            (voltage - float(output["baseline_millivolts"]))
            / (
                float(output["excitatory_reversal_millivolts"])
                - float(output["baseline_millivolts"])
            ),
            -1.0,
            1.0,
        ).astype(np.float32)
        targets = self.conductance["targets"]
        updated[targets] = (1.0 - self.leak[targets]) * state[targets] + self.leak[
            targets
        ] * target_drive
        history.insert(0, state.copy())
        del history[1:]
        return updated

    def run(self, stimulus: VisualStimulus) -> dict:
        return V7VisualProbe.run(self, stimulus)


def evaluate_v7_published_conductance(root: Path) -> dict:
    contract = V7Contract.load(root)
    config_path = root / CONDUCTANCE_CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    visual = contract.payload["controlled_vision"]
    stimuli = build_controlled_stimuli(
        width=visual["width"],
        height=visual["height"],
        frames=visual["frames_per_stimulus"],
    )
    calibration_stimuli = build_conductance_calibration_stimuli(config)
    results = {}
    for retinal_backend in config["retinal_backends"]:
        normalization = _collect_source_normalization(
            root,
            retinal_backend=retinal_backend,
            retinal_geometry=config["primary_retinal_geometry"],
            config=config,
        )
        probe = V7PublishedConductanceProbe(
            root, retinal_backend=retinal_backend, normalization=normalization
        )
        responses = {stimulus.name: probe.run(stimulus) for stimulus in stimuli}
        scores = score_v7_visual_responses(responses)
        t4 = [
            value
            for name, value in scores["direction_selectivity"].items()
            if name.startswith("T4")
        ]
        t4_median = float(np.median([value["contrast"] for value in t4]))
        t4_positive = float(np.mean([value["contrast"] > 0 for value in t4]))
        thresholds = config["gates"]
        gates = {
            "T4_cardinal_direction_contrast": t4_median
            >= thresholds["minimum_T4_cardinal_direction_contrast"],
            "all_T4_populations_positive": t4_positive
            >= thresholds["minimum_all_T4_populations_positive"],
            "overall_cardinal_direction_contrast": scores["summary"][
                "median_cardinal_direction_contrast"
            ]
            >= thresholds["minimum_overall_cardinal_direction_contrast"],
            "on_off_specialization": scores["summary"]["median_on_off_specialization"]
            >= thresholds["minimum_on_off_specialization"],
            "looming_contrast": scores["summary"]["median_known_looming_contrast"]
            >= thresholds["minimum_looming_contrast"],
            "mirror_response_error": scores["summary"][thresholds["mirror_metric"]]
            <= thresholds["maximum_mirror_response_error"],
        }
        results[retinal_backend] = {
            "normalization": {
                "sha256": normalization.sha256,
                "summary": normalization.summary,
            },
            "T4_summary": {
                "median_cardinal_direction_contrast": t4_median,
                "fraction_populations_positive": t4_positive,
            },
            "scores": scores,
            "gates": gates,
            "controlled_response_gates_pass": all(gates.values()),
            "responses": _public_visual_responses(responses),
            "conductance_targets": int(len(probe.conductance["targets"])),
            "source_edge_counts": {
                name: int(matrix.nnz) for name, matrix in probe.conductance["matrices"].items()
            },
        }
    passing = [name for name, result in results.items() if result["controlled_response_gates_pass"]]
    return {
        "protocol": {
            "version": 7,
            "name": config["name"],
            "deployment_enabled": False,
            "parameter_scan_allowed": False,
            "v7_config_sha256": contract.sha256,
            "v7_implementation_sha256": _sha256(root / V7_IMPLEMENTATION),
            "branched_implementation_sha256": _sha256(root / BRANCHED_IMPLEMENTATION),
            "candidate_config_sha256": _sha256(config_path),
            "candidate_implementation_sha256": _sha256(root / CONDUCTANCE_IMPLEMENTATION),
            "source_audit_sha256": _sha256(root / SOURCE_AUDIT),
            "driving_data_used": False,
            "direction_labels_used_for_normalization": False,
            "published_parameters_modified": False,
            "direct_input_type": "R1-R6",
            "calibration_stimuli": [
                {"name": stimulus.name, "sha256": stimulus.sha256}
                for stimulus in calibration_stimuli
            ],
        },
        "published_model": config["published_single_compartment_model"],
        "retinal_geometry": config["primary_retinal_geometry"],
        "retinal_results": results,
        "passing_retinal_backends": passing,
        "controlled_response_gates_pass": bool(passing),
        "topology_controls_complete": False,
        "advance_to_topology_controls": bool(passing),
        "advance_to_central_complex": False,
        "stop_reason": (
            "published conductance response gates failed"
            if not passing
            else "fresh topology controls required before any stage advance"
        ),
    }
