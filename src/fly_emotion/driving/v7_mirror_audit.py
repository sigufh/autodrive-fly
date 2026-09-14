"""Layerwise mirror-divergence audit after exactly paired R1--R6 input."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7 import V7_IMPLEMENTATION, V7Contract, build_controlled_stimuli
from fly_emotion.driving.v7_branched import (
    BRANCHED_CONFIG,
    BRANCHED_IMPLEMENTATION,
    V7BranchedT4Probe,
)
from fly_emotion.driving.v7_retina_audit import (
    RETINA_AUDIT_CONFIG,
    RETINA_AUDIT_IMPLEMENTATION,
)

MIRROR_AUDIT_IMPLEMENTATION = Path("src/fly_emotion/driving/v7_mirror_audit.py")
LAYER_TYPES = (
    "R1-R6",
    "L1",
    "L2",
    "L3",
    "L5",
    "Mi1",
    "Tm3",
    "Mi4",
    "Mi9",
    "C3",
    "CT1",
    "T4a",
    "T4b",
    "T4c",
    "T4d",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LayerMirrorProbe(V7BranchedT4Probe):
    def __init__(self, root: Path):
        super().__init__(
            root,
            retinal_backend="signed_frame_difference",
            retinal_geometry="balanced_exact_mirror_v1",
            brain_substeps=4,
            baseline_frames=8,
        )
        self.layer_populations = {}
        for cell_type in LAYER_TYPES:
            if cell_type == "R1-R6":
                self.layer_populations["R1-R6_L"] = self.retina.node_indices[self.retina.side < 0]
                self.layer_populations["R1-R6_R"] = self.retina.node_indices[self.retina.side > 0]
                continue
            for side in ("L", "R"):
                nodes = np.flatnonzero(
                    (self.node_types == cell_type)
                    & np.isin(
                        np.arange(self.graph.node_count),
                        self._population_side_nodes(side),
                    )
                ).astype(np.int32)
                if len(nodes):
                    self.layer_populations[f"{cell_type}_{side}"] = nodes

    def _population_side_nodes(self, side: str) -> np.ndarray:
        # Existing target populations expose soma-side membership for T4/LC. For upstream
        # layers load the same annotation column once and cache its graph-aligned labels.
        if not hasattr(self, "_soma_sides"):
            self._soma_sides = self._node_labels(
                self.root / "data/raw/malecns-v1.0/body-annotations.feather", "somaSide"
            )
        return np.flatnonzero(self._soma_sides == side).astype(np.int32)

    def run_layer_traces(self, stimulus) -> dict[str, list[float]]:
        state = np.zeros(self.graph.node_count, dtype=np.float32)
        history = [state.copy() for _ in range(self.branched["history_substeps"])]
        baseline = self._sample_retina(stimulus.frames[0])[self.retinal_permutation]
        baseline_drive = self._retinal_code(baseline, baseline, baseline)
        for _ in range(self.baseline_frames):
            drive = np.zeros_like(state)
            drive[self.retina.node_indices] = baseline_drive
            for _ in range(self.brain_substeps):
                state = self._advance(state, drive, history)
                state[self.retina.node_indices] = baseline_drive
        population_baseline = {
            name: float(np.mean(state[nodes])) for name, nodes in self.layer_populations.items()
        }
        previous = baseline.copy()
        traces = {name: [] for name in self.layer_populations}
        for image in stimulus.frames:
            sampled = self._sample_retina(image)[self.retinal_permutation]
            receptor_values = self._retinal_code(sampled, baseline, previous)
            previous = sampled
            drive = np.zeros_like(state)
            drive[self.retina.node_indices] = receptor_values
            for _ in range(self.brain_substeps):
                state = self._advance(state, drive, history)
                state[self.retina.node_indices] = receptor_values
            for name, nodes in self.layer_populations.items():
                traces[name].append(float(np.mean(state[nodes])) - population_baseline[name])
        return traces


def evaluate_v7_layerwise_mirror_audit(root: Path) -> dict:
    contract = V7Contract.load(root)
    probe = LayerMirrorProbe(root)
    stimuli = {item.name: item for item in build_controlled_stimuli()}
    pairs = (
        ("on_edge_left", "on_edge_right"),
        ("off_edge_left", "off_edge_right"),
        ("grating_left", "grating_right"),
        ("rotation_cw", "rotation_ccw"),
    )
    errors: dict[str, list[float]] = {cell_type: [] for cell_type in LAYER_TYPES}
    pair_reports = {}
    trace_hash = hashlib.sha256()
    for left_name, right_name in pairs:
        left = probe.run_layer_traces(stimuli[left_name])
        right = probe.run_layer_traces(stimuli[right_name])
        by_type = {}
        for cell_type in LAYER_TYPES:
            left_key = f"{cell_type}_L"
            right_key = f"{cell_type}_R"
            if left_key not in left or right_key not in right:
                continue
            a = np.asarray(left[left_key], dtype=np.float64)
            b = np.asarray(right[right_key], dtype=np.float64)
            scale = float(np.mean(np.abs(a)) + np.mean(np.abs(b)) + 1e-12)
            absolute_error = float(np.mean(np.abs(a - b)))
            error = float(absolute_error / scale)
            errors[cell_type].append(error)
            by_type[cell_type] = {
                "mirror_error": error,
                "response_scale": scale,
                "absolute_error": absolute_error,
                "left_population_size": int(len(probe.layer_populations[left_key])),
                "right_population_size": int(len(probe.layer_populations[right_key])),
            }
            trace_hash.update(a.astype(np.float32).tobytes())
            trace_hash.update(b.astype(np.float32).tobytes())
        pair_reports[f"{left_name}<->{right_name}"] = by_type
    summary = {
        cell_type: {
            "mean_mirror_error": float(np.mean(values)),
            "maximum_mirror_error": float(np.max(values)),
            "active_pair_mean_mirror_error": float(
                np.mean(
                    [
                        pair[cell_type]["mirror_error"]
                        for pair in pair_reports.values()
                        if cell_type in pair and pair[cell_type]["response_scale"] >= 1e-3
                    ]
                )
            )
            if any(
                cell_type in pair and pair[cell_type]["response_scale"] >= 1e-3
                for pair in pair_reports.values()
            )
            else None,
            "active_pairs": int(
                sum(
                    cell_type in pair and pair[cell_type]["response_scale"] >= 1e-3
                    for pair in pair_reports.values()
                )
            ),
        }
        for cell_type, values in errors.items()
        if values
    }
    ordered = [cell_type for cell_type in LAYER_TYPES if cell_type in summary]
    first_over_threshold = next(
        (cell_type for cell_type in ordered if summary[cell_type]["mean_mirror_error"] > 0.20),
        None,
    )
    first_active_over_threshold = next(
        (
            cell_type
            for cell_type in ordered
            if summary[cell_type]["active_pair_mean_mirror_error"] is not None
            and summary[cell_type]["active_pair_mean_mirror_error"] > 0.20
        ),
        None,
    )
    return {
        "protocol": {
            "version": 7,
            "name": "v7-layerwise-mirror-audit",
            "deployment_enabled": False,
            "retinal_geometry": "balanced_exact_mirror_v1",
            "retinal_backend": "signed_frame_difference",
            "implementation_sha256": _sha256(root / MIRROR_AUDIT_IMPLEMENTATION),
            "v7_config_sha256": contract.sha256,
            "v7_implementation_sha256": _sha256(root / V7_IMPLEMENTATION),
            "branched_config_sha256": _sha256(root / BRANCHED_CONFIG),
            "branched_implementation_sha256": _sha256(root / BRANCHED_IMPLEMENTATION),
            "retina_audit_config_sha256": _sha256(root / RETINA_AUDIT_CONFIG),
            "retina_audit_implementation_sha256": _sha256(root / RETINA_AUDIT_IMPLEMENTATION),
            "driving_data_used": False,
        },
        "stimulus_pairs": pair_reports,
        "layer_summary": summary,
        "first_layer_above_0_20_mean_error": first_over_threshold,
        "active_response_scale_threshold": 1e-3,
        "first_active_layer_above_0_20_mean_error": first_active_over_threshold,
        "trace_sha256": trace_hash.hexdigest(),
        "advance_to_central_complex": False,
    }
