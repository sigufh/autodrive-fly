from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.driving.v7 import V7Contract, V7VisualProbe
from fly_emotion.driving.v7_branched import V7BranchedT4Probe
from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    SOURCE_TYPES,
    V7PublishedConductanceProbe,
    _collect_source_normalization,
)
from fly_emotion.driving.v7_disinhibition import PresynapticThresholdProbe

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_stability.py")
CHECKPOINTS = (128, 512, 2048)
WINDOW = 128
TOLERANCE = 1e-5
MAX_LAG = 32


class TypedBackgroundProbe(V7BranchedT4Probe):
    def __init__(self, root: Path, *, retinal_backend: str):
        super().__init__(
            root,
            retinal_backend=retinal_backend,
            retinal_geometry="nested_t4_axis_v1",
            brain_substeps=4,
            baseline_frames=8,
        )
        self.correlator = None
        self.dynamics_backend = "typed_visual_subgraph_v1"

    def _advance(
        self, state: np.ndarray, drive: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        return V7VisualProbe._advance(self, state, drive, history)


def classify_trace(values: np.ndarray, *, tolerance: float = TOLERANCE) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("finite one-dimensional trace with at least two samples required")
    span = float(np.ptp(values))
    step = float(np.max(np.abs(np.diff(values))))
    lag_errors = {
        lag: float(max(np.ptp(values[phase::lag]) for phase in range(lag)))
        for lag in range(2, min(MAX_LAG, len(values) // 4) + 1)
    }
    period = next((lag for lag, error in lag_errors.items() if error <= tolerance), None)
    status = (
        "stationary_within_window"
        if span <= tolerance
        else ("periodic_within_window" if period is not None else "unresolved_drift_or_oscillation")
    )
    return {
        "status": status,
        "peak_to_peak": span,
        "maximum_step_change": step,
        "period_substeps": period if status == "periodic_within_window" else None,
        "best_lag_substeps": min(lag_errors, key=lag_errors.get) if lag_errors else None,
        "lag_maximum_errors": {str(lag): error for lag, error in lag_errors.items()},
    }


def summarize_tail(states: np.ndarray, labels: np.ndarray, body_ids: np.ndarray) -> dict:
    if states.ndim != 2 or states.shape[1] != len(labels) or len(labels) != len(body_ids):
        raise ValueError("state history and node labels must be aligned")
    if not np.all(np.isfinite(states)):
        raise ValueError("nonfinite neural state")
    span = np.ptp(states, axis=0)
    step = np.max(np.abs(np.diff(states, axis=0)), axis=0)
    groups = []
    for group in np.unique(labels):
        mask = labels == group
        groups.append(
            {
                "population": str(group),
                "cells": int(mask.sum()),
                "maximum_step_change": float(step[mask].max()),
                "maximum_peak_to_peak": float(span[mask].max()),
                "median_peak_to_peak": float(np.median(span[mask])),
                "cells_above_tolerance": int(np.count_nonzero(span[mask] > TOLERANCE)),
            }
        )
    groups.sort(key=lambda row: (-row["maximum_step_change"], row["population"]))
    top = np.lexsort((body_ids, -step))[:10]
    return {
        "window_samples": len(states),
        "cells": len(labels),
        "all_cells_stationary_within_window": bool(np.all(span <= TOLERANCE)),
        "maximum_step_change": float(step.max()),
        "maximum_peak_to_peak": float(span.max()),
        "cells_above_tolerance": int(np.count_nonzero(span > TOLERANCE)),
        "populations_ranked_by_step_change": groups,
        "top_cells": [
            {
                "body_id": int(body_ids[index]),
                "population": str(labels[index]),
                "trace": states[:, index].tolist(),
                **classify_trace(states[:, index]),
            }
            for index in top
        ],
    }


def run_constant_background(probe: V7VisualProbe, *, level: float = 0.5) -> dict:
    sampled = probe._sample_retina(np.full((24, 48), level, dtype=np.float32))
    receptor_drive = probe._retinal_code(sampled, sampled, sampled)
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    drive = np.zeros_like(state)
    drive[probe.retina.node_indices] = receptor_drive
    side = probe._node_labels(
        probe.root / "data/raw/malecns-v1.0/body-annotations.feather", "somaSide"
    )
    nodes = np.flatnonzero(probe.visual_subgraph_mask)
    labels = np.array(
        [f"{probe.node_types[node] or '<untyped>'}_{side[node] or '<unknown>'}" for node in nodes]
    )
    tail = deque(maxlen=WINDOW)
    checkpoints = {}
    full_delta = 0.0
    for substep in range(1, CHECKPOINTS[-1] + 1):
        previous = state
        state = probe._advance(state, drive, history)
        state[probe.retina.node_indices] = receptor_drive
        full_delta = float(np.max(np.abs(state - previous)))
        tail.append(state[nodes].copy())
        if substep in CHECKPOINTS:
            checkpoints[str(substep)] = {
                "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "last_substep_global_change": full_delta,
                **summarize_tail(np.asarray(tail), labels, probe.graph.body_ids[nodes]),
            }
    return {
        "background_level": level,
        "initial_state": "all zeros",
        "external_drive_is_zero": bool(np.all(receptor_drive == 0)),
        "retinal_drive_sha256": hashlib.sha256(receptor_drive.tobytes()).hexdigest(),
        "checkpoints": checkpoints,
    }


def cut_t4_feedback(
    adjacency: sparse.csr_matrix, node_types: np.ndarray, mode: str
) -> tuple[sparse.csr_matrix, dict]:
    if mode not in {"direct_inputs", "all_outputs"}:
        raise ValueError("unknown T4 feedback cut")
    source_t4 = np.isin(node_types, ("T4a", "T4b", "T4c", "T4d"))
    target_mask = (
        np.isin(node_types, SOURCE_TYPES)
        if mode == "direct_inputs"
        else np.ones(len(node_types), dtype=bool)
    )
    result = adjacency.copy()
    rows = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    removed = source_t4[adjacency.indices] & target_mask[rows]
    counts = {}
    removed_rows = rows[removed]
    for kind in np.unique(node_types[removed_rows]):
        selected = node_types[removed_rows] == kind
        counts[str(kind) or "<untyped>"] = int(selected.sum())
    metadata = {
        "mode": mode,
        "removed_edges": int(removed.sum()),
        "removed_normalized_weight_sum": float(np.sum(adjacency.data[removed], dtype=np.float64)),
        "postsynaptic_type_edge_counts": counts,
        "removed_edge_sha256": hashlib.sha256(
            removed_rows.astype(np.int32).tobytes()
            + adjacency.indices[removed].tobytes()
            + adjacency.data[removed].tobytes()
        ).hexdigest(),
        "remaining_weights_renormalized": False,
    }
    result.data[removed] = 0
    result.eliminate_zeros()
    return result, metadata


def _compact_checkpoints(result: dict) -> dict:
    checkpoints = {}
    for step, values in result["checkpoints"].items():
        groups = values["populations_ranked_by_step_change"]
        groups_hash = hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()
        checkpoints[step] = {
            key: value
            for key, value in values.items()
            if key != "populations_ranked_by_step_change"
        } | {
            "population_summary_sha256": groups_hash,
            "top_population_residuals": groups[:10],
            "T4_population_residuals": [
                group
                for group in groups
                if group["population"].startswith(("T4a_", "T4b_", "T4c_", "T4d_"))
            ],
        }
    return result | {"checkpoints": checkpoints}


def evaluate_v7_feedback_cut(root: Path) -> dict:
    contract = V7Contract.load(root)
    config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text())
    prior_path = root / "artifacts/v7-background-stability.json"
    reference = json.loads(prior_path.read_text())
    for relative, expected in reference["protocol"]["dependencies_sha256"].items():
        with (root / relative).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"stale stability baseline: {relative}")
    results = {}
    for backend in ("linear_luminance", "signed_frame_difference"):
        normalization = _collect_source_normalization(
            root, retinal_backend=backend, retinal_geometry="nested_t4_axis_v1", config=config
        )
        probe = V7PublishedConductanceProbe(
            root, retinal_backend=backend, normalization=normalization
        )
        adjacency = probe.adjacency
        models = {}
        for mode in ("intact", "direct_inputs", "all_outputs"):
            if mode == "intact":
                probe.adjacency = adjacency
                cut = {"mode": mode, "removed_edges": 0, "remaining_weights_renormalized": False}
            else:
                probe.adjacency, cut = cut_t4_feedback(adjacency, probe.node_types, mode)
            outcome = run_constant_background(probe)
            if mode == "intact":
                previous = reference["results"][backend]["models"]["V7PublishedConductanceProbe"]
                if outcome != previous:
                    raise ValueError("intact model failed exact frozen background replay")
            models[mode] = {"cut": cut, "trajectory": _compact_checkpoints(outcome)}
        probe.adjacency = adjacency
        results[backend] = {
            "normalization_sha256": normalization.sha256,
            "intact_matches_frozen_baseline": True,
            "models": models,
        }
    hashes = dict(reference["protocol"]["dependencies_sha256"])
    with prior_path.open("rb") as source:
        hashes[str(prior_path.relative_to(root))] = hashlib.file_digest(
            source, "sha256"
        ).hexdigest()
    return {
        "protocol": {
            "name": "v7-t4-feedback-cut-v1",
            "advance_allowed": False,
            "v7_config_sha256": contract.sha256,
            "dependencies_sha256": hashes,
            "cut_rule": "zero T4a-d source edges into Mi9/Tm3/Mi1/Mi4/C3, or every T4a-d output",
            "normalization": "same uncut calibration; remaining adjacency weights unchanged",
            "background": 0.5,
            "initial_state": "all zeros",
            "parameter_fitting": False,
            "checkpoints_substeps": list(CHECKPOINTS),
            "window_substeps": WINDOW,
            "driving_data_used": False,
            "population_ranking": "diagnostic, not used to select cuts",
        },
        "results": results,
        "limitations": [
            "All-output cuts also remove feedforward signalling; stability is not task competence.",
            "One constant background and one initial state do not establish general stability.",
            "Direct cuts test only one-hop T4 inputs; indirect feedback can remain.",
            "Removing edges is an explicit causal ablation, not an improved biological model.",
            "No motion battery, driving release, or biological oscillation claim is made.",
        ],
        "advance_to_central_complex": False,
    }


def evaluate_v7_background_stability(root: Path) -> dict:
    contract = V7Contract.load(root)
    config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text())
    results = {}
    for backend in ("linear_luminance", "signed_frame_difference"):
        normalization = _collect_source_normalization(
            root,
            retinal_backend=backend,
            retinal_geometry="nested_t4_axis_v1",
            config=config,
        )
        models = {}
        for cls in (TypedBackgroundProbe, V7PublishedConductanceProbe, PresynapticThresholdProbe):
            kwargs = {"retinal_backend": backend}
            if cls is not TypedBackgroundProbe:
                kwargs["normalization"] = normalization
            probe = cls(root, **kwargs)
            models[cls.__name__] = run_constant_background(probe)
        results[backend] = {"normalization_sha256": normalization.sha256, "models": models}
    paths = [
        IMPLEMENTATION,
        Path("src/fly_emotion/driving/v7.py"),
        Path("src/fly_emotion/driving/v7_branched.py"),
        Path("src/fly_emotion/driving/v7_conductance.py"),
        Path("src/fly_emotion/driving/v7_disinhibition.py"),
        Path("src/fly_emotion/driving/v7_source_audit.py"),
        Path("src/fly_emotion/driving/v7_retina_audit.py"),
        Path("src/fly_emotion/driving/retina.py"),
        Path("src/fly_emotion/driving/engine.py"),
        Path("src/fly_emotion/connectome/graph.py"),
        Path("configs/driving-v7.yaml"),
        Path("configs/driving-v7-branched-t4.yaml"),
        CONDUCTANCE_CONFIG,
        Path("configs/driving-v7-t4-source-audit.yaml"),
        Path("artifacts/v7-t4-source-audit.json"),
        Path("data/raw/malecns-v1.0/body-annotations.feather"),
        Path("data/raw/malecns-v1.0/body-neurotransmitters.feather"),
        Path("data/processed/malecns-v1.0/body_ids.npy"),
        Path("data/processed/malecns-v1.0/adjacency_raw.npz"),
        Path("data/processed/malecns-v1.0/adjacency_target_norm.npz"),
        Path("data/processed/malecns-v1.0/retina_map.npz"),
    ]
    hashes = {}
    for path in paths:
        with (root / path).open("rb") as source:
            hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-background-stability-v1",
            "exploratory": True,
            "advance_allowed": False,
            "parameter_fitting": False,
            "driving_data_used": False,
            "dependencies_sha256": hashes,
            "v7_config_sha256": contract.sha256,
            "geometry": "nested_t4_axis_v1",
            "background_level": 0.5,
            "checkpoints_substeps": list(CHECKPOINTS),
            "tail_window_substeps": WINDOW,
            "absolute_state_tolerance": TOLERANCE,
            "max_period_substeps": MAX_LAG,
            "period_test": "smallest lag with same-phase range <= tolerance; >=4 repetitions",
            "population_scope": "all frozen visual nodes; soma-side labels when present",
            "time_unit": "simulation substeps, not biological milliseconds",
        },
        "results": results,
        "limitations": [
            "Finite-window stationarity or periodicity is not a stability proof.",
            "Zero state with zero drive is a degenerate control, not physiological rest.",
            "Population residual ranking does not identify the causal feedback loop.",
            "Only one constant background and one initialization are tested per model and encoder.",
            "Periodic behaviour in this proxy is not evidence of physiological oscillations.",
        ],
        "advance_to_central_complex": False,
    }
