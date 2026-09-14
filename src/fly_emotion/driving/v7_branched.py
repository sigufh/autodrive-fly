"""Frozen three-branch T4 candidate following the retrospective source audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import yaml
from scipy import sparse

from fly_emotion.driving.v7 import (
    V7_IMPLEMENTATION,
    V7Contract,
    V7VisualProbe,
    _public_visual_responses,
    build_controlled_stimuli,
    score_v7_visual_responses,
)
from fly_emotion.driving.v7_retina_audit import build_balanced_retina_control
from fly_emotion.driving.v7_source_audit import AUDIT_CONFIG, AUDIT_IMPLEMENTATION

BRANCHED_CONFIG = Path("configs/driving-v7-branched-t4.yaml")
BRANCHED_IMPLEMENTATION = Path("src/fly_emotion/driving/v7_branched.py")
SOURCE_AUDIT = Path("artifacts/v7-t4-source-audit.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V7BranchedT4Probe(V7VisualProbe):
    """T4 probe with separate centre, proximal-ON and distal-OFF branches."""

    def __init__(
        self,
        root: Path,
        *,
        retinal_backend: str,
        retinal_geometry: str = "legacy_proxy_v2",
        brain_substeps: int,
        baseline_frames: int,
    ):
        self.branched_config = yaml.safe_load((root / BRANCHED_CONFIG).read_text(encoding="utf-8"))
        if self.branched_config.get("deployment_enabled"):
            raise ValueError("branched T4 candidate must not be deployed")
        if self.branched_config.get("parameter_scan_allowed"):
            raise ValueError("branched T4 parameters must remain frozen")
        if retinal_geometry not in self.branched_config["retinal_geometries"]:
            raise ValueError(f"unknown branched T4 geometry: {retinal_geometry}")
        super().__init__(
            root,
            brain_substeps=brain_substeps,
            baseline_frames=baseline_frames,
            retinal_backend=retinal_backend,
            retinal_geometry="legacy_proxy_v2",
            dynamics_backend="columnar_correlator_v1",
            control="real_malecns",
        )
        if retinal_geometry in {
            "nested_t4_axis_v1",
            "full_retina_eye_mass_normalized_v1",
            "balanced_count_nested_axis_v1",
        }:
            self.retinal_u, self.retinal_v = self._nested_t4_retinal_coordinates()
        self._retinal_drive_gain = np.ones(self.retina.size, dtype=np.float32)
        if retinal_geometry == "full_retina_eye_mass_normalized_v1":
            gains = self.branched_config["eye_mass_normalization"]
            self._retinal_drive_gain[self.retina.side < 0] = float(gains["left_gain"])
            self._retinal_drive_gain[self.retina.side > 0] = float(gains["right_gain"])
        if retinal_geometry in {
            "balanced_count_nested_axis_v1",
            "balanced_exact_mirror_v1",
        }:
            balanced = build_balanced_retina_control(root)
            self._retinal_drive_gain = self._retinal_drive_gain[balanced.source_indices]
            if retinal_geometry == "balanced_count_nested_axis_v1":
                self.retinal_u = self.retinal_u[balanced.source_indices]
                self.retinal_v = self.retinal_v[balanced.source_indices]
                self.retina = type(self.retina)(
                    node_indices=self.retina.node_indices[balanced.source_indices],
                    body_ids=self.retina.body_ids[balanced.source_indices],
                    u=self.retinal_u,
                    v=self.retinal_v,
                    side=self.retina.side[balanced.source_indices],
                    mapping_version=3,
                )
            else:
                self.retina = balanced.retina
                self.retinal_u = balanced.retina.u
                self.retinal_v = balanced.retina.v
            self.retinal_permutation = np.arange(self.retina.size, dtype=np.int32)
        self.retinal_geometry = retinal_geometry
        all_t4 = np.flatnonzero(np.isin(self.node_types, ("T4a", "T4b", "T4c", "T4d"))).astype(
            np.int32
        )
        branches = self.branched_config["branches"]
        matrices = {
            name: self._normalized_target_inputs(all_t4, tuple(branch["source_types"]))
            for name, branch in branches.items()
        }
        valid = np.ones(len(all_t4), dtype=bool)
        for matrix in matrices.values():
            valid &= np.diff(matrix.indptr) > 0
        self.branched = {
            "targets": all_t4[valid],
            "matrices": {name: matrix[valid] for name, matrix in matrices.items()},
            "branches": branches,
            "history_substeps": int(self.branched_config["history_substeps"]),
            "gain": float(self.branched_config["gain"]),
            "all_targets": int(len(all_t4)),
        }
        self._branch_population_masks = {
            name: np.isin(self.branched["targets"], nodes)
            for name, nodes in self.populations.items()
            if name.startswith("T4")
        }
        self._branch_step_trace: dict[str, dict[str, list[float]]] | None = None
        # Base run() owns history allocation and audit fields. This descriptor exposes only
        # the keys it needs; _advance below never invokes the superseded v1 correlator.
        self.correlator = {
            "targets": self.branched["targets"],
            "history_substeps": self.branched["history_substeps"],
        }
        self.dynamics_backend = self.branched_config["name"]

    def _retinal_code(
        self, values: np.ndarray, baseline: np.ndarray, previous: np.ndarray
    ) -> np.ndarray:
        encoded = super()._retinal_code(values, baseline, previous)
        return encoded * self._retinal_drive_gain

    def _nested_t4_retinal_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        audit_path = self.root / SOURCE_AUDIT
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        protocol = audit.get("protocol", {})
        if protocol.get("config_sha256") != _sha256(self.root / AUDIT_CONFIG):
            raise ValueError("nested T4 geometry source-audit config mismatch")
        if protocol.get("implementation_sha256") != _sha256(self.root / AUDIT_IMPLEMENTATION):
            raise ValueError("nested T4 geometry source-audit implementation mismatch")
        nested = audit.get("nested_branch_validation", {})
        if not nested.get("nested_validation_pass"):
            raise ValueError("nested T4 geometry requires passed anatomy cross-validation")
        transforms = audit["frozen_axis_calibration"]["transforms_by_eye"]

        annotations_path = self.root / "data/raw/malecns-v1.0/body-annotations.feather"
        processed = self.root / "data/processed/malecns-v1.0"
        annotations = feather.read_table(
            annotations_path,
            columns=["bodyId", "assignedOlHex1", "assignedOlHex2"],
            memory_map=True,
        ).to_pandas()
        known = annotations.dropna(subset=["assignedOlHex1", "assignedOlHex2"])
        body_ids = known["bodyId"].to_numpy(dtype=np.int64)
        nodes = np.searchsorted(self.graph.body_ids, body_ids)
        valid = nodes < self.graph.node_count
        valid[valid] &= self.graph.body_ids[nodes[valid]] == body_ids[valid]
        rows = known.iloc[np.flatnonzero(valid)]
        coordinates = np.full((self.graph.node_count, 2), np.nan, dtype=np.float64)
        coordinates[nodes[valid], 0] = rows["assignedOlHex1"].to_numpy(dtype=float)
        coordinates[nodes[valid], 1] = rows["assignedOlHex2"].to_numpy(dtype=float)

        raw = sparse.load_npz(processed / "adjacency_raw.npz")[:, self.retina.node_indices].tocsc()
        inferred = np.full((self.retina.size, 2), np.nan, dtype=np.float64)
        for column in range(self.retina.size):
            start, end = raw.indptr[column : column + 2]
            targets = raw.indices[start:end]
            weights = np.abs(raw.data[start:end]).astype(np.float64)
            located = np.all(np.isfinite(coordinates[targets]), axis=1)
            if np.any(located):
                inferred[column] = np.average(
                    coordinates[targets[located]], axis=0, weights=weights[located]
                )
        if not np.all(np.isfinite(inferred)):
            raise ValueError("nested T4 geometry cannot locate every retained receptor")

        projected = np.zeros_like(inferred)
        for eye, side in (("L", -1), ("R", 1)):
            mask = self.retina.side == side
            projected[mask] = inferred[mask] @ np.asarray(transforms[eye], dtype=np.float64)
        u = np.zeros(self.retina.size, dtype=np.float32)
        v = np.zeros(self.retina.size, dtype=np.float32)
        for side in (-1, 1):
            mask = self.retina.side == side
            horizontal = self._normalized(projected[mask, 0])
            u[mask] = 0.5 * horizontal if side < 0 else 0.5 + 0.5 * horizontal
            v[mask] = self._normalized(projected[mask, 1])
        return u, v

    @staticmethod
    def _polarized(values: np.ndarray, polarity: str) -> np.ndarray:
        if polarity == "positive":
            return np.maximum(values, 0.0)
        if polarity == "negative":
            return np.maximum(-values, 0.0)
        raise ValueError(f"unknown branch polarity: {polarity}")

    def _branch_activity(
        self, branch_name: str, state: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        branch = self.branched["branches"][branch_name]
        delay = int(branch["delay_substeps"])
        source_state = state if delay == 0 else history[min(delay - 1, len(history) - 1)]
        values = self.branched["matrices"][branch_name] @ source_state
        return self._polarized(values, str(branch["polarity"]))

    def _advance(
        self, state: np.ndarray, drive: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        recurrent = self.adjacency @ (state * self.source_sign)
        updated = ((1.0 - self.leak) * state + self.leak * np.tanh(1.8 * recurrent + drive)).astype(
            np.float32
        )
        branch_values = {
            name: self._branch_activity(name, state, history) for name in self.branched["branches"]
        }
        if self._branch_step_trace is not None:
            for branch_name, values in branch_values.items():
                for population, mask in self._branch_population_masks.items():
                    self._branch_step_trace[branch_name][population].append(
                        float(np.mean(values[mask]))
                    )
        net = np.zeros(len(self.branched["targets"]), dtype=np.float32)
        for name, values in branch_values.items():
            operation = self.branched["branches"][name]["operation"]
            if operation == "add":
                net += values
            elif operation == "subtract":
                net -= values
            else:
                raise ValueError(f"unknown branch operation: {operation}")
        target_drive = np.tanh(self.branched["gain"] * np.maximum(net, 0.0)).astype(np.float32)
        targets = self.branched["targets"]
        updated[targets] = (1.0 - self.leak[targets]) * state[targets] + self.leak[
            targets
        ] * target_drive
        history.insert(0, state.copy())
        del history[self.branched["history_substeps"] :]
        return updated

    def run(self, stimulus) -> dict:
        self._branch_step_trace = {
            branch_name: {population: [] for population in self._branch_population_masks}
            for branch_name in self.branched["branches"]
        }
        response = super().run(stimulus)
        baseline_steps = self.baseline_frames * self.brain_substeps
        response["branch_population_mean"] = {
            branch_name: {
                population: float(np.mean(values[baseline_steps:]))
                for population, values in populations.items()
            }
            for branch_name, populations in self._branch_step_trace.items()
        }
        self._branch_step_trace = None
        return response


def evaluate_v7_branched_t4_candidate(root: Path) -> dict:
    contract = V7Contract.load(root)
    config_path = root / BRANCHED_CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_audit_path = root / SOURCE_AUDIT
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("advance_to_columnar_branched_correlator_v2"):
        raise ValueError("source audit cannot itself authorize the branched candidate")
    if not source_audit.get("protocol", {}).get("exploratory"):
        raise ValueError("branched candidate requires the declared exploratory source audit")

    visual = contract.payload["controlled_vision"]
    stimuli = build_controlled_stimuli(
        width=visual["width"],
        height=visual["height"],
        frames=visual["frames_per_stimulus"],
    )
    results = {}
    for retinal_geometry in config["retinal_geometries"]:
        retinal_backends = (
            config["balanced_ablation_retinal_backends"]
            if retinal_geometry.startswith("balanced_")
            or retinal_geometry == "full_retina_eye_mass_normalized_v1"
            else config["retinal_backends"]
        )
        for retinal_backend in retinal_backends:
            result_name = f"{retinal_geometry}:{retinal_backend}"
            probe = V7BranchedT4Probe(
                root,
                retinal_backend=retinal_backend,
                retinal_geometry=retinal_geometry,
                brain_substeps=int(config["brain_substeps_per_frame"]),
                baseline_frames=int(config["baseline_frames"]),
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
            results[result_name] = {
                "retinal_geometry": retinal_geometry,
                "retinal_backend": retinal_backend,
                "T4_summary": {
                    "median_cardinal_direction_contrast": t4_median,
                    "fraction_populations_positive": t4_positive,
                },
                "scores": scores,
                "gates": gates,
                "controlled_response_gates_pass": all(gates.values()),
                "responses": _public_visual_responses(responses),
                "branched_targets": int(len(probe.branched["targets"])),
                "all_T4_targets": probe.branched["all_targets"],
                "branch_edges": {
                    name: int(matrix.nnz) for name, matrix in probe.branched["matrices"].items()
                },
            }
    passing = [
        backend for backend, result in results.items() if result["controlled_response_gates_pass"]
    ]
    return {
        "protocol": {
            "version": 7,
            "name": config["name"],
            "deployment_enabled": False,
            "parameter_scan_allowed": False,
            "v7_config_sha256": contract.sha256,
            "v7_implementation_sha256": _sha256(root / V7_IMPLEMENTATION),
            "candidate_config_sha256": _sha256(config_path),
            "candidate_implementation_sha256": _sha256(root / BRANCHED_IMPLEMENTATION),
            "source_audit_sha256": _sha256(source_audit_path),
            "driving_data_used": False,
            "stimulus_direction_labels_read_by_dynamics": False,
            "direct_input_type": "R1-R6",
            "parameter_source": "frozen literature roles plus retrospective anatomy audit",
        },
        "branches": config["branches"],
        "gain": config["gain"],
        "retinal_results": results,
        "passing_retinal_backends": passing,
        "controlled_response_gates_pass": bool(passing),
        "topology_controls_complete": False,
        "advance_to_topology_controls": bool(passing),
        "advance_to_central_complex": False,
        "stop_reason": (
            "branched T4 response gates failed"
            if not passing
            else "fresh topology controls required before any stage advance"
        ),
    }
