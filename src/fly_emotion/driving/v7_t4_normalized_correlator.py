from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_stage1_nested import CONFIG as NESTED_CONFIG
from fly_emotion.driving.v7_stage1_nested import build_condition_bundle
from fly_emotion.driving.v7_t4_source_resolved import (
    _frozen_axis_coordinates,
    _score_variant,
)

CONFIG = Path("configs/driving-v7-t4-normalized-correlator.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_normalized_correlator.py")


class NormalizedCorrelatorT4Probe(MotionSignConductanceProbe):
    def __init__(self, root: Path, normalization, gain: float, mode: str, source_audit: dict):
        super().__init__(root, retinal_backend="linear_luminance", normalization=normalization)
        if mode not in {"multiplicative", "additive"}:
            raise ValueError(f"unknown normalized correlator mode: {mode}")
        self.direction_gain = float(gain)
        self.direction_mode = mode
        coordinates = _frozen_axis_coordinates(
            root, self, source_audit["frozen_axis_calibration"]["transforms_by_eye"]
        )
        orientations = []
        for target in self.conductance["targets"]:
            row = self.adjacency.getrow(int(target))
            sources = row.indices
            weights = np.abs(row.data).astype(np.float64)
            horizontal = coordinates[sources, 0]

            def centroid(
                source_types: tuple[str, ...],
                source_nodes: np.ndarray = sources,
                positions: np.ndarray = horizontal,
                source_weights: np.ndarray = weights,
            ) -> float:
                mask = np.isin(self.node_types[source_nodes], source_types) & np.isfinite(
                    positions
                )
                return (
                    float(np.average(positions[mask], weights=source_weights[mask]))
                    if np.any(mask)
                    else np.nan
                )

            orientations.append(np.sign(centroid(("Mi4", "C3")) - centroid(("Mi1",))))
        self.target_orientation = np.nan_to_num(
            np.asarray(orientations), nan=0.0
        ).astype(np.float32)
        self.center_matrix = (
            self.conductance["matrices"]["Mi1"]
            + self.conductance["matrices"]["Tm3"]
        ).tocsr()
        self.proximal_matrix = (
            self.conductance["matrices"]["Mi4"]
            + self.conductance["matrices"]["C3"]
        ).tocsr()

    def _advance(
        self, state: np.ndarray, drive: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        recurrent = self.adjacency @ (state * self.source_sign)
        updated = (
            (1.0 - self.leak) * state + self.leak * np.tanh(1.8 * recurrent + drive)
        ).astype(np.float32)
        current = self._normalized_source_state(updated)
        previous = self._normalized_source_state(history[0])
        voltage = self._conductance_voltage(current)
        output = self.conductance["model"]["output_normalization"]
        base = np.clip(
            (voltage - float(output["baseline_millivolts"]))
            / (
                float(output["excitatory_reversal_millivolts"])
                - float(output["baseline_millivolts"])
            ),
            -1.0,
            1.0,
        )
        forward = (self.proximal_matrix @ current) * (self.center_matrix @ previous)
        reverse = (self.center_matrix @ current) * (self.proximal_matrix @ previous)
        correlation = self.target_orientation * (forward - reverse) / (
            np.abs(forward) + np.abs(reverse) + 1e-6
        )
        if self.direction_mode == "multiplicative":
            target_drive = base * (1.0 + self.direction_gain * correlation)
        else:
            target_drive = base + self.direction_gain * correlation
        target_drive = np.clip(target_drive, -1.0, 1.0).astype(np.float32)
        targets = self.conductance["targets"]
        updated[targets] = (
            (1.0 - self.leak[targets]) * state[targets]
            + self.leak[targets] * target_drive
        )
        history.insert(0, state.copy())
        del history[1:]
        return updated


def _pass_count(result: dict) -> int:
    return sum(
        score["passed"]
        for condition in result["per_condition_scores"].values()
        for head in condition.values()
        for score in head.values()
    )


def evaluate_v7_t4_normalized_correlator(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested = yaml.safe_load((root / NESTED_CONFIG).read_text())
    scoring_path = Path(config["scoring_config"])
    strict_scoring = yaml.safe_load((root / scoring_path).read_text())
    conductance_path = Path(config["conductance_config"])
    conductance = yaml.safe_load((root / conductance_path).read_text())
    source_path = Path(config["source_audit_evidence"])
    source_audit = json.loads((root / source_path).read_text())
    condition_ids = list(config["condition_ids"])
    conditions = {item["condition_id"]: item for item in nested["conditions"]}
    if any(conditions[name]["role"] != "tuning" for name in condition_ids):
        raise ValueError("normalized T4 correlator may consume tuning conditions only")
    condition_stimuli = {
        name: [
            item
            for item in build_condition_bundle(conditions[name])
            if item.family == "moving_edge" and item.direction in {"left", "right"}
        ]
        for name in condition_ids
    }
    normalization = _collect_source_normalization(
        root,
        retinal_backend=config["retinal_backend"],
        retinal_geometry=conductance["primary_retinal_geometry"],
        config=conductance,
    )
    scoring = {
        "target_populations": ["T4a_L", "T4a_R", "T4b_L", "T4b_R"],
        "direction_populations": strict_scoring["direction_populations"],
        "thresholds": strict_scoring["thresholds"],
        "gates": {
            "minimum_joint_valid_fraction": 0.80,
            "minimum_all_condition_success_fraction": 0.60,
        },
    }
    candidates = []
    for mode in config["correlator"]["modes"]:
        for gain in config["correlator"]["gains"]:
            probe = NormalizedCorrelatorT4Probe(
                root, normalization, float(gain), str(mode), source_audit
            )
            result = _score_variant(probe, condition_stimuli, scoring)
            candidates.append(
                {
                    "mode": mode,
                    "gain": float(gain),
                    "passed_gate_count": _pass_count(result),
                    **result,
                }
            )
    selected = min(
        candidates,
        key=lambda item: (-item["passed_gate_count"], item["gain"], item["mode"]),
    )
    passed = selected["strict_t4ab_tuning_passed"] and selected["passed_gate_count"] == 24
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(NESTED_CONFIG): _sha256(root / NESTED_CONFIG),
                str(scoring_path): _sha256(root / scoring_path),
                str(conductance_path): _sha256(root / conductance_path),
                str(source_path): _sha256(root / source_path),
            },
            "condition_ids": condition_ids,
            "candidate_count": len(candidates),
            "global_gain_shared_by_all_T4_targets": True,
            "subtype_labels_used_by_dynamics": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "external_final_evaluated": False,
            "runtime_modified": False,
        },
        "candidates": candidates,
        "selected_candidate": {
            key: selected[key] for key in ("mode", "gain", "passed_gate_count")
        },
        "selected_result": selected,
        "tuning_passed": bool(passed),
        "advance_to_calibration": bool(passed),
        "advance_to_navigation_release": False,
        "boundary": config["boundary"],
    }
