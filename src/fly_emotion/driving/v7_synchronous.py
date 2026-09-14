from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import V7Contract, build_controlled_stimuli, score_v7_visual_responses
from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    SOURCE_TYPES,
    V7PublishedConductanceProbe,
    _collect_source_normalization,
)
from fly_emotion.driving.v7_stability import _compact_checkpoints, run_constant_background

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_synchronous.py")


class SynchronousConductanceProbe(V7PublishedConductanceProbe):
    def _advance(
        self, state: np.ndarray, drive: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        recurrent = self.adjacency @ (state * self.source_sign)
        updated = ((1.0 - self.leak) * state + self.leak * np.tanh(1.8 * recurrent + drive)).astype(
            np.float32
        )
        voltage = self._conductance_voltage(self._normalized_source_state(state))
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


def evaluate_v7_synchronous_update(root: Path) -> dict:
    contract = V7Contract.load(root)
    config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text())
    stability_path = Path("artifacts/v7-background-stability.json")
    visual_path = Path("artifacts/v7-t4-conductance-candidate.json")
    stability = json.loads((root / stability_path).read_text())
    reference = json.loads((root / visual_path).read_text())
    hashes = dict(stability["protocol"]["dependencies_sha256"])
    for relative, expected in hashes.items():
        with (root / relative).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"stale baseline dependency: {relative}")
    for key, relative in {
        "v7_config_sha256": "configs/driving-v7.yaml",
        "v7_implementation_sha256": "src/fly_emotion/driving/v7.py",
        "branched_implementation_sha256": "src/fly_emotion/driving/v7_branched.py",
        "candidate_config_sha256": str(CONDUCTANCE_CONFIG),
        "candidate_implementation_sha256": "src/fly_emotion/driving/v7_conductance.py",
        "source_audit_sha256": "artifacts/v7-t4-source-audit.json",
    }.items():
        if reference["protocol"][key] != hashes[relative]:
            raise ValueError(f"visual and stability reference differ: {relative}")
    visual = contract.payload["controlled_vision"]
    stimuli = build_controlled_stimuli(
        width=visual["width"], height=visual["height"], frames=visual["frames_per_stimulus"]
    )
    results = {}
    for backend in ("linear_luminance", "signed_frame_difference"):
        normalization = _collect_source_normalization(
            root,
            retinal_backend=backend,
            retinal_geometry=config["primary_retinal_geometry"],
            config=config,
        )
        models = {}
        for cls in (V7PublishedConductanceProbe, SynchronousConductanceProbe):
            probe = cls(root, retinal_backend=backend, normalization=normalization)
            background = run_constant_background(probe)
            responses = {stimulus.name: probe.run(stimulus) for stimulus in stimuli}
            scores = score_v7_visual_responses(responses)
            if cls is V7PublishedConductanceProbe:
                if background != stability["results"][backend]["models"][cls.__name__]:
                    raise ValueError("original timing failed exact background replay")
                original = reference["retinal_results"][backend]
                if (
                    scores != original["scores"]
                    or normalization.sha256 != original["normalization"]["sha256"]
                ):
                    raise ValueError("original timing failed visual or normalization replay")
            t4 = [
                v["contrast"]
                for name, v in scores["direction_selectivity"].items()
                if name.startswith("T4")
            ]
            thresholds = config["gates"]
            summary = scores["summary"]
            gates = {
                "T4_direction": np.median(t4)
                >= thresholds["minimum_T4_cardinal_direction_contrast"],
                "all_T4_positive": np.mean(np.asarray(t4) > 0)
                >= thresholds["minimum_all_T4_populations_positive"],
                "overall_direction": summary["median_cardinal_direction_contrast"]
                >= thresholds["minimum_overall_cardinal_direction_contrast"],
                "on_off": summary["median_on_off_specialization"]
                >= thresholds["minimum_on_off_specialization"],
                "looming": summary["median_known_looming_contrast"]
                >= thresholds["minimum_looming_contrast"],
                "mirror": summary[thresholds["mirror_metric"]]
                <= thresholds["maximum_mirror_response_error"],
            }
            digest = hashlib.sha256()
            for name in SOURCE_TYPES:
                matrix = probe.conductance["matrices"][name]
                for part in (matrix.data, matrix.indices, matrix.indptr):
                    digest.update(part.tobytes())
            models[cls.__name__] = {
                "background": _compact_checkpoints(background),
                "scores": scores,
                "T4_median_direction_contrast": float(np.median(t4)),
                "T4_positive_population_fraction": float(np.mean(np.asarray(t4) > 0)),
                "gates": {name: bool(value) for name, value in gates.items()},
                "all_response_gates_pass": all(gates.values()),
                "target_ids_sha256": hashlib.sha256(
                    probe.graph.body_ids[probe.conductance["targets"]].tobytes()
                ).hexdigest(),
                "source_matrices_sha256": digest.hexdigest(),
                "stimuli": {
                    name: {
                        "sha256": response["stimulus_sha256"],
                        "retinal_drive_sha256": response["retinal_drive_sha256"],
                    }
                    for name, response in responses.items()
                },
            }
        old, new = models.values()
        if any(
            old[key] != new[key]
            for key in ("stimuli", "target_ids_sha256", "source_matrices_sha256")
        ):
            raise ValueError("timing-only comparator changed stimuli, targets or matrices")
        results[backend] = {
            "normalization_sha256": normalization.sha256,
            "original_replay_exact": True,
            "models": models,
        }
    for path in (IMPLEMENTATION, stability_path, visual_path):
        with (root / path).open("rb") as source:
            hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-conductance-synchronous-update-v1",
            "advance_allowed": False,
            "dependencies_sha256": hashes,
            "parameter_fitting": False,
            "driving_data_used": False,
            "change": "T4 conductance reads previous state instead of same-step upstream update",
            "normalization": "shared original uncut calibration per retinal backend",
            "background": "0.5; zero initial state; checkpoints 128/512/2048 microsteps",
            "visual_evidence": "reused 20-condition regression battery; not independent validation",
        },
        "results": results,
        "limitations": [
            "Synchronous scheduling is a numerical convention, not verified biological timing.",
            "One-step delay changes loop dynamics without identifying a specific feedback path.",
            "Lower background residuals do not prove direction selectivity or driving competence.",
            "Weights, normalizers, thresholds, geometry and default policies remain unchanged.",
        ],
        "advance_to_central_complex": False,
    }
