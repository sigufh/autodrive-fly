from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    V7PublishedConductanceProbe,
    _collect_source_normalization,
)
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_stage1_scoring import (
    looming_contrast_summary,
    mirror_error_summary,
    strict_contrast_summary,
)
from fly_emotion.driving.v7_stage1_split import build_stage1_split

CONFIG = Path("configs/driving-v7-stage1-development.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_stage1_development.py")
OPPOSITE_DIRECTION = {"left": "right", "right": "left", "up": "down", "down": "up"}


def _as_visual_stimulus(item) -> VisualStimulus:
    return VisualStimulus(
        name=item.identity,
        family=item.family,
        polarity=item.polarity,
        direction=item.direction,
        frames=item.frames,
        mirror_of=item.mirror_of,
    )


def run_signed_cell_responses(probe, stimulus) -> dict:
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history_length = max(
        1,
        int(probe.source_delays.max()),
        int(probe.correlator["history_substeps"]) if probe.correlator is not None else 0,
    )
    history = [state.copy() for _ in range(history_length)]
    baseline_image = stimulus.frames[0]
    baseline_values = probe._sample_retina(baseline_image)[probe.retinal_permutation]
    baseline_drive = probe._retinal_code(baseline_values, baseline_values, baseline_values)
    for _ in range(probe.baseline_frames):
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = baseline_drive
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = baseline_drive
    baselines = {name: state[nodes].astype(np.float64) for name, nodes in probe.populations.items()}
    peaks = {name: np.full(nodes.size, -np.inf) for name, nodes in probe.populations.items()}
    population_traces = {name: [] for name in probe.populations}
    previous = baseline_values.copy()
    retinal_hash = hashlib.sha256()
    for image in stimulus.frames:
        sampled = probe._sample_retina(image)[probe.retinal_permutation]
        receptor_values = probe._retinal_code(sampled, baseline_values, previous)
        previous = sampled
        retinal_hash.update(receptor_values.tobytes())
        drive = np.zeros_like(state)
        drive[probe.retina.node_indices] = receptor_values
        for _ in range(probe.brain_substeps):
            state = probe._advance(state, drive, history)
            state[probe.retina.node_indices] = receptor_values
        for name, nodes in probe.populations.items():
            delta = state[nodes].astype(np.float64) - baselines[name]
            peaks[name] = np.maximum(peaks[name], delta)
            population_traces[name].append(float(np.mean(delta)))
    return {
        "identity": stimulus.name,
        "retinal_drive_sha256": retinal_hash.hexdigest(),
        "peaks": peaks,
        "population_traces": {
            name: np.asarray(values, dtype=np.float64) for name, values in population_traces.items()
        },
    }


def _lookup(stimuli) -> dict:
    return {
        (
            item.family,
            item.polarity,
            item.direction,
            item.parameter_value,
            item.noise_standard_deviation,
            item.seed,
        ): item
        for item in stimuli
    }


def _median_cell_response(items, responses, population: str) -> np.ndarray:
    return np.median(
        np.stack([responses[item.identity]["peaks"][population] for item in items]), axis=0
    )


def _verify_evidence_dependencies(root: Path, evidence: dict, label: str) -> None:
    for path, expected in evidence.get("protocol", {}).get("dependencies_sha256", {}).items():
        if _sha256(root / path) != expected:
            raise ValueError(f"{label} has stale dependency: {path}")


def score_development(probe, stimuli, responses: dict, scoring: dict) -> dict:
    lookup = _lookup(stimuli)
    thresholds = scoring["thresholds"]
    direction = {}
    polarity = {}
    for population, preferred_direction in scoring["direction_populations"].items():
        expected_polarity = scoring["polarity_populations"][population]
        preferred_items = [
            item
            for item in stimuli
            if item.family == "moving_edge"
            and item.polarity == expected_polarity
            and item.direction == preferred_direction
        ]
        opposite_items = [
            lookup[
                (
                    item.family,
                    item.polarity,
                    OPPOSITE_DIRECTION[item.direction],
                    item.parameter_value,
                    item.noise_standard_deviation,
                    item.seed,
                )
            ]
            for item in preferred_items
        ]
        cell_ids = probe.graph.body_ids[probe.populations[population]]
        direction[population] = strict_contrast_summary(
            cell_ids,
            _median_cell_response(preferred_items, responses, population),
            _median_cell_response(opposite_items, responses, population),
            thresholds,
        )
        opposite_polarity = "off" if expected_polarity == "on" else "on"
        polarity_items = [
            lookup[
                (
                    item.family,
                    opposite_polarity,
                    item.direction,
                    item.parameter_value,
                    item.noise_standard_deviation,
                    item.seed,
                )
            ]
            for item in preferred_items
        ]
        polarity[population] = strict_contrast_summary(
            cell_ids,
            _median_cell_response(preferred_items, responses, population),
            _median_cell_response(polarity_items, responses, population),
            thresholds,
        )
    looming = {}
    for population in scoring["looming_populations"]:
        expansion = [
            item
            for item in stimuli
            if item.family == "looming" and item.polarity == "off" and item.direction == "expansion"
        ]
        receding = [
            lookup[
                (
                    item.family,
                    item.polarity,
                    "contraction",
                    item.parameter_value,
                    item.noise_standard_deviation,
                    item.seed,
                )
            ]
            for item in expansion
        ]
        static = [
            lookup[
                (
                    "static",
                    item.polarity,
                    "none",
                    item.parameter_value,
                    item.noise_standard_deviation,
                    item.seed,
                )
            ]
            for item in expansion
        ]
        cell_ids = probe.graph.body_ids[probe.populations[population]]
        looming[population] = looming_contrast_summary(
            cell_ids,
            _median_cell_response(expansion, responses, population),
            _median_cell_response(receding, responses, population),
            _median_cell_response(static, responses, population),
            thresholds,
        )
    mirrors = {}
    by_identity = {item.identity: item for item in stimuli}
    for item in stimuli:
        if item.identity > item.mirror_of:
            continue
        counterpart = by_identity[item.mirror_of]
        originals, reflected = [], []
        for population in sorted(probe.populations):
            if not population.endswith("_L"):
                continue
            other = population[:-2] + "_R"
            originals.append(responses[item.identity]["population_traces"][population])
            reflected.append(responses[counterpart.identity]["population_traces"][other])
        mirrors[f"{item.identity}<->{counterpart.identity}"] = mirror_error_summary(
            np.stack(originals), np.stack(reflected), thresholds
        )
    population_gates = {
        "all_direction_populations_pass": all(value["passed"] for value in direction.values()),
        "all_polarity_populations_pass": all(value["passed"] for value in polarity.values()),
        "all_looming_populations_pass": all(value["passed"] for value in looming.values()),
        "all_mirror_pairs_pass": all(value["passed"] for value in mirrors.values()),
    }
    return {
        "direction": direction,
        "polarity": polarity,
        "looming": looming,
        "mirror": mirrors,
        "gates": population_gates,
        "passed": all(population_gates.values()),
    }


def evaluate_v7_stage1_development(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    split_config_path = Path(config["stage1_config"])
    input_path = Path(config["input_evidence"])
    scoring_config_path = Path(config["scoring_config"])
    split_config = yaml.safe_load((root / split_config_path).read_text(encoding="utf-8"))
    input_evidence = json.loads((root / input_path).read_text(encoding="utf-8"))
    scoring = yaml.safe_load((root / scoring_config_path).read_text(encoding="utf-8"))
    scoring_evidence = json.loads((root / config["scoring_evidence"]).read_text(encoding="utf-8"))
    split_evidence = json.loads((root / config["stage1_evidence"]).read_text(encoding="utf-8"))
    _verify_evidence_dependencies(root, input_evidence, "input evidence")
    _verify_evidence_dependencies(root, scoring_evidence, "scoring evidence")
    _verify_evidence_dependencies(root, split_evidence, "split evidence")
    if not input_evidence["development_neural_evaluation_allowed"]:
        raise ValueError("development neural evaluation lacks passed input evidence")
    if (
        config["split"] != "development"
        or not config["evaluation_boundary"]["evaluate_development"]
    ):
        raise ValueError("development evaluation contract mismatch")
    if any(
        config["evaluation_boundary"][key]
        for key in ("evaluate_validation", "evaluate_ood", "evaluate_final", "fit_parameters")
    ):
        raise ValueError("development screen must not evaluate later splits or fit parameters")
    if not all(scoring_evidence["synthetic_controls"].values()):
        raise ValueError("strict scoring controls have not passed")
    if scoring_evidence["protocol"]["model_evaluated"]:
        raise ValueError("scoring contract must not contain model evaluation")
    stimuli = build_stage1_split(split_config)["development"]
    candidate_config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text(encoding="utf-8"))
    results = {}
    for backend in config["retinal_backends"]:
        normalization = _collect_source_normalization(
            root,
            retinal_backend=backend,
            retinal_geometry=config["retinal_geometry"],
            config=candidate_config,
        )
        probe = V7PublishedConductanceProbe(
            root, retinal_backend=backend, normalization=normalization
        )
        responses = {}
        manifest = []
        for item in stimuli:
            response = run_signed_cell_responses(probe, _as_visual_stimulus(item))
            responses[item.identity] = response
            manifest.append(
                {
                    "identity": item.identity,
                    "stimulus_sha256": item.sha256,
                    "retinal_drive_sha256": response["retinal_drive_sha256"],
                }
            )
        score = score_development(probe, stimuli, responses, scoring)
        results[backend] = {
            "normalization_sha256": normalization.sha256,
            "normalization_summary": normalization.summary,
            "stimulus_manifest": manifest,
            "strict_scores": score,
            "T4_conductance_target_count": int(len(probe.conductance["targets"])),
            "T4_all_target_count": int(probe.conductance["all_targets"]),
            "T4_conductance_target_fraction": float(
                len(probe.conductance["targets"]) / probe.conductance["all_targets"]
            ),
            "source_edge_counts": {
                name: int(matrix.nnz) for name, matrix in probe.conductance["matrices"].items()
            },
        }
    passing = [name for name, result in results.items() if result["strict_scores"]["passed"]]
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(split_config_path): _sha256(root / split_config_path),
        config["stage1_evidence"]: _sha256(root / config["stage1_evidence"]),
        str(input_path): _sha256(root / input_path),
        str(scoring_config_path): _sha256(root / scoring_config_path),
        config["scoring_evidence"]: _sha256(root / config["scoring_evidence"]),
        str(CONDUCTANCE_CONFIG): _sha256(root / CONDUCTANCE_CONFIG),
        config["candidate_implementation"]: _sha256(root / config["candidate_implementation"]),
        config["source_audit_evidence"]: _sha256(root / config["source_audit_evidence"]),
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "split": "development",
            "dependencies_sha256": dependencies,
            "parameter_fitting": False,
            "validation_evaluated": False,
            "ood_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "candidate_semantics": config["candidate_semantics"],
        "retinal_geometry": config["retinal_geometry"],
        "stimulus_count": len(stimuli),
        "retinal_results": results,
        "passing_retinal_backends": passing,
        "development_response_gates_pass": bool(passing),
        "topology_controls_performed": False,
        "advance_to_validation": False,
        "advance_to_ood": False,
        "advance_to_final": False,
        "advance_to_central_complex": False,
        "stop_reason": (
            "development strict response gates failed"
            if not passing
            else "fresh topology controls required before validation"
        ),
        "limitations": [
            (
                "Only T4 targets with all five declared source types use the "
                "published-parameter conductance proxy; remaining T4 cells stay legacy."
            ),
            "T5 and LPLC/LC outputs remain legacy recurrent states, not paper T5 dynamics.",
            "T5 subtype directions are internal labels, not the unresolved external 0/1 code map.",
            "The engineering split timebase is not applied to candidate dynamics.",
            "No model or retinal backend is selected by this development screen.",
        ],
    }
