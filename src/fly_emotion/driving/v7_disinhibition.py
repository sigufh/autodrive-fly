from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from fly_emotion.driving.v7 import V7Contract, build_controlled_stimuli, score_v7_visual_responses
from fly_emotion.driving.v7_branched import V7BranchedT4Probe
from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    SOURCE_TYPES,
    V7PublishedConductanceProbe,
    _collect_source_normalization,
)

IMPLEMENTATION = Path("src/fly_emotion/driving/v7_disinhibition.py")
DISTAL_BRANCH = "distal_delayed_off_inhibitory"


class MonotoneMi9Probe(V7BranchedT4Probe):
    def __init__(self, root: Path, *, retinal_backend: str):
        super().__init__(
            root,
            retinal_backend=retinal_backend,
            retinal_geometry="nested_t4_axis_v1",
            brain_substeps=4,
            baseline_frames=8,
        )
        self.branched = copy.deepcopy(self.branched)
        self.branched["branches"][DISTAL_BRANCH]["polarity"] = "positive"
        self.dynamics_backend = "t4-monotone-mi9-sign-v1"


class PresynapticThresholdProbe(V7PublishedConductanceProbe):
    def _conductance_voltage(self, source_state: np.ndarray) -> np.ndarray:
        model = self.conductance["model"]
        reversal = model["reversal_potentials_millivolts"]
        numerator = np.full(
            len(self.conductance["targets"]),
            reversal["leak"] * model["leak_conductance"],
            dtype=float,
        )
        denominator = np.full(
            len(self.conductance["targets"]), model["leak_conductance"], dtype=float
        )
        for name, parameters in model["source_parameters"].items():
            released = parameters["gain"] * np.maximum(source_state - parameters["threshold"], 0.0)
            conductance = self.conductance["matrices"][name] @ released
            denominator += conductance
            numerator += reversal[parameters["reversal"]] * conductance
        return numerator / denominator


def conductance_truth_table(model: dict) -> dict:
    voltages = {}
    for cls in (V7PublishedConductanceProbe, PresynapticThresholdProbe):
        probe = object.__new__(cls)
        probe.conductance = {
            "targets": np.array([0]),
            "model": model,
            "matrices": {
                name: sparse.csr_matrix(([1.0], ([0], [index])), shape=(1, 5))
                for index, name in enumerate(SOURCE_TYPES)
            },
        }
        cases = {}
        for name, tonic, excitation in (
            ("rest_inhibited", 1.0, 0.0),
            ("release_only", 0.0, 0.0),
            ("excitation_only", 1.0, 1.0),
            ("release_and_excitation", 0.0, 1.0),
        ):
            state = np.array(
                [
                    tonic if source == "Mi9" else excitation if source in ("Mi1", "Tm3") else 0.0
                    for source in SOURCE_TYPES
                ]
            )
            cases[name] = float(probe._conductance_voltage(state)[0])
        cases["interaction_excess_millivolts"] = (
            cases["release_and_excitation"]
            - cases["release_only"]
            - cases["excitation_only"]
            + cases["rest_inhibited"]
        )
        voltages[cls.__name__] = cases
    return {
        "inputs": "synthetic normalized source values, not measured resting activity",
        "models": voltages,
    }


def _resting_conductance(probe: V7PublishedConductanceProbe, level: float) -> dict:
    state = np.zeros(probe.graph.node_count, dtype=np.float32)
    history = [state.copy()]
    sampled = probe._sample_retina(np.full((24, 48), level, dtype=np.float32))
    drive_values = probe._retinal_code(sampled, sampled, sampled)
    drive = np.zeros_like(state)
    drive[probe.retina.node_indices] = drive_values
    delta = 0.0
    for _ in range(32 * 4):
        previous = state
        state = probe._advance(state, drive, history)
        state[probe.retina.node_indices] = drive_values
        delta = float(np.max(np.abs(state - previous)))
    source_state = probe._normalized_source_state(state)
    parameters = probe.conductance["model"]["source_parameters"]["Mi9"]
    matrix = probe.conductance["matrices"]["Mi9"]
    if isinstance(probe, PresynapticThresholdProbe):
        conductance = matrix @ (
            parameters["gain"] * np.maximum(source_state - parameters["threshold"], 0)
        )
    else:
        conductance = parameters["gain"] * np.maximum(
            matrix @ source_state - parameters["threshold"], 0
        )
    return {
        "background_level": level,
        "last_step_maximum_state_change": delta,
        "Mi9_conductance_median": float(np.median(conductance)),
        "fraction_targets_with_Mi9_conductance": float(np.mean(conductance > 1e-6)),
        "voltage_median_millivolts": float(np.median(probe._conductance_voltage(source_state))),
        "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
    }


def evaluate_v7_conductance_order(root: Path) -> dict:
    contract = V7Contract.load(root)
    config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text())
    visual = contract.payload["controlled_vision"]
    stimuli = build_controlled_stimuli(
        width=visual["width"], height=visual["height"], frames=visual["frames_per_stimulus"]
    )
    reference_path = Path("artifacts/v7-t4-conductance-candidate.json")
    reference = json.loads((root / reference_path).read_text())
    for key, path in {
        "v7_config_sha256": "configs/driving-v7.yaml",
        "v7_implementation_sha256": "src/fly_emotion/driving/v7.py",
        "branched_implementation_sha256": "src/fly_emotion/driving/v7_branched.py",
        "candidate_config_sha256": str(CONDUCTANCE_CONFIG),
        "candidate_implementation_sha256": "src/fly_emotion/driving/v7_conductance.py",
        "source_audit_sha256": "artifacts/v7-t4-source-audit.json",
    }.items():
        with (root / path).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != reference["protocol"][key]:
                raise ValueError(f"stale conductance reference: {path}")
    results = {}
    for backend in ("linear_luminance", "signed_frame_difference"):
        normalization = _collect_source_normalization(
            root,
            retinal_backend=backend,
            retinal_geometry=config["primary_retinal_geometry"],
            config=config,
        )
        models = {}
        for cls in (V7PublishedConductanceProbe, PresynapticThresholdProbe):
            probe = cls(root, retinal_backend=backend, normalization=normalization)
            rest = [_resting_conductance(probe, level) for level in (0.08, 0.5, 0.92)]
            responses = {item.name: probe.run(item) for item in stimuli}
            scores = score_v7_visual_responses(responses)
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
            matrix_digest = hashlib.sha256()
            for name in SOURCE_TYPES:
                matrix = probe.conductance["matrices"][name]
                for part in (matrix.data, matrix.indices, matrix.indptr):
                    matrix_digest.update(part.tobytes())
            models[cls.__name__] = {
                "source_matrices_sha256": matrix_digest.hexdigest(),
                "rest": rest,
                "scores": scores,
                "T4_median_direction_contrast": float(np.median(t4)),
                "T4_positive_population_fraction": float(np.mean(np.asarray(t4) > 0)),
                "gates": {name: bool(value) for name, value in gates.items()},
                "all_response_gates_pass": all(gates.values()),
                "target_ids_sha256": hashlib.sha256(
                    probe.graph.body_ids[probe.conductance["targets"]].tobytes()
                ).hexdigest(),
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
            raise ValueError("order comparator changed inputs, target IDs or source matrices")
        original = reference["retinal_results"][backend]
        if (
            old["scores"] != original["scores"]
            or normalization.sha256 != original["normalization"]["sha256"]
        ):
            raise ValueError("frozen conductance model did not reproduce archived evidence")
        results[backend] = {
            "normalization_sha256": normalization.sha256,
            "models": models,
            "original_scores_reproduced": True,
        }
    paths = [
        IMPLEMENTATION,
        reference_path,
        Path("src/fly_emotion/driving/v7_conductance.py"),
        Path("src/fly_emotion/driving/v7.py"),
        Path("src/fly_emotion/driving/v7_branched.py"),
        Path("src/fly_emotion/driving/v7_retina_audit.py"),
        Path("src/fly_emotion/driving/v7_source_audit.py"),
        Path("src/fly_emotion/driving/engine.py"),
        Path("src/fly_emotion/driving/retina.py"),
        Path("src/fly_emotion/connectome/graph.py"),
        Path("configs/driving-v7.yaml"),
        CONDUCTANCE_CONFIG,
        Path("configs/driving-v7-branched-t4.yaml"),
        Path("configs/driving-v7-t4-source-audit.yaml"),
        Path("artifacts/v7-t4-source-audit.json"),
        Path("data/raw/malecns-v1.0/body-annotations.feather"),
        Path("data/raw/malecns-v1.0/body-neurotransmitters.feather"),
        Path("data/processed/malecns-v1.0/body_ids.npy"),
        Path("data/processed/malecns-v1.0/retina_map.npz"),
        Path("data/processed/malecns-v1.0/adjacency_raw.npz"),
        Path("data/processed/malecns-v1.0/adjacency_target_norm.npz"),
    ]
    hashes = {}
    for path in paths:
        with (root / path).open("rb") as source:
            hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-conductance-order-comparison-v1",
            "advance_allowed": False,
            "dependencies_sha256": hashes,
            "v7_config_sha256": contract.sha256,
            "parameter_fitting": False,
            "published_parameters_modified": False,
            "rest_protocol": "32 frames x 4 substeps; finite adaptation, not proven equilibrium",
            "change": "type threshold after mean versus source threshold before same weighted sum",
            "normalization": "shared label-free calibration within each retinal backend",
            "stimulus_role": "reused regression battery, not independent validation",
        },
        "truth_table": conductance_truth_table(config["published_single_compartment_model"]),
        "results": results,
        "limitations": [
            "Source-wise release is a model hypothesis, not measured transmitter kinetics.",
            "Synthetic tonic input verifies algebra but cannot calibrate real resting activity.",
            "A frame-difference encoder has no absolute constant-luminance input.",
            "Per-source-type normalization still discards relative between-type weight mass.",
            "All-cell gates remain separate from algebraic truth-table checks.",
        ],
        "advance_to_central_complex": False,
    }


def synthetic_mi9_sweep() -> dict:
    signal = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float32)
    centre, proximal = 0.8, 0.1
    legacy = np.tanh(
        np.maximum(centre - proximal - V7BranchedT4Probe._polarized(signal, "negative"), 0)
    )
    revised = np.tanh(
        np.maximum(centre - proximal - V7BranchedT4Probe._polarized(signal, "positive"), 0)
    )
    return {
        "mi9_state": signal.tolist(),
        "centre": centre,
        "proximal": proximal,
        "legacy_target_drive": legacy.tolist(),
        "monotone_target_drive": revised.tolist(),
        "legacy_increases_inhibition_when_mi9_falls_below_zero": bool(legacy[0] < legacy[2]),
        "revised_nonincreasing_in_mi9_activity": bool(np.all(np.diff(revised) <= 0)),
        "revised_has_no_tonic_inhibition_below_zero": bool(revised[0] == revised[2]),
    }


def evaluate_v7_disinhibition(root: Path) -> dict:
    contract = V7Contract.load(root)
    visual = contract.payload["controlled_vision"]
    stimuli = build_controlled_stimuli(
        width=visual["width"], height=visual["height"], frames=visual["frames_per_stimulus"]
    )
    reference_path = root / "artifacts/v7-branched-t4-candidate.json"
    reference = json.loads(reference_path.read_text())
    for key, path in {
        "v7_config_sha256": "configs/driving-v7.yaml",
        "v7_implementation_sha256": "src/fly_emotion/driving/v7.py",
        "candidate_config_sha256": "configs/driving-v7-branched-t4.yaml",
        "candidate_implementation_sha256": "src/fly_emotion/driving/v7_branched.py",
        "source_audit_sha256": "artifacts/v7-t4-source-audit.json",
    }.items():
        with (root / path).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != reference["protocol"][key]:
                raise ValueError(f"stale frozen reference: {path}")
    results = {}
    for backend in ("linear_luminance", "signed_frame_difference"):
        probe = MonotoneMi9Probe(root, retinal_backend=backend)
        responses = {item.name: probe.run(item) for item in stimuli}
        scores = score_v7_visual_responses(responses)
        original = reference["retinal_results"][f"nested_t4_axis_v1:{backend}"]
        same_inputs = all(
            response[key] == original["responses"][name][key]
            for name, response in responses.items()
            for key in ("stimulus_sha256", "retinal_drive_sha256")
        )
        if not same_inputs:
            raise ValueError("sign-only comparison changed stimulus or retinal drive")
        t4 = [
            value["contrast"]
            for name, value in scores["direction_selectivity"].items()
            if name.startswith("T4")
        ]
        thresholds = probe.branched_config["gates"]
        summary = scores["summary"]
        gates = {
            "T4_cardinal_direction_contrast": float(np.median(t4))
            >= thresholds["minimum_T4_cardinal_direction_contrast"],
            "all_T4_populations_positive": float(np.mean(np.asarray(t4) > 0))
            >= thresholds["minimum_all_T4_populations_positive"],
            "overall_cardinal_direction_contrast": summary["median_cardinal_direction_contrast"]
            >= thresholds["minimum_overall_cardinal_direction_contrast"],
            "on_off_specialization": summary["median_on_off_specialization"]
            >= thresholds["minimum_on_off_specialization"],
            "looming_contrast": summary["median_known_looming_contrast"]
            >= thresholds["minimum_looming_contrast"],
            "mirror_response_error": summary[thresholds["mirror_metric"]]
            <= thresholds["maximum_mirror_response_error"],
        }
        results[backend] = {
            "same_inputs_as_frozen_reference": same_inputs,
            "T4_summary": {
                "median_cardinal_direction_contrast": float(np.median(t4)),
                "fraction_populations_positive": float(np.mean(np.asarray(t4) > 0)),
            },
            "scores": scores,
            "gates": gates,
            "controlled_response_gates_pass": all(gates.values()),
            "original_T4_summary": original["T4_summary"],
            "original_scores": original["scores"]["summary"],
            "original_gates": original["gates"],
            "branched_targets": len(probe.branched["targets"]),
            "branch_edges": {
                name: matrix.nnz for name, matrix in probe.branched["matrices"].items()
            },
            "stimuli": {
                name: {
                    "sha256": response["stimulus_sha256"],
                    "retinal_drive_sha256": response["retinal_drive_sha256"],
                    "response_sha256": hashlib.sha256(
                        b"".join(
                            np.asarray(
                                response["population_response_trace"][pop], dtype=np.float32
                            ).tobytes()
                            for pop in sorted(response["population_response_trace"])
                        )
                    ).hexdigest(),
                }
                for name, response in responses.items()
            },
        }
    paths = [
        IMPLEMENTATION,
        Path("src/fly_emotion/driving/v7.py"),
        Path("src/fly_emotion/driving/v7_branched.py"),
        Path("src/fly_emotion/driving/v7_retina_audit.py"),
        Path("src/fly_emotion/driving/v7_source_audit.py"),
        Path("src/fly_emotion/driving/retina.py"),
        Path("src/fly_emotion/connectome/graph.py"),
        Path("src/fly_emotion/driving/engine.py"),
        Path("configs/driving-v7.yaml"),
        Path("configs/driving-v7-branched-t4.yaml"),
        Path("configs/driving-v7-t4-source-audit.yaml"),
        Path("artifacts/v7-t4-source-audit.json"),
        Path("artifacts/v7-branched-t4-candidate.json"),
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
            "name": "v7-mi9-sign-only-comparison-v1",
            "deployment_enabled": False,
            "parameter_fitting": False,
            "driving_data_used": False,
            "dependencies_sha256": hashes,
            "change": "distal branch max(-state,0) -> max(state,0); subtraction unchanged",
            "frozen": "edges, target IDs, other branches, gains, delays, geometry and gates",
            "stimuli": "existing 20-condition regression battery, not independent validation",
            "reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8891015/",
        },
        "synthetic_sign_check": synthetic_mi9_sweep(),
        "results": results,
        "limitations": [
            "Proxy zero is not a measured resting potential; the revision lacks tonic inhibition.",
            "Fixing monotonicity does not validate a full biological disinhibition mechanism.",
            "Recurrent upstream responses can change indirectly when T4 output changes.",
            "This legacy branch error does not alone explain prior conductance-model failures.",
            "No covered-only filtering, driving assessment or stage advancement is performed.",
        ],
        "advance_to_central_complex": False,
    }
