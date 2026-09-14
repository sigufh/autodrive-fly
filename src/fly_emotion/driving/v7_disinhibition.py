from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7 import V7Contract, build_controlled_stimuli, score_v7_visual_responses
from fly_emotion.driving.v7_branched import V7BranchedT4Probe

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
