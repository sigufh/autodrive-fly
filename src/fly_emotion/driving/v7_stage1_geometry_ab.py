from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_conductance import (
    CONDUCTANCE_CONFIG,
    _collect_source_normalization,
)
from fly_emotion.driving.v7_geometry_sign import MotionSignConductanceProbe, _sha256
from fly_emotion.driving.v7_stage1_development import (
    _as_visual_stimulus,
    run_signed_cell_responses,
    score_development,
)
from fly_emotion.driving.v7_stage1_split import build_stage1_split

CONFIG = Path("configs/driving-v7-stage1-geometry-ab.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_stage1_geometry_ab.py")


def _verify_dependencies(root: Path, evidence: dict, label: str) -> None:
    for path, digest in evidence.get("protocol", {}).get("dependencies_sha256", {}).items():
        if _sha256(root / path) != digest:
            raise ValueError(f"{label} has stale dependency: {path}")


def evaluate_v7_stage1_geometry_ab(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    development_config_path = Path(config["development_config"])
    development_config = yaml.safe_load(
        (root / development_config_path).read_text(encoding="utf-8")
    )
    development_path = Path(config["development_evidence"])
    development = json.loads((root / development_path).read_text(encoding="utf-8"))
    geometry_path = Path(config["geometry_evidence"])
    geometry = json.loads((root / geometry_path).read_text(encoding="utf-8"))
    input_path = Path(config["input_evidence"])
    input_evidence = json.loads((root / input_path).read_text(encoding="utf-8"))
    scoring_evidence_path = Path(config["scoring_evidence"])
    scoring_evidence = json.loads((root / scoring_evidence_path).read_text(encoding="utf-8"))
    split_path = Path(development_config["stage1_config"])
    split_config = yaml.safe_load((root / split_path).read_text(encoding="utf-8"))
    scoring_path = Path(development_config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    for label, evidence in (
        ("development", development),
        ("geometry", geometry),
        ("input", input_evidence),
        ("scoring", scoring_evidence),
    ):
        _verify_dependencies(root, evidence, label)
    reversed_input = input_evidence["retinal_maps"]["reversed_nested_t4_axis_3344"]
    if not all(item["passed"] for item in reversed_input["encodings"].values()):
        raise ValueError("reversed nested geometry failed development input gates")
    if not all(scoring_evidence["synthetic_controls"].values()):
        raise ValueError("strict scoring controls have not passed")
    if development["development_response_gates_pass"]:
        raise ValueError("geometry A/B is only defined after failed reference development")
    boundary = config["boundary"]
    if not boundary["anatomy_supported_before_this_development_screen"]:
        raise ValueError("geometry intervention lacks prior anatomical support")
    if boundary["physiological_labels_changed"] or boundary["model_parameters_changed"]:
        raise ValueError("geometry A/B must not alter labels or model parameters")
    stimuli = build_stage1_split(split_config)["development"]
    candidate_config = yaml.safe_load((root / CONDUCTANCE_CONFIG).read_text(encoding="utf-8"))
    results = {}
    for backend in config["retinal_backends"]:
        normalization = _collect_source_normalization(
            root,
            retinal_backend=backend,
            retinal_geometry=development_config["retinal_geometry"],
            config=candidate_config,
        )
        probe = MotionSignConductanceProbe(
            root, retinal_backend=backend, normalization=normalization
        )
        responses = {}
        for item in stimuli:
            responses[item.identity] = run_signed_cell_responses(probe, _as_visual_stimulus(item))
        scores = score_development(probe, stimuli, responses, scoring)
        reference = development["retinal_results"][backend]["strict_scores"]
        results[backend] = {
            "normalization_sha256": normalization.sha256,
            "strict_scores": scores,
            "reference_pass_counts": {
                kind: sum(item["passed"] for item in reference[kind].values())
                for kind in ("direction", "polarity", "looming", "mirror")
            },
            "reversed_pass_counts": {
                kind: sum(item["passed"] for item in scores[kind].values())
                for kind in ("direction", "polarity", "looming", "mirror")
            },
            "T4_direction_median_changes": {
                population: (
                    scores["direction"][population]["median_signed_contrast"]
                    - reference["direction"][population]["median_signed_contrast"]
                )
                for population in scores["direction"]
                if population.startswith("T4")
            },
        }
    passing = [name for name, result in results.items() if result["strict_scores"]["passed"]]
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(development_config_path): _sha256(root / development_config_path),
        str(development_path): _sha256(root / development_path),
        str(geometry_path): _sha256(root / geometry_path),
        str(split_path): _sha256(root / split_path),
        development_config["stage1_evidence"]: _sha256(
            root / development_config["stage1_evidence"]
        ),
        str(scoring_path): _sha256(root / scoring_path),
        development_config["scoring_evidence"]: _sha256(
            root / development_config["scoring_evidence"]
        ),
        str(input_path): _sha256(root / input_path),
        str(scoring_evidence_path): _sha256(root / scoring_evidence_path),
        str(CONDUCTANCE_CONFIG): _sha256(root / CONDUCTANCE_CONFIG),
        development_config["candidate_implementation"]: _sha256(
            root / development_config["candidate_implementation"]
        ),
        development_config["source_audit_evidence"]: _sha256(
            root / development_config["source_audit_evidence"]
        ),
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "split": "development",
            "parameter_fitting": False,
            "physiological_labels_changed": False,
            "validation_evaluated": False,
            "ood_evaluated": False,
            "final_evaluated": False,
            "runtime_modified": False,
        },
        "geometry_intervention": config["candidate_geometry"],
        "candidate_semantics": development_config["candidate_semantics"],
        "stimulus_count": len(stimuli),
        "results": results,
        "passing_retinal_backends": passing,
        "development_response_gates_pass": bool(passing),
        "advance_to_validation": False,
        "advance_to_ood": False,
        "advance_to_final": False,
        "advance_to_central_complex": False,
        "limitations": [
            "The geometry sign was selected from prior anatomy and phase-motion diagnostics.",
            "This is a development A/B comparison, not independent validation.",
            "T5 and LPLC/LC remain legacy recurrent states, not published T5 dynamics.",
            "No model parameter, physiological label, runtime or downstream stage changed.",
        ],
    }
