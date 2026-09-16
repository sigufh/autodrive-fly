from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe
from fly_emotion.driving.v7_stage1_development import run_signed_cell_responses
from fly_emotion.driving.v7_stage1_nested import CONFIG as NESTED_CONFIG
from fly_emotion.driving.v7_stage1_scoring import strict_contrast_summary

CONFIG = Path("configs/driving-v7-lplc2-phenotype.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_lplc2_phenotype.py")
WIDTH = 48
HEIGHT = 24


def _local_radial_ring(
    duration: int,
    terminal_radius: float,
    half_width: float,
    noise: float,
    seed: int,
    *,
    outward: bool,
) -> np.ndarray:
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    distance = np.sqrt((xx - (WIDTH - 1) / 2) ** 2 + (yy - (HEIGHT - 1) / 2) ** 2)
    radii = np.linspace(1.0, terminal_radius, duration)
    if not outward:
        radii = radii[::-1]
    frames = np.full((duration, HEIGHT, WIDTH), 0.92, dtype=np.float32)
    for index, radius in enumerate(radii):
        frames[index, np.abs(distance - radius) <= half_width] = 0.08
    rng = np.random.default_rng(seed)
    perturbation = rng.normal(0.0, noise, frames.shape)
    perturbation = (perturbation + perturbation[:, :, ::-1]) / 2.0
    return np.clip(frames + perturbation, 0.0, 1.0).astype(np.float32)


def _target_joint_coverage(condition_scores: list[dict], body_ids: np.ndarray) -> dict:
    contrasts = np.asarray(
        [
            [value if value is not None else np.nan for value in score["contrast"]]
            for score in condition_scores
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(contrasts)
    success = valid & (contrasts >= 0.10)
    return {
        "target_count": int(len(body_ids)),
        "body_ids_sha256": hashlib.sha256(body_ids.tobytes()).hexdigest(),
        "joint_valid_count": int(np.count_nonzero(np.all(valid, axis=0))),
        "joint_valid_fraction": float(np.mean(np.all(valid, axis=0))),
        "all_condition_success_count": int(np.count_nonzero(np.all(success, axis=0))),
        "all_condition_success_fraction": float(np.mean(np.all(success, axis=0))),
    }


def evaluate_v7_lplc2_phenotype(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    nested = yaml.safe_load((root / NESTED_CONFIG).read_text())
    scoring_path = Path(config["scoring_config"])
    scoring = yaml.safe_load((root / scoring_path).read_text())
    conditions = {item["condition_id"]: item for item in nested["conditions"]}
    condition_ids = list(config["condition_ids"])
    if any(conditions[name]["role"] != "tuning" for name in condition_ids):
        raise ValueError("LPLC2 phenotype may consume tuning conditions only")
    probe = MassBalancedVisualProbe(root, config)
    scores: dict[str, dict] = {}
    stimulus_manifest = []
    half_width = float(config["stimulus"]["ring_half_width_pixels"])
    for condition_id in condition_ids:
        parameters = conditions[condition_id]["generator_parameters"]
        frames = {
            direction: _local_radial_ring(
                int(parameters["looming_duration_frames"]),
                float(parameters["terminal_radius_pixels"]),
                half_width,
                float(parameters["noise_standard_deviation"]),
                int(parameters["seed"]),
                outward=direction == "outward",
            )
            for direction in ("outward", "inward")
        }
        responses = {}
        for direction, stimulus_frames in frames.items():
            identity = f"{condition_id}:local_radial:off:{direction}"
            stimulus = VisualStimulus(
                identity, "local_radial", "off", direction, stimulus_frames, identity
            )
            responses[direction] = run_signed_cell_responses(probe, stimulus)
            stimulus_manifest.append(
                {
                    "identity": identity,
                    "frame_sha256": hashlib.sha256(stimulus_frames.tobytes()).hexdigest(),
                    "retinal_drive_sha256": responses[direction]["retinal_drive_sha256"],
                }
            )
        scores[condition_id] = {}
        for population in config["populations"]:
            body_ids = probe.graph.body_ids[probe.populations[population]]
            scores[condition_id][population] = strict_contrast_summary(
                body_ids,
                responses["outward"]["peaks"][population],
                responses["inward"]["peaks"][population],
                scoring["thresholds"],
            )
    joint = {}
    for population in config["populations"]:
        body_ids = probe.graph.body_ids[probe.populations[population]]
        joint[population] = _target_joint_coverage(
            [scores[name][population] for name in condition_ids], body_ids
        )
    passed = all(
        scores[condition][population]["passed"]
        for condition in condition_ids
        for population in config["populations"]
    ) and all(
        item["joint_valid_fraction"] >= 0.80
        and item["all_condition_success_fraction"] >= 0.60
        for item in joint.values()
    )
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(NESTED_CONFIG): _sha256(root / NESTED_CONFIG),
                str(scoring_path): _sha256(root / scoring_path),
            },
            "condition_ids": condition_ids,
            "stimulus_count": len(stimulus_manifest),
            "parameter_fit": False,
            "runtime_modified": False,
            "calibration_evaluated": False,
            "final_evaluated": False,
        },
        "stimulus_manifest": stimulus_manifest,
        "per_condition_scores": scores,
        "target_joint_coverage": joint,
        "strict_lplc2_local_radial_gate_passed": bool(passed),
        "advance_to_runtime_integration": False,
        "advance_to_navigation": False,
        "boundary": config["boundary"],
    }
