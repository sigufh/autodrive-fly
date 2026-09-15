from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.retina import RetinaMap
from fly_emotion.driving.v7_branched import V7BranchedT4Probe
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_retina_audit import build_balanced_retina_control, infer_retinal_columns
from fly_emotion.driving.v7_stage1_split import build_stage1_split

CONFIG = Path("configs/driving-v7-stage1-input-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_stage1_input_audit.py")


def encode_trace(values: np.ndarray, encoding: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("retinal values must be time-by-receptor")
    if encoding == "linear_luminance":
        return values.copy()
    if encoding == "signed_frame_difference":
        previous = np.concatenate((values[:1], values[:-1]), axis=0)
        baseline = np.maximum(np.abs(values[0]), 0.05)
        return np.clip((values - previous) / baseline, -1, 1).astype(np.float32)
    raise ValueError(f"unknown retinal encoding: {encoding}")


def _sample_stimulus(stimulus, retina) -> np.ndarray:
    return np.stack([retina.encode(frame) for frame in stimulus.frames])


def _dynamic_summary(encoded: np.ndarray) -> dict:
    ranges = np.max(encoded, axis=0) - np.min(encoded, axis=0)
    return {
        "minimum": float(np.min(encoded)),
        "maximum": float(np.max(encoded)),
        "dynamic_range": float(np.max(encoded) - np.min(encoded)),
        "dynamic_receptor_fraction_above_1e_6": float(np.mean(ranges > 1e-6)),
        "sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
    }


def _sign_agreement(stimuli, encoded: dict[str, np.ndarray], family: str) -> dict:
    on = [item for item in stimuli if item.family == family and item.polarity == "on"]
    lookup = {
        (
            item.family,
            item.direction,
            item.parameter_value,
            item.noise_standard_deviation,
            item.seed,
        ): item
        for item in stimuli
        if item.polarity == "off"
    }
    agreements = []
    pair_count = 0
    for item in on:
        key = (
            item.family,
            item.direction,
            item.parameter_value,
            item.noise_standard_deviation,
            item.seed,
        )
        other = lookup.get(key)
        if other is None:
            continue
        a = encoded[item.identity] - encoded[item.identity][0]
        b = encoded[other.identity] - encoded[other.identity][0]
        active = (np.abs(a) + np.abs(b)) > 1e-6
        if np.any(active):
            agreements.append(float(np.mean((a[active] * b[active]) <= 0)))
            pair_count += 1
    return {
        "pair_count": pair_count,
        "minimum_opposite_sign_fraction": min(agreements) if agreements else None,
        "median_opposite_sign_fraction": float(np.median(agreements)) if agreements else None,
    }


def _mirror_summary(stimuli, encoded: dict[str, np.ndarray], balanced) -> dict:
    by_identity = {item.identity: item for item in stimuli}
    errors = []
    pair_count = 0
    for item in stimuli:
        if item.identity > item.mirror_of:
            continue
        counterpart = by_identity[item.mirror_of]
        left = encoded[item.identity][:, balanced.left_positions]
        right = encoded[counterpart.identity][:, balanced.right_positions]
        error = float(np.max(np.abs(left - right)))
        errors.append(error)
        pair_count += 1
    return {
        "stimulus_pair_count": pair_count,
        "maximum_absolute_error": max(errors, default=0.0),
        "all_errors": errors,
    }


def _direction_null(stimuli, encoded: dict[str, np.ndarray]) -> dict:
    by_key = {
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
    values = []
    for item in stimuli:
        if item.family not in {"moving_edge", "translation"} or item.direction not in {
            "right",
            "down",
        }:
            continue
        opposite = {"right": "left", "down": "up"}[item.direction]
        other = by_key[
            (
                item.family,
                item.polarity,
                opposite,
                item.parameter_value,
                item.noise_standard_deviation,
                item.seed,
            )
        ]
        a = float(np.mean(np.square(encoded[item.identity])))
        b = float(np.mean(np.square(encoded[other.identity])))
        values.append(abs(a - b) / (a + b + 1e-12))
    return {
        "comparison_count": len(values),
        "maximum_absolute_population_energy_contrast": max(values, default=0.0),
    }


def receptor_separability_control(encoding: str) -> dict:
    rng = np.random.default_rng(20260915)
    values = rng.uniform(0.1, 0.9, size=(12, 7)).astype(np.float32)
    changed = values.copy()
    changed[5, 3] = np.clip(changed[5, 3] + 0.08, 0, 1)
    original = encode_trace(values, encoding)
    perturbed = encode_trace(changed, encoding)
    difference = np.abs(perturbed - original)
    other = np.ones(difference.shape, dtype=bool)
    other[:, 3] = False
    return {
        "perturbed_receptor_index": 3,
        "perturbed_time_index": 5,
        "maximum_other_receptor_change": float(np.max(difference[other])),
        "maximum_target_receptor_change": float(np.max(difference[:, 3])),
        "encoding_reads_direction_label": False,
    }


def evaluate_v7_stage1_input_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    split_config_path = Path(config["stage1_config"])
    split_evidence_path = Path(config["stage1_evidence"])
    split_config = yaml.safe_load((root / split_config_path).read_text(encoding="utf-8"))
    split_evidence = json.loads((root / split_evidence_path).read_text(encoding="utf-8"))
    if split_evidence["protocol"]["parameters_fitted"]:
        raise ValueError("stage-1 input audit cannot consume fitted split evidence")
    stimuli = build_stage1_split(split_config)[config["split"]]
    default_retina, _ = infer_retinal_columns(root)
    balanced = build_balanced_retina_control(root)
    nested_probe = V7BranchedT4Probe(
        root,
        retinal_backend="linear_luminance",
        retinal_geometry="nested_t4_axis_v1",
        brain_substeps=4,
        baseline_frames=8,
    )
    nested_retina = RetinaMap(
        node_indices=nested_probe.retina.node_indices.copy(),
        body_ids=nested_probe.retina.body_ids.copy(),
        u=nested_probe.retinal_u.copy(),
        v=nested_probe.retinal_v.copy(),
        side=nested_probe.retina.side.copy(),
        mapping_version=nested_probe.retina.mapping_version,
    )
    retinal_maps = {
        "default_3344": default_retina,
        "nested_t4_axis_3344": nested_retina,
        "balanced_1914": balanced.retina,
    }
    if list(retinal_maps) != config["retinal_maps"]:
        raise ValueError("stage-1 retinal map set differs from frozen input audit")
    thresholds = config["thresholds"]
    reports = {}
    all_gates = []
    for map_name, retina in retinal_maps.items():
        sampled = {item.identity: _sample_stimulus(item, retina) for item in stimuli}
        encodings = {}
        for encoding in config["retinal_encodings"]:
            encoded = {name: encode_trace(values, encoding) for name, values in sampled.items()}
            dynamic = {
                item.identity: _dynamic_summary(encoded[item.identity])
                for item in stimuli
                if item.family != "uniform"
            }
            family_ranges = {
                family: min(
                    result["dynamic_range"]
                    for identity, result in dynamic.items()
                    if next(item for item in stimuli if item.identity == identity).family == family
                )
                for family in thresholds["require_all_stimulus_families"]
                if family != "uniform"
            }
            family_coverage = {
                family: min(
                    result["dynamic_receptor_fraction_above_1e_6"]
                    for identity, result in dynamic.items()
                    if next(item for item in stimuli if item.identity == identity).family == family
                )
                for family in thresholds["require_all_stimulus_families"]
                if family != "uniform"
            }
            mirror = (
                _mirror_summary(stimuli, encoded, balanced) if map_name == "balanced_1914" else None
            )
            polarity = {
                family: _sign_agreement(stimuli, encoded, family)
                for family in ("moving_edge", "looming", "static", "translation")
            }
            direction_null = _direction_null(stimuli, encoded)
            separability = receptor_separability_control(encoding)
            coverage_thresholds = thresholds["minimum_dynamic_receptor_fraction_by_family"]
            gates = {
                "all_family_dynamic_ranges": min(family_ranges.values())
                >= float(thresholds["minimum_dynamic_range"]),
                "all_family_dynamic_receptor_coverage": all(
                    family_coverage[family] >= float(coverage_thresholds[family])
                    for family in family_coverage
                ),
                "all_on_off_pairs_have_opposite_sign": min(
                    value["minimum_opposite_sign_fraction"] for value in polarity.values()
                )
                >= float(thresholds["minimum_on_off_sign_agreement"]),
                "retinal_encoding_is_receptor_separable": separability[
                    "maximum_other_receptor_change"
                ]
                <= float(thresholds["maximum_cross_receptor_leakage"])
                and separability["maximum_target_receptor_change"]
                >= float(thresholds["minimum_changed_target_receptor_magnitude"]),
            }
            if mirror is not None:
                gates["balanced_pairs_are_exactly_mirrored"] = mirror[
                    "maximum_absolute_error"
                ] <= float(thresholds["maximum_balanced_pair_mirror_error"])
            encodings[encoding] = {
                "stimulus_dynamic_summary": dynamic,
                "minimum_dynamic_range_by_family": family_ranges,
                "minimum_dynamic_receptor_fraction_by_family": family_coverage,
                "on_off_sign_agreement_by_family": polarity,
                "independent_receptor_direction_null": direction_null,
                "receptor_separability_control": separability,
                "mirror_summary": mirror,
                "gates": gates,
                "passed": all(gates.values()),
            }
            all_gates.append(all(gates.values()))
        reports[map_name] = {
            "receptor_count": retina.size,
            "receptor_body_ids_sha256": hashlib.sha256(retina.body_ids.tobytes()).hexdigest(),
            "encodings": encodings,
        }
    input_gates_pass = all(all_gates)
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(split_config_path): _sha256(root / split_config_path),
        str(split_evidence_path): _sha256(root / split_evidence_path),
        config["retina_audit_config"]: _sha256(root / config["retina_audit_config"]),
        config["retina_audit_evidence"]: _sha256(root / config["retina_audit_evidence"]),
        config["branched_config"]: _sha256(root / config["branched_config"]),
        config["branched_implementation"]: _sha256(root / config["branched_implementation"]),
        config["source_audit_evidence"]: _sha256(root / config["source_audit_evidence"]),
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "split": config["split"],
            "dependencies_sha256": dependencies,
            "neural_evaluation_performed": False,
            "parameters_fitted": False,
            "runtime_modified": False,
        },
        "stimulus_count": len(stimuli),
        "stimulus_families": sorted({item.family for item in stimuli}),
        "retinal_maps": reports,
        "input_gates_pass": input_gates_pass,
        "development_neural_evaluation_allowed": input_gates_pass,
        "boundary": config["boundary"],
        "limitations": [
            "This audit tests sampled receptor input only, not neural direction selectivity.",
            "The balanced control discards unmatched receptors and is not a biological retina.",
            (
                "Finite-view per-receptor energy may differ across motion directions; the "
                "valid negative control is cross-receptor separability, not zero energy contrast."
            ),
            "Validation, OOD and reserved final stimuli were not sampled or evaluated.",
        ],
        "advance_to_validation": False,
        "advance_to_central_complex": False,
    }
