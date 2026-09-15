from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_stage1_split import (
    Stage1Stimulus,
    _complementary_pair,
    _condition_seed,
    _moving_edge,
    _paired_noise,
    _rotation,
    _symmetric_noise,
    _translation,
)

CONFIG = Path("configs/driving-v7-stage1-nested.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_stage1_nested.py")
WIDTH = 48
HEIGHT = 24


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _frame_digest(frames: np.ndarray) -> str:
    canonical = np.asarray(frames, dtype="<f4", order="C")
    metadata = {"shape": list(canonical.shape), "dtype": "<f4", "byteorder": "little"}
    digest = hashlib.sha256(b"fly-emotion/stage1/frame/v2\0")
    digest.update(_canonical_json(metadata))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _looming_with_radius(frames: int, radius: float, polarity: str, direction: str) -> np.ndarray:
    background, foreground = (0.08, 0.92) if polarity == "on" else (0.92, 0.08)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    cx, cy = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
    radii = np.linspace(1.0, radius, frames)
    if direction == "contraction":
        radii = radii[::-1]
    output = np.full((frames, HEIGHT, WIDTH), background, dtype=np.float32)
    for index, current_radius in enumerate(radii):
        output[index, (xx - cx) ** 2 + (yy - cy) ** 2 <= current_radius**2] = foreground
    return output


def _stimulus(
    condition_id: str,
    family: str,
    polarity: str,
    direction: str,
    parameter_name: str,
    parameter_value: float,
    noise: float,
    seed: int,
    frames: np.ndarray,
    mirror_of: str,
) -> Stage1Stimulus:
    identity = (
        f"{condition_id}:{family}:{polarity}:{direction}:"
        f"{parameter_name}={parameter_value:g}:noise={noise:g}:seed={seed}"
    )
    return Stage1Stimulus(
        identity,
        condition_id,
        family,
        polarity,
        direction,
        parameter_name,
        parameter_value,
        noise,
        seed,
        frames,
        mirror_of,
    )


def build_condition_bundle(condition: dict) -> list[Stage1Stimulus]:
    if condition["role"] not in {"tuning", "calibration"}:
        raise ValueError("only local tuning/calibration bundles may be generated")
    condition_id = str(condition["condition_id"])
    values = condition["generator_parameters"]
    seed = int(values["seed"])
    noise = float(values["noise_standard_deviation"])
    output: list[Stage1Stimulus] = []
    for level in (0.08, 0.50, 0.92):
        identity = f"{condition_id}:uniform:none:none:level={level:g}:noise={noise:g}:seed={seed}"
        frames = np.full((32, HEIGHT, WIDTH), level, dtype=np.float32)
        frames = _symmetric_noise(frames, noise, _condition_seed(seed, identity))
        output.append(
            Stage1Stimulus(
                identity,
                condition_id,
                "uniform",
                "none",
                "none",
                "level",
                level,
                noise,
                seed,
                frames,
                identity,
            )
        )
    speed = float(values["edge_speed_pixels_per_frame"])
    for primary_direction, mirror_direction, axis in (("right", "left", 2), ("down", "up", 1)):
        base = _moving_edge(WIDTH, HEIGHT, speed, "on", primary_direction)
        on = (
            _symmetric_noise(base, noise, _condition_seed(seed, f"edge:{primary_direction}"))
            if axis == 1
            else _complementary_pair(
                base, noise, _condition_seed(seed, f"edge:{primary_direction}")
            )[0]
        )
        for polarity, primary in (("on", on), ("off", (1.0 - on).astype(np.float32))):
            ids = {
                direction: (
                    f"{condition_id}:moving_edge:{polarity}:{direction}:"
                    f"speed_pixels_per_frame={speed:g}:noise={noise:g}:seed={seed}"
                )
                for direction in (primary_direction, mirror_direction)
            }
            for direction, frames in (
                (primary_direction, primary),
                (mirror_direction, np.flip(primary, axis=axis).copy()),
            ):
                mirror_of = (
                    ids[mirror_direction if direction == primary_direction else primary_direction]
                    if axis == 2
                    else ids[direction]
                )
                output.append(
                    Stage1Stimulus(
                        ids[direction],
                        condition_id,
                        "moving_edge",
                        polarity,
                        direction,
                        "speed_pixels_per_frame",
                        speed,
                        noise,
                        seed,
                        frames,
                        mirror_of,
                    )
                )
    duration = int(values["looming_duration_frames"])
    radius = float(values["terminal_radius_pixels"])
    for direction in ("expansion", "contraction"):
        on = _symmetric_noise(
            _looming_with_radius(duration, radius, "on", direction),
            noise,
            _condition_seed(seed, f"loom:{direction}"),
        )
        for polarity, frames in (("on", on), ("off", (1.0 - on).astype(np.float32))):
            item = _stimulus(
                condition_id,
                "looming",
                polarity,
                direction,
                "terminal_radius_pixels",
                radius,
                noise,
                seed,
                frames,
                "",
            )
            output.append(Stage1Stimulus(**{**item.__dict__, "mirror_of": item.identity}))
    for polarity in ("on", "off"):
        base = _looming_with_radius(duration, radius, polarity, "expansion")[-1]
        background = 0.08 if polarity == "on" else 0.92
        frames = np.repeat(base[None], duration, axis=0)
        frames[0] = background
        frames = _symmetric_noise(frames, noise, _condition_seed(seed, f"static:{polarity}"))
        item = _stimulus(
            condition_id,
            "static",
            polarity,
            "none",
            "terminal_radius_pixels",
            radius,
            noise,
            seed,
            frames,
            "",
        )
        output.append(Stage1Stimulus(**{**item.__dict__, "mirror_of": item.identity}))
    period = float(values["translation_period_pixels"])
    base = _translation(WIDTH, HEIGHT, period, "on", "right")
    on, _ = _complementary_pair(base, noise, _condition_seed(seed, "translation"))
    for polarity, primary in (("on", on), ("off", (1.0 - on).astype(np.float32))):
        ids = {
            direction: (
                f"{condition_id}:translation:{polarity}:{direction}:"
                f"period_pixels={period:g}:noise={noise:g}:seed={seed}"
            )
            for direction in ("right", "left")
        }
        for direction, frames in (("right", primary), ("left", np.flip(primary, axis=2).copy())):
            other = "left" if direction == "right" else "right"
            output.append(
                Stage1Stimulus(
                    ids[direction],
                    condition_id,
                    "translation",
                    polarity,
                    direction,
                    "period_pixels",
                    period,
                    noise,
                    seed,
                    frames,
                    ids[other],
                )
            )
    turns = float(values["rotation_turns_per_sequence"])
    cw_id = f"{condition_id}:rotation:mixed:clockwise:turns={turns:g}:noise={noise:g}:seed={seed}"
    ccw_id = (
        f"{condition_id}:rotation:mixed:counterclockwise:"
        f"turns={turns:g}:noise={noise:g}:seed={seed}"
    )
    clockwise, counterclockwise = _paired_noise(
        _rotation(WIDTH, HEIGHT, turns, "clockwise"), noise, _condition_seed(seed, cw_id), 2
    )
    for direction, frames, identity, mirror_of in (
        ("clockwise", clockwise, cw_id, ccw_id),
        ("counterclockwise", counterclockwise, ccw_id, cw_id),
    ):
        output.append(
            Stage1Stimulus(
                identity,
                condition_id,
                "rotation",
                "mixed",
                direction,
                "turns_per_sequence",
                turns,
                noise,
                seed,
                frames,
                mirror_of,
            )
        )
    return output


def _condition_manifest(condition: dict, stimuli: list[Stage1Stimulus]) -> dict:
    records = []
    for item in stimuli:
        records.append(
            {
                "identity": item.identity,
                "family": item.family,
                "polarity": item.polarity,
                "direction": item.direction,
                "parameter_name": item.parameter_name,
                "parameter_value": item.parameter_value,
                "noise_standard_deviation": item.noise_standard_deviation,
                "seed": item.seed,
                "shape": list(item.frames.shape),
                "dtype": "<f4",
                "frame_sha256": _frame_digest(item.frames),
                "raw_frame_sha256": item.sha256,
                "mirror_of": item.mirror_of,
            }
        )
    digest = hashlib.sha256(
        b"fly-emotion/stage1/condition/v2\0" + _canonical_json(records)
    ).hexdigest()
    return {
        "condition_id": condition["condition_id"],
        "role": condition["role"],
        "stimulus_count": len(records),
        "condition_manifest_sha256": digest,
        "stimuli": records,
    }


def evaluate_v7_stage1_nested(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    legacy_path = Path(config["legacy_split_evidence"])
    legacy = json.loads((root / legacy_path).read_text())
    old = legacy["split_manifests"][config["legacy_regression"]["source_split"]]
    expected = config["legacy_regression"]
    if (
        old["stimulus_count"] != expected["expected_stimulus_count"]
        or old["aggregate_sha256"] != expected["expected_aggregate_sha256"]
    ):
        raise ValueError("legacy regression identity drifted")
    conditions = config["conditions"]
    roles = [item["role"] for item in conditions]
    expected_roles = config["condition_contract"]["required_role_counts"]
    if len(conditions) != config["condition_contract"]["condition_count"] or any(
        roles.count(role) != count for role, count in expected_roles.items()
    ):
        raise ValueError("nested condition role cardinality mismatch")
    local_manifests = []
    all_hashes: set[str] = set()
    all_raw_hashes: set[str] = set()
    legacy_hashes = {item["sha256"] for item in old["stimuli"]}
    for condition in conditions:
        if condition["role"] == "external_final":
            if "generator_parameters" in condition:
                raise ValueError("external final parameters must not be present locally")
            continue
        manifest = _condition_manifest(condition, build_condition_bundle(condition))
        hashes = {item["frame_sha256"] for item in manifest["stimuli"]}
        raw_hashes = {item["raw_frame_sha256"] for item in manifest["stimuli"]}
        if all_hashes & hashes:
            raise ValueError("local nested conditions share frames")
        if all_raw_hashes & raw_hashes or legacy_hashes & raw_hashes:
            raise ValueError("nested condition shares raw frames with another or legacy data")
        all_hashes |= hashes
        all_raw_hashes |= raw_hashes
        local_manifests.append(manifest)
    final = next(item for item in conditions if item["role"] == "external_final")
    custody = final["external_manifest"]
    final_committed = bool(
        custody["commitment_sha256"] and custody["custodian"] and custody["committed_at_utc"]
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(legacy_path): _sha256(root / legacy_path),
            },
            "model_evaluated": False,
            "parameters_fitted": False,
            "runtime_modified": False,
        },
        "legacy_regression": {
            **expected,
            "stimulus_count": old["stimulus_count"],
            "aggregate_sha256": old["aggregate_sha256"],
            "verified": True,
            "decision_eligible": False,
        },
        "condition_contract": config["condition_contract"],
        "local_condition_manifests": local_manifests,
        "leakage_audit": {
            "local_frame_hash_overlap_count": 0,
            "legacy_raw_frame_hash_overlap_count": 0,
            "local_seeds_unique": len(
                {
                    item["generator_parameters"]["seed"]
                    for item in conditions
                    if item["role"] != "external_final"
                }
            )
            == 4,
            "legacy_can_affect_selection": False,
            "legacy_can_affect_calibration": False,
            "legacy_can_affect_final_gate": False,
        },
        "external_final": {
            "condition_id": final["condition_id"],
            **custody,
            "committed": final_committed,
            "locally_generatable": False,
            "evaluated": False,
        },
        "statistics": config["statistics"],
        "execution_policy": config["execution_policy"],
        "tuning_authorized": True,
        "calibration_authorized": False,
        "final_authorized": False,
        "advance_to_central_complex": False,
    }
