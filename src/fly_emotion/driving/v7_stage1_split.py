from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import V7_CONFIG, _sha256

CONFIG = Path("configs/driving-v7-stage1-split.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_stage1_split.py")


@dataclass(frozen=True)
class Stage1Stimulus:
    identity: str
    split: str
    family: str
    polarity: str
    direction: str
    parameter_name: str
    parameter_value: float
    noise_standard_deviation: float
    seed: int
    frames: np.ndarray
    mirror_of: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.frames.astype(np.float32, copy=False).tobytes()).hexdigest()


def _moving_edge(
    width: int, height: int, speed: float, polarity: str, direction: str
) -> np.ndarray:
    background, foreground = (0.08, 0.92) if polarity == "on" else (0.92, 0.08)
    axis = "x" if direction in {"left", "right"} else "y"
    extent = width if axis == "x" else height
    usable = extent - 5
    steps = int(np.ceil(usable / speed)) + 1
    positions = np.minimum(2 + np.arange(steps) * speed, extent - 3)
    if direction in {"left", "up"}:
        positions = positions[::-1]
    output = np.full((steps, height, width), background, dtype=np.float32)
    for index, position in enumerate(positions):
        centre = int(round(float(position)))
        if axis == "x":
            region = (
                (slice(None), slice(None, centre + 1))
                if direction == "right"
                else (slice(None), slice(centre, None))
            )
        else:
            region = (
                (slice(None, centre + 1), slice(None))
                if direction == "down"
                else (slice(centre, None), slice(None))
            )
        output[index][region] = foreground
    return output


def _looming(width: int, height: int, frames: int, polarity: str, direction: str) -> np.ndarray:
    background, foreground = (0.08, 0.92) if polarity == "on" else (0.92, 0.08)
    yy, xx = np.mgrid[:height, :width]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    radii = np.linspace(1.0, min(width, height) * 0.46, frames)
    if direction == "contraction":
        radii = radii[::-1]
    output = np.full((frames, height, width), background, dtype=np.float32)
    for index, radius in enumerate(radii):
        output[index, (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = foreground
    return output


def _translation(
    width: int, height: int, period: float, polarity: str, direction: str, frames: int = 32
) -> np.ndarray:
    _, xx = np.mgrid[:height, :width]
    sign = 1 if direction == "right" else -1
    contrast = 0.4 if polarity == "on" else -0.4
    return np.stack(
        [
            np.broadcast_to(
                0.5 + contrast * np.sin(2 * np.pi * xx / period - sign * phase),
                (height, width),
            ).astype(np.float32)
            for phase in np.linspace(0, 4 * np.pi, frames, endpoint=False)
        ]
    )


def _rotation(
    width: int, height: int, turns: float, direction: str, frames: int = 32
) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    x = (xx - (width - 1) / 2) / width
    y = (yy - (height - 1) / 2) / height
    radius = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)
    sign = 1 if direction == "clockwise" else -1
    phases = np.linspace(0, 2 * np.pi * turns, frames)
    return np.stack(
        [
            (0.5 + 0.4 * np.sin(18 * radius + 5 * (angle - sign * phase))).astype(np.float32)
            for phase in phases
        ]
    )


def _add_noise(frames: np.ndarray, standard_deviation: float, seed: int) -> np.ndarray:
    if standard_deviation == 0:
        return frames.copy()
    rng = np.random.default_rng(seed)
    return np.clip(frames + rng.normal(0, standard_deviation, frames.shape), 0, 1).astype(
        np.float32
    )


def _paired_noise(
    frames: np.ndarray, standard_deviation: float, seed: int, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    first = _add_noise(frames, standard_deviation, seed)
    return first, np.flip(first, axis=axis).copy()


def _condition_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _symmetric_noise(frames: np.ndarray, standard_deviation: float, seed: int) -> np.ndarray:
    if standard_deviation == 0:
        return frames.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, standard_deviation, frames.shape)
    noise = (noise + noise[:, :, ::-1]) / 2
    return np.clip(frames + noise, 0, 1).astype(np.float32)


def build_stage1_split(config: dict) -> dict[str, list[Stage1Stimulus]]:
    width = int(config["stimulus_geometry"]["width"])
    height = int(config["stimulus_geometry"]["height"])
    output = {}
    for split, values in config["splits"].items():
        stimuli = []
        for seed_index, seed in enumerate(values["seeds"]):
            noise = float(values["noise_standard_deviation"][seed_index])
            for level in config["stimulus_geometry"]["uniform_levels"]:
                identity = f"{split}:uniform:{level:g}:{noise:g}:{seed}"
                frames = np.full((32, height, width), float(level), dtype=np.float32)
                stimuli.append(
                    Stage1Stimulus(
                        identity,
                        split,
                        "uniform",
                        "none",
                        "none",
                        "level",
                        float(level),
                        noise,
                        int(seed),
                        _symmetric_noise(frames, noise, _condition_seed(int(seed), identity)),
                        identity,
                    )
                )
            for parameter in values["edge_speeds_pixels_per_frame"]:
                for polarity in config["polarities"]:
                    horizontal_ids = {
                        direction: (
                            f"{split}:edge:{polarity}:{direction}:{parameter:g}:{noise:g}:{seed}"
                        )
                        for direction in ("right", "left")
                    }
                    base = _moving_edge(width, height, float(parameter), polarity, "right")
                    right, left = _paired_noise(
                        base,
                        noise,
                        _condition_seed(int(seed), horizontal_ids["right"]),
                        2,
                    )
                    for direction, frames in (("right", right), ("left", left)):
                        identity = horizontal_ids[direction]
                        mirror_of = horizontal_ids["left" if direction == "right" else "right"]
                        stimuli.append(
                            Stage1Stimulus(
                                identity,
                                split,
                                "moving_edge",
                                polarity,
                                direction,
                                "speed_pixels_per_frame",
                                float(parameter),
                                noise,
                                int(seed),
                                frames,
                                mirror_of,
                            )
                        )
                    for direction in ("down", "up"):
                        identity = (
                            f"{split}:edge:{polarity}:{direction}:{parameter:g}:{noise:g}:{seed}"
                        )
                        frames = _moving_edge(width, height, float(parameter), polarity, direction)
                        stimuli.append(
                            Stage1Stimulus(
                                identity,
                                split,
                                "moving_edge",
                                polarity,
                                direction,
                                "speed_pixels_per_frame",
                                float(parameter),
                                noise,
                                int(seed),
                                _symmetric_noise(
                                    frames, noise, _condition_seed(int(seed), identity)
                                ),
                                identity,
                            )
                        )
            for parameter in values["looming_duration_frames"]:
                for polarity in config["polarities"]:
                    for direction in config["directions"]["looming"]:
                        identity = (
                            f"{split}:loom:{polarity}:{direction}:{parameter}:{noise:g}:{seed}"
                        )
                        frames = _looming(width, height, int(parameter), polarity, direction)
                        stimuli.append(
                            Stage1Stimulus(
                                identity,
                                split,
                                "looming",
                                polarity,
                                direction,
                                "duration_frames",
                                float(parameter),
                                noise,
                                int(seed),
                                _symmetric_noise(
                                    frames, noise, _condition_seed(int(seed), identity)
                                ),
                                identity,
                            )
                        )
            for parameter in values["translation_period_pixels"]:
                for polarity in config["polarities"]:
                    base = _translation(width, height, float(parameter), polarity, "right")
                    right_id = f"{split}:translation:{polarity}:right:{parameter}:{noise:g}:{seed}"
                    left_id = f"{split}:translation:{polarity}:left:{parameter}:{noise:g}:{seed}"
                    right, left = _paired_noise(
                        base, noise, _condition_seed(int(seed), right_id), 2
                    )
                    for direction, frames, identity, mirror_of in (
                        ("right", right, right_id, left_id),
                        ("left", left, left_id, right_id),
                    ):
                        stimuli.append(
                            Stage1Stimulus(
                                identity,
                                split,
                                "translation",
                                polarity,
                                direction,
                                "period_pixels",
                                float(parameter),
                                noise,
                                int(seed),
                                frames,
                                mirror_of,
                            )
                        )
            for parameter in values["rotation_turns_per_sequence"]:
                cw_id = f"{split}:rotation:clockwise:{parameter:g}:{noise:g}:{seed}"
                ccw_id = f"{split}:rotation:counterclockwise:{parameter:g}:{noise:g}:{seed}"
                base = _rotation(width, height, float(parameter), "clockwise")
                clockwise, counterclockwise = _paired_noise(
                    base, noise, _condition_seed(int(seed), cw_id), 2
                )
                for direction, frames, identity, mirror_of in (
                    ("clockwise", clockwise, cw_id, ccw_id),
                    ("counterclockwise", counterclockwise, ccw_id, cw_id),
                ):
                    stimuli.append(
                        Stage1Stimulus(
                            identity,
                            split,
                            "rotation",
                            "mixed",
                            direction,
                            "turns_per_sequence",
                            float(parameter),
                            noise,
                            int(seed),
                            frames,
                            mirror_of,
                        )
                    )
        output[split] = stimuli
    return output


def _split_digest(stimuli: list[Stage1Stimulus]) -> str:
    digest = hashlib.sha256()
    for item in stimuli:
        digest.update(item.identity.encode())
        digest.update(bytes.fromhex(item.sha256))
    return digest.hexdigest()


def evaluate_v7_stage1_split(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    visual = yaml.safe_load((root / V7_CONFIG).read_text(encoding="utf-8"))["controlled_vision"]
    timebase = config["engineering_timebase"]
    if timebase["neural_substeps_per_frame"] != visual["brain_substeps_per_frame"]:
        raise ValueError("engineering split substeps differ from v7 contract")
    if (
        timebase["nominal_substep_interval_milliseconds"] * timebase["neural_substeps_per_frame"]
        != timebase["frame_interval_milliseconds"]
    ):
        raise ValueError("engineering frame and solver intervals are inconsistent")
    if timebase["biologically_calibrated"]:
        raise ValueError("engineering split timebase must not claim biological calibration")
    splits = build_stage1_split(config)
    identities = {name: {item.identity for item in values} for name, values in splits.items()}
    frame_hashes = {name: {item.sha256 for item in values} for name, values in splits.items()}
    names = list(splits)
    overlap = {
        f"{left}:{right}": {
            "identity_overlap": len(identities[left] & identities[right]),
            "frame_hash_overlap": len(frame_hashes[left] & frame_hashes[right]),
        }
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    if any(value["identity_overlap"] or value["frame_hash_overlap"] for value in overlap.values()):
        raise ValueError("stage-1 stimulus splits are not disjoint")
    mirror_checks = {}
    for name, stimuli in splits.items():
        by_identity = {item.identity: item for item in stimuli}
        mirror_checks[name] = all(
            np.array_equal(item.frames[:, :, ::-1], by_identity[item.mirror_of].frames)
            for item in stimuli
        )
    if not all(mirror_checks.values()):
        raise ValueError("stage-1 stimulus mirror pairing is not exact")
    manifests = {}
    for name, stimuli in splits.items():
        manifest = {
            "evaluable": bool(config["splits"][name]["evaluable"]),
            "stimulus_count": len(stimuli),
            "aggregate_sha256": _split_digest(stimuli),
            "families": dict(
                sorted(
                    {
                        family: sum(x.family == family for x in stimuli)
                        for family in config["families"]
                    }.items()
                )
            ),
        }
        if name != "final":
            manifest["stimuli"] = [
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
                    "sha256": item.sha256,
                    "mirror_of": item.mirror_of,
                }
                for item in stimuli
            ]
        else:
            manifest["stimuli"] = None
            manifest["sealed"] = True
            manifest["evaluated"] = False
        manifests[name] = manifest
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(V7_CONFIG): _sha256(root / V7_CONFIG),
            },
            "model_evaluated": False,
            "parameters_fitted": False,
            "runtime_modified": False,
        },
        "engineering_timebase": timebase,
        "split_manifests": manifests,
        "cross_split_overlap": overlap,
        "exact_mirror_checks": mirror_checks,
        "release_boundary": config["release_boundary"],
        "limitations": [
            "The 10-ms frame interval is an engineering schedule, not a biological calibration.",
            "Stimulus-level disjointness does not create independent animals or cells.",
            "The sealed final aggregate digest proves identity but does not constitute evaluation.",
            "No v7 model, visual gate, navigation policy or driving task was evaluated.",
        ],
        "advance_to_model_fit": False,
        "advance_to_final_test": False,
        "advance_to_central_complex": False,
    }
