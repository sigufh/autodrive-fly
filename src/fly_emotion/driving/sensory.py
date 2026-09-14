"""Anatomy-backed optic-flow and self-motion sensory stimulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

SENSORY_PROFILES = (
    "front",
    "panorama",
    "panorama_flow",
    "panorama_flow_body",
)


def _present_nodes(body_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    nodes = np.searchsorted(body_ids, ids).astype(np.int32)
    valid = nodes < len(body_ids)
    valid[valid] &= body_ids[nodes[valid]] == ids[valid]
    return nodes[valid]


def horizontal_flow_proxy(previous: np.ndarray, current: np.ndarray, max_shift: int = 6) -> float:
    """Estimate signed panoramic image displacement without vehicle state."""
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("optic-flow frames must have equal 2-D shapes")
    before = previous.mean(axis=0).astype(np.float64)
    after = current.mean(axis=0).astype(np.float64)
    before -= before.mean()
    after -= after.mean()
    scale = float(np.linalg.norm(before) * np.linalg.norm(after))
    if scale < 1e-10:
        return 0.0
    shifts = range(-max_shift, max_shift + 1)
    scores = [float(np.dot(after, np.roll(before, shift)) / scale) for shift in shifts]
    return float(np.asarray(list(shifts))[int(np.argmax(scores))] / max_shift)


@dataclass(frozen=True)
class SensoryFrame:
    flow: float
    yaw_rate: float
    steering: float
    speed: float


@dataclass(frozen=True)
class AnatomySensoryProjection:
    flow_positive: np.ndarray
    flow_negative: np.ndarray
    haltere_left: np.ndarray
    haltere_right: np.ndarray
    proprio_left: np.ndarray
    proprio_right: np.ndarray

    @classmethod
    def from_annotations(
        cls, body_ids: np.ndarray, annotations_path: Path
    ) -> AnatomySensoryProjection:
        table = feather.read_table(
            annotations_path,
            columns=["bodyId", "type", "class", "subclass", "superclass", "rootSide"],
            memory_map=True,
        ).to_pandas()
        cell_type = table["type"].fillna("")
        root_side = table["rootSide"].fillna("")
        haltere = table["subclass"].fillna("").eq("haltere")
        proprio = table["class"].fillna("").eq("mechanosensory_proprioceptive") & table[
            "superclass"
        ].fillna("").str.startswith("sensory_ascending")

        def select(mask) -> np.ndarray:
            return _present_nodes(body_ids, table.loc[mask, "bodyId"].to_numpy(dtype=np.int64))

        return cls(
            flow_positive=select(cell_type.isin(["T4a", "T5a"])),
            flow_negative=select(cell_type.isin(["T4b", "T5b"])),
            haltere_left=select(haltere & root_side.eq("L")),
            haltere_right=select(haltere & root_side.eq("R")),
            proprio_left=select(proprio & root_side.eq("L")),
            proprio_right=select(proprio & root_side.eq("R")),
        )

    def add_drive(
        self, drive: np.ndarray, frame: SensoryFrame, *, optic_flow: bool, body: bool
    ) -> None:
        if optic_flow:
            positive = max(frame.flow, 0.0)
            negative = max(-frame.flow, 0.0)
            drive[self.flow_positive] += 0.30 * positive
            drive[self.flow_negative] += 0.30 * negative
        if body:
            yaw = float(np.clip(frame.yaw_rate / 1.2, -1.0, 1.0))
            steering = float(np.clip(frame.steering, -1.0, 1.0))
            speed = float(np.clip(abs(frame.speed) / 6.0, 0.0, 1.0))
            drive[self.haltere_left] += 0.40 * max(-yaw, 0.0)
            drive[self.haltere_right] += 0.40 * max(yaw, 0.0)
            drive[self.proprio_left] += 0.25 * max(-steering, 0.0) + 0.04 * speed
            drive[self.proprio_right] += 0.25 * max(steering, 0.0) + 0.04 * speed

    def summary(self) -> dict:
        return {
            "T4_T5_horizontal_flow": int(len(self.flow_positive) + len(self.flow_negative)),
            "haltere_yaw_rate": int(len(self.haltere_left) + len(self.haltere_right)),
            "ascending_proprioception": int(len(self.proprio_left) + len(self.proprio_right)),
            "haltere_by_side": [int(len(self.haltere_left)), int(len(self.haltere_right))],
            "proprioception_by_side": [int(len(self.proprio_left)), int(len(self.proprio_right))],
        }
