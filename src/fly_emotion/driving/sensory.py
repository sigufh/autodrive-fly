"""Anatomy-backed optic-flow and self-motion sensory stimulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy import sparse

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
    """Legacy global displacement estimate retained for report replay."""
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


def _non_circular_shift(before: np.ndarray, after: np.ndarray, max_shift: int) -> float:
    before = before.astype(np.float64) - float(before.mean())
    after = after.astype(np.float64) - float(after.mean())
    if float(np.linalg.norm(before) * np.linalg.norm(after)) < 1e-10:
        return 0.0
    scores = []
    shifts = range(-max_shift, max_shift + 1)
    for shift in shifts:
        if shift < 0:
            left, right = before[-shift:], after[:shift]
        elif shift > 0:
            left, right = before[:-shift], after[shift:]
        else:
            left, right = before, after
        scale = float(np.linalg.norm(left) * np.linalg.norm(right))
        scores.append(float(np.dot(left, right) / scale) if scale > 1e-10 else -1.0)
    return float(np.asarray(list(shifts))[int(np.argmax(scores))] / max_shift)


def regional_horizontal_flow_proxy(
    previous: np.ndarray, current: np.ndarray, *, regions: int = 6, max_shift: int = 3
) -> tuple[float, ...]:
    """Estimate local horizontal displacement without wrapping image edges."""
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("optic-flow frames must have equal 2-D shapes")
    if regions < 2 or previous.shape[1] % regions:
        raise ValueError("optic-flow width must divide evenly into at least two regions")
    before = previous.mean(axis=0)
    after = current.mean(axis=0)
    return tuple(
        _non_circular_shift(a, b, max_shift)
        for a, b in zip(np.split(before, regions), np.split(after, regions), strict=True)
    )


@dataclass(frozen=True)
class SensoryFrame:
    flow: float
    yaw_rate: float
    steering: float
    speed: float
    flow_regions: tuple[float, ...] = ()
    yaw_acceleration: float = 0.0
    steering_rate: float = 0.0
    longitudinal_acceleration: float = 0.0
    lateral_acceleration: float = 0.0


@dataclass(frozen=True)
class SensoryGains:
    """Explicit simulator-to-neuron scaling, kept separate from anatomy."""

    peripheral_visual: float = 0.001
    optic_flow: float = 0.0
    haltere_yaw_rate: float = 0.0
    haltere_yaw_acceleration: float = 0.0
    campaniform_lateral: float = 0.0
    campaniform_longitudinal: float = 0.0
    chordotonal_steering_rate: float = 0.0

    def scaled(
        self, *, peripheral_visual: float = 1.0, optic_flow: float = 1.0, body: float = 1.0
    ) -> SensoryGains:
        return SensoryGains(
            peripheral_visual=self.peripheral_visual * peripheral_visual,
            optic_flow=self.optic_flow * optic_flow,
            haltere_yaw_rate=self.haltere_yaw_rate * body,
            haltere_yaw_acceleration=self.haltere_yaw_acceleration * body,
            campaniform_lateral=self.campaniform_lateral * body,
            campaniform_longitudinal=self.campaniform_longitudinal * body,
            chordotonal_steering_rate=self.chordotonal_steering_rate * body,
        )


@dataclass(frozen=True)
class AnatomySensoryProjection:
    flow_positive: np.ndarray
    flow_negative: np.ndarray
    haltere_left: np.ndarray
    haltere_right: np.ndarray
    proprio_left: np.ndarray
    proprio_right: np.ndarray
    flow_positive_bins: tuple[np.ndarray, ...]
    flow_negative_bins: tuple[np.ndarray, ...]
    flow_unmapped: np.ndarray
    haltere_rate_left: np.ndarray
    haltere_rate_right: np.ndarray
    haltere_acceleration_left: np.ndarray
    haltere_acceleration_right: np.ndarray
    campaniform_lateral_left: np.ndarray
    campaniform_lateral_right: np.ndarray
    campaniform_longitudinal_left: np.ndarray
    campaniform_longitudinal_right: np.ndarray
    chordotonal_rate_left: np.ndarray
    chordotonal_rate_right: np.ndarray
    body_unmapped: np.ndarray

    @classmethod
    def from_annotations(
        cls,
        body_ids: np.ndarray,
        annotations_path: Path,
        adjacency: sparse.csr_matrix | None = None,
        flow_regions: int = 6,
    ) -> AnatomySensoryProjection:
        table = feather.read_table(
            annotations_path,
            columns=[
                "bodyId",
                "type",
                "class",
                "subclass",
                "superclass",
                "rootSide",
                "somaSide",
                "entryNerve",
                "assignedOlHex1",
                "assignedOlHex2",
            ],
            memory_map=True,
        ).to_pandas()
        cell_type = table["type"].fillna("")
        root_side = table["rootSide"].fillna("")
        haltere = table["subclass"].fillna("").eq("haltere")
        proprio = table["class"].fillna("").eq("mechanosensory_proprioceptive") & table[
            "superclass"
        ].fillna("").str.startswith("sensory_ascending")
        entry_nerve = table["entryNerve"].fillna("")
        subclass = table["subclass"].fillna("")
        typed_haltere = haltere & cell_type.ne("")
        haltere_rate = haltere & cell_type.eq("SApp")
        haltere_acceleration = typed_haltere & ~cell_type.eq("SApp")
        non_haltere_proprio = proprio & ~haltere
        lateral = non_haltere_proprio & subclass.eq("campaniform sensilla") & entry_nerve.eq(
            "ADMN"
        )
        longitudinal = (
            non_haltere_proprio
            & subclass.eq("campaniform sensilla")
            & entry_nerve.eq("DMetaN")
        )
        chordotonal = non_haltere_proprio & subclass.eq("chordotonal organ")
        mapped_body = haltere_rate | haltere_acceleration | lateral | longitudinal | chordotonal

        def select(mask) -> np.ndarray:
            return _present_nodes(body_ids, table.loc[mask, "bodyId"].to_numpy(dtype=np.int64))

        flow_positive = select(cell_type.isin(["T4a", "T5a"]))
        flow_negative = select(cell_type.isin(["T4b", "T5b"]))
        positive_bins, negative_bins, unmapped = cls._flow_bins(
            body_ids, table, adjacency, flow_regions=flow_regions
        )
        return cls(
            flow_positive=flow_positive,
            flow_negative=flow_negative,
            haltere_left=select(haltere & root_side.eq("L")),
            haltere_right=select(haltere & root_side.eq("R")),
            proprio_left=select(proprio & root_side.eq("L")),
            proprio_right=select(proprio & root_side.eq("R")),
            flow_positive_bins=positive_bins,
            flow_negative_bins=negative_bins,
            flow_unmapped=unmapped,
            haltere_rate_left=select(haltere_rate & root_side.eq("L")),
            haltere_rate_right=select(haltere_rate & root_side.eq("R")),
            haltere_acceleration_left=select(haltere_acceleration & root_side.eq("L")),
            haltere_acceleration_right=select(haltere_acceleration & root_side.eq("R")),
            campaniform_lateral_left=select(lateral & root_side.eq("L")),
            campaniform_lateral_right=select(lateral & root_side.eq("R")),
            campaniform_longitudinal_left=select(longitudinal & root_side.eq("L")),
            campaniform_longitudinal_right=select(longitudinal & root_side.eq("R")),
            chordotonal_rate_left=select(chordotonal & root_side.eq("L")),
            chordotonal_rate_right=select(chordotonal & root_side.eq("R")),
            body_unmapped=select((haltere | proprio) & ~mapped_body),
        )

    @staticmethod
    def _flow_bins(
        body_ids: np.ndarray,
        table,
        adjacency: sparse.csr_matrix | None,
        *,
        flow_regions: int,
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
        empty = tuple(np.empty(0, dtype=np.int32) for _ in range(flow_regions))
        flow_rows = table[table["type"].fillna("").isin(["T4a", "T5a", "T4b", "T5b"])]
        flow_ids = flow_rows["bodyId"].to_numpy(dtype=np.int64)
        flow_nodes = _present_nodes(body_ids, flow_ids)
        if adjacency is None:
            return empty, empty, flow_nodes
        coordinate_rows = table.dropna(subset=["assignedOlHex1", "assignedOlHex2"])
        coordinate_ids = coordinate_rows["bodyId"].to_numpy(dtype=np.int64)
        coordinate_nodes = _present_nodes(body_ids, coordinate_ids)
        coordinates = np.full(len(body_ids), np.nan, dtype=np.float64)
        coordinates[coordinate_nodes] = coordinate_rows.set_index("bodyId").loc[
            body_ids[coordinate_nodes], "assignedOlHex1"
        ].to_numpy(dtype=np.float64)
        row_by_id = flow_rows.set_index("bodyId")
        inferred = np.full(len(flow_nodes), np.nan, dtype=np.float64)
        sides = np.empty(len(flow_nodes), dtype=object)
        directions = np.empty(len(flow_nodes), dtype=np.int8)
        for index, node in enumerate(flow_nodes):
            row = row_by_id.loc[int(body_ids[node])]
            sides[index] = row["somaSide"]
            # Subtypes a/b encode front-to-back/back-to-front motion in each eye.
            # The global panorama axis runs left-rear -> front -> right-rear, so
            # the same egocentric preferred direction has opposite image signs
            # in the left and right eyes.
            front_to_back = row["type"] in {"T4a", "T5a"}
            directions[index] = (
                1 if (front_to_back and sides[index] == "R") else
                1 if (not front_to_back and sides[index] == "L") else
                -1
            )
            start, end = adjacency.indptr[node : node + 2]
            sources = adjacency.indices[start:end]
            weights = np.abs(adjacency.data[start:end]).astype(np.float64)
            known = np.isfinite(coordinates[sources])
            if np.any(known):
                inferred[index] = np.average(coordinates[sources[known]], weights=weights[known])
        global_u = np.full(len(flow_nodes), np.nan, dtype=np.float64)
        for side in ("L", "R"):
            mask = (sides == side) & np.isfinite(inferred)
            if not np.any(mask):
                continue
            values = inferred[mask]
            span = float(values.max() - values.min())
            local = (values - values.min()) / span if span > 0 else np.full(len(values), 0.5)
            global_u[mask] = (1.0 - local) * 0.5 if side == "L" else 0.5 + local * 0.5
        mapped = np.isfinite(global_u)
        bins = np.zeros(len(flow_nodes), dtype=np.int32)
        bins[mapped] = np.clip(
            (global_u[mapped] * flow_regions).astype(np.int32), 0, flow_regions - 1
        )
        positive = tuple(
            flow_nodes[mapped & (directions > 0) & (bins == region)]
            for region in range(flow_regions)
        )
        negative = tuple(
            flow_nodes[mapped & (directions < 0) & (bins == region)]
            for region in range(flow_regions)
        )
        return positive, negative, flow_nodes[~mapped]

    def add_drive(
        self,
        drive: np.ndarray,
        frame: SensoryFrame,
        *,
        optic_flow: bool,
        body: bool,
        gains: SensoryGains | None = None,
    ) -> None:
        gains = gains or SensoryGains()
        if optic_flow:
            if frame.flow_regions and self.flow_positive_bins:
                if len(frame.flow_regions) != len(self.flow_positive_bins):
                    raise ValueError("regional flow count does not match anatomical bins")
                for value, positive_nodes, negative_nodes in zip(
                    frame.flow_regions,
                    self.flow_positive_bins,
                    self.flow_negative_bins,
                    strict=True,
                ):
                    drive[positive_nodes] += gains.optic_flow * max(value, 0.0)
                    drive[negative_nodes] += gains.optic_flow * max(-value, 0.0)
            else:
                positive = max(frame.flow, 0.0)
                negative = max(-frame.flow, 0.0)
                drive[self.flow_positive] += gains.optic_flow * positive
                drive[self.flow_negative] += gains.optic_flow * negative
        if body:
            signals = self._body_drives(frame, gains)
            for name, nodes in self._body_groups().items():
                drive[nodes] += signals[name]

    def _body_groups(self) -> dict[str, np.ndarray]:
        return {
            "haltere_rate_left": self.haltere_rate_left,
            "haltere_rate_right": self.haltere_rate_right,
            "haltere_acceleration_left": self.haltere_acceleration_left,
            "haltere_acceleration_right": self.haltere_acceleration_right,
            "campaniform_lateral_left": self.campaniform_lateral_left,
            "campaniform_lateral_right": self.campaniform_lateral_right,
            "campaniform_longitudinal_left": self.campaniform_longitudinal_left,
            "campaniform_longitudinal_right": self.campaniform_longitudinal_right,
            "chordotonal_rate_left": self.chordotonal_rate_left,
            "chordotonal_rate_right": self.chordotonal_rate_right,
        }

    @staticmethod
    def _body_drives(frame: SensoryFrame, gains: SensoryGains) -> dict[str, float]:
        yaw_rate = float(np.clip(frame.yaw_rate / 1.2, -1.0, 1.0))
        yaw_acceleration = float(np.clip(frame.yaw_acceleration / 3.0, -1.0, 1.0))
        steering_rate = float(np.clip(frame.steering_rate / 1.0, -1.0, 1.0))
        lateral = float(np.clip(frame.lateral_acceleration / 5.0, -1.0, 1.0))
        longitudinal = float(np.clip(abs(frame.longitudinal_acceleration) / 5.0, 0.0, 1.0))
        return {
            "haltere_rate_left": gains.haltere_yaw_rate * max(-yaw_rate, 0.0),
            "haltere_rate_right": gains.haltere_yaw_rate * max(yaw_rate, 0.0),
            "haltere_acceleration_left": (
                gains.haltere_yaw_acceleration * max(-yaw_acceleration, 0.0)
            ),
            "haltere_acceleration_right": (
                gains.haltere_yaw_acceleration * max(yaw_acceleration, 0.0)
            ),
            "campaniform_lateral_left": (
                gains.campaniform_lateral * max(-lateral, 0.0)
            ),
            "campaniform_lateral_right": (
                gains.campaniform_lateral * max(lateral, 0.0)
            ),
            "campaniform_longitudinal_left": (
                gains.campaniform_longitudinal * longitudinal
            ),
            "campaniform_longitudinal_right": (
                gains.campaniform_longitudinal * longitudinal
            ),
            "chordotonal_rate_left": (
                gains.chordotonal_steering_rate * max(-steering_rate, 0.0)
            ),
            "chordotonal_rate_right": (
                gains.chordotonal_steering_rate * max(steering_rate, 0.0)
            ),
        }

    def diagnostics(
        self,
        activity: np.ndarray,
        frame: SensoryFrame,
        *,
        optic_flow: bool,
        body: bool,
        gains: SensoryGains | None = None,
    ) -> dict:
        """Report direct-drive scale and resulting activity by sensory group."""
        gains = gains or SensoryGains()
        if optic_flow and frame.flow_regions and self.flow_positive_bins:
            if len(frame.flow_regions) != len(self.flow_positive_bins):
                raise ValueError("regional flow count does not match anatomical bins")
            positive_l1 = sum(
                gains.optic_flow * max(value, 0.0) * len(nodes)
                for value, nodes in zip(
                    frame.flow_regions, self.flow_positive_bins, strict=True
                )
            )
            negative_l1 = sum(
                gains.optic_flow * max(-value, 0.0) * len(nodes)
                for value, nodes in zip(
                    frame.flow_regions, self.flow_negative_bins, strict=True
                )
            )
        else:
            positive_l1 = (
                gains.optic_flow * max(frame.flow, 0.0) * len(self.flow_positive)
                if optic_flow
                else 0.0
            )
            negative_l1 = (
                gains.optic_flow * max(-frame.flow, 0.0) * len(self.flow_negative)
                if optic_flow
                else 0.0
            )
        drives = {
            "flow_positive": (
                positive_l1 / len(self.flow_positive) if len(self.flow_positive) else 0.0
            ),
            "flow_negative": (
                negative_l1 / len(self.flow_negative) if len(self.flow_negative) else 0.0
            ),
        }
        body_drives = self._body_drives(frame, gains) if body else {
            name: 0.0 for name in self._body_groups()
        }
        drives.update(body_drives)
        groups = {
            "flow_positive": self.flow_positive,
            "flow_negative": self.flow_negative,
            **self._body_groups(),
        }
        result = {}
        for name, nodes in groups.items():
            values = np.abs(activity[nodes])
            result[name] = {
                "neurons": int(len(nodes)),
                "direct_drive": float(drives[name]),
                "mean_abs_activity": float(values.mean()) if len(values) else 0.0,
                "p95_abs_activity": float(np.quantile(values, 0.95)) if len(values) else 0.0,
                "saturated_fraction": float(np.mean(values >= 0.95)) if len(values) else 0.0,
            }
        result["total_direct_drive_l1"] = float(
            positive_l1
            + negative_l1
            + sum(drives[name] * len(nodes) for name, nodes in self._body_groups().items())
        )
        return result

    def summary(self) -> dict:
        return {
            "T4_T5_horizontal_flow": int(len(self.flow_positive) + len(self.flow_negative)),
            "T4_T5_spatially_mapped": int(
                sum(map(len, self.flow_positive_bins)) + sum(map(len, self.flow_negative_bins))
            ),
            "T4_T5_spatially_unmapped": int(len(self.flow_unmapped)),
            "T4_T5_spatial_bin_counts": [
                int(len(positive) + len(negative))
                for positive, negative in zip(
                    self.flow_positive_bins, self.flow_negative_bins, strict=True
                )
            ],
            "haltere_yaw_rate": int(len(self.haltere_left) + len(self.haltere_right)),
            "ascending_proprioception": int(len(self.proprio_left) + len(self.proprio_right)),
            "haltere_by_side": [int(len(self.haltere_left)), int(len(self.haltere_right))],
            "proprioception_by_side": [int(len(self.proprio_left)), int(len(self.proprio_right))],
            "body_mapped_unique": int(
                len(np.unique(np.concatenate(list(self._body_groups().values()))))
            ),
            "body_unmapped": int(len(self.body_unmapped)),
            "body_group_counts": {
                name: int(len(nodes)) for name, nodes in self._body_groups().items()
            },
        }
