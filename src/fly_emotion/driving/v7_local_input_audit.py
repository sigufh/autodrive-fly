from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from fly_emotion.driving.v7 import V7Contract, V7VisualProbe, VisualStimulus
from fly_emotion.driving.v7_retina_audit import infer_retinal_columns
from fly_emotion.driving.v7_temporal_audit import (
    ON_RESPONSE_SIGN,
    _step_trace,
    build_step_stimuli,
    summarize_step_response,
)

LOCAL_TYPES = ("Mi1", "Mi4", "Mi9", "C3")
SELECTION_SEED = "v7-local-input-20260915"
SITE_COUNT = 3
CENTRE_RADIUS = 1
OUTER_RADIUS = 3
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_local_input_audit.py")


def local_masks(width: int, height: int, x: int, y: int, side: int) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[:height, :width]
    eye = xx < width // 2 if side < 0 else xx >= width // 2
    distance_squared = (xx - x) ** 2 + (yy - y) ** 2
    return {
        "centre": eye & (distance_squared <= CENTRE_RADIUS**2),
        "annulus": eye
        & (distance_squared > CENTRE_RADIUS**2)
        & (distance_squared <= OUTER_RADIUS**2),
        "whole_eye": eye,
    }


def local_step(mask: np.ndarray, polarity: str) -> VisualStimulus:
    level = {"on": 0.8, "off": 0.2}[polarity]
    frames = np.full((48, *mask.shape), 0.5, dtype=np.float32)
    frames[16:, mask] = level
    return VisualStimulus(f"local_{polarity}", "local_step", polarity, "none", frames)


def select_local_sites(root: Path, probe: V7VisualProbe) -> tuple[list[dict], dict]:
    retina, assignments = infer_retinal_columns(root)
    if not np.array_equal(probe.retina.body_ids, retina.body_ids):
        raise ValueError("local audit requires the unchanged full retinal map")
    width, height = 48, 24
    x = np.rint(probe.retinal_u * (width - 1)).astype(np.int32)
    y = np.rint(probe.retinal_v * (height - 1)).astype(np.int32)
    columns = {
        side: set(map(tuple, assignments.coordinates[retina.side == side].tolist()))
        for side in (-1, 1)
    }
    annotations = feather.read_table(
        root / "data/raw/malecns-v1.0/body-annotations.feather",
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
        memory_map=True,
    ).to_pandas()
    available = annotations.loc[annotations["type"].isin(LOCAL_TYPES)].dropna(
        subset=["assignedOlHex1", "assignedOlHex2"]
    )
    by_column: dict[tuple, dict[str, list[int]]] = {}
    for row in available.itertuples(index=False):
        node = int(np.searchsorted(probe.graph.body_ids, row.bodyId))
        if node >= probe.graph.node_count or probe.graph.body_ids[node] != row.bodyId:
            continue
        key = (int(row.assignedOlHex1), int(row.assignedOlHex2), row.somaSide)
        by_column.setdefault(key, {}).setdefault(row.type, []).append(node)
    common = columns[-1] & columns[1]
    ordered = sorted(
        common,
        key=lambda coordinate: hashlib.sha256(
            f"{SELECTION_SEED}:{coordinate[0]}:{coordinate[1]}".encode()
        ).digest(),
    )
    eligible = []
    for coordinate in ordered:
        sides = []
        for side, eye in ((-1, "L"), (1, "R")):
            same_column = (retina.side == side) & np.all(
                assignments.coordinates == coordinate, axis=1
            )
            pixels = np.unique(np.column_stack((x[same_column], y[same_column])), axis=0)
            targets = by_column.get((*coordinate, eye), {})
            if len(pixels) != 1 or set(targets) != set(LOCAL_TYPES):
                break
            px, py = map(int, pixels[0])
            low_x, high_x = (0, width // 2 - 1) if side < 0 else (width // 2, width - 1)
            if not (
                low_x + OUTER_RADIUS <= px <= high_x - OUTER_RADIUS
                and OUTER_RADIUS <= py <= height - 1 - OUTER_RADIUS
            ):
                break
            sides.append(
                {
                    "eye": eye,
                    "side": side,
                    "pixel": [px, py],
                    "column_receptors": int(same_column.sum()),
                    "ambiguous_column_receptors": int(
                        np.count_nonzero(assignments.ambiguity[same_column] > 0)
                    ),
                    "target_nodes": {kind: sorted(nodes) for kind, nodes in targets.items()},
                    "target_body_ids": {
                        kind: probe.graph.body_ids[sorted(nodes)].tolist()
                        for kind, nodes in targets.items()
                    },
                }
            )
        if len(sides) == 2:
            eligible.append({"optic_hex": list(coordinate), "eyes": sides})
    selected = []
    for site in eligible:
        if any(
            np.linalg.norm(np.asarray(site["eyes"][index]["pixel"]) - other["eyes"][index]["pixel"])
            <= 2 * OUTER_RADIUS
            for other in selected
            for index in (0, 1)
        ):
            continue
        selected.append(site)
        if len(selected) == SITE_COUNT:
            break
    if len(selected) != SITE_COUNT:
        raise ValueError("insufficient anatomy-only nonoverlapping local sites")
    coverage = {}
    for kind in (*LOCAL_TYPES, "L3", "Tm3"):
        for eye in ("L", "R"):
            group = annotations[annotations["type"].eq(kind) & annotations["somaSide"].eq(eye)]
            coverage[f"{kind}_{eye}"] = {
                "cells": len(group),
                "coordinate_cells": int(
                    group[["assignedOlHex1", "assignedOlHex2"]].notna().all(axis=1).sum()
                ),
            }
    return selected, {
        "coordinate_coverage": coverage,
        "common_columns": len(common),
        "eligible_columns": len(eligible),
        "selected_columns": len(selected),
        "seed": SELECTION_SEED,
        "selection": "hash order; same-column targets in both eyes; pixel margin and separation",
        "functional_responses_used": False,
    }


def evaluate_v7_local_input_audit(root: Path) -> dict:
    contract = V7Contract.load(root)
    results = {}
    sites = None
    selection = None
    for backend in ("linear_luminance", "signed_frame_difference"):
        probe = V7VisualProbe(
            root,
            retinal_backend=backend,
            retinal_geometry="legacy_proxy_v2",
            dynamics_backend="typed_visual_subgraph_v1",
            brain_substeps=4,
            baseline_frames=32,
        )
        if sites is None:
            sites, selection = select_local_sites(root, probe)
        nodes = np.unique(
            np.concatenate(
                [
                    np.asarray(group, dtype=np.int32)
                    for site in sites
                    for eye in site["eyes"]
                    for group in eye["target_nodes"].values()
                ]
            )
        )
        if np.intersect1d(nodes, probe.retina.node_indices).size:
            raise ValueError("readout cells must not receive direct stimulus")
        uniform_stimuli = build_step_stimuli(48, 24)
        uniform = {item.polarity: _step_trace(probe, item, nodes) for item in uniform_stimuli}
        rows = []
        for site in sites:
            for eye in site["eyes"]:
                masks = local_masks(48, 24, *eye["pixel"], eye["side"])
                for region, mask in masks.items():
                    input_mask = probe._sample_retina(mask.astype(np.float32)).astype(bool)
                    for polarity in ("on", "off"):
                        stimulus = local_step(mask, polarity)
                        trace = _step_trace(probe, stimulus, nodes) - uniform["sham"]
                        groups = {}
                        for kind, target_nodes in eye["target_nodes"].items():
                            indices = np.searchsorted(nodes, target_nodes)
                            sign = (
                                ON_RESPONSE_SIGN.get(kind)
                                if polarity == "on" and region == "centre"
                                else None
                            )
                            groups[kind] = summarize_step_response(
                                trace[:, indices], onset=16, expected_sign=sign
                            )
                        rows.append(
                            {
                                "optic_hex": site["optic_hex"],
                                "eye": eye["eye"],
                                "pixel": eye["pixel"],
                                "region": region,
                                "polarity": polarity,
                                "stimulus_sha256": stimulus.sha256,
                                "stimulated_pixels": int(mask.sum()),
                                "stimulated_receptors": int(input_mask.sum()),
                                "other_eye_stimulated_receptors": int(
                                    np.count_nonzero(
                                        input_mask & (probe.retina.side != eye["side"])
                                    )
                                ),
                                "populations": groups,
                            }
                        )
                for polarity in ("on", "off"):
                    rows.append(
                        {
                            "optic_hex": site["optic_hex"],
                            "eye": eye["eye"],
                            "pixel": eye["pixel"],
                            "region": "whole_screen",
                            "polarity": polarity,
                            "stimulus_sha256": next(
                                item.sha256 for item in uniform_stimuli if item.polarity == polarity
                            ),
                            "stimulated_pixels": 48 * 24,
                            "stimulated_receptors": probe.retina.size,
                            "other_eye_stimulated_receptors": int(
                                np.count_nonzero(probe.retina.side != eye["side"])
                            ),
                            "populations": {
                                kind: summarize_step_response(
                                    (uniform[polarity] - uniform["sham"])[
                                        :, np.searchsorted(nodes, target_nodes)
                                    ],
                                    onset=16,
                                    expected_sign=None,
                                )
                                for kind, target_nodes in eye["target_nodes"].items()
                            },
                        }
                    )
        results[backend] = rows
    dependencies = [
        IMPLEMENTATION,
        Path("src/fly_emotion/driving/v7.py"),
        Path("src/fly_emotion/driving/v7_temporal_audit.py"),
        Path("src/fly_emotion/driving/v7_retina_audit.py"),
        Path("src/fly_emotion/driving/retina.py"),
        Path("src/fly_emotion/connectome/graph.py"),
        Path("src/fly_emotion/driving/engine.py"),
        Path("configs/driving-v7.yaml"),
        Path("data/raw/malecns-v1.0/body-annotations.feather"),
        Path("data/raw/malecns-v1.0/body-neurotransmitters.feather"),
        Path("data/processed/malecns-v1.0/body_ids.npy"),
        Path("data/processed/malecns-v1.0/adjacency_raw.npz"),
        Path("data/processed/malecns-v1.0/adjacency_target_norm.npz"),
        Path("data/processed/malecns-v1.0/retina_map.npz"),
    ]
    hashes = {}
    for path in dependencies:
        with (root / path).open("rb") as source:
            hashes[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "protocol": {
            "name": "v7-local-input-audit-v1",
            "exploratory": True,
            "advance_allowed": False,
            "v7_config_sha256": contract.sha256,
            "dependencies_sha256": hashes,
            "geometry": "legacy_proxy_v2",
            "image_width": 48,
            "image_height": 24,
            "dynamics": "typed_visual_subgraph_v1",
            "direct_input": "R1-R6 only",
            "target_direct_input_overlap": 0,
            "baseline_frames": 32,
            "pre_step_frames": 16,
            "post_step_frames": 32,
            "brain_substeps_per_frame": 4,
            "centre_radius_pixels": CENTRE_RADIUS,
            "outer_radius_pixels": OUTER_RADIUS,
            "reference": "same-frame whole-screen sham",
            "qualitative_polarity_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8891015/",
            "expected_sign_scope": "centre ON only; no polarity expectation assigned to annulus",
            "on_level": 0.8,
            "off_level": 0.2,
            "background_level": 0.5,
            "parameter_fitting": False,
            "driving_data_used": False,
            "selection": selection,
        },
        "sites": sites,
        "responses": results,
        "limitations": [
            "Three anatomy-selected columns are exploratory samples, not independent animals.",
            "Optic-hex and legacy camera coordinates are proxies, not measured receptive fields.",
            "Centre and annulus have unequal areas and stimulated receptor counts.",
            "whole_eye means camera hemifield; boundary receptors may belong to the other eye.",
            "Anatomy-covered site selection is not representative of all Mi9 cells.",
            "Latencies are simulation frames and late peaks can be window-censored.",
            "Tm3 and left-eye L3 lack optic-hex annotations and are excluded from local readout.",
            "This audit does not modify candidate parameters, visual gates or default runtime.",
        ],
        "advance_to_central_complex": False,
    }
