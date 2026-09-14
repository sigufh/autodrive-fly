import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7 import V7VisualProbe
from fly_emotion.driving.v7_local_input_audit import (
    LOCAL_TYPES,
    OUTER_RADIUS,
    SITE_COUNT,
    local_masks,
    local_step,
    select_local_sites,
)

ROOT = Path(__file__).parents[1]


def test_local_masks_are_disjoint_within_eye_and_no_clipped_ring() -> None:
    masks = local_masks(48, 24, 12, 12, -1)
    assert masks["centre"].sum() == 5
    assert masks["annulus"].sum() == 24
    assert not np.any(masks["centre"] & masks["annulus"])
    assert np.all((masks["centre"] | masks["annulus"]) <= masks["whole_eye"])
    assert not masks["whole_eye"][:, 24:].any()
    right = local_masks(48, 24, 35, 12, 1)
    for region, mask in masks.items():
        assert np.array_equal(mask[:, ::-1], right[region])


def test_local_stimuli_share_prehistory_and_touch_only_mask() -> None:
    mask = local_masks(48, 24, 12, 12, -1)["centre"]
    on = local_step(mask, "on")
    off = local_step(mask, "off")
    assert np.array_equal(on.frames[:16], off.frames[:16])
    assert np.all(on.frames[:, ~mask] == 0.5)
    assert np.all(off.frames[:, ~mask] == 0.5)
    assert on.direction == off.direction == "none"
    np.testing.assert_allclose(on.frames + off.frames, 1.0)


def test_anatomy_only_site_selection_is_deterministic_and_covered() -> None:
    probe = V7VisualProbe(
        ROOT, dynamics_backend="typed_visual_subgraph_v1", brain_substeps=4, baseline_frames=32
    )
    sites, metadata = select_local_sites(ROOT, probe)
    repeated, repeated_metadata = select_local_sites(ROOT, probe)
    assert sites == repeated
    assert metadata == repeated_metadata
    assert metadata["functional_responses_used"] is False
    assert len(sites) == SITE_COUNT
    for site in sites:
        assert [eye["eye"] for eye in site["eyes"]] == ["L", "R"]
        for eye in site["eyes"]:
            assert set(eye["target_nodes"]) == set(LOCAL_TYPES)
            assert eye["column_receptors"] > 0
            masks = local_masks(48, 24, *eye["pixel"], eye["side"])
            assert masks["annulus"].sum() == 24
            driven = probe._sample_retina(masks["centre"].astype(np.float32)) > 0
            assert driven.any()
            assert not driven[probe.retina.side != eye["side"]].any()
            for kind, nodes in eye["target_nodes"].items():
                assert np.all(probe.node_types[nodes] == kind)
                assert not np.intersect1d(nodes, probe.retina.node_indices).size
    assert metadata["coordinate_coverage"]["L3_L"]["coordinate_cells"] == 0
    assert metadata["coordinate_coverage"]["Tm3_R"]["coordinate_cells"] == 0
    for first, second in combinations(sites, 2):
        for index in (0, 1):
            assert (
                np.linalg.norm(
                    np.asarray(first["eyes"][index]["pixel"]) - second["eyes"][index]["pixel"]
                )
                > 2 * OUTER_RADIUS
            )


def test_saved_local_audit_covers_all_sites_conditions_and_frozen_dependencies() -> None:
    report = json.loads((ROOT / "artifacts/v7-local-input-audit.json").read_text())
    protocol = report["protocol"]
    assert protocol["advance_allowed"] is False
    assert protocol["selection"]["functional_responses_used"] is False
    assert report["advance_to_central_complex"] is False
    for relative, expected in protocol["dependencies_sha256"].items():
        with (ROOT / relative).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    expected_conditions = {
        (tuple(site["optic_hex"]), eye["eye"], region, polarity)
        for site in report["sites"]
        for eye in site["eyes"]
        for region in ("centre", "annulus", "whole_eye", "whole_screen")
        for polarity in ("on", "off")
    }
    for rows in report["responses"].values():
        assert len(rows) == len(expected_conditions)
        assert {
            (tuple(row["optic_hex"]), row["eye"], row["region"], row["polarity"]) for row in rows
        } == expected_conditions
        for row in rows:
            assert len(row["stimulus_sha256"]) == 64
            if row["region"] in ("centre", "annulus"):
                assert row["other_eye_stimulated_receptors"] == 0
            for summary in row["populations"].values():
                assert summary["maximum_pre_step_difference"] == 0
                assert len(summary["population_mean_trace"]) == 48
                if row["region"] != "centre":
                    assert summary["expected_peak_sign"] is None
