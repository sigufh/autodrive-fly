import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from fly_emotion.connectome.graph import load_graph
from fly_emotion.driving.v7 import V7VisualProbe
from fly_emotion.driving.v7_local_input_audit import (
    LOCAL_TYPES,
    OUTER_RADIUS,
    SITE_COUNT,
    BaselineHeldRetinaProbe,
    classify_column_coverage,
    local_masks,
    local_step,
    select_local_sites,
    select_receptor_masks,
    summarize_input_coverage,
)
from fly_emotion.driving.v7_retina_audit import infer_retinal_columns

ROOT = Path(__file__).parents[1]


def test_input_coverage_preserves_unknown_weights_and_absent_targets() -> None:
    matrix = sparse.csr_matrix([[2, 3, 5, 7], [0, 0, 0, 0], [4, 0, 0, 0]])
    labels = np.array(["covered", "uncovered", "missing_coordinate", "missing_side"])
    result = summarize_input_coverage(matrix, labels)
    assert result["total_synapse_weight"] == 21
    assert result["synapse_weight_by_coverage"] == {
        "covered": 6,
        "uncovered": 3,
        "missing_coordinate": 5,
        "missing_side": 7,
    }
    assert result["covered_weight_fraction"] == pytest.approx(6 / 21)
    assert result["targets_without_input"] == 1
    assert result["targets_with_all_weight_covered"] == 1
    assert result["targets_without_covered_input"] == 0
    empty = summarize_input_coverage(sparse.csr_matrix((3, 0), dtype=int), np.array([]))
    assert empty["targets_without_input"] == 3
    assert empty["covered_weight_fraction"] is None
    with pytest.raises(ValueError, match="declared coverage"):
        summarize_input_coverage(matrix, np.array(["covered"] * 3 + ["invalid"]))


def test_t4_input_coverage_retains_all_targets_and_conserves_raw_weights() -> None:
    report = json.loads((ROOT / "artifacts/v7-t4-input-coverage.json").read_text())
    assert report["protocol"]["neural_responses_used"] is False
    assert report["advance_to_central_complex"] is False
    targets = report["targets"]
    assert len(targets["body_ids"]) == len(set(targets["body_ids"])) == 6861
    assert sum(report["target_counts"].values()) == 6861
    types, sides = np.asarray(targets["types"]), np.asarray(targets["sides"])
    sources = targets["synapse_weights_by_source_and_coverage"]
    declared_weight = 0
    for population, groups in report["populations"].items():
        kind, eye = population.split("_")
        mask = (types == kind) & (sides == (-1 if eye == "L" else 1))
        for source, arrays in sources.items():
            row = groups[source]
            assert row["target_count"] == int(mask.sum())
            assert row["total_synapse_weight"] == sum(row["synapse_weight_by_coverage"].values())
            for label, values in arrays.items():
                assert len(values) == 6861
                assert (
                    int(np.asarray(values)[mask].sum()) == row["synapse_weight_by_coverage"][label]
                )
            declared_weight += row["total_synapse_weight"]
        for branch, members in report["branches"].items():
            assert groups[branch]["total_synapse_weight"] == sum(
                groups[x]["total_synapse_weight"] for x in members
            )
        for unknown in ("Tm3", "CT1"):
            assert (
                groups[unknown]["synapse_weight_by_coverage"]["missing_coordinate"]
                == groups[unknown]["total_synapse_weight"]
            )
    graph = load_graph(ROOT / "data/processed/malecns-v1.0", normalized=False)
    nodes = np.searchsorted(graph.body_ids, targets["body_ids"])
    assert declared_weight + report["omitted_from_declared_branches"]["synapse_weight"] == int(
        graph.adjacency[nodes].sum(dtype=np.int64)
    )
    for relative, expected in report["protocol"]["dependencies_sha256"].items():
        with (ROOT / relative).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected


def test_receptor_masks_match_count_side_and_are_disjoint() -> None:
    coordinates = np.array([[0, 0], [0, 0], [1, 0], [2, 0], [6, 0], [7, 0], [0, 0]])
    sides = np.array([-1, -1, -1, -1, -1, -1, 1])
    ids = np.array([70, 60, 50, 40, 30, 20, 10])
    masks = select_receptor_masks(coordinates, sides, ids, np.array([0, 0]), -1)
    assert masks["same_column"].tolist() == [0, 1]
    assert masks["nearest_other_columns"].tolist() == [2, 3]
    assert set(masks["far_other_columns"]) == {4, 5}
    for mask in masks.values():
        assert len(mask) == 2
        assert np.all(sides[mask] == -1)
    assert len(np.unique(np.concatenate(list(masks.values())))) == 6
    perm = np.array([3, 6, 4, 0, 5, 1, 2])
    reordered = select_receptor_masks(
        coordinates[perm], sides[perm], ids[perm], np.array([0, 0]), -1
    )
    for name in masks:
        assert set(ids[masks[name]]) == set(ids[perm][reordered[name]])


def test_receptor_masks_reject_uncovered_target_and_insufficient_controls() -> None:
    with pytest.raises(ValueError, match="same-column"):
        select_receptor_masks(
            np.array([[1, 1]]), np.array([-1]), np.array([1]), np.array([0, 0]), -1
        )
    with pytest.raises(ValueError, match="insufficient"):
        select_receptor_masks(
            np.array([[0, 0]]), np.array([-1]), np.array([1]), np.array([0, 0]), -1
        )


@pytest.mark.parametrize("backend", ["linear_luminance", "signed_frame_difference"])
def test_baseline_hold_changes_only_selected_external_receptor_drive(backend: str) -> None:
    probe = BaselineHeldRetinaProbe(ROOT, retinal_backend=backend)
    baseline = np.full(probe.retina.size, 0.5, dtype=np.float32)
    values = np.full(probe.retina.size, 0.8, dtype=np.float32)
    original = probe._retinal_code(values, baseline, baseline)
    original_baseline = probe._retinal_code(baseline, baseline, baseline)
    probe.held_receptor_positions = np.array([0, 2, 5])
    masked = probe._retinal_code(values, baseline, baseline)
    expected = original.copy()
    expected[[0, 2, 5]] = original_baseline[[0, 2, 5]]
    assert np.array_equal(masked, expected)
    assert np.array_equal(probe._retinal_code(baseline, baseline, baseline), original_baseline)
    probe.held_receptor_positions = np.empty(0, dtype=np.int32)
    assert np.array_equal(probe._retinal_code(values, baseline, baseline), original)


def test_coverage_classification_keeps_missing_separate_and_respects_eye() -> None:
    coordinates = np.array([[1, 2], [1, 2], [2, 3], [np.nan, np.nan], [1, 2]])
    sides = np.array([-1, 1, -1, -1, 0])
    receptors = np.array([[1, 2], [1, 2], [2, 3]])
    receptor_sides = np.array([-1, -1, 1])
    labels, counts = classify_column_coverage(coordinates, sides, receptors, receptor_sides)
    assert labels.tolist() == [
        "covered",
        "uncovered",
        "uncovered",
        "missing_coordinate",
        "missing_side",
    ]
    assert counts.tolist() == [2, 0, 0, -1, -1]


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


def test_mi9_coverage_groups_partition_cells_and_reconstruct_response_statistics() -> None:
    report = json.loads((ROOT / "artifacts/v7-local-input-audit.json").read_text())
    coverage = report["mi9_coverage"]
    assert report["protocol"]["coverage_uses_response_labels"] is False
    assert len(coverage["body_ids"]) == len(set(coverage["body_ids"])) == 1775
    sides = np.asarray(coverage["sides"])
    labels = np.asarray(coverage["labels"])
    for index, label in enumerate(labels):
        count = coverage["same_column_receptor_count"][index]
        assert (count is None) == (label in {"missing_coordinate", "missing_side"})
        if label == "covered":
            assert count > 0
        elif label == "uncovered":
            assert count == 0
    for backend in coverage["responses"].values():
        for eye, side in (("L", -1), ("R", 1)):
            groups = backend["groups"][eye]
            assert (
                sum(group["on"]["cells"] for name, group in groups.items() if name != "all")
                == groups["all"]["on"]["cells"]
            )
            for label, group in groups.items():
                mask = sides == side
                if label != "all":
                    mask &= labels == label
                for polarity in ("on", "off"):
                    peaks = np.asarray(backend["per_cell_responses"][polarity]["signed_peak"])[mask]
                    assert len(peaks) == group[polarity]["cells"]
                    assert np.isclose(
                        np.mean(peaks < -1e-6), group[polarity]["negative_peak_fraction"]
                    )
                    assert np.isclose(np.median(peaks), group[polarity]["median_signed_peak"])


def test_coverage_extension_reproduces_previous_all_mi9_response_statistics() -> None:
    report = json.loads((ROOT / "artifacts/v7-local-input-audit.json").read_text())
    previous = json.loads((ROOT / "artifacts/v7-temporal-input-audit.json").read_text())
    for backend, responses in report["mi9_coverage"]["responses"].items():
        for eye in ("L", "R"):
            assert (
                responses["groups"][eye]["all"]
                == previous["step_responses"][backend]["populations"][f"Mi9_{eye}"]
            )


def test_saved_mask_audit_binds_dependencies_and_matched_receptor_sets() -> None:
    report = json.loads((ROOT / "artifacts/v7-receptor-mask-audit.json").read_text())
    local = json.loads((ROOT / "artifacts/v7-local-input-audit.json").read_text())
    protocol = report["protocol"]
    assert protocol["advance_allowed"] is False
    assert protocol["sites_previously_observed"] is True
    assert report["advance_to_central_complex"] is False
    assert report["sites"] == local["sites"]
    for relative, expected in protocol["dependencies_sha256"].items():
        with (ROOT / relative).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    retina, assignments = infer_retinal_columns(ROOT)
    for backend, rows in report["responses"].items():
        assert len(rows) == 6
        for row in rows:
            masks = select_receptor_masks(
                assignments.coordinates,
                retina.side,
                retina.body_ids,
                np.asarray(row["optic_hex"]),
                -1 if row["eye"] == "L" else 1,
            )
            assert set(row["conditions"]) == {"intact", *masks}
            for name, values in row["conditions"].items():
                assert values["sham_matches_intact"] is True
                if name != "intact":
                    assert values["held_receptor_body_ids"] == retina.body_ids[masks[name]].tolist()
                    assert values["held_receptor_count"] == len(masks["same_column"])
                for polarity, response in values["responses"].items():
                    assert response["maximum_pre_step_difference"] == 0.0
                    trace = np.asarray(response["population_mean_trace"])[16:]
                    reference = np.asarray(
                        row["conditions"]["intact"]["responses"][polarity]["population_mean_trace"]
                    )[16:]
                    assert np.isclose(
                        response["mean_post_step_shift_vs_intact"], np.mean(trace - reference)
                    )
                    if name == "intact":
                        previous = next(
                            v
                            for v in local["responses"][backend]
                            if v["optic_hex"] == row["optic_hex"]
                            and v["eye"] == row["eye"]
                            and v["region"] == "whole_screen"
                            and v["polarity"] == polarity
                        )
                        assert (
                            response["population_mean_trace"]
                            == previous["populations"]["Mi9"]["population_mean_trace"]
                        )
