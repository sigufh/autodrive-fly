import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_emotion.driving.v7_coverage_response import (
    align_by_body_id,
    coverage_bins,
    summarize_direction,
)


def test_coverage_bins_keep_unknown_denominator_and_boundary_cases() -> None:
    labels, fraction, total = coverage_bins(
        {
            "covered": np.array([0, 0, 0, 1, 1, 1]),
            "uncovered": np.array([0, 0, 2, 2, 1, 0]),
            "missing_coordinate": np.array([0, 4, 1, 0, 0, 3]),
            "missing_side": np.zeros(6, dtype=int),
        }
    )
    assert labels.tolist() == [
        "absent",
        "unknown_only",
        "zero_covered",
        "below_half",
        "at_least_half",
        "below_half",
    ]
    np.testing.assert_allclose(fraction, [0, 0, 0, 1 / 3, 0.5, 0.25])
    np.testing.assert_array_equal(total, [0, 4, 3, 3, 2, 4])


def test_response_alignment_is_by_unique_body_id_not_array_position() -> None:
    np.testing.assert_array_equal(
        align_by_body_id(np.array([30, 10, 20]), np.array([10, 20, 30])), [1, 2, 0]
    )
    with pytest.raises(ValueError, match="unique"):
        align_by_body_id(np.array([10, 10]), np.array([10]))
    with pytest.raises(ValueError, match="missing"):
        align_by_body_id(np.array([10, 20]), np.array([15]))
    with pytest.raises(ValueError, match="missing"):
        align_by_body_id(np.array([10, 20]), np.array([30]))


def test_direction_summary_preserves_silence_and_empty_bins() -> None:
    preferred = np.array([2.0, 0, 0])
    opposite = np.array([1.0, 0, 0])
    off = np.array([0.0, 0, 0])
    summary = summarize_direction(preferred, opposite, off, np.array([True, True, True]))
    assert summary["cells"] == 3
    assert summary["median_direction_contrast"] == 0
    assert summary["direction_positive_fraction"] == pytest.approx(1 / 3)
    assert summary["fraction_direction_pair_above_1e_6"] == pytest.approx(1 / 3)
    empty = summarize_direction(preferred, opposite, off, np.zeros(3, dtype=bool))
    assert empty["cells"] == 0
    assert empty["median_direction_contrast"] is None


def test_saved_coverage_response_has_complete_id_partition_and_reproducible_summaries() -> None:
    root = Path(__file__).parents[1]
    report = json.loads((root / "artifacts/v7-t4-coverage-response.json").read_text())
    coverage = json.loads((root / "artifacts/v7-t4-input-coverage.json").read_text())
    assert report["targets"]["body_ids"] == coverage["targets"]["body_ids"]
    assert len(set(report["targets"]["body_ids"])) == 6861
    assert report["advance_to_central_complex"] is False
    for relative, expected in report["protocol"]["dependencies_sha256"].items():
        with (root / relative).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    populations = np.asarray(report["targets"]["populations"])
    for backend in report["results"].values():
        assert all(backend["replay_checks"].values())
        assert all(backend["stimulus_replay_checks"].values())
        assert sum(backend["per_cell"]["branched_equation_applied"]) == 6852
        p = np.asarray(backend["per_cell"]["preferred_on_response"])
        o = np.asarray(backend["per_cell"]["opposite_on_response"])
        off = np.asarray(backend["per_cell"]["preferred_off_response"])
        for group, strata in report["coverage_strata"].items():
            labels = np.asarray(strata["labels"])
            assert len(labels) == 6861
            for population, summaries in backend["by_source_coverage"][group].items():
                assert sum(row["cells"] for row in summaries.values()) == np.count_nonzero(
                    populations == population
                )
                for label, row in summaries.items():
                    mask = (populations == population) & (labels == label)
                    assert row == summarize_direction(p, o, off, mask)
        assert backend["all_cells"] == summarize_direction(p, o, off, np.ones(6861, dtype=bool))
