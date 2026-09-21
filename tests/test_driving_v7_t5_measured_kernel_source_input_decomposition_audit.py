import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-measured-kernel-source-input-decomposition-audit.json"


def test_source_input_decomposition_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["functional_stimulus_evaluated"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_source_input_categories_are_exhaustive_and_mutually_exclusive() -> None:
    report = json.loads(REPORT.read_text())
    assert report["all_direct_input_edges_partitioned_exactly_once"] is True
    assert set(report["source_results"]) == {"Tm1", "Tm2", "Tm4", "Tm9"}
    expected_counts = {"Tm1": 1777, "Tm2": 1766, "Tm4": 1670, "Tm9": 1771}
    for source, result in report["source_results"].items():
        assert result["target_count"] == expected_counts[source]
        assert np.isclose(result["category_fraction_sum"], 1.0)
        assert result["has_recurrent_or_target_feedback_input"] is True


def test_tm_recurrence_and_t5_feedback_are_present_before_measured_fir() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "Tm1": (0.06015856247042119, 0.001090083357878112),
        "Tm2": (0.037499553052929104, 0.0021449748973974593),
        "Tm4": (0.16887416522181767, 0.00122188195270657),
        "Tm9": (0.2180908219952677, 0.010337831504127412),
    }
    for source, fractions in expected.items():
        categories = report["source_results"][source]["category_results"]
        assert np.isclose(
            categories["other_Tm_type"]["normalized_absolute_input_fraction"],
            fractions[0],
        )
        assert np.isclose(
            categories["T5_feedback"]["normalized_absolute_input_fraction"],
            fractions[1],
        )
    assert report["every_source_has_recurrent_or_target_feedback_input"] is True


def test_update_order_and_replacement_boundary_remain_explicit() -> None:
    report = json.loads(REPORT.read_text())
    assert set(report["update_ordering"].values()) == {True}
    assert report["feedforward_only_source_drive_available_from_current_trace"] is False
    assert report["measured_kernel_replacement_of_source_dynamics_evaluated"] is False
    assert report["source_dynamics_replacement_requires_explicit_intervention"] is True
    assert report["authorize_source_dynamics_replacement"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
