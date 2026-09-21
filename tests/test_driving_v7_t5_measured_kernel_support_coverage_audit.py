import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-measured-kernel-support-coverage-audit.json"


def test_support_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["stimulus_count"] == 120
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_author_filter_direction_is_visible_but_export_is_not_self_contained() -> None:
    semantics = json.loads(REPORT.read_text())["author_convolution_semantics"]
    assert semantics["stored_kernel_passed_to_lfilter_without_reversal"] is True
    assert semantics["causal_index_zero_interpretation_supported"] is True
    assert semantics["lfilter_explicitly_imported_in_exported_module"] is False
    assert semantics["exported_module_self_contained_execution_verified"] is False


def test_local_edge_trace_covers_peaks_but_less_than_half_of_kernel_mass() -> None:
    report = json.loads(REPORT.read_text())
    window = report["window"]
    assert window["kernel_length_samples"] == 499
    assert window["kernel_sample_interval_milliseconds"] == 10.0
    assert window["kernel_nominal_support_milliseconds"] == 4990.0
    assert window["kernel_maximum_discrete_lag_milliseconds"] == 4980.0
    assert window["post_baseline_trace_samples"] == 27
    assert window["post_baseline_trace_duration_milliseconds"] == 270.0
    assert window["maximum_kernel_lag_used_milliseconds"] == 260.0
    assert np.isclose(window["kernel_coefficient_fraction_used"], 27 / 499)
    expected = {
        "Tm1": (6, 60.0, 0.4361246593526668),
        "Tm2": (5, 50.0, 0.4082240394891599),
        "Tm4": (7, 70.0, 0.4683527541643402),
        "Tm9": (8, 80.0, 0.4716852908248215),
    }
    assert set(report["source_results"]) == set(expected)
    for source, values in expected.items():
        item = report["source_results"][source]
        assert item["population_kernel_absolute_peak_index"] == values[0]
        assert item["population_kernel_absolute_peak_lag_milliseconds"] == values[1]
        assert item["absolute_peak_within_scored_trace"] is True
        assert np.isclose(item["scored_prefix_L1_mass_fraction"], values[2])
        assert item["full_support_coverage_passed"] is False


def test_support_boundary_prevents_full_kernel_or_direction_claims() -> None:
    report = json.loads(REPORT.read_text())
    assert report["coverage_gate"][
        "all_population_absolute_peaks_within_scored_trace"
    ] is True
    assert report["coverage_gate"][
        "all_population_kernel_L1_mass_coverage_passed"
    ] is False
    assert report["early_support_negative_result_available"] is True
    assert report["full_kernel_support_evaluated"] is False
    assert report["full_support_negative_conclusion_authorized"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["direction_scoring_performed"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
