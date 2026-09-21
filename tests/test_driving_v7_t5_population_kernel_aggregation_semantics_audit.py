import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-population-kernel-aggregation-semantics-audit.json"


def test_aggregation_semantics_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_author_default_baseline_none_is_used_at_bounded_call_sites() -> None:
    report = json.loads(REPORT.read_text())
    semantics = report["author_function_semantics"]
    assert semantics == {
        "baseline_default_is_none": True,
        "start_baseline_uses_first_sample": True,
        "end_baseline_uses_last_100_sample_mean": True,
        "default_branch_uses_zero_offset": True,
        "population_mean_is_row_weighted": True,
    }
    sites = report["bounded_call_sites"]
    assert {name: item["call_count"] for name, item in sites.items()} == {
        "figure2": 8,
        "flash_analysis": 1,
        "figure5": 0,
        "figure6a_checkpoint": 0,
        "figure6b_checkpoint": 0,
    }
    assert report["call_summary"] == {
        "call_count": 9,
        "calls_with_explicit_baseline": 0,
        "all_observed_calls_use_default_baseline_none": True,
    }
    assert report["current_no_baseline_matches_author_default"] is True
    assert report["alternative_tail_baseline_variant_authorized"] is False


def test_recording_id_balancing_is_explicit_and_not_exact_author_row_mean() -> None:
    report = json.loads(REPORT.read_text())
    sources = report["source_results"]
    assert sources["Tm1"]["payload_row_count"] == 8
    assert sources["Tm1"]["unique_recording_id_count"] == 7
    assert sources["Tm1"]["duplicate_recording_id_row_count"] == 1
    assert sources["Tm1"]["author_row_weighted_equals_v7_equal_recording_id_weighted"] is False
    assert np.isclose(
        sources["Tm1"]["row_weighted_vs_equal_recording_id_correlation"],
        0.9995598341098436,
    )
    assert np.isclose(
        sources["Tm1"][
            "row_weighted_vs_equal_recording_id_maximum_absolute_difference"
        ],
        0.001330646526801052,
    )
    for source in ("Tm2", "Tm4", "Tm9"):
        assert sources[source]["duplicate_recording_id_row_count"] == 0
        assert sources[source]["author_row_weighted_equals_v7_equal_recording_id_weighted"] is True
    current = report["current_v7_semantics"]
    assert current["deliberate_recording_id_balancing"] is True
    assert current["matches_author_row_weighted_population_mean_exactly"] is False
    assert report["author_row_weighted_aggregation_exactly_reproduced"] is False


def test_aggregation_semantics_do_not_authorize_transfer() -> None:
    report = json.loads(REPORT.read_text())
    assert report["population_kernel_transfer_authorized"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"]["recording_id_is_not_biological_individual_id"] is True
