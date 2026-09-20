import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-t5-peak-latency-audit.json"


def test_peak_latency_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_saline_supports_relative_tm9_delay_but_oa_does_not_fully_separate() -> None:
    report = json.loads(REPORT.read_text())
    saline = report["state_results"]["saline"]
    oa = report["state_results"]["OA"]
    assert saline["pooled_fast_median_milliseconds"] == 50.0
    assert saline["Tm9_median_milliseconds"] == 80.0
    assert saline["Tm9_median_minus_each_fast_source_milliseconds"] == {
        "Tm1": 30.0,
        "Tm2": 30.0,
        "Tm4": 15.0,
    }
    assert oa["pooled_fast_median_milliseconds"] == 40.0
    assert oa["Tm9_median_milliseconds"] == 50.0
    assert oa["Tm9_median_minus_each_fast_source_milliseconds"]["Tm2"] == 0.0
    assert report["gates"][
        "saline_Tm9_median_slower_than_each_Tm1_Tm2_Tm4_median"
    ] is True
    assert report["gates"][
        "OA_Tm9_median_slower_than_each_Tm1_Tm2_Tm4_median"
    ] is False
    assert report["gates"]["Tm9_delay_ordering_state_invariant"] is False
    assert report["common_recording_id_state_deltas"]["Tm1"][
        "OA_minus_saline_peak_latency_milliseconds"
    ] == [-10.0, -30.0, -20.0, 0.0]


def test_descriptive_latency_does_not_authorize_v7_delay_transfer() -> None:
    report = json.loads(REPORT.read_text())
    assert report["relative_Tm9_delay_candidate_supported_in_saline"] is True
    assert report["relative_Tm9_delay_candidate_supported_across_states"] is False
    assert report["gates"]["CT1_peak_latency_available"] is False
    assert report["gates"]["v7_source_delay_substeps_biologically_calibrated"] is False
    assert report["gates"]["independent_peak_latency_validation_available"] is False
    assert report["authorize_source_delay_transfer_to_v7"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
