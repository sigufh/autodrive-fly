import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-t5-frequency-tuning-audit.json"


def test_frequency_tuning_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_tm9_frequency_result_depends_on_the_summary_definition() -> None:
    report = json.loads(REPORT.read_text())
    saline = report["state_results"]["saline"]["sources"]
    oa = report["state_results"]["OA"]["sources"]
    saline_medians = {
        source: values["record_median_preferred_frequency_hz"]
        for source, values in saline.items()
    }
    assert saline_medians == {
        "Tm1": 0.95,
        "Tm2": 0.95,
        "Tm4": 0.675,
        "Tm9": 0.475,
    }
    assert {
        source: values["author_normalized_population_curve_peak_frequency_hz"]
        for source, values in saline.items()
    } == {"Tm1": 0.7, "Tm2": 0.8, "Tm4": 0.5, "Tm9": 0.8}
    oa_medians = {
        source: values["record_median_preferred_frequency_hz"]
        for source, values in oa.items()
    }
    assert oa_medians == {
        "Tm1": 2.325,
        "Tm2": 2.8,
        "Tm4": 1.45,
        "Tm9": 0.25,
    }
    assert report["gates"][
        "Tm9_record_median_lower_than_each_fast_source_in_both_states"
    ] is True
    assert report["gates"][
        "Tm9_author_population_curve_peak_lower_than_each_fast_source_in_both_states"
    ] is False
    assert report["gates"]["preferred_frequency_summary_invariant"] is False


def test_frequency_shape_does_not_authorize_source_transfer() -> None:
    report = json.loads(REPORT.read_text())
    assert report["relative_low_frequency_Tm9_shape_evidence_available"] is True
    assert report["gates"]["absolute_gain_available"] is False
    assert report["gates"]["CT1_frequency_tuning_available"] is False
    assert report["gates"]["independent_frequency_tuning_validation_available"] is False
    assert report["authorize_source_frequency_transfer_to_v7"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
