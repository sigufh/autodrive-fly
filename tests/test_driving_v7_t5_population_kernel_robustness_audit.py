import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-population-kernel-robustness-audit.json"


def test_population_kernel_robustness_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["threshold_source"] == (
        "configs/driving-v7-fig3-source-kernel-robustness.yaml"
    )
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_three_sources_pass_but_tm2_fails_inherited_robustness_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["passing_sources"] == ["Tm1", "Tm4", "Tm9"]
    assert report["failing_sources"] == ["Tm2"]
    expected = {
        "Tm1": (0.9839966520222216, 0.870974175447607, 0.7498781432736034, 0.8415390174658499),
        "Tm2": (0.9749612228281109, 0.785694273562359, 0.6463991254279933, 0.76950266256901),
        "Tm4": (0.9955158014292839, 0.9388219062006047, 0.876058815401695, 0.9371672351651636),
        "Tm9": (0.9927554808138923, 0.8860370097481005, 0.8478525671341767, 0.9202411553154414),
    }
    for source, values in expected.items():
        item = report["source_results"][source]
        assert np.isclose(item["mean_median_kernel_correlation"], values[0])
        assert np.isclose(item["held_out_recording_id_vs_rest"]["summary"]["median"], values[1])
        assert np.isclose(item["held_out_recording_id_vs_rest"]["summary"]["minimum"], values[2])
        assert np.isclose(
            item["exhaustive_near_equal_recording_id_partitions"]["summary"]["p05"],
            values[3],
        )
    assert report["source_results"]["Tm2"]["gates"] == {
        "mean_median_kernel_correlation": True,
        "median_leave_one_recording_id_out_correlation": False,
        "minimum_leave_one_recording_id_out_correlation": True,
        "partition_correlation_p05": False,
    }
    assert report["all_population_kernel_robustness_gates_passed"] is False


def test_jackknife_stability_does_not_replace_recording_consistency() -> None:
    report = json.loads(REPORT.read_text())
    for item in report["source_results"].values():
        jackknife = item["population_mean_jackknife"]
        assert jackknife["minimum_correlation_to_full_mean"] > 0.97
        assert jackknife["maximum_absolute_peak_shift_samples"] <= 2
        assert jackknife["peak_sign_stable_in_every_fold"] is True
        assert jackknife["used_as_transfer_gate"] is False


def test_recording_ids_do_not_unlock_biological_transfer() -> None:
    report = json.loads(REPORT.read_text())
    assert report["recording_id_is_biological_individual_id"] is False
    assert report["independent_biological_validation_available"] is False
    assert report["authorize_population_kernel_transfer"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
