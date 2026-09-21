import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-measured-kernel-substep-sensitivity.json"


def test_substep_sensitivity_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["stimulus_count_per_mode"] == 120
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_substep_sensitivity_binds_reference_and_standard_update_counts() -> None:
    report = json.loads(REPORT.read_text())
    timebase = report["timebase_sensitivity"]
    assert timebase["stimulus_frame_interval_milliseconds"] == 10.0
    assert timebase["kernel_sample_interval_milliseconds"] == 10.0
    assert timebase["tested_brain_updates_per_frame"] == [1, 4]
    assert timebase["reference_diagnostic_updates_per_frame"] == 1
    assert timebase["standard_offline_updates_per_frame"] == 4
    assert timebase["solver_interval_biologically_calibrated"] is False
    assert report["fixed_target_denominator"] == 6719
    assert report["valid_target_count"] == 6715


def test_no_candidate_passes_both_substep_resolutions() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "summed_filtered_source_centroid_projection": (
            5.1355203439692785,
            0.9998562782073322,
        ),
        "fast_pool_vs_Tm9_centroid_difference": (
            2.421487215968884,
            0.9277551543401545,
        ),
        "temporal_difference_filtered_Tm_pair_reichardt": (
            2.1529500436227584,
            0.9762604521295588,
        ),
    }
    assert set(report["candidate_results"]) == set(expected)
    for name, ratios in expected.items():
        result = report["candidate_results"][name]
        standard = result["by_brain_updates_per_frame"]["4"]
        assert np.isclose(standard["shuffle_to_ordered_residual_energy_ratio"], ratios[0])
        assert np.isclose(standard["static_to_ordered_energy_ratio"], ratios[1])
        assert standard["passed"] is False
        assert result["passed_every_update_count"] is False
    assert report["standard_substep_negative_result_reproduced"] is True
    assert report["cross_substep_temporal_identifiability_passed"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["direction_scoring_performed"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False


def test_substep_sensitivity_keeps_physical_transfer_closed() -> None:
    report = json.loads(REPORT.read_text())
    boundary = report["boundary"]
    assert boundary["numerical_substep_sensitivity_only"] is True
    assert boundary["does_not_calibrate_probe_solver_interval"] is True
    assert boundary["does_not_establish_external_recording_to_solver_alignment"] is True
    assert boundary["source_activity_driven_only_from_R1_R6_propagation"] is True
    assert boundary["subtype_and_direction_labels_not_read_by_dynamics"] is True
    assert report["authorize_physical_source_dynamics_transfer"] is False
