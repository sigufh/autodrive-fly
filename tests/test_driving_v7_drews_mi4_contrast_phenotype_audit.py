import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-drews-mi4-contrast-phenotype-audit.json"


def test_drews_contrast_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["pickle_security"]["restricted_unpickler_used"] is True
    assert report["pickle_security"]["arbitrary_repository_code_executed"] is False


def test_drews_f1_reproduction_uses_the_published_analysis_window() -> None:
    report = json.loads(REPORT.read_text())
    analysis = report["analysis_contract"]
    assert analysis["stimulus_motion_window_seconds"] == [0.0, 4.0]
    assert analysis["motion_window_right_open"] is True
    assert analysis["stimulus_frequency_hz"] == 1.0
    assert analysis["samples_per_trial_in_motion_window"] == 48
    assert analysis["trial_labels"] == [0, 1, 2]
    expected_counts = {
        "Mi1": (20, 5, 2520),
        "Tm3": (21, 8, 2646),
        "Mi4": (20, 13, 2520),
        "Mi9": (21, 9, 2646),
    }
    for source, counts in expected_counts.items():
        result = report["cell_type_results"][source]
        assert (
            result["ROI_count"],
            result["pseudonymous_fly_count"],
            result["condition_trial_group_count"],
        ) == counts


def test_mi4_qualitative_phenotype_is_robust_but_not_transferable() -> None:
    report = json.loads(REPORT.read_text())
    results = report["cell_type_results"]
    assert results["Mi4"]["median_fly_delta"] > 0
    assert results["Mi4"]["positive_fly_count"] == 12
    assert results["Mi4"]["leave_one_fly_out_median_min"] > 0
    for source in ("Mi1", "Tm3"):
        assert results[source]["median_fly_delta"] < 0
        assert results[source]["leave_one_fly_out_median_max"] < 0
    assert report["Mi4_tonic_weak_surround_suppression_qualitatively_reproduced"] is True
    assert report["independent_study_Mi4_calcium_phenotype_available"] is True
    assert report["preregistered_independent_dynamic_validation_available"] is False
    assert report["full_Mi4_temporal_kernel_identified"] is False
    assert report["experimental_Mi4_membrane_voltage_available"] is False
    assert report["C3_direct_recording_available"] is False
    assert report["recording_to_MaleCNS_body_crosswalk_found"] is False
    assert report["authorize_Mi4_source_kernel_transfer"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["boundary"][
        "retrospective_descriptive_reproduction_not_preregistered_validation"
    ] is True
