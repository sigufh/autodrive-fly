import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-temporal-shuffle-input-energy-audit.json"


def test_t4_shuffle_energy_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["stimulus_count"] == 8
    assert report["protocol"]["brain_substeps_per_frame"] == 4
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_t4_shuffle_is_retinal_energy_matched_and_source_near_matched() -> None:
    report = json.loads(REPORT.read_text())
    ratios = report["energy_ratios"]["temporal_shuffle_to_ordered"]
    assert report["frame_multiset_preserved_for_every_shuffle"] is True
    assert ratios["pixel_temporal_difference"] > 6.9
    assert np.isclose(ratios["R1_R6_drive"], 1.0)
    assert report["R1_R6_mean_absolute_energy_preserved"] is True
    assert report["all_source_state_energy_ratios_within_five_percent"] is True
    assert all(1.02 < value < 1.05 for value in ratios["centered_source_state"].values())


def test_energy_audit_retains_the_negative_temporal_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["original_shuffle_output_ratio"] > 1.9
    assert report["original_shuffle_output_gate_passed"] is False
    assert report["input_energy_mismatch_explains_original_output_failure"] is False
    assert report["original_temporal_identifiability_failure_retained"] is True
    assert report["authorize_new_energy_normalized_gate"] is False
    assert report["authorize_T4_functional_precheck"] is False
