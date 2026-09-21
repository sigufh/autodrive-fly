import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-increment-order-control-replication.json"


def test_increment_order_replication_is_preregistered_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["preregistration_commit"] == (
        "7a239d602f6b589bfc1fe516176b8ac512145f5a"
    )
    assert report["protocol"]["preregistration_sha256"] == (
        "2f2e5b9d23b3c18a5416c0047881f221d04d97e2805f7e4009d14d1a2a6fb415"
    )
    assert report["protocol"]["discovery_condition_excluded"] is True
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_replication_input_contract_passes_every_condition_and_update() -> None:
    report = json.loads(REPORT.read_text())
    assert list(report["condition_results"]) == ["S1-T02", "S1-T03"]
    assert report["input_validity_passed_every_condition_and_update"] is True
    expected_lengths = {"S1-T02": 14, "S1-T03": 9}
    for condition, result in report["condition_results"].items():
        assert result["role"] == "tuning"
        assert result["stimulus_count"] == 120
        for update, values in result["by_brain_updates_per_frame"].items():
            assert update in {"1", "4"}
            assert len(values["increment_order"]) == expected_lengths[condition]
            assert all(values["input_gates"].values())
            assert np.allclose(
                list(values["R1_R6_mean_absolute_energy_ratio_summary"].values()),
                1.0,
                atol=1e-12,
            )
            assert all(
                0.998 < ratio < 1.001
                for ratio in values["lamina_only_source_energy_ratio"].values()
            )


def test_no_candidate_passes_the_preregistered_replication_gate() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "S1-T02": {
            "1": [1.00284398318869, 1.0892274724774895, 1.2957417076195203],
            "4": [1.0060882227106522, 1.084717718691444, 1.2366253578925497],
        },
        "S1-T03": {
            "1": [1.0016376982747357, 1.066355733716158, 1.2714122216092767],
            "4": [1.0036064852216553, 1.0688790749446417, 1.2259874086367528],
        },
    }
    for condition, by_update in expected.items():
        for update, ratios in by_update.items():
            candidates = report["condition_results"][condition][
                "by_brain_updates_per_frame"
            ][update]["candidate_output_results"]
            assert np.allclose(
                [
                    value["shuffle_to_ordered_residual_energy_ratio"]
                    for value in candidates.values()
                ],
                ratios,
            )
            assert all(value["passed"] is False for value in candidates.values())
    assert report["candidate_replication_passed_every_condition_and_update"] == {
        "summed_filtered_source_centroid_projection": False,
        "fast_pool_vs_Tm9_centroid_difference": False,
        "temporal_difference_filtered_Tm_pair_reichardt": False,
    }
    assert report["same_candidate_passed_every_condition_and_update"] is False
    assert report["replication_gate_passed"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["direction_scoring_performed"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
