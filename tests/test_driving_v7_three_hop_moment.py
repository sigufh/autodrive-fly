import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-three-hop-moment.json"


def test_three_hop_moment_is_read_only_and_uses_known_sources() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_ids"] == ["S1-T01", "S1-T02", "S1-T03"]
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["runtime_modified"] is False
    assert report["path_summary"]["T4"]["reachable_target_fraction"] > 0.99
    assert report["path_summary"]["T5"]["reachable_target_fraction"] > 0.99
    assert report["interpretation"] == {
        "diagnostic_output": "target-specific two-dimensional path-weighted optical flow",
        "subtype_label_used_for_final_projection": True,
        "subtype_label_used_for_activity_generation": False,
        "may_validate_target_cell_direction_selectivity": False,
        "main_score_is_not_a_T4_T5_scalar_activity_response": True,
    }


def test_three_hop_main_pass_is_rejected_by_temporal_shuffle() -> None:
    report = json.loads(REPORT.read_text())
    assert report["bilateral_direction_populations"] == [
        "T4a",
        "T4b",
        "T4c",
        "T4d",
        "T5a",
        "T5b",
        "T5c",
        "T5d",
    ]
    shuffled = report["controls"]["temporal_shuffle"]
    assert shuffled["direction_pass_count"] == 16
    assert shuffled["polarity_pass_count"] == 16
    assert shuffled["passed_as_failure_control"] is False
    assert report["controls"]["static_sham"]["passed_as_failure_control"] is True
    assert report["controls"]["coordinate_shuffle"]["evaluated"] is False
    assert report["controls"]["direction_reversal"]["evaluated"] is False
    assert report["temporal_shuffle_control_passed"] is False
    attenuation = report["temporal_shuffle_energy_attenuation"]
    assert attenuation["maximum_allowed_ratio"] == 0.5
    assert attenuation["attenuated_population_count"] == 0
    assert attenuation["population_count"] == 16
    assert attenuation["all_populations_attenuated"] is False
    assert (
        min(attenuation["shuffle_to_ordered_mean_absolute_energy_ratio_by_population"].values())
        > 3.0
    )
    assert attenuation["diagnostic_only_does_not_authorize_candidate"] is True
    assert report["strict_three_hop_gates_passed"] is False
    assert report["advance_to_target_dynamics"] is False
    assert report["advance_to_calibration"] is False
