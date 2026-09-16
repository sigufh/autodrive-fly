import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-lplc-typed-screen.json"


def test_typed_lplc_screen_is_tuning_only_and_separates_mechanisms() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["condition_ids"] == ["LPLC-T01", "LPLC-T02", "LPLC-T03"]
    assert protocol["stimulus_count"] == 33
    assert protocol["shared_generic_looming_rule"] is False
    assert protocol["parameter_fit"] is False
    assert protocol["target_activity_injection"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["external_final_evaluated"] is False


def test_typed_lplc_mechanisms_fail_without_hiding_behind_mirror_pass() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "LPLC1_L": {
            "near_vs_miss",
            "back_to_front_vs_front_to_back",
            "object_vs_rotating_background",
        },
        "LPLC1_R": {
            "near_vs_miss",
            "back_to_front_vs_front_to_back",
            "object_vs_rotating_background",
        },
        "LPLC2_L": {"outward_vs_inward", "outward_vs_motion_free", "outward_vs_translation"},
        "LPLC2_R": {"outward_vs_inward", "outward_vs_motion_free", "outward_vs_translation"},
        "LC4_L": {"angular_velocity"},
        "LC4_R": {"angular_velocity"},
    }
    actual = {
        name: set(values) for name, values in report["population_consistency"].items()
    }
    assert actual == expected
    for mechanisms in report["population_consistency"].values():
        for result in mechanisms.values():
            assert result["passing_condition_count"] == 0
            assert result["required_condition_count"] == 3
            assert result["passed"] is False
    assert all(item["passed"] for item in report["mirror_summary"].values())
    assert report["typed_lplc_lc4_gates_passed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False
