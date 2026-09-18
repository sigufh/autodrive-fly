import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-native-direction-waveform-audit.json"


def test_t5_native_waveform_audit_is_hash_bound_and_zero_fit() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["repository_commit"] == (
        "fe52053dda84d49a124e6c1f141dd461eba9630c"
    )
    assert report["protocol"]["recording_file_count"] == 17
    assert report["protocol"]["paired_condition_count"] == 134
    assert report["protocol"]["parameter_fit"] is False


def test_t5_native_waveforms_fail_strict_leave_one_file_out_gate() -> None:
    report = json.loads(REPORT.read_text())
    assert report["condition_count"] == 12
    assert report["passing_condition_count"] == 1
    for name, condition in report["conditions"].items():
        assert condition["correlation_summary"]["fixed_denominator"] == condition[
            "recording_file_count"
        ]
        assert condition["passed"] is (name == "width=4:duration=20ms")
    assert report["conditions"]["width=4:duration=20ms"]["recording_file_count"] == 2
    assert report["all_native_direction_waveform_gates_passed"] is False
    assert report["authorize_T5_direction_template"] is False
    assert report["advance_to_T5_fit"] is False
    assert report["boundary"][
        "recording_file_is_not_claimed_as_independent_fly_identity"
    ] is True
    assert report["boundary"]["LPLC_and_vehicle_experiments_remain_frozen"] is True
