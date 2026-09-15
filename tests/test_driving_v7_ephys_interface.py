import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from fly_emotion.driving.v7_ephys_interface import (
    CONFIG,
    NativeTimedTrace,
    TimedArray,
    _array_summary,
    load_offline_bundle,
)

ROOT = Path(__file__).parents[1]


def test_timed_array_has_explicit_time_only_when_axis_exists() -> None:
    timed = TimedArray(np.zeros((2, 4)), ("cell", "time"), 1.0, "training")
    np.testing.assert_array_equal(timed.time_milliseconds, [0, 1, 2, 3])
    untimed = TimedArray(np.zeros((2, 36)), ("cell", "direction"), None, "validation")
    assert untimed.time_milliseconds is None
    summary = _array_summary(untimed)
    assert summary["time_start_milliseconds"] is None
    assert summary["time_end_milliseconds"] is None


def test_offline_bundle_rejects_non_native_timebase(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen at 1 ms"):
        load_offline_bundle(tmp_path, 2.0)


def test_native_t5_trace_preserves_ragged_time_and_is_read_only() -> None:
    trace = NativeTimedTrace(
        np.array([0.0, 1.0, 0.5]),
        np.array([-20.0, -15.0, -10.0]),
        "within_cell_condition_generalization",
    )
    assert trace.quantity == "baseline_subtracted_membrane_voltage_millivolts"
    assert trace.values.flags.writeable is False
    assert trace.time_milliseconds.flags.writeable is False
    with pytest.raises(ValueError, match="equal lengths"):
        NativeTimedTrace(np.zeros(2), np.zeros(3), "invalid")


def test_split_contract_has_no_final_test_or_verified_cell_holdout() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    split = config["split_contract"]
    assert split["fit_allowed"] is False
    assert split["t5_data_available"] is True
    assert split["t5_same_cells_used_for_fit_and_prediction"] is True
    assert split["t5_independent_cell_holdout"] is False
    assert split["t5_resampled_to_1khz"] is False
    assert split["fig5_is_independent_stimulus_condition"] is True
    assert split["fig3_fig5_cell_disjointness_verified"] is False
    assert split["fig5_is_independent_input_dataset"] is False
    assert split["final_test_available"] is False
    assert config["dataset_roles"]["final_held_out"] == []


def test_saved_interface_is_hash_bound_native_1khz_and_runtime_isolated() -> None:
    report = json.loads((ROOT / "artifacts/v7-ephys-interface.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fitting"] is False
    assert report["protocol"]["runtime_imports_interface"] is False
    assert report["protocol"]["raw_files_committed"] is False
    assert report["available_final_test"] is False
    assert report["advance_to_parameter_fit"] is False
    assert report["advance_to_central_complex"] is False
    boundary = report["interface_boundary"]
    assert boundary["preserve_native_1khz_samples"] is True
    assert boundary["permit_resampling"] is False
    assert boundary["connect_to_v7_runtime"] is False
    assert boundary["convert_millivolts_to_normalized_drive"] is False
    assert boundary["inject_target_activity"] is False
    arrays = report["arrays"]
    assert arrays["fig3_T4_voltage"]["shape"] == [33, 2, 2, 8000]
    assert arrays["fig3_T4_voltage"]["sample_interval_milliseconds"] == 1.0
    assert arrays["fig3_T4_voltage"]["time_end_milliseconds"] == 7999.0
    assert arrays["fig3_T4_voltage"]["role"] == "training_reproduction"
    assert arrays["fig5_groups"]["gfp"]["delta_voltage"]["shape"] == [25, 36]
    assert arrays["fig5_groups"]["gfp"]["delta_voltage"]["sample_interval_milliseconds"] is None
    assert arrays["fig5_groups"]["gfp"]["delta_voltage"]["role"] == (
        "external_condition_validation"
    )
    assert len(report["verified_files"]) == 13
    t5 = report["t5_conductance"]
    assert t5["cell_count"] == 17
    assert t5["fit_allowed"] is False
    assert t5["fit_trace_count"] == 635
    assert t5["condition_trace_counts"] == {
        "single_bar": 1398,
        "moving_bar": 268,
        "minimal_motion": 1633,
        "moving_grating": 96,
        "static_grating": 176,
    }
    assert t5["native_sample_intervals_milliseconds"] == {"2.5": 1905, "5": 1666}
    assert t5["spfr_is_exact_subset_of_all"] is True
    assert all(item["all_finite"] for item in arrays["fig3_inputs"].values())
    for path in (
        ROOT / "src/fly_emotion/driving/engine.py",
        ROOT / "apps/api/src/fly_emotion_api/main.py",
    ):
        assert "v7_ephys_interface" not in path.read_text(encoding="utf-8")
