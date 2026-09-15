import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_ephys_audit import (
    CONFIG,
    _literal_assignments,
    _npy_header,
    _shift,
)

ROOT = Path(__file__).parents[1]


def test_shift_fills_boundaries_instead_of_wrapping() -> None:
    values = np.arange(6, dtype=float)
    np.testing.assert_array_equal(_shift(values, 2), [0, 0, 0, 1, 2, 3])
    np.testing.assert_array_equal(_shift(values, -2), [2, 3, 4, 5, 5, 5])
    np.testing.assert_array_equal(_shift(values, 0), values)


def test_literal_assignment_parser_does_not_execute_notebook_code() -> None:
    source = "safe = 3\nunsafe = dangerous()\nvalues = [1, 2]\n"
    assert _literal_assignments(source, {"safe", "unsafe", "values"}) == {
        "safe": 3,
        "values": [1, 2],
    }


def test_npy_header_detects_object_arrays_without_loading_pickle(tmp_path: Path) -> None:
    numeric = tmp_path / "numeric.npy"
    objects = tmp_path / "objects.npy"
    np.save(numeric, np.zeros((2, 3), dtype=np.float64))
    np.save(objects, np.asarray({"unsafe": object()}, dtype=object))
    assert _npy_header(numeric)["contains_python_objects"] is False
    header = _npy_header(objects)
    assert header["contains_python_objects"] is True
    assert header["shape"] == []


def test_frozen_source_manifest_has_unique_ids_and_complete_fig3_set() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    files = config["files"]
    assert len({item["id"] for item in files.values()}) == len(files)
    assert config["dataset"]["license"] == "CC BY 4.0"
    assert config["dataset"]["release_date"] == "2021-12-14"
    assert config["paper_protocol"]["stimuli"] == ["on", "off"]
    assert {f"fig3_{name}.npy" for name in ("Mi9", "Tm3", "Mi1", "Mi4", "C3")} <= set(files)
    assert {"fig3_T4v.npy", "fig3_T4r.npy", "fig3_led.npy", "fig3.ipynb"} <= set(files)
    for item in files.values():
        assert len(item["md5"]) == 32
        assert len(item["sha256"]) == 64
        assert item["size"] > 0


def test_saved_ephys_audit_is_hash_bound_and_reconstructible() -> None:
    report = json.loads((ROOT / "artifacts/v7-electrophysiology-audit.json").read_text())
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    assert (
        report["protocol"]["config_sha256"]
        == hashlib.sha256((ROOT / CONFIG).read_bytes()).hexdigest()
    )
    assert (
        report["protocol"]["implementation_sha256"]
        == hashlib.sha256(
            (ROOT / "src/fly_emotion/driving/v7_ephys_audit.py").read_bytes()
        ).hexdigest()
    )
    assert report["protocol"]["raw_files_committed"] is False
    assert report["protocol"]["parameter_fitting_performed"] is False
    assert report["advance_to_parameter_fit"] is False
    assert report["advance_to_central_complex"] is False
    assert set(report["verified_files"]) == set(config["files"])
    for name, item in report["verified_files"].items():
        expected = config["files"][name]
        assert item["id"] == expected["id"]
        assert item["bytes"] == expected["size"]
        assert item["md5"] == expected["md5"]
        assert item["sha256"] == expected["sha256"]
    assert report["safe_loading"]["fig1d_object_array_loaded"] is False
    assert report["safe_loading"]["replay_arrays_loaded_with_allow_pickle_false"] is True
    assert report["safe_loading"]["T4_resistance_array_used_in_replay"] is False
    assert report["array_headers"]["fig1d_receptive_fields.npy"]["contains_python_objects"]
    assert report["array_headers"]["fig1e_receptive_fields.npy"]["shape"] == [17, 17, 210, 46]
    replay = report["paper_model_replay"]
    assert replay["input_cells"] == {"Mi9": 29, "Tm3": 12, "Mi1": 24, "Mi4": 19, "C3": 16}
    assert replay["T4_cells"] == 33
    assert replay["direction_synthesis"]["shift_samples"] == 160
    assert report["paper_protocol"]["actual_source_array_order"] == [
        "Mi9",
        "Tm3",
        "Mi1",
        "Mi4",
        "C3",
    ]
    assert report["paper_protocol"]["onoff_comment_disagrees_with_array_order"] is True
    assert replay["display_window_milliseconds"] == [2500, 5500]
    assert set(replay["conditions"]) == {"on_pd", "on_nd", "off_pd", "off_nd"}
    assert replay["led_half_intensity_crossing_milliseconds"] == {
        "on_pd": 4004,
        "on_nd": 4005,
        "off_pd": 3998,
        "off_nd": 3998,
    }
    assert 1.7 < replay["pooled"]["rmse_millivolts"] < 1.8
    assert 0.85 < replay["pooled"]["pearson_correlation"] < 0.86
    assert replay["direction_peak_difference"]["on"]["observed_peak_pd_minus_nd_millivolts"] > 10
    assert replay["direction_peak_difference"]["off"]["predicted_peak_pd_minus_nd_millivolts"] < 1
    provenance = report["parameter_provenance"]
    assert provenance["current_config_parameter_values_match"] is True
    assert provenance["optimizer_code_present_in_fig3_notebook"] is False
    assert provenance["replay_is_independent_validation"] is False
    assert report["current_v7_model_differences"]["direct_fig3_reproduction"] is False
