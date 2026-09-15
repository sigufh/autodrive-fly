import hashlib
import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_t5_conductance_audit import CONFIG

ROOT = Path(__file__).parents[1]


def test_t5_conductance_config_freezes_non_advancing_split() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    assert config["repository"]["commit"] == "fe52053dda84d49a124e6c1f141dd461eba9630c"
    assert config["paper"]["recorded_cells"] == 17
    assert config["model_protocol"]["paper_random_initializations_per_cell"] == 1000
    assert config["model_protocol"]["script_attempts_per_seed"] == 10
    boundary = config["interface_boundary"]
    assert boundary["read_only"] is True
    assert boundary["fit_allowed"] is False
    assert boundary["preserve_native_ragged_time_vectors"] is True
    assert boundary["resample_to_1khz"] is False
    assert boundary["independent_cell_holdout"] is False


def test_saved_t5_conductance_audit_is_complete_and_hash_bound() -> None:
    report = json.loads((ROOT / "artifacts/v7-t5-conductance-audit.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fitting"] is False
    assert report["protocol"]["model_simulation"] is False
    assert report["repository"]["file_count"] == 38
    assert report["repository"]["total_file_bytes"] == 13_686_897
    assert len(report["repository"]["files"]) == 38
    assert len(report["cells"]) == 17
    assert all(cell["all_values_finite"] for cell in report["cells"])
    assert report["aggregate"]["training_width2_conditions"] == 635
    assert report["aggregate"]["training_width2_samples"] == 124_381
    assert report["aggregate"]["single_bar_conditions"] == 1398
    assert report["aggregate"]["moving_bar_conditions"] == 268
    assert report["aggregate"]["minimal_motion_conditions"] == 1633
    assert report["aggregate"]["moving_grating_conditions"] == 96
    assert report["aggregate"]["static_grating_conditions"] == 176
    assert report["subset_verification"]["spfr_conditions_found_in_all"] == 973
    assert (
        report["subset_verification"]["all_spfr_traces_byte_exact_with_condition_matched_all_trace"]
        is True
    )
    semantics = report["data_semantics"]
    assert semantics["response_quantity"] == "baseline_subtracted_membrane_voltage"
    assert semantics["native_sample_intervals_milliseconds"] == {"2.5": 1905, "5": 1666}
    split = report["split_contract"]
    assert split["same_cells_used_for_fit_and_prediction"] is True
    assert split["independent_cell_holdout"] is False
    assert split["untouched_final_test"] is False
    assert report["optimizer_reconciliation"]["paper_1000_start_selection_reproduced"] is False
    assert report["advance_to_T5_fit"] is False
    assert report["advance_to_visual_gate"] is False
    assert report["advance_to_central_complex"] is False
