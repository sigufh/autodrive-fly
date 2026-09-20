import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-direction-code-provenance-audit.json"


def test_direction_code_provenance_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_figure4_codes_are_relative_PD_ND_not_absolute_direction() -> None:
    evidence = json.loads(REPORT.read_text())["figure4_moving_bar_direction_code"]
    assert evidence["direction_codes_present"] == [0, 1]
    assert evidence["cells_with_both_codes"] == 17
    assert evidence["relative_mapping"] == {"0": "ND", "1": "PD"}
    assert evidence["relative_PD_ND_mapping_verified"] is True
    assert evidence["record_level_native_coordinate_motion_mapping_verified"] is True
    assert evidence["native_coordinate_mapping"] == {
        "0": "decreasing_receptive_field_position_coordinate",
        "1": "increasing_receptive_field_position_coordinate",
    }
    assert evidence["moving_bar_pair_count"] == 134
    assert len(evidence["per_record_direction_code_counts"]) == 17
    assert evidence["absolute_screen_or_body_motion_mapping_verified"] is False
    assert evidence["absolute_direction_phrase_hits_in_extracted_analysis_code"] == []


def test_kohn_portes_records_have_no_direction_code_or_nested_protocol() -> None:
    records = json.loads(REPORT.read_text())["kohn_portes_drifting_grating_records"]
    assert records["record_count"] == 26
    assert records["file_count"] == 8
    assert records["direction_like_fields"] == []
    assert records["nested_protocol_metadata_fields"] == []
    assert records["selected_field_values"]["direction_mb"] == {
        "present_count": 0,
        "missing_count": 26,
    }
    assert records["selected_field_values"]["stimulus_name"] == {"driftinggrating_05hz": 26}
    assert records["selected_field_values"]["wn_orientation"] == {"null": 26}
    assert records["all_historical_pickle_paths_have_one_blob_version"] is True
    assert all(
        item["commit_presence_count"] == 10 and item["unique_blob_count"] == 1
        for item in records["pickle_history"].values()
    )


def test_generator_capability_does_not_authorize_imputation_or_transfer() -> None:
    report = json.loads(REPORT.read_text())
    generator = report["motyxia2_generator_and_history"]
    assert generator["generator_direction_parameter_unit"] == "degrees"
    assert generator["generator_default_direction_degrees"] == 0.0
    assert generator["direction_is_passed_to_grating_generator"] is True
    assert generator["exact_token_path_hits"] == {
        "direction_mb": [],
        "driftinggrating_05hz": [],
        "jko14": [],
        "motyxia_log_ephys_rig2": [],
    }
    assert generator["generator_default_assigned_to_records"] is False
    assert report["direction_code_provenance_complete_for_kohn_portes"] is False
    assert report["authorize_kohn_portes_direction_conditioning"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["boundary"]["figure4_and_kohn_portes_are_distinct_datasets"] is True
