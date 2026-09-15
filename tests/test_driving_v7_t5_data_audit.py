import hashlib
import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_t5_data_audit import (
    CONFIG,
    _datacite_metadata,
    _plain,
    evidence_classification,
)

ROOT = Path(__file__).parents[1]


def test_html_description_is_normalized_without_executing_content() -> None:
    assert _plain("<b>T5</b><br>whole-cell &amp; code") == "T5 whole-cell & code"


def test_datacite_parser_requires_declared_scope_and_size() -> None:
    raw = json.dumps(
        {
            "data": {
                "attributes": {
                    "doi": "10.example/test",
                    "titles": [{"title": "T5 data"}],
                    "descriptions": [{"description": "singleBarStT5 raw recordings"}],
                    "sizes": ["123 Bytes"],
                    "rightsList": [{"rightsIdentifier": "cc-by-4.0"}],
                }
            }
        }
    ).encode()
    expected = {
        "doi": "10.example/test",
        "expected_size_bytes": 123,
        "required_description_phrases": ["singleBarStT5", "raw recordings"],
        "stimulus_scope": "test",
    }
    result = _datacite_metadata(raw, expected)
    assert result["size_bytes"] == 123
    assert result["license_identifiers"] == ["cc-by-4.0"]


def test_evidence_modalities_do_not_collapse_to_voltage() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    whole = evidence_classification("whole_cell_electrophysiology", config["allowed_roles"])
    asap = evidence_classification("two_photon_ASAP2f_voltage_imaging", config["allowed_roles"])
    calcium = evidence_classification("two_photon_calcium_imaging", config["allowed_roles"])
    topology = evidence_classification("connectome_structure", config["allowed_roles"])
    assert whole["supports_absolute_voltage_calibration"] is True
    assert asap["supports_relative_voltage_validation"] is True
    assert asap["supports_absolute_voltage_calibration"] is False
    assert calcium["supports_calcium_activity_validation"] is True
    assert calcium["supports_absolute_voltage_calibration"] is False
    assert topology["supports_topology_only"] is True


def test_saved_T5_data_audit_is_hash_bound_and_non_advancing() -> None:
    report = json.loads((ROOT / "artifacts/v7-t5-data-audit.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["observed_on"] == "2026-09-15"
    assert report["protocol"]["large_raw_files_downloaded"] is False
    candidate = report["whole_cell_candidate"]
    assert candidate["paper_doi"] == "10.1016/j.cub.2021.09.072"
    assert candidate["classification"]["supports_absolute_voltage_calibration"] is True
    assert candidate["candidate_discovered"] is True
    assert candidate["file_manifest_retrieved"] is False
    assert candidate["raw_data_verified"] is False
    assert candidate["usable_in_current_interface"] is False
    assert candidate["large_dataset_bytes_not_downloaded"] == 10_626_450_860
    assert candidate["datasets"]["figure_2_single_bar"]["size_bytes"] == 6_320_065_220
    assert candidate["datasets"]["figure_4_minimal_motion"]["size_bytes"] == 4_306_385_640
    assert candidate["datasets"]["unified_model"]["size_bytes"] == 3_630_019
    assert all(item["metadata_status_code"] == 200 for item in candidate["datasets"].values())
    status = report["interface_status"]
    assert status["current_interface_contains_T5_data"] is False
    assert status["absolute_T5_voltage_candidate_discovered"] is True
    assert status["absolute_T5_voltage_files_verified"] is False
    assert status["T5_fit_allowed"] is False
    other = report["other_modalities"]
    assert other["wienecke_2018_voltage_imaging"]["signal_unit"] == (
        "inverted_relative_fluorescence"
    )
    assert other["wienecke_2018_voltage_imaging"]["absolute_millivolts_possible"] is False
    assert other["ramos_2021_calcium"]["supports_calcium_activity_validation"] is True
    assert other["shinomiya_2025_connectome"]["supports_topology_only"] is True
    assert report["advance_to_T5_fit"] is False
    assert report["advance_to_visual_gate"] is False
    assert report["advance_to_central_complex"] is False
