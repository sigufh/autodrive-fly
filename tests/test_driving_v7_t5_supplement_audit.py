import hashlib
import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_t5_supplement_audit import CONFIG

ROOT = Path(__file__).parents[1]


def test_t5_supplement_contract_forbids_inference_and_execution() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    assert len(config["expected_free_parameters"]) == 10
    assert len(config["expected_fixed_parameters"]) == 3
    boundary = config["audit_boundary"]
    assert boundary["execute_office_content"] is False
    assert boundary["extract_xml_text_only"] is True
    assert boundary["fit_parameters"] is False
    assert boundary["infer_missing_cell_parameters"] is False


def test_saved_t5_supplement_audit_has_bounds_but_no_fitted_vectors() -> None:
    report = json.loads((ROOT / "artifacts/v7-t5-supplement-audit.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    source = report["source"]
    assert source["member_size_bytes"] == 121_076
    assert source["member_sha256"] == (
        "b0933c4d99363b3807d41c9fdffa9277ece3ced9bc0f9c187abc38cfe780daa0"
    )
    assert source["bundle_sha256_is_identity_constraint"] is False
    contract = report["parameter_contract"]
    assert contract["free_parameter_count"] == 10
    assert contract["table_rows_including_header"] == 14
    assert contract["fitted_cell_parameter_values_present"] is False
    assert contract["fixed_parameters"] == {
        "excitatory_reversal_millivolts": 0.0,
        "inhibitory_reversal_millivolts": -74.0,
        "resting_potential_millivolts": -65.0,
    }
    replay = report["replay_status"]
    assert replay["zero_fit_replay_performed"] is False
    assert replay["measured_model_comparison_performed"] is False
    assert report["advance_to_T5_replay"] is False
    assert report["advance_to_T5_fit"] is False
    assert report["advance_to_visual_gate"] is False
