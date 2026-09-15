import hashlib
import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_looming_mechanism_audit import CONFIG

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-looming-mechanism-audit.json"


def test_looming_mechanism_contract_keeps_target_types_distinct() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    contracts = config["target_contracts"]
    assert set(contracts) == {"LPLC1", "LPLC2", "LC4"}
    assert len({item["output_role"] for item in contracts.values()}) == 3
    assert all(
        item["current_shared_looming_score_is_sufficient"] is False
        for item in contracts.values()
    )
    boundary = config["boundary"]
    assert boundary["literature_audit_only"] is True
    assert boundary["fit_parameters"] is False
    assert boundary["treat_calcium_as_membrane_voltage"] is False
    assert boundary["treat_connectivity_as_function"] is False
    assert boundary["evaluate_final"] is False


def test_saved_looming_mechanism_audit_exposes_missing_stimulus_axes() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    contracts = report["target_contracts"]
    assert contracts["LPLC1"]["coverage_fraction"] == 0.0
    assert contracts["LPLC2"]["coverage_fraction"] == 4 / 7
    assert contracts["LC4"]["coverage_fraction"] == 0.0
    assert "radius_velocity_ratio" in contracts["LPLC2"][
        "missing_from_current_stage1_battery"
    ]
    assert "angular_velocity" in contracts["LC4"][
        "missing_from_current_stage1_battery"
    ]
    assert report["advance_to_typed_looming_model"] is False
    assert report["advance_to_validation"] is False
    assert report["advance_to_central_complex"] is False
