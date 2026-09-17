import hashlib
import json
from pathlib import Path

import pytest

from fly_emotion.driving.v7_parameter_matched_baselines import parameter_count

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-parameter-matched-baselines.json"


def test_parameter_count_formulas_match_declared_tensors() -> None:
    assert parameter_count("linear", 450, 3) == 1353
    assert parameter_count("one_hidden_layer_mlp", 450, 3, 3) == 1365
    assert parameter_count("single_layer_gru", 450, 3, 1) == 1365
    with pytest.raises(ValueError, match="positive width"):
        parameter_count("single_layer_gru", 450, 3, 0)
    with pytest.raises(ValueError, match="unknown baseline"):
        parameter_count("transformer", 450, 3, 1)


def test_parameter_matched_baseline_protocol_is_hash_bound_but_not_run() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["input_dimension"] == 450
    assert report["protocol"]["output_dimension"] == 3
    assert report["protocol"]["reference_trainable_parameter_budget"] == 1353
    assert report["protocol"]["preprocessing_shared_identically"] is True
    assert report["parameter_count_gate_passed"] is True
    assert report["architectures"]["linear"]["trainable_parameter_count"] == 1353
    assert report["architectures"]["one_hidden_layer_mlp"]["hidden_width"] == 3
    assert report["architectures"]["single_layer_gru"]["hidden_width"] == 1
    for result in report["architectures"].values():
        assert sum(result["parameter_tensors"].values()) == result[
            "trainable_parameter_count"
        ]
        assert result["relative_parameter_difference"] <= 0.01
    assert report["data_contract"]["tuning_conditions"] == [
        "NAV-T01",
        "NAV-T02",
        "NAV-T03",
    ]
    assert report["data_contract"]["calibration_access_authorized"] is False
    assert report["data_contract"]["external_final_access_authorized"] is False
    assert report["training_contract"]["maximum_optimizer_steps_per_fold"] == 2000
    assert report["training_contract"]["optimizer"] == "AdamW"
    assert report["training_contract"]["initialization_seeds"] == [
        20260917,
        20260918,
        20260919,
    ]
    assert report["training_contract"]["evaluation_interval_steps"] == 25
    assert report["training_contract"]["early_stopping_patience_evaluations"] == 20
    assert report["training_contract"]["calibration_used_for_early_stopping"] is False
    assert report["training_contract"]["external_final_used_for_early_stopping"] is False
    assert report["training_contract"]["models_trained"] is False
    assert report["evaluation_performed"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_final"] is False
