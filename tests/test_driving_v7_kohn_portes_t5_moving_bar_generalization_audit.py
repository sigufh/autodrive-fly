import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-t5-moving-bar-generalization-audit.json"


def test_moving_bar_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit_for_v7"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_static_coefficients_produce_gain_free_moving_bar_shape_evidence() -> None:
    report = json.loads(REPORT.read_text())
    assert len(report["condition_results"]) == 24
    assert report["condition_order"][0] == "saline:0.02:2.25"
    assert report["condition_order"][-1] == "OA:0.16:9"
    for state, expected in {
        "saline": (0.15869915062032694, 0.5104766152581159),
        "OA": (0.17245674320932294, 0.6757813700333978),
    }.items():
        summary = report["state_summaries"][state]
        assert summary["condition_count"] == 12
        assert summary["all_predicted_DSI_positive"] is True
        assert np.isclose(summary["mean_absolute_DSI_error"], expected[0])
        assert np.isclose(summary["predicted_target_DSI_correlation"], expected[1])
    assert report["gates"]["static_bar_coefficients_reused_without_refit"] is True
    assert report["gates"]["gain_free_predicted_DSI_available"] is True


def test_moving_bar_gain_fit_is_not_independent_validation_or_transfer() -> None:
    report = json.loads(REPORT.read_text())
    assert all(
        item["moving_bar_gain_fit_and_score_samples_disjoint"] is False
        for item in report["condition_results"]
    )
    assert report["gates"]["stable_target_cell_ids_available"] is False
    assert report["gates"]["independent_target_cell_holdout_available"] is False
    assert report["gates"]["absolute_physical_direction_mapping_verified"] is False
    assert report["gates"]["source_absolute_gain_preserved"] is False
    assert report["independent_moving_bar_validation_available"] is False
    assert report["authorize_moving_bar_generalization_transfer_to_v7"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
