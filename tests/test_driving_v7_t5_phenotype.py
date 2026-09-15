import hashlib
import json
from pathlib import Path

import pytest
import yaml

from fly_emotion.driving.v7_t5_phenotype import CONFIG, direction_contrast

ROOT = Path(__file__).parents[1]


def test_direction_contrast_is_label_symmetric_and_zero_safe() -> None:
    assert direction_contrast(2.0, 6.0) == pytest.approx(2 / 3)
    assert direction_contrast(6.0, 2.0) == pytest.approx(-2 / 3)
    assert direction_contrast(0.0, 0.0) == 0.0


def test_t5_phenotype_contract_forbids_posthoc_pd_assignment() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    boundary = config["interpretation_boundary"]
    assert boundary["repository_maps_numeric_code_to_PD_ND"] is False
    assert boundary["infer_PD_from_larger_response"] is False
    assert boundary["fit_allowed"] is False
    assert boundary["model_scoring_allowed"] is False
    assert boundary["change_visual_gate"] is False


def test_saved_t5_phenotype_retains_pairs_controls_and_label_uncertainty() -> None:
    report = json.loads((ROOT / "artifacts/v7-t5-phenotype.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fitting"] is False
    assert report["protocol"]["model_simulation"] is False
    assert len(report["moving_bar_pairs"]) == 134
    assert {item["cell_id"] for item in report["moving_bar_pairs"]} == set(range(1, 18))
    summary = report["summary"]["all_pairs"]
    assert summary["pair_count"] == 134
    assert summary["positive_count"] == 130
    assert summary["negative_count"] == 4
    assert summary["median_contrast"] == pytest.approx(0.4922425318290533)
    assert report["summary"]["cells_with_positive_median"] == 17
    assert report["negative_controls"]["swap_direction_codes_median_contrast"] == pytest.approx(
        -summary["median_contrast"]
    )
    assert report["negative_controls"]["swap_negation_maximum_error"] == 0.0
    boundary = report["label_boundary"]
    assert boundary["biological_PD_code_assigned"] is None
    assert boundary["infer_PD_from_larger_response"] is False
    assert report["advance_to_T5_fit"] is False
    assert report["advance_to_visual_gate"] is False
    assert report["advance_to_central_complex"] is False
