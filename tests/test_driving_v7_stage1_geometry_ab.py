import hashlib
import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_stage1_geometry_ab import CONFIG

ROOT = Path(__file__).parents[1]


def test_geometry_ab_contract_changes_only_previously_supported_geometry() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    assert config["candidate_geometry"] == {
        "reference": "nested_t4_axis_v1",
        "intervention": "reverse_each_eye_continuous_uv",
        "formula": "u'=0.5-u(L), u'=1.5-u(R), v'=1-v",
    }
    boundary = config["boundary"]
    assert boundary["anatomy_supported_before_this_development_screen"] is True
    assert boundary["physiological_labels_changed"] is False
    assert boundary["model_parameters_changed"] is False
    assert boundary["parameter_fitting"] is False
    assert boundary["development_only"] is True
    assert boundary["validation_evaluated"] is False
    assert boundary["ood_evaluated"] is False
    assert boundary["final_evaluated"] is False


def test_saved_geometry_ab_retains_strict_results_and_never_advances_directly() -> None:
    report = json.loads((ROOT / "artifacts/v7-stage1-geometry-ab.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["split"] == "development"
    assert report["protocol"]["parameter_fitting"] is False
    assert report["protocol"]["physiological_labels_changed"] is False
    assert report["protocol"]["validation_evaluated"] is False
    assert report["protocol"]["ood_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False
    assert report["stimulus_count"] == 172
    assert report["advance_to_validation"] is False
    assert report["advance_to_final"] is False
    assert report["advance_to_central_complex"] is False
    for result in report["results"].values():
        assert set(result["reference_pass_counts"]) == {
            "direction",
            "polarity",
            "looming",
            "mirror",
        }
        assert set(result["reversed_pass_counts"]) == {"direction", "polarity", "looming", "mirror"}
        scores = result["strict_scores"]
        assert len(scores["direction"]) == 16
        assert len(scores["polarity"]) == 16
        assert len(scores["looming"]) == 6
