import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-neural-corridor-dagger.json"


def test_dagger_is_tuning_only_and_pair_isolated() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    protocol = report["protocol"]
    assert protocol["tuning_only"] is True
    assert protocol["held_out_pairs_never_labeled_for_fit"] is True
    assert protocol["teacher_actions_injected"] is False
    assert protocol["calibration_evaluated"] is False
    assert protocol["external_final_evaluated"] is False
    for summary in report["round_summaries"]:
        for fold in summary["folds"]:
            assert not set(fold["training_seeds"]) & set(fold["held_out_seeds"])


def test_dagger_does_not_recover_closed_loop_generalization() -> None:
    report = json.loads(REPORT.read_text())
    summaries = report["round_summaries"]
    assert [item["aggregation_rounds"] for item in summaries] == [0, 1, 2]
    assert [item["held_out_success_count"] for item in summaries] == [0, 0, 0]
    assert [item["held_out_obstacles_passed"] for item in summaries] == [18, 20, 14]
    assert report["selected_rounds"] == 1
    assert report["cross_validation_passed"] is False
    assert report["advance_to_full_tuning"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_navigation_release"] is False
