import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-lplc2-phenotype.json"


def test_lplc2_phenotype_is_tuning_only_and_nonadvancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_ids"] == ["S1-T01", "S1-T02", "S1-T03"]
    assert report["protocol"]["stimulus_count"] == 6
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["final_evaluated"] is False
    assert report["strict_lplc2_local_radial_gate_passed"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["advance_to_navigation"] is False


def test_lplc2_phenotype_preserves_full_target_denominators() -> None:
    report = json.loads(REPORT.read_text())
    expected = {"LPLC2_L": 94, "LPLC2_R": 91}
    for condition in ("S1-T01", "S1-T02", "S1-T03"):
        for population, count in expected.items():
            score = report["per_condition_scores"][condition][population]
            assert score["cell_count"] == count
            assert score["valid_cell_count"] < 0.80 * count
            assert score["passed"] is False
    for population, count in expected.items():
        joint = report["target_joint_coverage"][population]
        assert joint["target_count"] == count
        assert joint["joint_valid_fraction"] < 0.80
