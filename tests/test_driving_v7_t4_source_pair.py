import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

import fly_emotion.driving.v7_t4_source_pair as source_pair
from fly_emotion.driving.v7_t4_source_pair import _pair_moments

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-pair.json"


def test_T4_source_pair_audit_is_preregistered_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["preregistration_commit"] == (
        "9581fe0"
    )
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_T4_source_pair_grid_and_denominators_are_preserved() -> None:
    report = json.loads(REPORT.read_text())
    assert report["fixed_T4_population_denominator"] == 6861
    assert report["full_source_valid_target_count"] == 6749
    assert list(report["pair_results"]) == [
        "Mi1_x_Mi4",
        "Mi1_x_C3",
        "Tm3_x_Mi4",
        "Tm3_x_C3",
    ]
    for result in report["pair_results"].values():
        assert 0 < result["valid_target_count"] <= 6749
        assert set(result["mode_results"]) == {
            "ordered",
            "temporal_shuffle",
            "static_sham",
        }
        assert all(result["retained_matrix_nonzero_count"].values())


def test_T4_source_pair_audit_never_authorizes_selection_or_downstream_stage() -> None:
    report = json.loads(REPORT.read_text())
    assert report["pair_evaluated"] is True
    assert report["descriptive_pair_decomposition_only"] is True
    assert report["any_pair_satisfies_all_eight_direction_and_polarity_populations"] is False
    assert report["authorize_source_pair_candidate"] is False
    assert report["authorize_source_removal"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False


def test_pair_mask_keeps_only_the_named_fast_and_delayed_source_types() -> None:
    node_types = np.asarray(["Mi1", "Tm3", "Mi4", "C3"], dtype=object)
    matrices = {
        f"{channel}_{moment}": sparse.csr_matrix(np.ones((2, 4)))
        for channel in ("fast", "delayed")
        for moment in ("mass", "x", "y")
    }
    moments = {
        "matrices": matrices,
        "valid": np.asarray([True, True]),
        "axes": np.ones((2, 2)),
    }
    selected, retained = _pair_moments(moments, node_types, "Tm3", "C3")
    for name, matrix in selected["matrices"].items():
        expected_column = 1 if name.startswith("fast_") else 3
        assert matrix.nnz == 2
        assert set(matrix.indices) == {expected_column}
        assert retained[name] == 2
    assert selected["valid"].tolist() == [True, True]


def test_unobserved_pair_outputs_fail_closed(tmp_path: Path, monkeypatch) -> None:
    config_path = Path("pair.yaml")
    config = {
        "pair_outputs_observed": False,
        "preregistration_evidence": "not-read.json",
        "preregistration_commit": "not-read",
        "preregistration_sha256": "not-read",
    }
    (tmp_path / config_path).write_text(yaml.safe_dump(config))
    monkeypatch.setattr(source_pair, "CONFIG", config_path)
    with pytest.raises(ValueError, match="outputs have not been authorized"):
        source_pair.evaluate_v7_t4_source_pair(tmp_path)
