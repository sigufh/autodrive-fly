import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-descending-path-audit.json"


def test_descending_path_audit_is_structure_only_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["search"] == "sparse_directed_breadth_first_search"
    assert report["protocol"]["matrix_power_used"] is False
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["boundary"]["shortest_paths_are_not_functional_propagation"] is True


def test_all_descending_targets_have_enumerated_bilateral_shortest_paths() -> None:
    report = json.loads(REPORT.read_text())
    expected = {
        "DNa02": {"L": (1, 12), "R": (1, 12)},
        "DNa01": {"L": (2, 545), "R": (2, 492)},
        "DNp20": {"L": (2, 7), "R": (3, 5831)},
    }
    for target_type, side_expected in expected.items():
        records = report["target_shortest_paths"][target_type]
        actual = {
            item["target"]["soma_side"]: (
                item["shortest_hops"],
                item["shortest_path_count"],
            )
            for item in records
        }
        assert actual == side_expected
        assert all(item["all_shortest_paths_enumerated"] for item in records)
        assert all(len(item["all_shortest_paths_sha256"]) == 64 for item in records)
        assert all(item["unique_shortest_path_dag_edges"] for item in records)
        assert all(item["stored_example_count"] <= 8 for item in records)
        assert report["population_reachability"][target_type][
            "bilateral_reachability_passed"
        ] is True
    assert report["structural_bilateral_path_gate_passed"] is True
    assert report["authorize_fixed_input_functional_propagation_precheck"] is True


def test_action_equivalence_is_not_mislabeled_as_neural_readout_equivalence() -> None:
    report = json.loads(REPORT.read_text())
    boundary = report["algebraic_action_readout_boundary"]
    assert boundary["reported_action_equivalence"] is True
    assert boundary["arms_consume_real_DNa01_DNa02_DNp20_states"] is False
    assert boundary["equivalence_is_formula_level_not_neural_state_level"] is True
    assert boundary["real_neural_state_readout_performed"] is False
    assert boundary["neural_readout_equivalence_established"] is False
    assert report["functional_propagation_evaluated"] is False
    assert report["functional_neural_readout_validated"] is False
    assert report["advance_to_navigation_release"] is False
    assert report["advance_to_mushroom_body"] is False
