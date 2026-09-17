import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-lc4-input-speed-precheck.json"


def test_lc4_input_speed_precheck_is_frozen_and_non_advancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "LPLC-T01"
    assert report["protocol"]["position_count"] == 15
    assert report["protocol"]["duration_frames"] == [32, 16, 8]
    assert report["protocol"]["source_pool"] == "all_direct_non_LC4_presynaptic_nodes"
    assert report["protocol"]["LC4_recurrent_sources_excluded"] is True
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["calibration_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_lc4_direct_input_lacks_full_denominator_positive_speed_code() -> None:
    report = json.loads(REPORT.read_text())
    derivative = report["readouts"]["peak_positive_input_derivative"]
    assert derivative["L"]["valid_cell_count"] == 1
    assert derivative["R"]["valid_cell_count"] == 3
    assert derivative["L"]["positive_slope_fraction_all_cells"] == 1 / 71
    assert derivative["R"]["positive_slope_fraction_all_cells"] == 3 / 55
    assert derivative["L"]["passed"] is False
    assert derivative["R"]["passed"] is False
    peak = report["readouts"]["input_peak"]
    assert peak["L"]["median_response_by_speed"][0] > peak["L"][
        "median_response_by_speed"
    ][-1]
    assert peak["R"]["median_response_by_speed"][0] > peak["R"][
        "median_response_by_speed"
    ][-1]
    assert report["mirror"]["peak_positive_input_derivative"]["passed"] is True
    excitatory = report["readouts"]["peak_positive_excitatory_input_derivative"]
    assert excitatory["L"]["valid_cell_count"] == 0
    assert excitatory["R"]["valid_cell_count"] == 0
    inhibitory = report["readouts"]["peak_positive_inhibitory_input_derivative"]
    assert inhibitory["L"]["monotonic_cell_count"] == 63
    assert inhibitory["R"]["monotonic_cell_count"] == 47
    assert inhibitory["L"]["valid_cell_count"] == 0
    assert inhibitory["R"]["valid_cell_count"] == 0
    assert report["LC4_input_speed_precheck_passed"] is False
    assert report["expand_to_three_conditions"] is False
    assert report["advance_to_calibration"] is False
    assert report["advance_to_runtime_integration"] is False


def test_lc4_direct_input_structure_is_distinct_from_lplc2() -> None:
    report = json.loads(REPORT.read_text())
    for side in "LR":
        rows = report["direct_input_structure"][side]["source_types_by_synapse_weight"]
        top_five = [item["type"] for item in rows[:5]]
        assert top_five == ["T2", "TmY3", "Tm4", "Tm2", "Tm3"]
        target_count = report["direct_input_structure"][side]["target_count"]
        assert all(item["target_count"] == target_count for item in rows[:5])
