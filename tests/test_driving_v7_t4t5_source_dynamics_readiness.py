import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4t5-source-dynamics-readiness.json"


def test_t4t5_source_dynamics_readiness_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["functional_stimulus_evaluated"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_t4t5_source_dynamics_gate_stops_new_candidates_and_downstream_work() -> None:
    report = json.loads(REPORT.read_text())
    assert report["passing_gates"] == [
        "T4_crossfit_structure_axis",
        "T5_crossfit_structure_axis",
        "shared_physical_v7_timebase",
    ]
    assert set(report["failing_gates"]) == {
        "T4_source_dynamics_transfer",
        "T5_source_dynamics_transfer",
        "independent_dynamic_validation",
    }
    assert report["FlyVis_time_constants_transferable"] is False
    assert report["T4_T5_source_dynamics_ready"] is False
    assert report["authorize_new_T4_or_T5_functional_candidate"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
