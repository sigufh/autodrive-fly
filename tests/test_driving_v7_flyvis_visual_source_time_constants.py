import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-flyvis-visual-source-time-constants.json"


def test_flyvis_visual_source_time_constants_are_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["repository_commit"] == (
        "92b3845cc426dd309a1a0e1b3890156c42e14021"
    )
    assert report["protocol"]["checkpoint_pickle_executed"] is False
    assert report["training_contract"]["model_count"] == 50
    assert report["training_contract"]["cell_type_count"] == 65


def test_flyvis_visual_source_time_constants_do_not_authorize_transfer() -> None:
    report = json.loads(REPORT.read_text())
    assert all(not item["missing"] for item in report["source_coverage"].values())
    assert set(report["source_time_constants"]) == {
        "C3",
        "CT1(Lo1)",
        "Mi1",
        "Mi4",
        "Tm1",
        "Tm2",
        "Tm3",
        "Tm4",
        "Tm9",
    }
    assert report["gates"]["every_source_in_connectome"] is True
    assert report["gates"]["source_physiology_supervision_available"] is False
    assert report["gates"]["MaleCNS_topology_and_body_mapping_available"] is False
    assert report["gates"]["independent_dynamic_validation_available"] is False
    assert report["FlyVis_visual_source_time_constants_transferable"] is False
    assert report["authorize_source_dynamics_response_audit"] is False
    assert report["advance_to_T4_or_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
