import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-flyvis-c3-time-constant-audit.json"


def test_flyvis_audit_is_commit_archive_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["repository_commit"] == (
        "92b3845cc426dd309a1a0e1b3890156c42e14021"
    )
    archive = report["protocol"]["pretrained_archive"]
    assert archive["google_drive_file_id"] == "13cJr2nMn89j-jBAd5RduYRJpBcXwoNrC"
    assert archive["bytes"] == 3_417_042
    assert archive["sha256"] == (
        "71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1"
    )
    assert archive["entry_count"] == 353
    assert archive["integrity_verified"] is True
    assert report["protocol"]["checkpoint_pickle_executed"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_flyvis_C3_time_constant_is_not_identified_across_ensemble() -> None:
    report = json.loads(REPORT.read_text())
    training = report["training_contract"]
    assert training["task"] == "Sintel_optic_flow"
    assert training["dataset_dt_seconds"] == 0.02
    assert training["model_count"] == 50
    assert training["cell_type_count"] == 65
    c3 = report["source_time_constants"]["C3"]
    assert c3["models_at_or_below_solver_dt"] == 21
    assert c3["fraction_at_or_below_solver_dt"] == 0.42
    assert 0.0195 < c3["minimum_seconds"] < 0.0197
    assert 0.066 < c3["median_seconds"] < 0.068
    assert 0.54 < c3["maximum_seconds"] < 0.55
    assert report["C3_identifiability"]["all_models_above_solver_dt"] is False


def test_flyvis_C3_parameter_does_not_authorize_MaleCNS_transfer() -> None:
    report = json.loads(REPORT.read_text())
    gates = report["C3_identifiability"]
    assert gates["C3_physiology_supervision_available"] is False
    assert gates["independent_C3_holdout_available"] is False
    assert gates["MaleCNS_topology_used"] is False
    assert gates["stable_FlyVis_type_to_MaleCNS_body_mapping_available"] is False
    assert report["C3_time_constant_transfer_authorized"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["boundary"]["trained_on_optic_flow_not_C3_physiology"] is True
    assert report["boundary"]["uses_FIB25_FIB19_average_filter_connectome_not_MaleCNS"] is True
    assert report["boundary"]["T5_and_LPLC_remain_frozen"] is True
