import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t4-source-pool-camera-frame-discovery.json"


def test_camera_frame_discovery_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "LPLC-T01"
    assert report["protocol"]["source_coordinate_frame"] == "camera_image_xy"
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_camera_vertical_axis_discovers_only_bilateral_t4d() -> None:
    report = json.loads(REPORT.read_text())
    assert report["direction_pass_counts"]["pairwise_distal_vector"] == 2
    assert report["bilateral_passing_subtypes"]["pairwise_distal_vector"] == [
        "d"
    ]
    assert report["discovered_bilateral_readouts"] == [
        "pairwise_distal_vector"
    ]
    assert report["discovered_bilateral_subtypes"] == {
        "pairwise_distal_vector": ["d"]
    }
    assert report["post_hoc_candidate_discovered"] is True
    assert report["population_scores"]["T4d_L"][
        "pairwise_distal_vector"
    ]["passed"] is True
    assert report["population_scores"]["T4d_R"][
        "pairwise_distal_vector"
    ]["passed"] is True


def test_post_hoc_discovery_does_not_authorize_or_advance() -> None:
    report = json.loads(REPORT.read_text())
    assert report["independent_replication_preregistered"] is False
    assert report["independent_replication_evaluated"] is False
    assert report["authorize_new_target_formula"] is False
    assert report["advance_to_T4_calibration"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["boundary"]["candidate_selected_after_observing_discovery_output"] is True
