import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-c3-citation-graph-audit.json"


def test_C3_citation_graph_is_hash_and_revision_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["repository"]["local_and_remote_head"] == (
        "7fa5829e37d566e02beaaa87efd6a0f1de4e48c0"
    )


def test_bounded_graph_size_and_unresolved_ID_boundary_are_explicit() -> None:
    graph = json.loads(REPORT.read_text())["citation_graph"]
    assert graph == {
        "root_count": 2,
        "root_reference_ID_count": 134,
        "resolved_reference_count": 132,
        "unresolved_reference_ID_count": 2,
        "citing_record_count": 110,
        "unique_citing_work_count": 110,
        "union_unique_work_count": 240,
    }


def test_Pang_candidate_and_Dryad_are_L1_L2_not_C3() -> None:
    report = json.loads(REPORT.read_text())
    candidate = report["high_relevance_candidate"]
    assert candidate["final_doi"] == "10.1016/j.cub.2024.11.064"
    assert candidate["measurement_modality"] == "two_photon_ASAP2f_voltage_imaging"
    assert candidate["directly_recorded_neuron_types"] == ["L1", "L2"]
    assert candidate["full_text_source_term_hits"]["C3"] == 0
    assert candidate["C3_direct_recording_found"] is False
    dryad = report["Dryad"]
    assert dryad["version_number"] == 3
    assert dryad["file_count"] == 75
    assert dryad["total_declared_bytes"] == 49_600_989_798
    assert dryad["all_files_have_SHA256"] is True
    assert dryad["filename_hits"]["C3"] == []
    assert dryad["filename_hits"]["L1"]
    assert dryad["filename_hits"]["L2"]
    assert dryad["bulk_download_performed"] is False
    assert dryad["bulk_download_authorized_for_C3_audit"] is False


def test_citation_graph_does_not_change_C3_or_downstream_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["new_independent_C3_direct_recording_candidate_found"] is False
    assert report["authorize_C3_source_dynamics_transfer"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["boundary"]["citation_graph_is_bounded_not_global_literature_exhaustion"] is True
