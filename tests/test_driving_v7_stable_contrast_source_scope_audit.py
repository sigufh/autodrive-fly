import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-stable-contrast-source-scope-audit.json"


def test_stable_contrast_scope_is_hash_and_revision_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["repository"]["local_and_remote_head"] == (
        "529eefa8399bc849c0bfe8001bcaccf4ca72ec65"
    )


def test_paper_records_T5_pathway_calcium_not_Mi4_C3_physiology() -> None:
    report = json.loads(REPORT.read_text())
    assert report["paper"]["doi"] == "10.1038/s41467-024-52724-5"
    assert report["paper_source_term_hits"]["Mi4"] == 0
    assert report["paper_source_term_hits"]["C3"] == 0
    assert {"Tm1", "Tm2", "Tm4", "Tm9", "Dm12"} <= set(
        report["directly_recorded_neuron_types"]
    )
    assert report["measurement_modalities"] == [
        "GCaMP6f_calcium",
        "iGluSnFR_glutamate",
    ]
    assert report["Mi4_direct_physiology_found"] is False
    assert report["C3_direct_physiology_found"] is False


def test_Mi4_C3_Zenodo_files_are_anatomical_proofreading_tables() -> None:
    report = json.loads(REPORT.read_text())
    assert report["Zenodo"]["processed_ZIP_entry_count"] == 635
    assert report["Zenodo"]["full_processed_archive_downloaded"] is False
    assert report["Zenodo"]["full_raw_archive_downloaded"] is False
    assert report["proofreading_workbooks"]["Mi4"]["row_count"] == 723
    assert report["proofreading_workbooks"]["C3"]["row_count"] == 678
    for inventory in report["proofreading_workbooks"].values():
        assert inventory["hemispheres"] == ["R"]
        assert inventory["proofreading_statuses"] == ["Y"]
        assert inventory["time_field_found"] is False
        assert inventory["response_field_found"] is False
        assert inventory["stimulus_field_found"] is False
        assert inventory["recording_term_found_in_values"] is False
    assert report["Mi4_C3_files_are_connectome_proofreading_only"] is True
    assert report["external_recording_to_MaleCNS_crosswalk_found"] is False


def test_stable_contrast_evidence_does_not_change_transfer_or_downstream_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["authorize_Mi4_C3_source_dynamics_transfer"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["boundary"][
        "T5_source_calcium_does_not_substitute_for_T4_inhibitory_sources"
    ] is True
