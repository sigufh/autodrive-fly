import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-axolotl-availability-audit.json"


def test_axolotl_availability_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_related_project_is_found_but_repository_content_is_not_publicly_readable() -> None:
    report = json.loads(REPORT.read_text())
    project = report["related_project"]
    assert project["id"] == 4_848_981
    assert project["path"] == "rbehnialab/axolotl"
    assert project["visibility"] == "public"
    assert project["page_marks_project_nonempty"] is True
    assert project["graphql_repository"] is None
    assert project["endpoint_statuses"]["branches"] == 404
    assert project["endpoint_statuses"]["tree"] == 403
    assert project["public_fork_count"] == 0
    assert report["source_global_absence_claimed"] is False


def test_public_history_and_package_indexes_do_not_supply_the_source() -> None:
    report = json.loads(REPORT.read_text())
    history = report["flexible_filtering_history"]
    assert history["branch_tree_entry_counts"] == {
        "master_tree": 95,
        "dev_tree": 95,
    }
    assert history["commit_count"] == 28
    assert history["commit_tree_count"] == 28
    assert history["unique_blob_count"] == 83
    assert history["unique_text_blob_count"] == 33
    assert history["Figure6_import_occurrence_count"] == 5
    assert set(history["source_definition_hits"].values()) == {False}
    packages = report["package_indexes"]
    assert packages["PyPI_first_upload_year"] == 2024
    assert packages["same_name_packages_are_unrelated_LLM_software"] is True
    assert report["wayback_snapshot_count"] == 0
    assert report["gates"]["Figure6_axolotl_source_recovered"] is False
    assert report["authorize_Tm_to_T5_model_transfer_to_v7"] is False
    assert report["authorize_T5_functional_precheck"] is False
