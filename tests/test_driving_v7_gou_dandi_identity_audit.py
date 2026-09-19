import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-gou-dandi-identity-audit.json"


def test_gou_dandi_identity_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_all_assets_have_unique_dandi_subject_and_session_metadata() -> None:
    index = json.loads(REPORT.read_text())["asset_index"]
    assert index["asset_count"] == 282
    assert index["manifest_asset_count"] == 282
    assert index["JSON_and_YAML_asset_ids_match"] is True
    assert index["JSON_and_YAML_paths_and_sizes_match"] is True
    assert index["unique_asset_path_count"] == 282
    assert index["unique_subject_id_count"] == 282
    assert index["unique_session_start_count"] == 282
    assert index["one_unique_subject_per_asset"] is True
    assert index["one_unique_session_per_asset"] is True
    assert index["participant_fields"] == ["age", "identifier", "schemaKey", "sex", "species"]
    assert index["session_fields"] == ["description", "name", "schemaKey", "startDate"]
    assert index["genotype_field_present"] is False
    assert index["cell_type_field_present"] is False


def test_no_explicit_dryad_to_dandi_identity_crosswalk_is_claimed() -> None:
    report = json.loads(REPORT.read_text())
    labels = report["Dryad_identity_labels"]
    assert labels["distinct_fliesUsed_label_count"] == 66
    assert labels["exact_token_matches_in_DANDI_paths_or_participant_fields"] == {}
    assert labels["exact_match_count"] == 0
    representative = report["representative_NWB_range_inspection"]
    assert representative["subject"]["subject_id"] == "10176231931525660316"
    assert representative["imaging_plane"]["imaging_rate_hz"] == 13.0
    assert representative["imaging_plane"]["indicator"] == "GC6f"
    assert representative["imaging_plane"]["location"] == "T4T5"
    assert representative["scratch_group_keys"] == []
    assert representative["processing_group_keys"] == []
    conclusions = report["identity_conclusions"]
    assert conclusions["DANDI_asset_level_stable_participant_IDs_available"] is True
    assert conclusions["representative_NWB_original_Dryad_fly_ID_available"] is False
    assert conclusions["Dryad_fliesUsed_to_DANDI_subject_crosswalk_verified"] is False
    assert conclusions["Dryad_processed_rows_have_stable_biological_IDs"] is False
    assert conclusions["fit_held_out_biological_ID_disjointness_authorized"] is False
    assert report["authorize_source_dynamics_transfer"] is False


def test_public_repository_history_contains_no_conversion_crosswalk() -> None:
    report = json.loads(REPORT.read_text())
    history = report["public_repository_history"]
    assert history["repository"] == "dandisets/001205"
    assert history["combined_distinct_commit_count_in_API_snapshots"] == 20
    assert history["draft_tree_truncated"] is False
    assert history["release_tree_truncated"] is False
    assert history["git_annex_tree_truncated"] is False
    assert history["draft_and_release_path_sets_equal"] is True
    assert history["release_NWB_pointer_count"] == 282
    assert history["release_code_file_count"] == 0
    assert history["git_annex_content_log_count"] == 286
    assert history["git_annex_web_log_count"] == 282
    assert history["git_annex_code_file_count"] == 0
    assert history["conversion_script_or_crosswalk_found"] is False
    organization = report["ClarkLabCode_public_repository_index"]
    assert organization["repository_count"] == 24
    assert organization["name_or_description_matches_for_bounded_terms"] == []
    assert organization["conversion_or_crosswalk_repository_identified"] is False
    assert report["boundary"][
        "repository_history_conclusions_are_limited_to_public_DANDI_and_ClarkLabCode_snapshots"
    ] is True
