import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-hao-asap7y-candidate-audit.json"


def test_ASAP7y_candidate_is_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["raw_snapshot_identity"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["paper"]["doi"] == "10.64898/2026.05.27.728040"


def test_ASAP7y_is_real_fly_voltage_evidence_but_cell_types_are_unresolved() -> None:
    report = json.loads(REPORT.read_text())
    scope = report["verified_scope"]
    assert scope["Drosophila_in_vivo_voltage_imaging"] is True
    assert scope["measurement_modality"] == "two_photon_ASAP7y_voltage_imaging"
    assert scope["millisecond_subcellular_subthreshold_resolution"] is True
    assert scope["visual_system_model_cell_type_count"] == 717
    assert scope["cites_Groschner_2022"] is True
    cell_types = report["cell_type_resolution"]
    assert cell_types["full_text_retrieved"] is False
    assert cell_types["public_author_figure_named_examples"] == ["Dm9", "MeLo13"]
    assert cell_types["complete_experimental_cell_type_set_resolved"] is False
    assert cell_types["experimental_Drosophila_cell_types_named_in_paper_sources"] == []
    assert cell_types["Mi4_direct_recording_verified"] is False
    assert cell_types["C3_direct_recording_verified"] is False
    assert cell_types["candidate_classification"] == "unresolved_high_value_candidate"
    locator = report["locator_evidence"]
    assert locator["author_handle"] == "michaelzlin.bsky.social"
    assert locator["post_11_visual_labels"] == ["Dm9", "MeLo13"]
    assert locator["counts_as_numeric_payload"] is False
    assert locator["resolves_complete_paper_cell_type_set"] is False


def test_author_dissertation_has_a_bounded_access_date_not_a_cell_type_resolution() -> None:
    report = json.loads(REPORT.read_text())
    dissertation = report["author_dissertation"]
    assert dissertation["purl"] == "https://purl.stanford.edu/cg111nm7996"
    assert dissertation["title"] == (
        "Voltage imaging for revealing neuronal dynamics across scales"
    )
    assert dissertation["fulltext_filename"] == (
        "PhDThesis_YKH_final-augmented.pdf"
    )
    assert dissertation["fulltext_file_count"] == 1
    assert dissertation["access_probe_status"] == 401
    assert dissertation["restricted_until"] == "2027-03-14"
    assert dissertation["public_abstract_confirms_fly_visual_dendritic_voltage"] is True
    assert dissertation["public_abstract_names_experimental_cell_types"] is False
    assert dissertation["fulltext_retrieved"] is False
    assert dissertation["same_experimental_cohort_as_preprint_verified"] is False
    assert dissertation["resolves_complete_preprint_cell_type_set"] is False


def test_successful_public_indexes_do_not_overclaim_global_payload_absence() -> None:
    indexes = json.loads(REPORT.read_text())["public_repository_indexes"]
    assert indexes == {
        "openRxiv_MECA_location_found": False,
        "GitHub_ASAP7y_repository_count": 0,
        "GitHub_exact_title_repository_count": 0,
        "DataCite_DOI_related_object_count": 0,
        "Dryad_DOI_dataset_count": 0,
        "Zenodo_ASAP7y_record_count": 0,
        "ClandininLab_public_repository_count": 31,
        "ClandininLab_paper_specific_name_or_description_hits": [],
        "ClandininLab_listing_resolves_paper_payload": False,
        "successful_index_numeric_payload_found": False,
        "global_payload_absence_claimed": False,
    }


def test_wayback_capture_is_abstract_only_and_does_not_resolve_cell_types() -> None:
    archived = json.loads(REPORT.read_text())["archived_biorxiv_page"]
    assert archived["capture_status"] == 200
    assert archived["capture_mimetype"] == "text/html"
    assert archived["captured_view"] == "abstract_only"
    assert archived["required_source_term_hits"] == {"Mi4": False, "C3": False}
    assert archived["linked_routes_present"] == {
        "full_text": True,
        "PDF": True,
        "supplementary_material": True,
    }
    assert archived["linked_route_contents_retrieved"] is False
    assert archived["resolves_complete_paper_cell_type_set"] is False


def test_official_tdm_bucket_requires_authenticated_requester_pays_access() -> None:
    access = json.loads(REPORT.read_text())["access_boundaries"]
    assert access["bioRxiv_TDM_bucket"] == "biorxiv-src-monthly"
    assert access["bioRxiv_TDM_probe_status"] == 403
    assert access["bioRxiv_TDM_requester_pays_authentication_required"] is True
    assert access["bioRxiv_TDM_payload_inventory_readable"] is False


def test_europe_pmc_annotations_are_abstract_only_locator_evidence() -> None:
    text_mining = json.loads(REPORT.read_text())["Europe_PMC_text_mining"]
    assert text_mining["article_id"] == "PPR:PPR1242681"
    assert text_mining["annotation_count"] == 6
    assert text_mining["exact_terms"] == [
        "membranes",
        "organization",
        "mice",
        "flies",
        "photon",
        "Drosophila",
    ]
    assert text_mining["Mi4_annotation_hit"] is False
    assert text_mining["C3_annotation_hit"] is False
    assert text_mining["annotation_scope"] == "abstract_only_because_in_PMC_is_false"
    assert text_mining["resolves_complete_experimental_cell_type_set"] is False


def test_unresolved_candidate_does_not_change_transfer_or_downstream_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["public_numeric_payload_verified"] is False
    assert report["authorize_Mi4_C3_candidate_classification"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["advance_to_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_runtime_integration"] is False
    assert report["boundary"][
        "high_value_candidate_kept_unresolved_until_cell_types_are_verified"
    ] is True
    assert report["boundary"][
        "author_dissertation_abstract_does_not_substitute_for_restricted_fulltext"
    ] is True
